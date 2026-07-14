"""Distill an MPC score policy for score-space PPS steering.

This is the score-space replacement for the old policy distillation path.  The
reference target is not a pi0/pi05 teacher action.  Instead, labels are generated
by querying the same FK/cost MPC score estimator used at evaluation time:

    s_ref(x_t, o, t) = MPC.estimate_mbd_score_action_prox(x_t, o, context, t)

For each cached observation, generation starts at Gaussian action noise and
follows the complete online ``mbd_score_action_prox`` reverse trajectory.  Each
visited state is paired with its MPC score target.  Expert actions are not used
to construct the diffusion states.

The script has two explicit stages so expensive MPC labels can be inspected and
reused:

  python scripts/train_mpc_proxy_score_pytorch.py generate-cache \
      --config score_ref_weight \
      --hdf5_path /home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5 \
      --cache_path ../data/weight/ref_action_prox_reverse_512x8_n0.8.npz

  python scripts/train_mpc_proxy_score_pytorch.py train \
      --config score_ref_weight \
      --hdf5_path /home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5 \
      --cache_path ../data/weight/ref_action_prox_reverse_512x8_n0.8.npz \
      --exp_name ref --overwrite
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import random
import shutil
import sys
import time
from typing import Any

_OPENPI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_OPENPI_DIR)
_OPENPI_SRC_DIR = os.path.join(_OPENPI_DIR, "src")
for _path in (_REPO_DIR, _OPENPI_SRC_DIR, os.path.dirname(os.path.abspath(__file__))):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import h5py
import jax
import numpy as np
import safetensors.torch
import torch
import torch.utils.data
import tqdm
import wandb

from openpi import transforms as _transforms
from openpi.models import model as _model
import openpi.models.proxy_score_config
import openpi.models_pytorch.proxy_score_pytorch as _proxy_score
import openpi.training.config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.shared import normalize as _normalize
from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig
from sim_free_mpc.ddim import ddim_iteration_alphas

from train_proxy_score_pytorch import (
    cleanup_ddp,
    ensure_tensor_loss,
    get_model_parameters,
    init_logging,
    init_wandb,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    setup_ddp,
)


DEFAULT_BASE_CONFIG = "pi05_droid_jointpos"
DEFAULT_BASE_CHECKPOINT_DIR = os.path.join(
    _OPENPI_DIR,
    "checkpoints",
    "pytorch",
    "pi05_droid_jointpos",
)
DEFAULT_PROMPT = "put pear and apple on the scale"
CACHE_FORMAT_VERSION = 3
CACHE_LABEL_TYPE = "mpc_score_action_prox_reverse_trajectory"
CACHE_STATE_SOURCE = "action_prox_reverse_trajectory_from_gaussian"
ACTION_PROX_NOISE_SCHEDULE = "mpc_noise_times_sqrt_one_minus_alpha_bar"


def _norm_stats_fingerprint(norm_stats: dict[str, Any] | None) -> str | None:
    if norm_stats is None:
        return None
    digest = hashlib.sha256()
    for key in sorted(norm_stats):
        digest.update(key.encode("utf-8"))
        stat = norm_stats[key]
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stat, field, None)
            if value is None:
                digest.update(f"{field}:None".encode("utf-8"))
                continue
            arr = np.asarray(value)
            digest.update(field.encode("utf-8"))
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _build_data_pipeline(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None and getattr(config.data, "norm_stats_dir", None):
        norm_stats_dir = pathlib.Path(getattr(config.data, "norm_stats_dir"))
        candidates = [
            norm_stats_dir,
            pathlib.Path(_OPENPI_DIR) / norm_stats_dir,
            pathlib.Path(_REPO_DIR) / norm_stats_dir,
        ]
        for candidate in candidates:
            if (candidate / "norm_stats.json").exists():
                data_config = dataclasses.replace(
                    data_config,
                    norm_stats=_normalize.load(candidate),
                )
                break
    input_transform = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    return data_config, input_transform


def _demo_sample(
    demo: h5py.Group,
    step_idx: int,
    *,
    action_horizon: int,
    prompt: str,
) -> dict[str, Any]:
    action_start = step_idx + 1
    action_end = action_start + action_horizon
    return {
        "exterior_image_1_left": demo["obs/table_cam"][step_idx],
        "wrist_image_left": demo["obs/wrist_cam"][step_idx],
        "joint_position": np.asarray(demo["obs/joint_pos"][step_idx][:7], dtype=np.float32),
        "gripper_position": np.asarray(demo["obs/gripper_pos"][step_idx][:1], dtype=np.float32),
        "actions": np.asarray(demo["obs/joint_actions"][action_start:action_end], dtype=np.float32),
        "prompt": prompt,
    }


class _ActionDecoder:
    """Minimal policy interface needed by sim-free MPC action decoding."""

    def __init__(self, norm_stats: dict[str, Any], *, use_quantile_norm: bool):
        self._metadata = {
            "output_norm_stats": {
                key: value for key, value in norm_stats.items() if key in ("state", "actions")
            },
            "use_quantile_norm": use_quantile_norm,
        }


def _torch_inputs(input_transform, sample: dict[str, Any], device: torch.device):
    inputs = input_transform(jax.tree.map(lambda x: x, sample))
    return jax.tree.map(
        lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...],
        inputs,
    )


def _pose(group: h5py.Group, path: str, step_idx: int) -> np.ndarray | None:
    if path not in group:
        return None
    arr = np.asarray(group[path][step_idx], dtype=np.float32)
    return arr


def _object_dict(demo: h5py.Group, step_idx: int) -> dict[str, dict[str, np.ndarray]]:
    objects = {}
    root = demo.get("states/rigid_object")
    if root is None:
        return objects
    for name in root.keys():
        path = f"states/rigid_object/{name}/root_pose"
        pose = _pose(demo, path, step_idx)
        if pose is None or pose.shape[-1] < 7:
            continue
        objects[name] = {
            "pos": pose[:3],
            "quat": pose[3:7],
        }
    return objects


def _heuristic_weight_subtasks(objects: dict[str, dict[str, np.ndarray]], eef_pos: np.ndarray) -> dict[str, bool]:
    pear = objects.get("pear", {}).get("pos")
    apple = objects.get("apple", {}).get("pos")
    scale = objects.get("scale", {}).get("pos")
    subtasks: dict[str, bool] = {}
    if pear is not None and scale is not None:
        pear_xy_scale = float(np.linalg.norm(pear[:2] - scale[:2]))
        subtasks["pear_on_scale"] = bool(pear_xy_scale < 0.09 and pear[2] > scale[2] + 0.04)
        pear_near_ee = float(np.linalg.norm(pear - eef_pos[:3])) < 0.10
        pear_lifted = pear[2] > scale[2] + 0.08
        subtasks["grasp_pear"] = bool(pear_near_ee or pear_lifted or subtasks["pear_on_scale"])
    if apple is not None:
        apple_near_ee = float(np.linalg.norm(apple - eef_pos[:3])) < 0.10
        apple_lifted = scale is not None and apple[2] > scale[2] + 0.08
        subtasks["grasp_apple"] = bool(subtasks.get("pear_on_scale", False) and (apple_near_ee or apple_lifted))
    return subtasks


def _mpc_context(
    demo: h5py.Group,
    step_idx: int,
    *,
    task: str,
    subtask_mode: str,
) -> dict[str, Any]:
    robot_pose = _pose(demo, "states/articulation/robot/root_pose", step_idx)
    eef_pos = np.asarray(demo["obs/eef_pos"][step_idx], dtype=np.float32)
    objects = _object_dict(demo, step_idx)
    subtasks = (
        _heuristic_weight_subtasks(objects, eef_pos)
        if subtask_mode == "heuristic"
        else {}
    )
    context = {
        "task": task,
        "subtasks": subtasks,
        "joint_pos": np.asarray(demo["obs/joint_pos"][step_idx], dtype=np.float32),
        "joint_vel": np.asarray(demo["obs/joint_vel"][step_idx], dtype=np.float32)
        if "obs/joint_vel" in demo
        else None,
        "eef_pos": eef_pos,
        "eef_quat": np.asarray(demo["obs/eef_quat"][step_idx], dtype=np.float32)
        if "obs/eef_quat" in demo
        else None,
        "gripper_pos": np.asarray(demo["obs/gripper_pos"][step_idx], dtype=np.float32),
        "robot_root_pos": robot_pose[:3] if robot_pose is not None else None,
        "robot_root_quat": robot_pose[3:7] if robot_pose is not None else None,
        "objects": objects,
    }
    return {key: value for key, value in context.items() if value is not None}


def _time_from_iteration(
    *,
    iteration: int,
    num_iterations: int,
    num_train_timesteps: int,
) -> float:
    ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=num_iterations,
        num_train_timesteps=num_train_timesteps,
    )
    step_ratio = int(num_train_timesteps) // int(num_iterations)
    timestep = int((int(num_iterations) - 1 - int(iteration)) * step_ratio)
    return timestep / max(float(num_train_timesteps - 1), 1.0)


def _sample_indices(
    hdf5_path: pathlib.Path,
    *,
    action_horizon: int,
    max_trajectories: int | None,
    stride: int,
    seed: int,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    with h5py.File(hdf5_path, "r") as f:
        all_indices = []
        for demo_name in sorted(f["data"].keys()):
            demo = f["data"][demo_name]
            length = len(demo["obs/joint_actions"])
            num_windows = length - action_horizon
            if num_windows <= 0:
                continue
            all_indices.extend((demo_name, step) for step in range(0, num_windows, stride))
    rng.shuffle(all_indices)
    if max_trajectories is not None:
        all_indices = all_indices[:max_trajectories]
    return all_indices


def generate_cache(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config)
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    base_config = _config.get_config(args.base_config)
    base_data_config = base_config.data.create(base_config.assets_dirs, base_config.model)
    if base_data_config.asset_id is None:
        raise ValueError(f"Base config {args.base_config!r} has no norm-stat asset_id.")
    base_norm_stats = _checkpoints.load_norm_stats(
        pathlib.Path(args.base_checkpoint_dir) / "assets",
        base_data_config.asset_id,
    )
    score_data_config, input_transform = _build_data_pipeline(config)
    if _norm_stats_fingerprint(base_norm_stats) != _norm_stats_fingerprint(score_data_config.norm_stats):
        raise ValueError(
            "Base policy and score config norm_stats differ. Use one shared norm_stats source "
            "before generating MPC score labels."
        )
    if base_data_config.use_quantile_norm != score_data_config.use_quantile_norm:
        raise ValueError("Base policy and score config use different normalization modes.")
    action_decoder = _ActionDecoder(
        base_norm_stats,
        use_quantile_norm=base_data_config.use_quantile_norm,
    )

    planner = SimFreeMPC(
        action_decoder,
        SimFreeMPCConfig(
            task_name=args.task,
            num_samples=args.mpc_num_samples,
            iterations=args.mpc_iterations,
            noise=args.mpc_noise,
            temperature=args.mpc_temperature,
            beta_opt_iter=args.mpc_beta_opt_iter,
            beta_horizon=args.mpc_beta_horizon,
            action_dims=config.model.action_dim,
            joint_delta_clip=args.mpc_joint_delta_clip,
            cost_style=args.mpc_cost,
            optimize_space="action",
            ddim_num_train_timesteps=config.model.ddim_num_train_timesteps,
            interpolate=args.mpc_interpolate,
            control_frequency=args.control_frequency,
            interpolate_frequency=args.interpolate_frequency,
        ),
    )

    indices = _sample_indices(
        pathlib.Path(args.hdf5_path),
        action_horizon=config.model.action_horizon,
        max_trajectories=args.max_trajectories,
        stride=args.stride,
        seed=args.seed,
    )
    if not indices:
        raise ValueError("No valid HDF5 windows found for MPC score label generation.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    demo_names: list[str] = []
    step_indices: list[int] = []
    trajectory_ids: list[int] = []
    iterations: list[int] = []
    times: list[float] = []
    diffusion_states: list[np.ndarray] = []
    target_scores: list[np.ndarray] = []
    min_costs: list[float] = []
    score_norms: list[float] = []

    num_iterations = int(args.num_steps) + 1
    if num_iterations > config.model.ddim_num_train_timesteps:
        raise ValueError("--num_steps + 1 must be <= ddim_num_train_timesteps.")

    with h5py.File(args.hdf5_path, "r") as f:
        pbar = tqdm.tqdm(indices, desc="MPC reverse trajectories")
        for trajectory_id, (demo_name, step_idx) in enumerate(pbar):
            demo = f["data"][demo_name]
            sample = _demo_sample(
                demo,
                step_idx,
                action_horizon=config.model.action_horizon,
                prompt=args.prompt,
            )
            model_inputs = _torch_inputs(input_transform, sample, device)
            x_t = torch.randn(
                (1, config.model.action_horizon, config.model.action_dim),
                device=device,
                dtype=torch.float32,
            )

            context = _mpc_context(
                demo,
                step_idx,
                task=args.task,
                subtask_mode=args.subtask_mode,
            )
            with torch.no_grad():
                for iteration in range(num_iterations):
                    score, diagnostics = planner.estimate_mbd_score_action_prox(
                        x_t,
                        model_inputs,
                        context,
                        iteration=iteration,
                        num_iterations=num_iterations,
                    )

                    demo_names.append(demo_name)
                    step_indices.append(int(step_idx))
                    trajectory_ids.append(trajectory_id)
                    iterations.append(iteration)
                    times.append(
                        _time_from_iteration(
                            iteration=iteration,
                            num_iterations=num_iterations,
                            num_train_timesteps=config.model.ddim_num_train_timesteps,
                        )
                    )
                    diffusion_states.append(x_t[0].detach().cpu().numpy().astype(np.float32))
                    target_scores.append(score[0].detach().cpu().numpy().astype(np.float32))
                    min_costs.append(float(diagnostics.get("cost_min", np.nan)))
                    score_norms.append(float(diagnostics.get("score_norm", np.nan)))

                    if iteration + 1 < num_iterations:
                        x_t = planner.step_from_score(
                            x_t,
                            score,
                            iteration=iteration,
                            num_iterations=num_iterations,
                            update_mode="mbd_score",
                            active_dims=config.model.action_dim,
                        )

            pbar.set_postfix(
                {
                    "labels": len(diffusion_states),
                    "cost_min": f"{min_costs[-1]:.3f}",
                    "score_norm": f"{score_norms[-1]:.2f}",
                }
            )

    metadata = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "label_type": CACHE_LABEL_TYPE,
        "state_source": CACHE_STATE_SOURCE,
        "initial_state_distribution": "standard_gaussian",
        "trajectory_update": "mbd_score",
        "num_trajectories": len(indices),
        "labels_per_trajectory": int(num_iterations),
        "config": args.config,
        "base_config": args.base_config,
        "base_checkpoint_dir": str(args.base_checkpoint_dir),
        "hdf5_path": str(args.hdf5_path),
        "task": args.task,
        "prompt": args.prompt,
        "num_steps": int(args.num_steps),
        "num_iterations": int(num_iterations),
        "ddim_num_train_timesteps": int(config.model.ddim_num_train_timesteps),
        "score_model_action_dim": int(config.model.action_dim),
        "score_model_action_horizon": int(config.model.action_horizon),
        "stored_action_dim": int(diffusion_states[0].shape[-1]) if diffusion_states else None,
        "norm_stats_fingerprint": _norm_stats_fingerprint(score_data_config.norm_stats),
        "use_quantile_norm": bool(score_data_config.use_quantile_norm),
        "subtask_mode": args.subtask_mode,
        "mpc": {
            "num_samples": int(args.mpc_num_samples),
            "iterations": int(args.mpc_iterations),
            "proposal_center": "current_noisy_action",
            "noise_schedule": ACTION_PROX_NOISE_SCHEDULE,
            "noise": float(args.mpc_noise),
            "temperature": float(args.mpc_temperature),
            "beta_opt_iter": float(args.mpc_beta_opt_iter),
            "beta_horizon": float(args.mpc_beta_horizon),
            "joint_delta_clip": float(args.mpc_joint_delta_clip),
            "cost_style": args.mpc_cost,
            "interpolate": bool(args.mpc_interpolate),
        },
    }
    cache_path = pathlib.Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        demo_name=np.asarray(demo_names),
        step_index=np.asarray(step_indices, dtype=np.int64),
        trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
        iteration=np.asarray(iterations, dtype=np.int64),
        time=np.asarray(times, dtype=np.float32),
        x_t=np.asarray(diffusion_states, dtype=np.float32),
        score=np.asarray(target_scores, dtype=np.float32),
        cost_min=np.asarray(min_costs, dtype=np.float32),
        score_norm=np.asarray(score_norms, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logging.info("Wrote %s MPC score labels to %s", len(demo_names), cache_path)


class MPCScoreDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        hdf5_path: str,
        cache_path: str,
        config: _config.TrainConfig,
        prompt: str,
    ):
        self.hdf5_path = hdf5_path
        self.cache = np.load(cache_path, allow_pickle=False)
        self.demo_names = self.cache["demo_name"].astype(str)
        self.step_indices = self.cache["step_index"].astype(np.int64)
        self.trajectory_ids = self.cache["trajectory_id"].astype(np.int64)
        self.iterations = self.cache["iteration"].astype(np.int64)
        self.diffusion_states = self.cache["x_t"].astype(np.float32)
        self.target_scores = self.cache["score"].astype(np.float32)
        self.times = self.cache["time"].astype(np.float32)
        self.prompt = prompt
        self.config = config
        self.data_config, self.input_transform = _build_data_pipeline(config)
        self._h5 = None

        metadata = json.loads(str(self.cache["metadata_json"].item()))
        self.metadata = metadata
        if int(metadata.get("cache_format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError(
                "MPC score cache format is stale. Regenerate it with the current action-prox sampler."
            )
        if metadata.get("label_type") != CACHE_LABEL_TYPE:
            raise ValueError("MPC score cache does not contain action-prox reverse trajectories.")
        if metadata.get("state_source") != CACHE_STATE_SOURCE:
            raise ValueError("MPC score cache x_t states were not sampled from base reverse denoising.")
        if metadata.get("initial_state_distribution") != "standard_gaussian":
            raise ValueError("MPC score cache trajectories do not start from standard Gaussian noise.")
        if metadata.get("trajectory_update") != "mbd_score":
            raise ValueError("MPC score cache does not use the online MBD reverse update.")
        mpc_metadata = metadata.get("mpc", {})
        if mpc_metadata.get("proposal_center") != "current_noisy_action":
            raise ValueError("MPC score cache does not use z_t as the action-prox proposal center.")
        if mpc_metadata.get("noise_schedule") != ACTION_PROX_NOISE_SCHEDULE:
            raise ValueError(
                "MPC score cache does not use mpc_noise * sqrt(1 - alpha_bar) proposals."
            )
        proposal_noise = float(mpc_metadata.get("noise", float("nan")))
        if not np.isfinite(proposal_noise) or proposal_noise < 0.0:
            raise ValueError("MPC score cache has an invalid action-prox noise coefficient.")
        expected_fingerprint = _norm_stats_fingerprint(self.data_config.norm_stats)
        if metadata.get("norm_stats_fingerprint") != expected_fingerprint:
            raise ValueError(
                "MPC score cache norm_stats fingerprint does not match this training config. "
                "Regenerate the cache with the shared base/task/ref norm_stats."
            )
        if int(metadata.get("ddim_num_train_timesteps", -1)) != int(config.model.ddim_num_train_timesteps):
            raise ValueError("MPC score cache DDIM scheduler does not match this training config.")
        expected_horizon = config.model.action_horizon
        if self.diffusion_states.ndim != 3 or self.diffusion_states.shape[1] != expected_horizon:
            raise ValueError(
                "MPC score cache x_t must have shape [N, action_horizon, action_dim]; "
                f"got {self.diffusion_states.shape}."
            )
        if self.diffusion_states.shape != self.target_scores.shape:
            raise ValueError(
                "MPC score cache x_t/score shapes differ: "
                f"{self.diffusion_states.shape} vs {self.target_scores.shape}."
            )
        if self.diffusion_states.shape[-1] < config.model.action_dim:
            raise ValueError(
                f"MPC score cache has {self.diffusion_states.shape[-1]} action dims; "
                f"the model requires {config.model.action_dim}."
            )
        size = self.diffusion_states.shape[0]
        if not (
            len(self.demo_names)
            == len(self.step_indices)
            == len(self.trajectory_ids)
            == len(self.iterations)
            == len(self.times)
            == size
        ):
            raise ValueError("MPC score cache arrays have inconsistent sample counts.")
        labels_per_trajectory = int(metadata.get("labels_per_trajectory", -1))
        num_trajectories = int(metadata.get("num_trajectories", -1))
        if labels_per_trajectory <= 0 or num_trajectories <= 0:
            raise ValueError("MPC score cache has invalid reverse-trajectory metadata.")
        if labels_per_trajectory != int(metadata.get("num_iterations", -1)):
            raise ValueError("MPC score cache trajectory length does not match num_iterations.")
        if size != labels_per_trajectory * num_trajectories:
            raise ValueError("MPC score cache does not contain every step of every reverse trajectory.")
        expected_iterations = np.tile(np.arange(labels_per_trajectory), num_trajectories)
        expected_trajectory_ids = np.repeat(np.arange(num_trajectories), labels_per_trajectory)
        if not np.array_equal(self.iterations, expected_iterations):
            raise ValueError("MPC score cache reverse iterations are incomplete or out of order.")
        if not np.array_equal(self.trajectory_ids, expected_trajectory_ids):
            raise ValueError("MPC score cache trajectory ids are incomplete or out of order.")
        expected_times = np.asarray(
            [
                _time_from_iteration(
                    iteration=iteration,
                    num_iterations=labels_per_trajectory,
                    num_train_timesteps=int(metadata["ddim_num_train_timesteps"]),
                )
                for iteration in range(labels_per_trajectory)
            ],
            dtype=np.float32,
        )
        if not np.allclose(self.times.reshape(num_trajectories, -1), expected_times[None, :]):
            raise ValueError("MPC score cache diffusion times do not match reverse iterations.")
        demo_grid = self.demo_names.reshape(num_trajectories, labels_per_trajectory)
        step_grid = self.step_indices.reshape(num_trajectories, labels_per_trajectory)
        if np.any(demo_grid != demo_grid[:, :1]) or np.any(step_grid != step_grid[:, :1]):
            raise ValueError("MPC score cache changes observation inside a reverse trajectory.")
        if not np.isfinite(self.diffusion_states).all() or not np.isfinite(self.target_scores).all():
            raise ValueError("MPC score cache contains non-finite x_t or score values.")
        if not np.isfinite(self.times).all() or np.any((self.times < 0.0) | (self.times > 1.0)):
            raise ValueError("MPC score cache contains invalid normalized diffusion times.")

    def __len__(self) -> int:
        return int(self.diffusion_states.shape[0])

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        demo = self._file()["data"][self.demo_names[idx]]
        raw = _demo_sample(
            demo,
            int(self.step_indices[idx]),
            action_horizon=self.config.model.action_horizon,
            prompt=self.prompt,
        )
        inputs = self.input_transform(jax.tree.map(lambda x: x, raw))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)), inputs)
        return (
            inputs,
            torch.from_numpy(self.diffusion_states[idx]),
            torch.from_numpy(self.target_scores[idx]),
            torch.tensor(self.times[idx], dtype=torch.float32),
        )


def _collate_cache_batch(batch):
    inputs, x_t, score, time_cond = zip(*batch, strict=True)
    inputs = torch.utils.data.default_collate(inputs)
    return (
        _model.Observation.from_dict(inputs),
        torch.stack(x_t, dim=0),
        torch.stack(score, dim=0),
        torch.stack(time_cond, dim=0),
    )


def train(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config)
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.train_steps is not None:
        config = dataclasses.replace(config, num_train_steps=args.train_steps)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.checkpoint_base_dir is not None:
        config = dataclasses.replace(config, checkpoint_base_dir=args.checkpoint_base_dir)
    config = dataclasses.replace(
        config,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        wandb_enabled=not bool(args.no_wandb),
    )
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")

    use_ddp, local_rank, device = setup_ddp()
    rank = torch.distributed.get_rank() if use_ddp else 0
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    is_main = rank == 0
    set_seed(config.seed, rank)

    resuming = False
    if config.resume:
        if not config.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume."
            )
        resuming = True
    elif config.overwrite and config.checkpoint_dir.exists() and is_main:
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)
    elif not config.overwrite and config.checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint directory {config.checkpoint_dir} already exists; use --resume or --overwrite."
        )

    if use_ddp:
        torch.distributed.barrier()

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    dataset = MPCScoreDataset(
        hdf5_path=args.hdf5_path,
        cache_path=args.cache_path,
        config=config,
        prompt=args.prompt,
    )
    if is_main:
        logging.info(
            "Loaded reverse cache: trajectories=%s labels=%s labels_per_trajectory=%s",
            dataset.metadata["num_trajectories"],
            len(dataset),
            dataset.metadata["labels_per_trajectory"],
        )
    if config.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size={config.batch_size} must be divisible by world_size={world_size}."
        )
    local_batch_size = config.batch_size // world_size
    if len(dataset) < local_batch_size * world_size:
        raise ValueError(
            f"MPC score cache has {len(dataset)} samples, fewer than global batch_size={config.batch_size}."
        )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        if use_ddp
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate_cache_batch,
        persistent_workers=args.num_workers > 0,
    )
    data_config = dataset.data_config

    model = _proxy_score.ProxyScorePytorch(config.model).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        get_model_parameters(model),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = load_checkpoint(model, optimizer, config.checkpoint_dir, device) if resuming else 0

    def lr_schedule(step: int):
        warmup_steps = config.lr_schedule.warmup_steps
        peak_lr = config.lr_schedule.peak_lr
        decay_steps = config.lr_schedule.decay_steps
        end_lr = config.lr_schedule.decay_lr
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="MPC score ref")
        if is_main
        else None
    )
    metrics = []
    start_time = time.time()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    while global_step < config.num_train_steps:
        try:
            observation, x_t, score_target, time_cond = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            observation, x_t, score_target, time_cond = next(data_iter)

        observation = move_to_device(observation, device)
        x_t = x_t.to(torch.float32).to(device)
        score_target = score_target.to(torch.float32).to(device)
        time_cond = time_cond.to(torch.float32).to(device)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(global_step)

        losses = model(
            observation,
            x_t,
            time=time_cond,
            score_target=score_target,
        )
        losses = ensure_tensor_loss(losses, device)
        loss = losses.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            get_model_parameters(model),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        metrics.append(
            {
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
            }
        )
        completed_step = global_step + 1
        if is_main and completed_step % config.log_interval == 0 and metrics:
            elapsed = time.time() - start_time
            avg_loss = sum(item["loss"] for item in metrics) / len(metrics)
            avg_lr = sum(item["lr"] for item in metrics) / len(metrics)
            avg_grad_norm = sum(item["grad_norm"] for item in metrics) / len(metrics)
            logging.info(
                "step=%s mpc_score_loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                completed_step,
                avg_loss,
                avg_lr,
                avg_grad_norm,
                elapsed,
            )
            if config.wandb_enabled:
                wandb.log(
                    {
                        "mpc_score_loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "time_per_step": elapsed / config.log_interval,
                    },
                    step=completed_step,
                )
            metrics = []
            start_time = time.time()

        global_step = completed_step
        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "mpc_score_loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="ProxyScore training config name.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("generate-cache")
    _add_config_arg(cache_parser)
    cache_parser.add_argument("--hdf5_path", required=True)
    cache_parser.add_argument("--cache_path", required=True)
    cache_parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    cache_parser.add_argument("--base_checkpoint_dir", default=DEFAULT_BASE_CHECKPOINT_DIR)
    cache_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    cache_parser.add_argument("--task", default="weight")
    cache_parser.add_argument("--device", default="cuda")
    cache_parser.add_argument("--seed", type=int, default=0)
    cache_parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of observation-conditioned reverse trajectories.",
    )
    cache_parser.add_argument("--stride", type=int, default=4)
    cache_parser.add_argument("--num_steps", type=int, default=10)
    cache_parser.add_argument("--subtask_mode", choices=("heuristic", "empty"), default="heuristic")
    cache_parser.add_argument("--mpc_num_samples", type=int, default=512)
    cache_parser.add_argument("--mpc_iterations", type=int, default=8)
    cache_parser.add_argument("--mpc_noise", type=float, default=0.8)
    cache_parser.add_argument("--mpc_temperature", type=float, default=0.1)
    cache_parser.add_argument("--mpc_beta_opt_iter", type=float, default=1.0)
    cache_parser.add_argument("--mpc_beta_horizon", type=float, default=1.0)
    cache_parser.add_argument("--mpc_joint_delta_clip", type=float, default=0.15)
    cache_parser.add_argument(
        "--mpc_cost",
        default="grasp_flow",
        choices=("priority", "ref_style", "explore", "grasp_flow"),
    )
    cache_parser.add_argument("--mpc_interpolate", action="store_true")
    cache_parser.add_argument("--control_frequency", type=float, default=40.0)
    cache_parser.add_argument("--interpolate_frequency", type=float, default=5.0)
    cache_parser.set_defaults(func=generate_cache)

    train_parser = subparsers.add_parser("train")
    _add_config_arg(train_parser)
    train_parser.add_argument("--hdf5_path", required=True)
    train_parser.add_argument("--cache_path", required=True)
    train_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    train_parser.add_argument("--exp_name", default="ref")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--no_wandb", action="store_true")
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--train_steps", type=int, default=None)
    train_parser.add_argument("--batch_size", type=int, default=None)
    train_parser.add_argument("--checkpoint_base_dir", default=None)
    train_parser.set_defaults(func=train)
    return parser


def main() -> None:
    init_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
