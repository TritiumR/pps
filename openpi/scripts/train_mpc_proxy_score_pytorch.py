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

Training groups every reverse trajectory by observation.  Its 11 score labels
share one DINO visual prefix, while a persistent mmap sidecar stores the
once-resized uint8 observations for all DataLoader workers and DDP ranks.

A third stage, ``train-bc``, skips the MPC labels entirely: the same architecture is
trained as a diffusion-BC x0 predictor on the demo's next-H action chunks (the
"does injecting actual demo content help" A/B against the MPC-labelled proxy):

  python scripts/train_mpc_proxy_score_pytorch.py train-bc \
      --config score_task_capsule --config_name score_task_stack \
      --hdf5_path ../data/mg_stack/demo_224.hdf5 --demo_stride 10 \
      --exp_name task_bc_n20
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import json
import logging
import math
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
from tools.ref_observation_split import split_observations, write_split_manifest
from tools.ref_action_dataset import MPCActionChunkDataset
from tools.ref_batching import (
    resolve_ref_batch_layout,
    sample_cached_label_indices,
    sample_trajectory_indices,
)

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
CACHE_FORMAT_VERSION = 5
CACHE_LABEL_TYPE = "mpc_epsilon_action_prox_reverse_trajectory"
CACHE_STATE_SOURCE = "action_prox_reverse_trajectory_from_gaussian"
ACTION_PROX_NOISE_SCHEDULE = "mpc_noise_times_sqrt_one_minus_alpha_bar"
OBSERVATION_CACHE_FORMAT_VERSION = 1
OBSERVATION_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


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


def _file_sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ACTION_NORM_STATS_FILENAME = "action_norm_stats.json"


@dataclasses.dataclass(frozen=True)
class ChunkActionNormalize(_transforms.DataTransformFn):
    """Per-(row, dim) affine action normalize; the Normalize twin that Normalize cannot express.

    openpi's Normalize indexes stats on the last axis only, so every chunk row shares one
    scale. The DeltaActions target a[t+1+h] - q[t] grows ~7x from row 0 to row 14, which puts
    the executed rows far below unit scale (and far below the diffusion noise) whatever the
    per-dim constants are. Stats shaped [H, D] give every row unit scale.
    """

    mean: np.ndarray  # [H, D]
    std: np.ndarray  # [H, D]

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)
        dim = actions.shape[-1]
        mean = self.mean[: actions.shape[-2], :dim]
        std = self.std[: actions.shape[-2], :dim]
        data["actions"] = (actions - mean) / (std + 1e-6)
        return data


def unnormalize_chunk_actions(
    actions: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    """Inverse of ChunkActionNormalize (the serving half of the round trip)."""
    actions = np.asarray(actions, dtype=np.float32)
    horizon, dim = actions.shape[-2], actions.shape[-1]
    mean = np.asarray(stats["mean"], dtype=np.float32)[:horizon, :dim]
    std = np.asarray(stats["std"], dtype=np.float32)[:horizon, :dim]
    return (actions * (std + 1e-6) + mean).astype(np.float32)


def demo_action_norm_stats(
    hdf5_path: str,
    *,
    action_horizon: int,
    demo_names: list[str],
    pooled: bool = False,
    action_offset: int = 1,
    gripper_center: float = 0.5,
    gripper_scale: float = 0.5,
) -> dict[str, Any]:
    """Scale of the DeltaActions target a[t+offset+h] - q[t], measured on the training demos.

    Mirrors eval_mg.demo_delta_stats (per-dim std of the demos' own joint deltas) on the exact
    quantity that reaches the loss. Arm rows are centred at 0 and the gripper keeps eval_mg's
    ChunkDecodePolicy coding, so proxy model space matches the planner's.
    """
    arm = 7
    per_row = [[] for _ in range(action_horizon)]
    with h5py.File(hdf5_path, "r") as f:
        for name in demo_names:
            demo = f["data"][name]
            joint_actions = np.asarray(demo["obs/joint_actions"], dtype=np.float64)
            joint_pos = np.asarray(demo["obs/joint_pos"], dtype=np.float64)[:, :arm]
            windows = len(joint_actions) - action_horizon
            if windows <= 0:
                continue
            for h in range(action_horizon):
                start = action_offset + h
                per_row[h].append(
                    joint_actions[start : start + windows, :arm] - joint_pos[:windows]
                )
    std = np.stack([np.concatenate(rows, 0).std(0) for rows in per_row])  # [H, arm]
    raw_mean = np.stack([np.concatenate(rows, 0).mean(0) for rows in per_row])
    if pooled:
        # demo_delta_stats' own reduction: one per-dim scale for the whole chunk.
        std = np.broadcast_to(np.sqrt((std**2).mean(0)), std.shape).copy()
    mean = np.zeros((action_horizon, arm + 1), dtype=np.float32)
    scale = np.ones((action_horizon, arm + 1), dtype=np.float32)
    mean[:, arm] = gripper_center
    scale[:, :arm] = std
    scale[:, arm] = gripper_scale
    return {
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "source": "demo_action_norm_stats",
        "pooled": bool(pooled),
        "action_offset": int(action_offset),
        "hdf5_path": str(hdf5_path),
        "action_horizon": int(action_horizon),
        "num_demos": len(demo_names),
        # Diagnostics only: the arm is centred at 0 to match eval_mg's ChunkDecodePolicy.
        "arm_delta_mean": raw_mean.tolist(),
    }


def load_action_norm_stats(path: str | pathlib.Path) -> dict[str, np.ndarray] | None:
    """Read an action_norm_stats.json into float32 [H, D] arrays (None if absent)."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {
        "mean": np.asarray(raw["mean"], dtype=np.float32),
        "std": np.asarray(raw["std"], dtype=np.float32),
        "meta": {k: v for k, v in raw.items() if k not in ("mean", "std")},
    }


def find_action_norm_stats(checkpoint_dir: str | pathlib.Path):
    """Locate the stats a checkpoint was trained with (step dir, then the run root)."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    for candidate in (
        checkpoint_dir / ACTION_NORM_STATS_FILENAME,
        checkpoint_dir.parent / ACTION_NORM_STATS_FILENAME,
    ):
        stats = load_action_norm_stats(candidate)
        if stats is not None:
            return stats, candidate
    return None, None


def _build_data_pipeline(config: _config.TrainConfig, action_norm_stats=None):
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
    norm_stats = data_config.norm_stats
    action_norm = []
    if action_norm_stats is not None:
        # Actions leave the shared Normalize and take the demo-measured [H, D] scale instead;
        # state/images keep the pi05_droid constants so the visual prefix is unchanged.
        norm_stats = {k: v for k, v in norm_stats.items() if k != "actions"}
        action_norm = [
            ChunkActionNormalize(action_norm_stats["mean"], action_norm_stats["std"])
        ]
    input_transform = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *action_norm,
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
    action_offset: int = 1,
) -> dict[str, Any]:
    # action_offset 1 is the shipped alignment; 0 makes row 0 the demo's IMMEDIATE next
    # action (joint_actions[t] == joint_pos[t+1]), which is what the verified replay shim
    # executes (setup/07_relabel_replay_gate.py, 19/20 on square).
    action_start = step_idx + action_offset
    action_end = action_start + action_horizon
    actions = np.asarray(demo["obs/joint_actions"][action_start:action_end], dtype=np.float32)
    if len(actions) < action_horizon:
        all_actions = demo["obs/joint_actions"]
        if len(actions):
            pad_value = actions[-1]
        elif len(all_actions):
            pad_value = np.asarray(all_actions[-1], dtype=np.float32)
        else:
            raise ValueError("Demo has no joint actions.")
        padding = np.repeat(pad_value[None], action_horizon - len(actions), axis=0)
        actions = np.concatenate([actions, padding], axis=0)
    return {
        "exterior_image_1_left": demo["obs/table_cam"][step_idx],
        "wrist_image_left": demo["obs/wrist_cam"][step_idx],
        "joint_position": np.asarray(demo["obs/joint_pos"][step_idx][:7], dtype=np.float32),
        "gripper_position": np.asarray(demo["obs/gripper_pos"][step_idx][:1], dtype=np.float32),
        "actions": actions,
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


def _load_task_module(path: str):
    """Load an external offline-task hook by file path (the --task_module surface).

    Out-of-repo benchmarks (e.g. the MuJoCo/MimicGen port) register their stage ladders without
    adding machine-specific paths here. The module mirrors the in-repo capsule hook and may expose:
    prepare_planner(planner), attach_priority_cost(planner, cost_cfg), episode_signals(demo),
    frame_context(context, demo, step, sig, heuristic=...).
    """
    import importlib.util

    module_path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(f"offline_task_{module_path.stem}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
            if length <= 0:
                continue
            all_indices.extend((demo_name, step) for step in range(0, length, stride))
    rng.shuffle(all_indices)
    if max_trajectories is not None:
        all_indices = all_indices[:max_trajectories]
    return all_indices


def _shard_indices(indices, shard, num_shards):
    """Deterministic disjoint shards for multi-GPU cache generation, sharded BY DEMO.

    Two properties the labels depend on, which a post-shuffle round-robin over frames broke:

    - a demo's frames all land in one shard, and
    - each shard is processed in ascending (demo, step) order.

    MonotoneStages clamps a frame's stage to the highest stage seen at any lower step of that demo,
    but it can only see steps already processed. Splitting a demo across shards, or feeding it
    shuffled, therefore made the stage label depend on the shard count and the shuffle seed. Which
    frames are SAMPLED still depends on the seed (that is the point of the shuffle); which stage a
    sampled frame gets no longer does.

    Demos vary in length, so shards are balanced only approximately.
    """
    by_demo: dict[str, list[tuple[str, int]]] = {}
    for demo_name, step in indices:
        by_demo.setdefault(demo_name, []).append((demo_name, step))
    if num_shards > 1:
        keep = sorted(by_demo)[shard::num_shards]
        by_demo = {d: by_demo[d] for d in keep}
    return sorted((idx for frames in by_demo.values() for idx in frames))


def generate_cache(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config)
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")
    if config.model.prediction_type != "epsilon":
        raise ValueError(
            f"{args.config!r} must use prediction_type='epsilon' for ref distillation."
        )

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
    action_stats = load_action_norm_stats(args.base_action_stats)
    if action_stats is None:
        raise FileNotFoundError(f"Base demo action stats not found: {args.base_action_stats}")
    base_action_norm = base_norm_stats.get("actions")
    if base_action_norm is None:
        raise ValueError("Base/task norm stats do not contain actions.")
    for field in ("mean", "std"):
        embedded = np.asarray(getattr(base_action_norm, field), dtype=np.float32)
        standalone = np.asarray(action_stats[field], dtype=np.float32)
        if embedded.shape != standalone.shape or not np.allclose(
            embedded, standalone, rtol=0.0, atol=1e-7
        ):
            raise ValueError(
                f"{field} mismatch between task checkpoint stats and --base_action_stats."
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
            cost_executable_actions=bool(args.cost_executable_actions),
            joint_delta_clip=args.mpc_joint_delta_clip,
            logit_norm=args.mpc_logit_norm,
            ancestral_eta=args.mpc_ancestral_eta,
            cost_style=args.mpc_cost,
            optimize_space="action",
            ddim_num_train_timesteps=config.model.ddim_num_train_timesteps,
            interpolate=args.mpc_interpolate,
            control_frequency=args.control_frequency,
            interpolate_frequency=args.interpolate_frequency,
        ),
    )
    task_module = _load_task_module(args.task_module) if args.task_module else None
    if task_module is not None and hasattr(task_module, "prepare_planner"):
        # e.g. the MimicGen port swaps in its fitted PandaGripper FK, as its eval harness does.
        task_module.prepare_planner(planner)
    if args.mpc_cost == "priority":
        # Our stage-aware CompositeCost, attached exactly as the eval bridge does. The per-frame
        # stage structure it reads comes from vlm_dp.offline_context at the _mpc_context call.
        import yaml
        if not args.vlm_cost_config:
            raise SystemExit("--mpc_cost priority requires --vlm_cost_config (the cost YAML).")
        with open(os.path.join(_REPO_DIR, args.vlm_cost_config)) as fh:
            cost_cfg = yaml.safe_load(fh)
        if task_module is not None:
            # The external registry owns the ladder AND its representability guard.
            task_module.attach_priority_cost(planner, cost_cfg)
        elif args.task != "weight":
            raise SystemExit(
                "--mpc_cost priority labelling wires the weight stage ladder "
                "(or an external --task_module); capsule uses --mpc_cost capsule_flow."
            )
        else:
            from vlm_dp.offline_context import attach_priority_cost
            attach_priority_cost(planner, cost_cfg)

    indices = _sample_indices(
        pathlib.Path(args.hdf5_path),
        action_horizon=config.model.action_horizon,
        max_trajectories=args.max_trajectories,
        stride=args.stride,
        seed=args.seed,
    )
    indices = _shard_indices(indices, args.obs_shard, args.obs_num_shards)
    if not indices:
        raise ValueError("No valid HDF5 windows found for MPC score label generation.")
    num_observations = len(indices)
    trajectories_per_observation = int(args.trajectories_per_observation)
    if trajectories_per_observation <= 0:
        raise ValueError("--trajectories_per_observation must be positive.")
    indices = [index for index in indices for _ in range(trajectories_per_observation)]
    logging.info(
        "shard %s/%s: observations=%s trajectories=%s K=%s",
        args.obs_shard, args.obs_num_shards, num_observations, len(indices),
        trajectories_per_observation,
    )
    external_signals: dict[str, Any] = {}
    weight_signals: dict[str, dict[str, Any]] = {}
    capsule_signals: dict[str, dict[str, Any]] = {}
    if args.task == "capsule":
        # Registry ladder (vlm_dp.offline_context): per-demo latched events feed the
        # capsule_flow context; the weight path below is untouched.
        from vlm_dp.offline_context import capsule_episode_signals, capsule_flow_context

    _label_seed = getattr(args, "label_seed", None)
    torch.manual_seed(args.seed if _label_seed is None else _label_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed if _label_seed is None else _label_seed)
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
            if task_module is not None:
                sig = external_signals.get(demo_name)
                if sig is None:
                    sig = external_signals[demo_name] = task_module.episode_signals(demo)
                context = task_module.frame_context(
                    context, demo, step_idx, sig,
                    heuristic=args.subtask_mode == "heuristic",
                )
            if args.mpc_cost == "priority" and task_module is None:
                from vlm_dp.offline_context import weight_episode_signals, weight_frame_context
                sig = weight_signals.get(demo_name)
                if sig is None:
                    sig = weight_signals[demo_name] = weight_episode_signals(demo)
                context = weight_frame_context(context, sig, step_idx)
            if args.task == "capsule":
                sig = capsule_signals.get(demo_name)
                if sig is None:
                    sig = capsule_signals[demo_name] = capsule_episode_signals(demo)
                context = capsule_flow_context(
                    context, demo, step_idx, sig,
                    heuristic=args.subtask_mode == "heuristic",
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
                    alpha_bar, _ = ddim_iteration_alphas(
                        iteration=iteration,
                        num_iterations=num_iterations,
                        num_train_timesteps=config.model.ddim_num_train_timesteps,
                    )
                    epsilon = -math.sqrt(max(1.0 - alpha_bar, 1e-6)) * score
                    target_scores.append(
                        epsilon[0].detach().cpu().numpy().astype(np.float32)
                    )
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
        "num_observations": int(num_observations),
        "trajectories_per_observation": int(trajectories_per_observation),
        "labels_per_trajectory": int(num_iterations),
        "config": args.config,
        "base_config": args.base_config,
        "base_checkpoint_dir": str(args.base_checkpoint_dir),
        "base_action_stats": str(args.base_action_stats),
        "base_action_stats_sha256": _file_sha256(args.base_action_stats),
        "hdf5_path": str(args.hdf5_path),
        "obs_shard": int(args.obs_shard),
        "obs_num_shards": int(args.obs_num_shards),
        "seed": int(args.seed),
        "label_seed": int(args.seed if args.label_seed is None else args.label_seed),
        "task": args.task,
        "task_module": str(args.task_module) if args.task_module else None,
        "vlm_cost_config": args.vlm_cost_config,
        "teacher_code_sha256": {
            "cache_generator": _file_sha256(__file__),
            "planner": _file_sha256(pathlib.Path(_REPO_DIR) / "sim_free_mpc/planner.py"),
            "offline_context": _file_sha256(pathlib.Path(_REPO_DIR) / "vlm_dp/offline_context.py"),
            "cost_config": _file_sha256(pathlib.Path(_REPO_DIR) / args.vlm_cost_config),
        },
        "prompt": args.prompt,
        "num_steps": int(args.num_steps),
        "num_iterations": int(num_iterations),
        "prediction_type": "epsilon",
        "target_transform": "epsilon=-sqrt(1-alpha_bar_t)*mpc_score",
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
            "proposal_center": "current_noisy_action" if args.mpc_cost == "priority" else "noisy_action_div_sqrt_alpha",
            "sampler": "base",
            "noise_schedule": ACTION_PROX_NOISE_SCHEDULE,
            "noise": float(args.mpc_noise),
            "temperature": float(args.mpc_temperature),
            "beta_opt_iter": float(args.mpc_beta_opt_iter),
            "beta_horizon": float(args.mpc_beta_horizon),
            "joint_delta_clip": float(args.mpc_joint_delta_clip),
            "cost_style": args.mpc_cost,
            "interpolate": bool(args.mpc_interpolate),
            "cost_executable_actions": bool(args.cost_executable_actions),
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
        epsilon=np.asarray(target_scores, dtype=np.float32),
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
        observation_cache_path: str | None = None,
        build_observation_cache: bool = True,
    ):
        self.hdf5_path = hdf5_path
        self.cache = np.load(cache_path, allow_pickle=False)
        self.demo_names = self.cache["demo_name"].astype(str)
        self.step_indices = self.cache["step_index"].astype(np.int64)
        self.trajectory_ids = self.cache["trajectory_id"].astype(np.int64)
        self.iterations = self.cache["iteration"].astype(np.int64)
        self.diffusion_states = self.cache["x_t"].astype(np.float32)
        self.target_scores = self.cache["epsilon"].astype(np.float32)
        self.times = self.cache["time"].astype(np.float32)
        self.prompt = prompt
        self.config = config
        self.data_config, self.input_transform = _build_data_pipeline(config)

        metadata = json.loads(str(self.cache["metadata_json"].item()))
        self.metadata = metadata
        if int(metadata.get("cache_format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError(
                "MPC score cache format is stale. Regenerate it with the current action-prox sampler."
            )
        if metadata.get("label_type") != CACHE_LABEL_TYPE:
            raise ValueError("MPC score cache does not contain action-prox reverse trajectories.")
        if metadata.get("prediction_type") != "epsilon" or config.model.prediction_type != "epsilon":
            raise ValueError("MPC ref cache and model must both use epsilon prediction.")
        if bool(metadata.get("use_quantile_norm")):
            raise ValueError("New Base/ref cache must use demo mean/std, not quantile normalization.")
        if not bool(config.model.bidirectional_attention):
            raise ValueError("New ref training config must use bidirectional attention.")
        if metadata.get("state_source") != CACHE_STATE_SOURCE:
            raise ValueError("MPC score cache x_t states were not sampled from base reverse denoising.")
        if metadata.get("initial_state_distribution") != "standard_gaussian":
            raise ValueError("MPC score cache trajectories do not start from standard Gaussian noise.")
        if metadata.get("trajectory_update") != "mbd_score":
            raise ValueError("MPC score cache does not use the online MBD reverse update.")
        mpc_metadata = metadata.get("mpc", {})
        if mpc_metadata.get("sampler") != "base":
            raise ValueError("MPC ref cache must use the Base sampler.")
        if mpc_metadata.get("cost_style") != "priority":
            raise ValueError("MPC ref cache must use the priority cost.")
        if not bool(mpc_metadata.get("cost_executable_actions")):
            raise ValueError("MPC ref cache must score executable actions.")
        if mpc_metadata.get("proposal_center") != "current_noisy_action":
            raise ValueError(
                "MPC score cache does not use x_t as the priority/base-sampler "
                "action-prox proposal center."
            )
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
        num_observations = int(metadata.get("num_observations", -1))
        trajectories_per_observation = int(metadata.get("trajectories_per_observation", -1))
        if num_observations <= 0 or trajectories_per_observation <= 0:
            raise ValueError("MPC score cache has invalid observation/K metadata.")
        if num_trajectories != num_observations * trajectories_per_observation:
            raise ValueError("MPC score cache trajectory count does not equal observations * K.")
        if labels_per_trajectory != int(metadata.get("num_steps", -1)) + 1:
            raise ValueError("MPC score cache must save 11 levels for 10 updates.")
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

        self.labels_per_trajectory = labels_per_trajectory
        self.num_trajectories = num_trajectories
        self.num_observations = num_observations
        self.trajectories_per_observation = trajectories_per_observation
        self.labels_per_observation = labels_per_trajectory * trajectories_per_observation
        self.trajectory_demo_names = demo_grid[:, 0]
        self.trajectory_step_indices = step_grid[:, 0]
        trajectory_index_grid = np.arange(num_trajectories).reshape(num_observations, trajectories_per_observation)
        obs_demo_grid = self.trajectory_demo_names.reshape(num_observations, trajectories_per_observation)
        obs_step_grid = self.trajectory_step_indices.reshape(num_observations, trajectories_per_observation)
        if (
            np.any(obs_demo_grid != obs_demo_grid[:, :1])
            or np.any(obs_step_grid != obs_step_grid[:, :1])
        ):
            raise ValueError("K trajectories for one observation are not contiguous or share different observations.")
        self.observation_trajectory_indices = trajectory_index_grid
        self.unique_demo_names = obs_demo_grid[:, 0]
        self.unique_step_indices = obs_step_grid[:, 0]
        self.trajectory_observation_indices = np.repeat(np.arange(num_observations), trajectories_per_observation)

        if observation_cache_path is None:
            observation_cache_path = f"{cache_path}.observations"
        self.observation_cache_path = pathlib.Path(observation_cache_path)
        expected_metadata = self._observation_cache_metadata()
        if not self._observation_cache_matches(expected_metadata):
            if not build_observation_cache:
                raise FileNotFoundError(
                    f"Shared observation cache is missing or stale: {self.observation_cache_path}"
                )
            self._build_observation_cache(expected_metadata)
        self._load_observation_cache()

    def __len__(self) -> int:
        return self.num_observations

    @property
    def num_labels(self) -> int:
        return int(self.diffusion_states.shape[0])

    def _observation_cache_metadata(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for demo_name, step_idx in zip(
            self.unique_demo_names,
            self.unique_step_indices,
            strict=True,
        ):
            digest.update(str(demo_name).encode("utf-8"))
            digest.update(np.asarray(step_idx, dtype=np.int64).tobytes())

        if self.metadata.get("observation_source") == "eval_steering_live_model_inputs":
            return {
                "format_version": int(self.metadata["observation_cache_format_version"]),
                "source": "eval_steering_live_model_inputs",
                "num_observations": int(len(self.unique_demo_names)),
                "prompt": self.prompt,
                "norm_stats_fingerprint": _norm_stats_fingerprint(self.data_config.norm_stats),
                "use_quantile_norm": bool(self.data_config.use_quantile_norm),
                "image_keys": list(OBSERVATION_IMAGE_KEYS),
                "image_shape": [224, 224, 3],
            }

        hdf5_stat = pathlib.Path(self.hdf5_path).stat()
        return {
            "format_version": OBSERVATION_CACHE_FORMAT_VERSION,
            "hdf5_size": int(hdf5_stat.st_size),
            "observation_key_fingerprint": digest.hexdigest(),
            "num_observations": int(len(self.unique_demo_names)),
            "prompt": self.prompt,
            "norm_stats_fingerprint": _norm_stats_fingerprint(self.data_config.norm_stats),
            "use_quantile_norm": bool(self.data_config.use_quantile_norm),
            "image_keys": list(OBSERVATION_IMAGE_KEYS),
            "image_shape": [224, 224, 3],
        }

    def _observation_cache_matches(self, expected_metadata: dict[str, Any]) -> bool:
        metadata_path = self.observation_cache_path / "metadata.json"
        if not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if metadata != expected_metadata:
            return False
        required_files = [
            "images.npy",
            "image_masks.npy",
            "states.npy",
            "tokenized_prompt.npy",
            "tokenized_prompt_mask.npy",
        ]
        return all((self.observation_cache_path / name).exists() for name in required_files)

    def _build_observation_cache(self, metadata: dict[str, Any]) -> None:
        cache_path = self.observation_cache_path
        for stale_tmp_path in cache_path.parent.glob(f"{cache_path.name}.tmp-*"):
            shutil.rmtree(stale_tmp_path)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
        tmp_path.mkdir(parents=True)

        num_observations = len(self.unique_demo_names)
        images = np.lib.format.open_memmap(
            tmp_path / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(num_observations, len(OBSERVATION_IMAGE_KEYS), 224, 224, 3),
        )
        image_masks = np.lib.format.open_memmap(
            tmp_path / "image_masks.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(num_observations, len(OBSERVATION_IMAGE_KEYS)),
        )
        states = np.lib.format.open_memmap(
            tmp_path / "states.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_observations, self.config.model.action_dim),
        )
        tokenized_prompt = None
        tokenized_prompt_mask = None

        logging.info(
            "Building shared observation cache with %s unique observations at %s",
            num_observations,
            cache_path,
        )
        with h5py.File(self.hdf5_path, "r") as h5_file:
            iterator = zip(self.unique_demo_names, self.unique_step_indices, strict=True)
            iterator = tqdm.tqdm(iterator, total=num_observations, desc="Observation cache")
            for observation_idx, (demo_name, step_idx) in enumerate(iterator):
                demo = h5_file["data"][str(demo_name)]
                raw = _demo_sample(
                    demo,
                    int(step_idx),
                    action_horizon=self.config.model.action_horizon,
                    prompt=self.prompt,
                )
                inputs = self.input_transform(jax.tree.map(lambda x: x, raw))
                for image_idx, image_key in enumerate(OBSERVATION_IMAGE_KEYS):
                    image = np.asarray(inputs["image"][image_key])
                    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                        raise ValueError(
                            f"Expected uint8 224x224 image for {image_key}, got "
                            f"shape={image.shape} dtype={image.dtype}."
                        )
                    images[observation_idx, image_idx] = image
                    image_masks[observation_idx, image_idx] = bool(
                        inputs["image_mask"][image_key]
                    )
                states[observation_idx] = np.asarray(inputs["state"], dtype=np.float32)
                if tokenized_prompt is None:
                    tokenized_prompt = np.asarray(inputs["tokenized_prompt"], dtype=np.int64)
                    tokenized_prompt_mask = np.asarray(
                        inputs["tokenized_prompt_mask"], dtype=np.bool_
                    )

        images.flush()
        image_masks.flush()
        states.flush()
        if tokenized_prompt is None or tokenized_prompt_mask is None:
            raise ValueError("Cannot build an empty observation cache.")
        np.save(tmp_path / "tokenized_prompt.npy", tokenized_prompt)
        np.save(tmp_path / "tokenized_prompt_mask.npy", tokenized_prompt_mask)
        (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        if cache_path.exists():
            shutil.rmtree(cache_path)
        tmp_path.rename(cache_path)
        logging.info("Finished shared observation cache: %s", cache_path)

    def _load_observation_cache(self) -> None:
        cache_path = self.observation_cache_path
        # Copy-on-write mmap keeps the underlying pages shared across DDP ranks and
        # DataLoader workers while allowing zero-copy torch views without warnings.
        self.cached_images = np.load(cache_path / "images.npy", mmap_mode="c")
        self.cached_image_masks = np.load(cache_path / "image_masks.npy", mmap_mode="c")
        self.cached_states = np.load(cache_path / "states.npy", mmap_mode="c")
        self.cached_tokenized_prompt = np.load(
            cache_path / "tokenized_prompt.npy", mmap_mode="c"
        )
        self.cached_tokenized_prompt_mask = np.load(
            cache_path / "tokenized_prompt_mask.npy", mmap_mode="c"
        )

    def __getitem__(self, idx: int):
        observation_idx = int(idx)
        inputs = {
            "image": {
                image_key: torch.from_numpy(self.cached_images[observation_idx, image_idx])
                for image_idx, image_key in enumerate(OBSERVATION_IMAGE_KEYS)
            },
            "image_mask": {
                image_key: torch.as_tensor(
                    bool(self.cached_image_masks[observation_idx, image_idx]),
                    dtype=torch.bool,
                )
                for image_idx, image_key in enumerate(OBSERVATION_IMAGE_KEYS)
            },
            "state": torch.from_numpy(self.cached_states[observation_idx]),
            "tokenized_prompt": torch.from_numpy(self.cached_tokenized_prompt),
            "tokenized_prompt_mask": torch.from_numpy(self.cached_tokenized_prompt_mask),
        }
        label_start = idx * self.labels_per_observation
        label_end = label_start + self.labels_per_observation
        return (
            inputs,
            torch.from_numpy(self.diffusion_states[label_start:label_end]),
            torch.from_numpy(self.target_scores[label_start:label_end]),
            torch.from_numpy(self.times[label_start:label_end]),
        )


def _collate_cache_batch(batch):
    inputs, x_t, score, time_cond = zip(*batch, strict=True)
    inputs = torch.utils.data.default_collate(inputs)
    return (
        inputs,
        torch.stack(x_t, dim=0),
        torch.stack(score, dim=0),
        torch.stack(time_cond, dim=0),
    )


class BCDemoActionDataset(torch.utils.data.Dataset):
    """Demo-BC frames: score-pipeline obs + the demo's normalized model-space action chunk.

    The A/B twin of MPCScoreDataset: same _demo_sample schema and input transform (so the
    same serve path applies), but the target is the demo's next-H action chunk rather than an
    MPC score label.  x_t/alpha/noise are drawn inside the model's standard diffusion recipe.
    """

    def __init__(
        self,
        *,
        hdf5_path: str,
        config: _config.TrainConfig,
        prompt: str,
        demo_stride: int = 1,
        demo_offset: int = 0,
        stride: int = 1,
        action_norm_stats=None,
        exclude_demos: set[str] | None = None,
        action_offset: int = 1,
    ):
        self.hdf5_path = hdf5_path
        self.prompt = prompt
        self.config = config
        self.action_offset = int(action_offset)
        self.data_config, self.input_transform = _build_data_pipeline(
            config, action_norm_stats=action_norm_stats
        )
        exclude_demos = exclude_demos or set()
        with h5py.File(hdf5_path, "r") as f:
            self.demo_names = [
                name
                for name in sorted(f["data"].keys())[demo_offset::demo_stride]
                if name not in exclude_demos
            ]
            self.index: list[tuple[str, int]] = []
            for name in self.demo_names:
                num_windows = (
                    len(f["data"][name]["obs/joint_actions"]) - config.model.action_horizon
                )
                self.index.extend((name, step) for step in range(0, max(num_windows, 0), stride))
        if not self.index:
            raise ValueError("No BC windows found for the selected demo subset.")
        self._file = None

    def __len__(self) -> int:
        return len(self.index)

    def _demo(self, name: str) -> h5py.Group:
        if self._file is None:
            # Lazy per-worker handle: h5py files must not cross fork boundaries.
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file["data"][name]

    def __getitem__(self, idx: int):
        demo_name, step = self.index[idx]
        raw = _demo_sample(
            self._demo(demo_name),
            step,
            action_horizon=self.config.model.action_horizon,
            prompt=self.prompt,
            action_offset=self.action_offset,
        )
        inputs = self.input_transform(jax.tree.map(lambda x: x, raw))
        actions = torch.from_numpy(np.asarray(inputs["actions"], dtype=np.float32))
        sample = {
            "image": {
                key: torch.from_numpy(np.asarray(inputs["image"][key]))
                for key in OBSERVATION_IMAGE_KEYS
            },
            "image_mask": {
                key: torch.as_tensor(bool(inputs["image_mask"][key]), dtype=torch.bool)
                for key in OBSERVATION_IMAGE_KEYS
            },
            "state": torch.from_numpy(np.asarray(inputs["state"], dtype=np.float32)),
            "tokenized_prompt": torch.from_numpy(
                np.asarray(inputs["tokenized_prompt"], dtype=np.int64)
            ),
            "tokenized_prompt_mask": torch.from_numpy(
                np.asarray(inputs["tokenized_prompt_mask"], dtype=np.bool_)
            ),
        }
        return sample, actions


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )


class _WeightEMA:
    """Exponential moving average of the trainable weights.

    DDP keeps every rank's weights in lockstep, so each rank tracks an identical
    shadow and only rank 0 ever writes it.
    """

    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().to(torch.float32).clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(param.detach().to(torch.float32), 1.0 - self.decay)

    @contextlib.contextmanager
    def swapped_in(self, module: torch.nn.Module):
        """Hold the EMA weights in the module (checkpoint writing), then restore."""
        backup = {}
        with torch.no_grad():
            for name, param in module.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(shadow.to(param.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in module.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])


def _optimizer_parameters(model: torch.nn.Module, *, drop_frozen: bool) -> list:
    """Optimizer parameters; today's grouping unless the encoder is deliberately frozen.

    Filtering unconditionally would also drop the always-frozen lm_head and change the
    optimizer state layout of every existing checkpoint.
    """
    params = list(get_model_parameters(model))
    if not drop_frozen:
        return params
    return [param for param in params if param.requires_grad]


def _is_checkpoint_step(global_step: int, config: _config.TrainConfig) -> bool:
    # Mirrors save_checkpoint's own gate in train_proxy_score_pytorch.
    return (
        global_step % config.save_interval == 0 and global_step > 0
    ) or global_step == config.num_train_steps


def _copy_action_norm_stats(config, global_step: int) -> None:
    """Keep each step dir self-contained so a copied checkpoint carries its own stats."""
    src = config.checkpoint_dir / ACTION_NORM_STATS_FILENAME
    dst_dir = config.checkpoint_dir / f"{global_step}"
    if src.exists() and dst_dir.is_dir():
        shutil.copyfile(src, dst_dir / ACTION_NORM_STATS_FILENAME)


def _save_bc_checkpoint(
    model, optimizer, global_step, config, is_main, data_config, ema
) -> None:
    """save_checkpoint, with the EMA weights as model.safetensors when EMA is on."""
    if ema is None:
        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
        if is_main and _is_checkpoint_step(global_step, config):
            _copy_action_norm_stats(config, global_step)
        return
    if not (is_main and _is_checkpoint_step(global_step, config)):
        return
    module = _unwrap_model(model)
    with ema.swapped_in(module):
        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
    # Live weights stay next to the EMA ones for resume and EMA-vs-raw comparisons.
    safetensors.torch.save_model(
        module, config.checkpoint_dir / f"{global_step}" / "model_raw.safetensors"
    )
    _copy_action_norm_stats(config, global_step)


def _random_shift(image: torch.Tensor, shift_px: int) -> torch.Tensor:
    """DrQ shift: replicate-pad by shift_px, then crop back at a per-sample offset."""
    channels_first = image.shape[1] == 3
    x = image if channels_first else image.permute(0, 3, 1, 2)
    batch, channels, height, width = x.shape
    padded = torch.nn.functional.pad(
        x.to(torch.float32), (shift_px,) * 4, mode="replicate"
    )
    rows = torch.randint(
        0, 2 * shift_px + 1, (batch, 1, 1, 1), device=x.device
    ) + torch.arange(height, device=x.device).view(1, 1, height, 1)
    padded = torch.gather(
        padded, 2, rows.expand(batch, channels, height, padded.shape[3])
    )
    cols = torch.randint(
        0, 2 * shift_px + 1, (batch, 1, 1, 1), device=x.device
    ) + torch.arange(width, device=x.device).view(1, 1, 1, width)
    out = torch.gather(padded, 3, cols.expand(batch, channels, height, width)).to(
        image.dtype
    )
    return out if channels_first else out.permute(0, 2, 3, 1)


def _augment_observation(observation, shift_px: int):
    """Random-shift both cameras, independently per sample and per camera (train only).

    Independent shifts: the table and wrist cameras are separate sensors, so a shared
    offset would model a rigid jitter that does not exist and halve the augmentation's
    diversity.
    """
    if shift_px <= 0:
        return observation
    images = {
        key: _random_shift(image, shift_px)
        for key, image in observation.images.items()
    }
    return dataclasses.replace(observation, images=images)


def _held_out_demos(hdf5_path: str, val_demos: int) -> set[str]:
    """The last `val_demos` demos by numeric index (the offline gate's split)."""
    if val_demos <= 0:
        return set()
    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    if val_demos >= len(names):
        raise ValueError(f"--val_demos {val_demos} leaves no training demos.")
    return set(names[-val_demos:])


def _bc_action_norm_stats(args, config, val_demos: set[str]):
    """Resolve --action_norm into [H, D] stats (None keeps the pi05_droid constants)."""
    if args.action_norm == "droid":
        return None
    if args.action_norm != "demo_delta":
        raise ValueError(f"Unknown --action_norm {args.action_norm!r}.")
    with h5py.File(args.hdf5_path, "r") as f:
        names = [
            name
            for name in sorted(f["data"].keys())[args.demo_offset :: args.demo_stride]
            if name not in val_demos
        ]
    raw = demo_action_norm_stats(
        args.hdf5_path,
        action_horizon=config.model.action_horizon,
        demo_names=names,
        pooled=bool(args.action_norm_pooled),
        action_offset=int(args.action_offset),
    )
    return {
        "mean": np.asarray(raw["mean"], dtype=np.float32),
        "std": np.asarray(raw["std"], dtype=np.float32),
        "meta": {k: v for k, v in raw.items() if k not in ("mean", "std")},
    }


def _write_action_norm_stats(checkpoint_dir: pathlib.Path, stats) -> None:
    """Persist the stats beside the run so the serve path cannot drift from training."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(stats["meta"])
    payload["mean"] = stats["mean"].tolist()
    payload["std"] = stats["std"].tolist()
    (checkpoint_dir / ACTION_NORM_STATS_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def train_bc(args: argparse.Namespace) -> None:
    """Standard epsilon diffusion training on clean action chunks."""
    config = _apply_train_overrides(_config.get_config(args.config), args)
    if getattr(args, "wandb_project", None):
        config = dataclasses.replace(config, project_name=args.wandb_project)
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")
    # B re-noises each clean teacher chunk online and predicts the sampled epsilon.
    model_overrides = {"prediction_type": args.prediction_type}
    if getattr(args, "action_expert_variant", None):
        model_overrides["action_expert_variant"] = args.action_expert_variant
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, **model_overrides)
    )

    use_ddp, local_rank, device = setup_ddp()
    rank = torch.distributed.get_rank() if use_ddp else 0
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    is_main = rank == 0
    set_seed(config.seed, rank)

    resuming = _prepare_checkpoint_dir(config, is_main)
    if use_ddp:
        torch.distributed.barrier()

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        if config.wandb_enabled and wandb.run is not None and getattr(args, "wandb_run_name", None):
            wandb.run.name = args.wandb_run_name
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    cache_mode = bool(args.cache_path)
    batch_layout = None
    if cache_mode:
        if config.model.prediction_type != "epsilon":
            raise ValueError("Teacher action-chunk B must use standard epsilon training.")
        val_demos = set()
        action_norm_stats = None
        grouped_dataset = MPCScoreDataset(
            hdf5_path=args.hdf5_path,
            cache_path=args.cache_path,
            config=config,
            prompt=args.prompt,
            observation_cache_path=args.observation_cache_path,
            build_observation_cache=False,
        )
        dataset = MPCActionChunkDataset(grouped_dataset)
        train_obs, val_obs, split_manifest = split_observations(
            dataset.unique_demo_names,
            dataset.unique_step_indices,
            val_fraction=args.val_fraction,
            seed=args.split_seed,
        )
        train_dataset = torch.utils.data.Subset(dataset, train_obs.tolist())
        batch_layout = resolve_ref_batch_layout(
            global_target_batch=config.batch_size,
            trajectories_per_observation=dataset.trajectories_per_observation,
            world_size=world_size,
            targets_per_observation=args.targets_per_observation,
        )
        if is_main:
            write_split_manifest(
                config.checkpoint_dir / "observation_split.json", split_manifest
            )
            logging.info(
                "Teacher action split: train_obs=%s val_obs=%s train_chunks=%s",
                len(train_obs), len(val_obs),
                len(train_dataset) * dataset.trajectories_per_observation,
            )
    else:
        if args.targets_per_observation is not None:
            raise ValueError("--targets_per_observation requires --cache_path.")
        val_demos = _held_out_demos(args.hdf5_path, args.val_demos)
        action_norm_stats = _bc_action_norm_stats(args, config, val_demos)
        if is_main and action_norm_stats is not None:
            _write_action_norm_stats(config.checkpoint_dir, action_norm_stats)
        if use_ddp:
            torch.distributed.barrier()
        dataset = BCDemoActionDataset(
            hdf5_path=args.hdf5_path,
            config=config,
            prompt=args.prompt,
            demo_stride=args.demo_stride,
            demo_offset=args.demo_offset,
            stride=args.stride,
            action_norm_stats=action_norm_stats,
            exclude_demos=val_demos,
            action_offset=args.action_offset,
        )
        train_dataset = dataset
    if is_main:
        logging.info(
            "BC action_norm=%s action_offset=%s held_out=%s",
            args.action_norm, args.action_offset, sorted(val_demos)
        )
        if action_norm_stats is not None:
            logging.info(
                "BC action scale row0=%s row%s=%s",
                np.round(action_norm_stats["std"][0, :7], 5),
                config.model.action_horizon - 1,
                np.round(action_norm_stats["std"][-1, :7], 5),
            )
        logging.info(
            "BC dataset: demos=%s windows=%s prediction_type=%s",
            len(dataset.demo_names),
            len(dataset),
            config.model.prediction_type,
        )
        logging.info(
            "BC recipe: freeze_dino=%s ema_decay=%s aug_shift_px=%s expert=%s",
            config.model.freeze_dino_encoder,
            args.ema_decay,
            args.aug_shift_px,
            config.model.action_expert_variant,
        )
        (config.checkpoint_dir / "bc_metadata.json").write_text(
            json.dumps(
                {
                    "mode": "teacher_action_chunk_epsilon" if cache_mode else "demo_bc_x0",
                    "prediction_type": config.model.prediction_type,
                    "hdf5_path": str(args.hdf5_path),
                    "cache_path": str(args.cache_path) if cache_mode else None,
                    "num_train_chunks": (
                        len(train_dataset) * dataset.trajectories_per_observation
                        if cache_mode else len(train_dataset)
                    ),
                    "grouped_batch": dataclasses.asdict(batch_layout) if cache_mode else None,
                    "val_fraction": float(args.val_fraction),
                    "split_seed": int(args.split_seed),
                    "prompt": args.prompt,
                    "demo_stride": int(args.demo_stride),
                    "demo_offset": int(args.demo_offset),
                    "stride": int(args.stride),
                    "num_demos": len(dataset.demo_names),
                    "num_windows": (
                        len(dataset) * dataset.trajectories_per_observation
                        if cache_mode else len(dataset)
                    ),
                    "num_observations": len(dataset) if cache_mode else None,
                    "demo_names": list(dataset.demo_names),
                    "freeze_dino_encoder": bool(config.model.freeze_dino_encoder),
                    "action_expert_variant": str(config.model.action_expert_variant),
                    "ema_decay": float(args.ema_decay),
                    "aug_shift_px": int(args.aug_shift_px),
                    "action_norm": str(args.action_norm),
                    "action_norm_pooled": bool(args.action_norm_pooled),
                    "action_offset": int(args.action_offset),
                    "val_demos": sorted(val_demos),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if cache_mode:
        assert batch_layout is not None
        local_batch = batch_layout.local_observation_batch
        per_rank_batch = [local_batch] * world_size
    else:
        # Non-cache demo BC keeps its historical flat-example batching.
        per_rank_batch = [
            config.batch_size // world_size
            + (1 if r < config.batch_size % world_size else 0)
            for r in range(world_size)
        ]
        local_batch = per_rank_batch[rank]
    if local_batch <= 0:
        raise ValueError(f"batch_size={config.batch_size} too small for world_size={world_size}.")
    if is_main:
        if cache_mode:
            logging.info(
                "Teacher action grouped batch: target_batch=%s observations=%s "
                "targets_per_observation=%s observations_per_rank=%s",
                batch_layout.global_target_batch,
                batch_layout.global_observation_batch,
                batch_layout.targets_per_observation,
                batch_layout.local_observation_batch,
            )
        else:
            logging.info("BC batch: global=%s per_rank=%s", config.batch_size, per_rank_batch)
        if config.wandb_enabled and wandb.run is not None:
            wandb.config.update(
                {"global_batch_size": config.batch_size,
                 "per_rank_batch_sizes": per_rank_batch,
                 "world_size": world_size,
                 "grouped_batch": dataclasses.asdict(batch_layout) if cache_mode else None},
                allow_val_change=True,
            )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        if use_ddp
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_config = dataset.data_config

    model = _proxy_score.ProxyScorePytorch(config.model).to(device)
    if getattr(args, "init_from", None):
        init_path = os.fspath(args.init_from)
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "model.safetensors")
        safetensors.torch.load_model(model, init_path, device=str(device))
        logging.info("Initialized BC proxy from %s", init_path)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    trainable_params = _optimizer_parameters(
        model, drop_frozen=bool(config.model.freeze_dino_encoder)
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = load_checkpoint(model, optimizer, config.checkpoint_dir, device) if resuming else 0
    module = _unwrap_model(model)
    ema = _WeightEMA(module, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and resuming:
        # model.safetensors holds the EMA weights (just loaded, so the shadow is correct);
        # the live weights come back from the sidecar.
        raw_path = config.checkpoint_dir / f"{global_step}" / "model_raw.safetensors"
        if raw_path.exists():
            safetensors.torch.load_model(module, raw_path, device=str(device))
        else:
            logging.warning("No %s; resuming EMA from the EMA weights.", raw_path)
    lr_schedule = _make_lr_schedule(config)

    model.train()
    metric_name = "teacher_action_epsilon_loss" if cache_mode else "bc_x0_loss"
    progress_desc = "Teacher action epsilon" if cache_mode else "Demo BC x0"
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc=progress_desc)
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
            input_batch, actions = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            input_batch, actions = next(data_iter)

        if cache_mode:
            assert batch_layout is not None
            trajectory_indices = sample_trajectory_indices(
                actions.shape[0],
                batch_layout.trajectories_per_observation,
                batch_layout.targets_per_observation,
            )
            observation_indices = torch.arange(actions.shape[0])[:, None]
            actions = actions[observation_indices, trajectory_indices]

        input_batch = move_to_device(input_batch, device)
        observation = _model.Observation.from_dict(input_batch)
        if args.aug_shift_px:
            # After normalization, before the model's own train-time preprocessing.
            observation = _augment_observation(observation, args.aug_shift_px)
        actions = actions.to(torch.float32).to(device)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(global_step)

        losses = model(observation, actions)
        losses = ensure_tensor_loss(losses, device)
        loss = losses.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optimizer.step()
        if ema is not None:
            ema.update(module)
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
                "step=%s %s=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                completed_step,
                metric_name,
                avg_loss,
                avg_lr,
                avg_grad_norm,
                elapsed,
            )
            if config.wandb_enabled:
                step_time = elapsed / config.log_interval
                wandb.log(
                    {
                        metric_name: avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "time_per_step": step_time,
                        "it_per_s": 1.0 / max(step_time, 1e-9),
                        "global_step": completed_step,
                    },
                    step=completed_step,
                )
            metrics = []
            start_time = time.time()

        global_step = completed_step
        _save_bc_checkpoint(
            model, optimizer, global_step, config, is_main, data_config, ema
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    metric_name: f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def _apply_train_overrides(config: _config.TrainConfig, args: argparse.Namespace):
    if getattr(args, "config_name", None):
        # Name override for checkpoint pathing/logging when a registry twin does not exist yet
        # (e.g. score_task_stack reuses the score_task_capsule ProxyScoreConfig unchanged).
        config = dataclasses.replace(config, name=args.config_name)
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.train_steps is not None:
        config = dataclasses.replace(config, num_train_steps=args.train_steps)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.checkpoint_base_dir is not None:
        config = dataclasses.replace(config, checkpoint_base_dir=args.checkpoint_base_dir)
    return dataclasses.replace(
        config,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        wandb_enabled=not bool(args.no_wandb),
    )


def _prepare_checkpoint_dir(config: _config.TrainConfig, is_main: bool) -> bool:
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
    return resuming


def _make_lr_schedule(config: _config.TrainConfig):
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

    return lr_schedule


def train(args: argparse.Namespace) -> None:
    config = _apply_train_overrides(_config.get_config(args.config), args)
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")

    use_ddp, local_rank, device = setup_ddp()
    rank = torch.distributed.get_rank() if use_ddp else 0
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    is_main = rank == 0
    set_seed(config.seed, rank)

    resuming = _prepare_checkpoint_dir(config, is_main)

    if use_ddp:
        torch.distributed.barrier()

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    cache_sync_group = None
    if use_ddp:
        # Cache preparation can take longer than NCCL's 10-minute watchdog.
        # Keep the default NCCL group idle and synchronize this one-time CPU job
        # through a long-timeout Gloo group instead.
        cache_sync_group = torch.distributed.new_group(
            backend="gloo",
            timeout=datetime.timedelta(hours=2),
        )

    dataset = None
    if is_main:
        dataset = MPCScoreDataset(
            hdf5_path=args.hdf5_path,
            cache_path=args.cache_path,
            config=config,
            prompt=args.prompt,
            observation_cache_path=args.observation_cache_path,
            build_observation_cache=True,
        )
    if cache_sync_group is not None:
        torch.distributed.barrier(group=cache_sync_group)
        torch.distributed.destroy_process_group(cache_sync_group)
    if dataset is None:
        dataset = MPCScoreDataset(
            hdf5_path=args.hdf5_path,
            cache_path=args.cache_path,
            config=config,
            prompt=args.prompt,
            observation_cache_path=args.observation_cache_path,
            build_observation_cache=False,
        )
    if is_main:
        logging.info(
            "Loaded grouped reverse cache: observations=%s trajectories=%s K=%s "
            "labels=%s labels_per_observation=%s",
            dataset.num_observations,
            dataset.metadata["num_trajectories"],
            dataset.trajectories_per_observation,
            dataset.num_labels,
            dataset.labels_per_observation,
        )
    train_indices, val_indices, split_manifest = split_observations(
        dataset.unique_demo_names,
        dataset.unique_step_indices,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
    if is_main:
        write_split_manifest(config.checkpoint_dir / "observation_split.json", split_manifest)
        logging.info(
            "Observation split: train=%s val=%s seed=%s",
            len(train_indices), len(val_indices), args.split_seed,
        )
    batch_layout = resolve_ref_batch_layout(
        global_target_batch=config.batch_size,
        trajectories_per_observation=dataset.trajectories_per_observation,
        world_size=world_size,
        targets_per_observation=args.targets_per_observation,
    )
    local_observation_batch = batch_layout.local_observation_batch
    if len(train_dataset) < batch_layout.global_observation_batch:
        raise ValueError(
            f"MPC score cache has {len(train_dataset)} training observations, fewer than the "
            f"global observation batch={batch_layout.global_observation_batch}."
        )
    if is_main:
        logging.info(
            "Score-cache grouped batch: target_batch=%s observations=%s "
            "targets_per_observation=%s sampling=one_level_per_distinct_trajectory "
            "observations_per_rank=%s",
            batch_layout.global_target_batch,
            batch_layout.global_observation_batch,
            batch_layout.targets_per_observation,
            batch_layout.local_observation_batch,
        )
        (config.checkpoint_dir / "train_metadata.json").write_text(
            json.dumps(
                {
                    "mode": "cached_teacher_epsilon",
                    "batching": dataclasses.asdict(batch_layout),
                    "sampling": "one_random_level_per_distinct_trajectory",
                    "cache_path": str(args.cache_path),
                    "train_steps": int(config.num_train_steps),
                    "seed": int(config.seed),
                    "split_seed": int(args.split_seed),
                },
                indent=2,
                sort_keys=True,
            )
        )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        if use_ddp
        else None
    )
    # The mmap cache already contains training-ready tensors.  Forking loader
    # workers after CUDA/NCCL initialization can leave one DDP rank waiting on
    # its input queue while the other rank blocks in an all-reduce.  Loading
    # these small mmap slices in the rank process is both cheaper and safer.
    if args.num_workers != 0 and is_main:
        logging.info(
            "Shared mmap observation cache uses num_workers=0; ignoring requested num_workers=%s.",
            args.num_workers,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_observation_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate_cache_batch,
    )
    data_config = dataset.data_config

    model = _proxy_score.ProxyScorePytorch(config.model).to(device)

    if getattr(args, 'init_from', None):

        _ip = os.fspath(args.init_from)

        if os.path.isdir(_ip):

            _ip = os.path.join(_ip, 'model.safetensors')

        safetensors.torch.load_model(model, _ip, device=str(device))

        logging.info('Initialized score proxy from %s', _ip)
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

    lr_schedule = _make_lr_schedule(config)

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
            input_batch, x_t, score_target, time_cond = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            input_batch, x_t, score_target, time_cond = next(data_iter)

        label_indices = sample_cached_label_indices(
            x_t.shape[0],
            dataset.trajectories_per_observation,
            dataset.labels_per_trajectory,
            batch_layout.targets_per_observation,
        )
        observation_indices = torch.arange(x_t.shape[0])[:, None]
        x_t = x_t[observation_indices, label_indices]
        score_target = score_target[observation_indices, label_indices]
        time_cond = time_cond[observation_indices, label_indices]

        # Keep cached images as compact uint8 tensors through the worker/pinned-memory
        # path. Conversion to float, channel permutation, and augmentations happen as
        # one batched operation on the GPU.
        input_batch = move_to_device(input_batch, device)
        observation = _model.Observation.from_dict(input_batch)
        x_t = x_t.to(torch.float32).to(device)
        score_target = score_target.to(torch.float32).to(device)
        time_cond = time_cond.to(torch.float32).to(device)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(global_step)

        losses = model(
            observation,
            x_t,
            time=time_cond,
            model_output_target=score_target,
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
    cache_parser.add_argument("--base_action_stats", required=True)
    cache_parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    cache_parser.add_argument("--base_checkpoint_dir", default=DEFAULT_BASE_CHECKPOINT_DIR)
    cache_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    cache_parser.add_argument("--task", default="weight")
    cache_parser.add_argument(
        "--task_module",
        default=None,
        help="File path of an external offline-task hook module (episode_signals/frame_context/"
             "attach_priority_cost/prepare_planner) for tasks outside this repo, e.g. the "
             "MimicGen stack port. Used with --mpc_cost priority + --vlm_cost_config.",
    )
    cache_parser.add_argument("--device", default="cuda")
    cache_parser.add_argument("--seed", type=int, default=0)
    cache_parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of unique observations before K-fold trajectory expansion.",
    )
    cache_parser.add_argument(
        "--trajectories_per_observation",
        type=int,
        default=1,
        help="Independent Gaussian reverse trajectories generated for every observation.",
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
    cache_parser.add_argument("--cost_executable_actions", action="store_true")
    cache_parser.add_argument("--mpc_logit_norm", choices=("raw", "std"), default="raw")
    cache_parser.add_argument("--obs_shard", type=int, default=0)
    cache_parser.add_argument("--label_seed", type=int, default=None,
                              help="Torch RNG seed for proposal draws only (index shuffle keeps "
                              "--seed). Same --seed + different --label_seed = same tuples, fresh "
                              "draws: the label-noise repeatability probe.")
    cache_parser.add_argument("--obs_num_shards", type=int, default=1)
    cache_parser.add_argument("--mpc_ancestral_eta", type=float, default=0.0)
    cache_parser.add_argument(
        "--mpc_cost",
        default="grasp_flow",
        # capsule_flow: the capsule twin of grasp_flow (open lid / grasp pod / place pod).
        choices=("priority", "ref_style", "explore", "grasp_flow", "grasp_flow_fake", "capsule_flow"),
    )
    cache_parser.add_argument(
        "--vlm_cost_config",
        default=None,
        help="Cost YAML (repo-relative) for --mpc_cost priority, e.g. "
             "vlm_dp/configs/parity23_stall_release.yaml. Required for priority.",
    )
    cache_parser.add_argument("--mpc_interpolate", action="store_true")
    cache_parser.add_argument(
        "--mpc_interpolation_method",
        choices=("bspline", "linear"),
        default="bspline",
    )
    cache_parser.add_argument("--control_frequency", type=float, default=40.0)
    cache_parser.add_argument("--interpolate_frequency", type=float, default=5.0)
    cache_parser.set_defaults(func=generate_cache)

    train_parser = subparsers.add_parser("train")
    _add_config_arg(train_parser)
    train_parser.add_argument("--hdf5_path", required=True)
    train_parser.add_argument("--cache_path", required=True)
    train_parser.add_argument(
        "--observation_cache_path",
        default=None,
        help="Shared mmap observation cache directory (default: <cache_path>.observations).",
    )
    train_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    train_parser.add_argument(
        "--config_name",
        default=None,
        help="Override the config's name for checkpoint pathing/logging (the model and data "
             "config are untouched), e.g. score_task_stack over --config score_task_capsule.",
    )
    train_parser.add_argument("--exp_name", default="ref")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--no_wandb", action="store_true")
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--train_steps", type=int, default=None)
    train_parser.add_argument("--batch_size", type=int, default=None)
    train_parser.add_argument(
        "--targets_per_observation",
        type=int,
        default=None,
        help="Grouped cache targets per observation (default: K). For K=8 and "
             "--batch_size 32, both A/B use 4 observations x 8 targets.",
    )
    train_parser.add_argument("--val_fraction", type=float, default=0.1)
    train_parser.add_argument("--split_seed", type=int, default=42)
    train_parser.add_argument("--checkpoint_base_dir", default=None)
    train_parser.add_argument("--init_from", default=None,
                              help="Checkpoint dir (or model.safetensors) to initialize the model "
                              "from before training -- the shared-init half of PPS C2.")
    train_parser.set_defaults(func=train)

    bc_parser = subparsers.add_parser(
        "train-bc",
        help="Standard epsilon diffusion training on clean action chunks.",
    )
    _add_config_arg(bc_parser)
    bc_parser.add_argument("--hdf5_path", required=True)
    bc_parser.add_argument(
        "--cache_path",
        default=None,
        help="v5 epsilon cache; when set, train on each trajectory's final teacher chunk.",
    )
    bc_parser.add_argument("--observation_cache_path", default=None)
    bc_parser.add_argument("--val_fraction", type=float, default=0.1)
    bc_parser.add_argument("--split_seed", type=int, default=42)
    bc_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    bc_parser.add_argument(
        "--config_name",
        default=None,
        help="Override the config's name for checkpoint pathing/logging, e.g. score_task_stack.",
    )
    bc_parser.add_argument("--exp_name", default="bc")
    bc_parser.add_argument("--demo_stride", type=int, default=1,
                           help="Demo subset: sorted demo names [--demo_offset::--demo_stride].")
    bc_parser.add_argument("--demo_offset", type=int, default=0)
    bc_parser.add_argument("--stride", type=int, default=1,
                           help="Frame stride inside each selected demo.")
    bc_parser.add_argument("--overwrite", action="store_true")
    bc_parser.add_argument("--resume", action="store_true")
    bc_parser.add_argument("--no_wandb", action="store_true")
    bc_parser.add_argument("--num_workers", type=int, default=2)
    bc_parser.add_argument("--train_steps", type=int, default=None)
    bc_parser.add_argument("--batch_size", type=int, default=None)
    bc_parser.add_argument(
        "--targets_per_observation",
        type=int,
        default=None,
        help="With --cache_path, grouped targets per observation (default: K). "
             "Must match A for a controlled comparison.",
    )
    bc_parser.add_argument("--checkpoint_base_dir", default=None)
    bc_parser.add_argument("--init_from", default=None,
                           help="Checkpoint dir (or model.safetensors) to initialize from.")
    bc_parser.add_argument("--wandb_project", default=None,
                           help="Override config.project_name for wandb.")
    bc_parser.add_argument("--wandb_run_name", default=None,
                           help="Rename the wandb run (exp_name still drives checkpoint paths).")
    # Low-data recipe knobs; every default reproduces the pre-recipe behaviour.
    bc_parser.add_argument("--ema_decay", type=float, default=0.0,
                           help="EMA of the trainable weights (0 = off). When >0 the checkpoint's "
                           "model.safetensors holds the EMA weights and model_raw.safetensors the "
                           "live ones, so serving picks up EMA unchanged. Try 0.999.")
    bc_parser.add_argument("--aug_shift_px", type=int, default=0,
                           help="DrQ pad-and-random-crop shift in pixels, applied to both cameras "
                           "independently per sample, train only (0 = off). Try 4 at 224px.")
    bc_parser.add_argument("--action_expert_variant", default=None,
                           help="Override the config's action expert size, e.g. gemma_2m. Serve "
                           "such a checkpoint with the matching serve --action_expert_variant.")
    bc_parser.add_argument("--action_norm", default="droid",
                           choices=("droid", "demo_delta"),
                           help="Action normalisation. 'droid' (default) keeps the pi05_droid "
                           "quantile constants -- the shipped behaviour, ~13x too wide for MG "
                           "deltas. 'demo_delta' measures the scale of the DeltaActions target "
                           "on the training demos and writes it next to the checkpoint.")
    bc_parser.add_argument("--action_norm_pooled", action="store_true",
                           help="demo_delta with one per-dim scale for the whole chunk "
                           "(eval_mg.demo_delta_stats' reduction) instead of per chunk row.")
    bc_parser.add_argument("--action_offset", type=int, default=1,
                           help="Chunk row 0 is joint_actions[t + offset]. 1 (default) is the "
                           "shipped alignment; 0 makes row 0 the demo's immediate next action, "
                           "which is what setup/07_relabel_replay_gate.py's shim replays.")
    bc_parser.add_argument("--val_demos", type=int, default=0,
                           help="Hold out the last N demos by numeric index for the offline gate.")
    bc_parser.add_argument("--prediction_type", default="x0",
                           choices=("x0", "epsilon", "score", "regress"),
                           help="Training head. 'regress' is plain chunked regression (no noising) "
                           "and cannot be served through the score-space path.")
    bc_parser.set_defaults(func=train_bc)
    return parser


def main() -> None:
    init_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
