"""Distill an MPC score policy for score-space PPS steering.

This is the score-space replacement for the old policy distillation path.  The
reference target is not a pi0/pi05 teacher action.  Instead, labels are generated
by querying the same FK/cost MPC score estimator used at evaluation time:

    s_ref(x_t, o, t) = MPC.estimate_mbd_score_action_prox(x_t, o, context, t)

The script has two explicit stages so expensive MPC labels can be inspected and
reused:

  python scripts/train_mpc_proxy_score_pytorch.py generate-cache \
      --config proxy_score_mpc_weight_jointpos \
      --hdf5_path /home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5 \
      --cache_path ../data/weight/mpc_score_ref_labels.npz

  python scripts/train_mpc_proxy_score_pytorch.py train \
      --config proxy_score_mpc_weight_jointpos \
      --hdf5_path /home/chuanruo/diffusion_policy/data/weight/generated_dataset.hdf5 \
      --cache_path ../data/weight/mpc_score_ref_labels.npz \
      --exp_name reference --overwrite
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
from openpi.policies import policy_config
import openpi.training.config as _config
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


def _data_config_and_transform(config: _config.TrainConfig):
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


def _raw_hdf5_sample(
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


def _raw_policy_observation(
    demo: h5py.Group,
    step_idx: int,
    *,
    action_horizon: int,
    prompt: str,
) -> dict[str, Any]:
    sample = _raw_hdf5_sample(
        demo,
        step_idx,
        action_horizon=action_horizon,
        prompt=prompt,
    )
    return {
        "observation/exterior_image_1_left": sample["exterior_image_1_left"],
        "observation/wrist_image_left": sample["wrist_image_left"],
        "observation/joint_position": sample["joint_position"],
        "observation/gripper_position": sample["gripper_position"],
        "actions": sample["actions"],
        "prompt": prompt,
    }


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


def _mpc_context_from_hdf5(
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
    max_labels: int | None,
    stride: int,
    seed: int,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    with h5py.File(hdf5_path, "r") as f:
        all_indices = []
        for demo_name in sorted(f["data"].keys()):
            demo = f["data"][demo_name]
            length = len(demo["obs/joint_actions"])
            max_step = length - action_horizon - 1
            if max_step <= 0:
                continue
            all_indices.extend((demo_name, step) for step in range(0, max_step, stride))
    rng.shuffle(all_indices)
    if max_labels is not None:
        all_indices = all_indices[:max_labels]
    return all_indices


def generate_cache(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config)
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    base_config = _config.get_config(args.base_config)
    base_policy = policy_config.create_trained_policy(
        base_config,
        args.base_checkpoint_dir,
        pytorch_device=device_name,
    )
    score_data_config, _ = _data_config_and_transform(config)
    base_norm_stats = (getattr(base_policy, "_metadata", {}) or {}).get("output_norm_stats")
    if _norm_stats_fingerprint(base_norm_stats) != _norm_stats_fingerprint(score_data_config.norm_stats):
        raise ValueError(
            "Base policy and score config norm_stats differ. Use one shared norm_stats source "
            "before generating MPC score labels."
        )

    planner = SimFreeMPC(
        base_policy,
        SimFreeMPCConfig(
            task_name=args.task,
            num_samples=args.mpc_num_samples,
            iterations=args.mpc_iterations,
            noise=args.mpc_noise,
            temperature=args.mpc_temperature,
            beta_opt_iter=args.mpc_beta_opt_iter,
            beta_horizon=args.mpc_beta_horizon,
            action_dims=config.model.action_dim,
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
        max_labels=args.max_labels,
        stride=args.stride,
        seed=args.seed,
    )
    if not indices:
        raise ValueError("No valid HDF5 windows found for MPC score label generation.")

    rng = np.random.default_rng(args.seed)
    demo_names: list[str] = []
    step_indices: list[int] = []
    iterations: list[int] = []
    times: list[float] = []
    x_t_values: list[np.ndarray] = []
    score_values: list[np.ndarray] = []
    cost_min: list[float] = []
    score_norm: list[float] = []

    num_iterations = int(args.num_steps) + 1
    if num_iterations > config.model.ddim_num_train_timesteps:
        raise ValueError("--num_steps + 1 must be <= ddim_num_train_timesteps.")

    with h5py.File(args.hdf5_path, "r") as f:
        pbar = tqdm.tqdm(indices, desc="MPC score labels")
        for demo_name, step_idx in pbar:
            demo = f["data"][demo_name]
            raw_obs = _raw_policy_observation(
                demo,
                step_idx,
                action_horizon=config.model.action_horizon,
                prompt=args.prompt,
            )
            base_obs, base_inputs = base_policy.obs_to_input(raw_obs)
            clean_actions = base_inputs["actions"].to(device=device, dtype=torch.float32)

            iteration = int(rng.integers(0, num_iterations))
            alpha, _ = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=num_iterations,
                num_train_timesteps=config.model.ddim_num_train_timesteps,
            )
            sqrt_alpha = float(np.sqrt(max(alpha, 1e-12)))
            sqrt_beta = float(np.sqrt(max(1.0 - alpha, 1e-12)))
            noise = torch.randn_like(clean_actions, device=device)
            x_t = sqrt_alpha * clean_actions + sqrt_beta * noise

            context = _mpc_context_from_hdf5(
                demo,
                step_idx,
                task=args.task,
                subtask_mode=args.subtask_mode,
            )
            score, diagnostics = planner.estimate_mbd_score_action_prox(
                x_t,
                base_inputs,
                context,
                iteration=iteration,
                num_iterations=num_iterations,
            )

            demo_names.append(demo_name)
            step_indices.append(int(step_idx))
            iterations.append(iteration)
            times.append(
                _time_from_iteration(
                    iteration=iteration,
                    num_iterations=num_iterations,
                    num_train_timesteps=config.model.ddim_num_train_timesteps,
                )
            )
            x_t_values.append(x_t[0].detach().cpu().numpy().astype(np.float32))
            score_values.append(score[0].detach().cpu().numpy().astype(np.float32))
            cost_min.append(float(diagnostics.get("cost_min", np.nan)))
            score_norm.append(float(diagnostics.get("score_norm", np.nan)))
            pbar.set_postfix({"cost_min": f"{cost_min[-1]:.3f}", "score_norm": f"{score_norm[-1]:.2f}"})

    metadata = {
        "label_type": "mpc_score_action_prox",
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
        "stored_action_dim": int(x_t_values[0].shape[-1]) if x_t_values else None,
        "norm_stats_fingerprint": _norm_stats_fingerprint(score_data_config.norm_stats),
        "use_quantile_norm": bool(score_data_config.use_quantile_norm),
        "subtask_mode": args.subtask_mode,
        "mpc": {
            "num_samples": int(args.mpc_num_samples),
            "iterations": int(args.mpc_iterations),
            "noise": float(args.mpc_noise),
            "temperature": float(args.mpc_temperature),
            "beta_opt_iter": float(args.mpc_beta_opt_iter),
            "beta_horizon": float(args.mpc_beta_horizon),
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
        iteration=np.asarray(iterations, dtype=np.int64),
        time=np.asarray(times, dtype=np.float32),
        x_t=np.asarray(x_t_values, dtype=np.float32),
        score=np.asarray(score_values, dtype=np.float32),
        cost_min=np.asarray(cost_min, dtype=np.float32),
        score_norm=np.asarray(score_norm, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logging.info("Wrote %s MPC score labels to %s", len(demo_names), cache_path)


class MpcScoreCacheDataset(torch.utils.data.Dataset):
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
        self.x_t = self.cache["x_t"].astype(np.float32)
        self.score = self.cache["score"].astype(np.float32)
        self.time = self.cache["time"].astype(np.float32)
        self.prompt = prompt
        self.config = config
        self.data_config, self.input_transform = _data_config_and_transform(config)
        self._h5 = None

        metadata = json.loads(str(self.cache["metadata_json"].item()))
        expected_fingerprint = _norm_stats_fingerprint(self.data_config.norm_stats)
        if metadata.get("norm_stats_fingerprint") != expected_fingerprint:
            raise ValueError(
                "MPC score cache norm_stats fingerprint does not match this training config. "
                "Regenerate the cache with the shared base/task/ref norm_stats."
            )
        if int(metadata.get("ddim_num_train_timesteps", -1)) != int(config.model.ddim_num_train_timesteps):
            raise ValueError("MPC score cache DDIM scheduler does not match this training config.")

    def __len__(self) -> int:
        return int(self.x_t.shape[0])

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        demo = self._file()["data"][self.demo_names[idx]]
        raw = _raw_hdf5_sample(
            demo,
            int(self.step_indices[idx]),
            action_horizon=self.config.model.action_horizon,
            prompt=self.prompt,
        )
        inputs = self.input_transform(jax.tree.map(lambda x: x, raw))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)), inputs)
        return (
            inputs,
            torch.from_numpy(self.x_t[idx]),
            torch.from_numpy(self.score[idx]),
            torch.tensor(self.time[idx], dtype=torch.float32),
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


def train_from_cache(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config)
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    config = dataclasses.replace(
        config,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        wandb_enabled=not bool(args.no_wandb),
    )
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(f"{args.config!r} must use ProxyScoreConfig.")

    use_ddp, local_rank, device = setup_ddp()
    is_main = local_rank == 0
    set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        if not config.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume."
            )
        resuming = True
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    dataset = MpcScoreCacheDataset(
        hdf5_path=args.hdf5_path,
        cache_path=args.cache_path,
        config=config,
        prompt=args.prompt,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate_cache_batch,
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

    optim = torch.optim.AdamW(
        get_model_parameters(model),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = load_checkpoint(model, optim, config.checkpoint_dir, device) if resuming else 0

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
    infos = []
    start_time = time.time()
    data_iter = iter(train_loader)
    while global_step < config.num_train_steps:
        try:
            observation, x_t, score_target, time_cond = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            observation, x_t, score_target, time_cond = next(data_iter)

        observation = move_to_device(observation, device)
        x_t = x_t.to(torch.float32).to(device)
        score_target = score_target.to(torch.float32).to(device)
        time_cond = time_cond.to(torch.float32).to(device)

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        losses = model(
            observation,
            x_t,
            time=time_cond,
            score_target=score_target,
        )
        losses = ensure_tensor_loss(losses, device)
        loss = losses.mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            get_model_parameters(model),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        infos.append(
            {
                "loss": float(loss.detach().cpu()),
                "lr": float(optim.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
            }
        )
        completed_step = global_step + 1
        if is_main and completed_step % config.log_interval == 0 and infos:
            elapsed = time.time() - start_time
            avg_loss = sum(info["loss"] for info in infos) / len(infos)
            avg_lr = sum(info["lr"] for info in infos) / len(infos)
            avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)
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
            infos = []
            start_time = time.time()

        global_step = completed_step
        save_checkpoint(model, optim, global_step, config, is_main, data_config)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "mpc_score_loss": f"{loss.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
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

    gen = subparsers.add_parser("generate-cache")
    _add_config_arg(gen)
    gen.add_argument("--hdf5_path", required=True)
    gen.add_argument("--cache_path", required=True)
    gen.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    gen.add_argument("--base_checkpoint_dir", default=DEFAULT_BASE_CHECKPOINT_DIR)
    gen.add_argument("--prompt", default=DEFAULT_PROMPT)
    gen.add_argument("--task", default="weight")
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--max_labels", type=int, default=None)
    gen.add_argument("--stride", type=int, default=4)
    gen.add_argument("--num_steps", type=int, default=10)
    gen.add_argument("--subtask_mode", choices=("heuristic", "empty"), default="heuristic")
    gen.add_argument("--mpc_num_samples", type=int, default=512)
    gen.add_argument("--mpc_iterations", type=int, default=2)
    gen.add_argument("--mpc_noise", type=float, default=0.35)
    gen.add_argument("--mpc_temperature", type=float, default=0.15)
    gen.add_argument("--mpc_beta_opt_iter", type=float, default=1.0)
    gen.add_argument("--mpc_beta_horizon", type=float, default=1.0)
    gen.add_argument("--mpc_cost", default="grasp_flow", choices=("priority", "ref_style", "explore", "grasp_flow"))
    gen.add_argument("--mpc_interpolate", action="store_true")
    gen.add_argument("--control_frequency", type=float, default=15.0)
    gen.add_argument("--interpolate_frequency", type=float, default=5.0)
    gen.set_defaults(func=generate_cache)

    train = subparsers.add_parser("train")
    _add_config_arg(train)
    train.add_argument("--hdf5_path", required=True)
    train.add_argument("--cache_path", required=True)
    train.add_argument("--prompt", default=DEFAULT_PROMPT)
    train.add_argument("--exp_name", default="reference")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no_wandb", action="store_true")
    train.add_argument("--num_workers", type=int, default=2)
    train.set_defaults(func=train_from_cache)
    return parser


def main() -> None:
    init_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
