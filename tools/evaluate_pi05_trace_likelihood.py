#!/usr/bin/env python3
"""Evaluate pi0.5 conditional action likelihood on batched rollout state traces.

The trace stores simulator state rather than pixels.  This script recreates the weight
environment, resets with each recorded seed (recovering randomized camera extrinsics), restores
the logged physical state, renders fresh policy observations, and scores the following 15
executed actions with the pi0.5 probability-flow ODE.  Divergence is estimated with fixed
Rademacher Hutchinson probes along each ODE path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from tqdm.auto import tqdm

_REPO_DIR = Path(__file__).resolve().parents[1]
_ISAACLAB_DIR = _REPO_DIR / "IsaacLab"
for _package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _source = str(_ISAACLAB_DIR / "source" / _package)
    if _source not in sys.path:
        sys.path.insert(0, _source)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

from isaaclab.app import AppLauncher
from sim_free_mpc.trace_action_chunks import BatchedReplanTracker, weight_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay PPS state traces and estimate pi0.5 conditional action likelihood."
    )
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument("--prompt", default="put pear and apple on the scale")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_DIR / "openpi/checkpoints/pytorch/pi05_droid_jointpos",
    )
    parser.add_argument("--config", default="pi05_droid_jointpos_weight_demo_meanstd")
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        default=(
            _REPO_DIR
            / "openpi/checkpoints/score_task_weight_demo_meanstd/"
            "task_eps_bidir_demo_meanstd/30000/assets/cn356/isaaclab_weight"
        ),
        help="Directory containing norm_stats.json. Defaults to the task rollout's demo mean/std stats.",
    )
    parser.add_argument("--output", type=Path, default=Path("pi05_hutchinson_likelihood.jsonl"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and retain existing output rows, then score only missing chunks.",
    )
    parser.add_argument("--action-horizon", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument(
        "--chunk-stride",
        "--frame-stride",
        dest="chunk_stride",
        type=int,
        default=1,
        help="Score every Nth policy replan/action-chunk state. The default scores every chunk.",
    )
    parser.add_argument(
        "--max-samples-per-seed",
        type=int,
        default=0,
        help="Evenly subsample this many chunks per seed; the default 0 evaluates every generated chunk.",
    )
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed allow-list.")
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--hutchinson-probes", type=int, default=1)
    parser.add_argument(
        "--policy-batch-size",
        type=int,
        default=4,
        help="Number of rendered observations scored together by pi0.5.",
    )
    parser.add_argument("--estimator-seed", type=int, default=20260816)
    parser.add_argument("--time-eps", type=float, default=1.0e-3)
    parser.add_argument("--progress-position", type=int, default=0)
    parser.add_argument(
        "--progress-fd",
        type=int,
        default=-1,
        help="Write the progress bar to this inherited file descriptor; stderr if negative.",
    )
    return parser


@dataclasses.dataclass(frozen=True)
class TraceSample:
    trace: Path
    group_index: int
    lane: int
    seed: int
    step: int
    phase: str
    state: dict[str, Any]
    actions: list[list[float]]


def _eligible_samples(
    trace_path: Path,
    *,
    horizon: int,
    stride: int,
    allowed_seeds: set[int] | None,
) -> Iterable[TraceSample]:
    windows: dict[tuple[int, int], deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=horizon + 1)
    )
    trackers: dict[int, BatchedReplanTracker] = {}
    chunk_indices: dict[tuple[int, int], int] = defaultdict(int)
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A live rollout may have one partial final line. All complete frames remain usable.
                continue
            if record.get("event") != "frame":
                continue
            group_index = int(record["group_index"])
            tracker = trackers.setdefault(group_index, BatchedReplanTracker())
            generated_chunk = tracker.observe_frame(int(record["step"]), record["lanes"])
            for lane_record in record["lanes"]:
                if not lane_record.get("valid", False):
                    continue
                seed = int(lane_record["seed"])
                if allowed_seeds is not None and seed not in allowed_seeds:
                    continue
                key = (group_index, int(lane_record["lane"]))
                window = windows[key]
                lane_record["_generated_action_chunk"] = generated_chunk
                window.append(lane_record)
                if len(window) != horizon + 1:
                    continue
                condition = window[0]
                step = int(record["step"]) - horizon
                if step < 0 or not condition.get("active", False):
                    continue
                if not condition.get("_generated_action_chunk", False):
                    continue
                chunk_index = chunk_indices[key]
                chunk_indices[key] += 1
                if chunk_index % stride:
                    continue
                future_frames = list(window)[1:]
                # The action entering a terminal frame is real, but held actions after that frame
                # are not policy samples and must not enter an action chunk likelihood.
                if any(not frame.get("active", False) for frame in future_frames[:-1]):
                    continue
                future_actions = [frame.get("action") for frame in future_frames]
                if any(action is None for action in future_actions):
                    continue
                yield TraceSample(
                    trace=trace_path,
                    group_index=group_index,
                    lane=int(condition["lane"]),
                    seed=seed,
                    step=step,
                    phase=weight_phase(condition.get("subtasks", {})),
                    state=condition,
                    actions=future_actions,
                )


def load_samples(args: argparse.Namespace) -> list[TraceSample]:
    allowed = None
    if args.seeds:
        allowed = {int(token) for token in args.seeds.split(",") if token.strip()}
    by_seed: dict[int, list[TraceSample]] = defaultdict(list)
    for trace in args.traces:
        if not trace.is_file():
            raise FileNotFoundError(trace)
        for sample in _eligible_samples(
            trace,
            horizon=args.action_horizon,
            stride=args.chunk_stride,
            allowed_seeds=allowed,
        ):
            by_seed[sample.seed].append(sample)

    selected = []
    for seed in sorted(by_seed):
        candidates = sorted(by_seed[seed], key=lambda sample: sample.step)
        limit = int(args.max_samples_per_seed)
        if limit > 0 and len(candidates) > limit:
            indices = [round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
            candidates = [candidates[index] for index in dict.fromkeys(indices)]
        selected.extend(candidates)
    return sorted(selected, key=lambda sample: (sample.seed, sample.step))


def _sample_key(sample: TraceSample) -> tuple[str, int, int, int, int]:
    return (str(sample.trace), sample.group_index, sample.lane, sample.seed, sample.step)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    try:
        return (
            str(row["trace"]),
            int(row["group_index"]),
            int(row["lane"]),
            int(row["seed"]),
            int(row["step"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed likelihood row identity: {row!r}") from error


def _load_resume_rows(
    path: Path,
    args: argparse.Namespace,
    valid_keys: set[tuple[str, int, int, int, int]],
) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"--resume requires a nonempty existing output: {path}")
    rows = []
    seen = set()
    expected = {
        "action_horizon": args.action_horizon,
        "active_action_dim": args.action_dim,
        "ode_steps": args.ode_steps,
        "hutchinson_probes": args.hutchinson_probes,
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            key = _row_key(row)
            if key not in valid_keys:
                raise ValueError(
                    f"Existing row {path}:{line_number} does not belong to the selected traces: {key}"
                )
            if key in seen:
                raise ValueError(f"Duplicate existing likelihood row in {path}:{line_number}: {key}")
            for field, value in expected.items():
                if int(row.get(field, -1)) != value:
                    raise ValueError(
                        f"Resume setting mismatch at {path}:{line_number}: "
                        f"{field}={row.get(field)!r}, expected {value}"
                    )
            seen.add(key)
            rows.append(row)
    return rows


def _restore_state(env, state: dict[str, Any]) -> None:
    import torch

    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
    origin = env.scene.env_origins[0]
    for name, entity in state["entities"].items():
        if name not in env.scene.keys():
            raise KeyError(f"Trace entity {name!r} is absent from replay scene")
        asset = env.scene[name]
        root_values = entity.get("root_state_env")
        if root_values is not None:
            root = torch.as_tensor(root_values, dtype=torch.float32, device=env.device)
            if root.ndim != 1:
                raise ValueError(f"Object collections are not supported in replay: {name}")
            root = root.clone()
            root[:3] += origin
            asset.write_root_state_to_sim(root.unsqueeze(0), env_ids=env_ids)
        if "joint_pos" in entity:
            joint_pos = torch.as_tensor(entity["joint_pos"], dtype=torch.float32, device=env.device)[None]
            joint_vel = torch.as_tensor(entity["joint_vel"], dtype=torch.float32, device=env.device)[None]
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            if "joint_pos_target" in entity:
                target = torch.as_tensor(entity["joint_pos_target"], dtype=torch.float32, device=env.device)[None]
                asset.set_joint_position_target(target, env_ids=env_ids)
            if "joint_vel_target" in entity:
                target = torch.as_tensor(entity["joint_vel_target"], dtype=torch.float32, device=env.device)[None]
                asset.set_joint_velocity_target(target, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.render()
    env.scene.update(dt=0.0)


def _to_numpy_unbatched(value):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value[0] if value.ndim and value.shape[0] == 1 else value


def _raw_policy_observation(policy_obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    joint_pos = _to_numpy_unbatched(policy_obs["joint_pos"])
    return {
        "observation/joint_position": joint_pos[:7],
        "observation/gripper_position": joint_pos[7:8],
        "observation/exterior_image_1_left": _to_numpy_unbatched(policy_obs["table_cam"]),
        "observation/wrist_image_left": _to_numpy_unbatched(policy_obs["wrist_cam"]),
        "prompt": prompt,
    }


def _concatenate_observations(observations):
    """Concatenate OpenPI Observation pytrees along their existing batch dimension."""
    import torch

    def concatenate(values):
        first = values[0]
        if first is None:
            if any(value is not None for value in values):
                raise ValueError("Cannot batch a mixture of present and absent observation fields")
            return None
        if torch.is_tensor(first):
            return torch.cat(values, dim=0)
        if isinstance(first, dict):
            if any(value.keys() != first.keys() for value in values[1:]):
                raise ValueError("Observation dictionary keys differ across batch items")
            return {key: concatenate([value[key] for value in values]) for key in first}
        if dataclasses.is_dataclass(first):
            return dataclasses.replace(
                first,
                **{
                    field.name: concatenate([getattr(value, field.name) for value in values])
                    for field in dataclasses.fields(first)
                },
            )
        raise TypeError(f"Unsupported batched observation field: {type(first)!r}")

    return concatenate(observations)


def _physical_log_jacobian(norm_stats, horizon: int, action_dim: int) -> float:
    import numpy as np

    std = np.asarray(norm_stats["actions"].std, dtype=np.float64)[:action_dim]
    if np.any(std <= 0):
        raise ValueError(f"Non-positive action std in active dimensions: {std}")
    # y=(a-mean)/std, hence log p(a)=log p(y)-sum(log std).
    return -float(horizon * np.log(std + 1.0e-6).sum())


def main() -> None:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("traces", nargs="+", type=Path, help="One or more state_trace.jsonl files.")
    args = parser.parse_args()
    args.enable_cameras = True
    if args.policy_batch_size < 1:
        parser.error("--policy-batch-size must be positive")
    all_samples = load_samples(args)
    if not all_samples:
        raise SystemExit("No complete action chunks matched the requested traces/seeds/stride.")
    existing_rows = (
        _load_resume_rows(args.output, args, {_sample_key(sample) for sample in all_samples})
        if args.resume
        else []
    )
    completed_keys = {_row_key(row) for row in existing_rows}
    samples = [sample for sample in all_samples if _sample_key(sample) not in completed_keys]
    print(
        f"Selected {len(all_samples)} chunks across "
        f"{len({sample.seed for sample in all_samples})} seeds; "
        f"completed={len(existing_rows)} remaining={len(samples)}",
        flush=True,
    )
    progress_stream = (
        os.fdopen(args.progress_fd, "w", closefd=False) if args.progress_fd >= 0 else sys.stderr
    )
    progress = tqdm(
        total=len(all_samples), initial=len(existing_rows), desc="pi0.5 initialization", unit="chunk",
        position=args.progress_position, dynamic_ncols=True, file=progress_stream,
    )

    simulation_app = AppLauncher(args).app
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from openpi.policies import policy_config
    from openpi.training import checkpoints, config
    from sim_free_mpc.hutchinson_likelihood import estimate_probability_flow_log_likelihood

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = (samples or all_samples)[0].seed
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    train_config = config.get_config(args.config)
    if train_config.model.model_type.value != "pi05":
        raise ValueError(f"Config {args.config!r} is not pi0.5")
    norm_stats = checkpoints.load_norm_stats(norm_stats_dir=str(args.norm_stats_dir))
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        norm_stats=norm_stats,
        pytorch_device=args.device,
    )
    model = policy._model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.action_horizon != model.config.action_horizon:
        raise ValueError(
            f"Trace action horizon {args.action_horizon} != model horizon {model.config.action_horizon}"
        )
    if not 1 <= args.action_dim <= model.config.action_dim:
        raise ValueError(f"Invalid active action dimension {args.action_dim}")
    if policy.metadata.get("use_quantile_norm", False):
        raise ValueError(
            "Physical-coordinate Jacobian currently requires mean/std normalization; "
            "use the default demo-mean/std config or add quantile Jacobian handling."
        )

    physical_log_jacobian = _physical_log_jacobian(
        norm_stats, args.action_horizon, args.action_dim
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    rows = list(existing_rows)
    current_seed = None
    progress.set_description("pi0.5 Hutchinson")
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as output:
        for batch_start in range(0, len(samples), args.policy_batch_size):
            batch_samples = samples[batch_start : batch_start + args.policy_batch_size]
            observations = []
            normalized_action_items = []
            replay_errors = []
            for sample in batch_samples:
                if sample.seed != current_seed:
                    env.reset(seed=sample.seed)
                    current_seed = sample.seed
                _restore_state(env, sample.state)
                env_obs = env.observation_manager.compute(update_history=False)
                replay_joint = _to_numpy_unbatched(env_obs["policy"]["joint_pos"])
                logged_joint = np.asarray(sample.state["policy_state"]["joint_pos"])
                replay_errors.append(float(np.max(np.abs(replay_joint - logged_joint))))
                raw = _raw_policy_observation(env_obs["policy"], args.prompt)
                raw["actions"] = np.asarray(sample.actions, dtype=np.float32)
                observation, transformed = policy.obs_to_input(raw)
                observations.append(observation)
                normalized_action_items.append(
                    transformed["actions"][..., : args.action_dim].to(torch.float32)
                )

            batched_observation = _concatenate_observations(observations)
            normalized_actions = torch.cat(normalized_action_items, dim=0)
            with torch.no_grad():
                past_key_values, prefix_pad_masks, padded_state = model.compute_prefix_cache(
                    batched_observation
                )

            def velocity_fn(active_state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
                padding = torch.zeros(
                    (*active_state.shape[:-1], model.config.action_dim - args.action_dim),
                    dtype=active_state.dtype,
                    device=active_state.device,
                )
                full_state = torch.cat((active_state, padding), dim=-1)
                full_velocity = model.denoise_step(
                    padded_state, prefix_pad_masks, past_key_values, full_state, time
                )
                return full_velocity[..., : args.action_dim]

            estimate = estimate_probability_flow_log_likelihood(
                velocity_fn,
                normalized_actions,
                num_steps=args.ode_steps,
                num_probes=args.hutchinson_probes,
                seed=[
                    args.estimator_seed + sample.seed * 100_000 + sample.step
                    for sample in batch_samples
                ],
                time_start=args.time_eps,
                time_end=1.0 - args.time_eps,
            )
            dimensions = args.action_horizon * args.action_dim
            for batch_index, sample in enumerate(batch_samples):
                log_prob_normalized = float(estimate.log_prob[batch_index].item())
                row = {
                    "trace": str(sample.trace),
                    "group_index": sample.group_index,
                    "lane": sample.lane,
                    "seed": sample.seed,
                    "step": sample.step,
                    "frame": sample.step,
                    "task_phase": sample.phase,
                    "action_source": "next_15_executed_actions",
                    "action_horizon": args.action_horizon,
                    "active_action_dim": args.action_dim,
                    "dimensions": dimensions,
                    "ode_steps": args.ode_steps,
                    "hutchinson_probes": args.hutchinson_probes,
                    "policy_batch_size": len(batch_samples),
                    "log_prob_normalized": log_prob_normalized,
                    "log_prob_physical": log_prob_normalized + physical_log_jacobian,
                    "nll_per_dim_normalized": -log_prob_normalized / dimensions,
                    "prior_log_prob": float(estimate.prior_log_prob[batch_index].item()),
                    "divergence_integral": float(
                        estimate.divergence_integral[batch_index].item()
                    ),
                    "replay_joint_max_abs_error": replay_errors[batch_index],
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
                print(
                    f"[{len(rows)}/{len(all_samples)}] seed={sample.seed} step={sample.step} "
                    f"logp={log_prob_normalized:.3f} "
                    f"nll/dim={row['nll_per_dim_normalized']:.4f}",
                    flush=True,
                )
            output.flush()
            last_sample = batch_samples[-1]
            progress.set_postfix(
                seed=last_sample.seed,
                frame=last_sample.step,
                phase=last_sample.phase,
                batch=len(batch_samples),
            )
            progress.update(len(batch_samples))
    progress.close()

    normalized = np.asarray([row["log_prob_normalized"] for row in rows], dtype=np.float64)
    physical = np.asarray([row["log_prob_physical"] for row in rows], dtype=np.float64)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "norm_stats_dir": str(args.norm_stats_dir),
        "prompt": args.prompt,
        "sample_count": len(rows),
        "resumed_from_count": len(existing_rows),
        "seed_count": len({row["seed"] for row in rows}),
        "mean_log_prob_normalized": float(normalized.mean()),
        "std_log_prob_normalized": float(normalized.std()),
        "mean_log_prob_physical": float(physical.mean()),
        "mean_nll_per_dim_normalized": float(-normalized.mean() / (args.action_horizon * args.action_dim)),
        "physical_log_jacobian": physical_log_jacobian,
        "selection": {
            "chunk_stride": args.chunk_stride,
            "max_samples_per_seed": args.max_samples_per_seed,
            "semantics": "every policy replan state with 15 subsequently executed actions",
        },
        "estimator": {
            "kind": "probability_flow_ode_hutchinson_rademacher",
            "ode_steps": args.ode_steps,
            "probes": args.hutchinson_probes,
            "time_interval": [args.time_eps, 1.0 - args.time_eps],
            "active_action_projection": args.action_dim,
            "policy_batch_size": args.policy_batch_size,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} and {summary_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
