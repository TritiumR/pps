#!/usr/bin/env python3
"""Evaluate normalized action-chunk likelihood under the PPS VLM cost energy.

For task-only traces, the VLM bridge state was not recorded.  This evaluator rebuilds the
reference fake-VLM plan with GT simulator state, replays every recorded frame through the bridge
stage machine, and normalizes ``exp(-cost / temperature)`` with scrambled Sobol quadrature over
the executable physical action domain.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from tqdm.auto import tqdm

_REPO_DIR = Path(__file__).resolve().parents[1]
_ISAACLAB_DIR = _REPO_DIR / "IsaacLab"
for _package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _source = str(_ISAACLAB_DIR / "source" / _package)
    if _source not in sys.path:
        sys.path.insert(0, _source)

from isaaclab.app import AppLauncher
from sim_free_mpc.trace_action_chunks import BatchedReplanTracker, weight_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument("--task-key", default="weight")
    parser.add_argument("--vlm-cost", default="rekep_fake_vlm",
                        choices=("gt", "rekep_fake", "rekep_fake_vlm"))
    parser.add_argument("--cost-config", type=Path,
                        default=_REPO_DIR / "vlm_dp/configs/test_configs/simple_auth.yaml")
    parser.add_argument("--output", type=Path, default=Path("vlm_cost_quadrature.jsonl"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Validate existing rows and score only missing action chunks.")
    parser.add_argument("--action-horizon", type=int, default=15)
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
    parser.add_argument("--steps-per-inference", type=int, default=4)
    parser.add_argument("--joint-delta", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--quadrature-points", type=int, default=16384)
    parser.add_argument("--quadrature-scrambles", type=int, default=4)
    parser.add_argument("--cost-batch-size", type=int, default=2048)
    parser.add_argument("--quadrature-seed", type=int, default=20260816)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed allow-list.")
    parser.add_argument("--progress-position", type=int, default=0)
    parser.add_argument(
        "--progress-fd",
        type=int,
        default=-1,
        help="Write the progress bar to this inherited file descriptor; stderr if negative.",
    )
    return parser


def _read_trajectories(paths: list[Path], allowed: set[int] | None):
    trajectories: dict[tuple[Path, int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        trackers: dict[int, BatchedReplanTracker] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") != "frame":
                    continue
                step = int(record["step"])
                group = int(record["group_index"])
                tracker = trackers.setdefault(group, BatchedReplanTracker())
                generated_chunk = tracker.observe_frame(step, record["lanes"])
                for lane in record["lanes"]:
                    if not lane.get("valid", False):
                        continue
                    seed = int(lane["seed"])
                    if allowed is not None and seed not in allowed:
                        continue
                    lane["_generated_action_chunk"] = generated_chunk
                    trajectories[(path, group, int(lane["lane"]))].append((step, lane))
    return trajectories


def _select_starts(frames, horizon: int, stride: int, limit: int):
    by_step = {step: lane for step, lane in frames}
    candidates = []
    for step, lane in frames:
        if not lane.get("active", False) or not lane.get("_generated_action_chunk", False):
            continue
        future = [by_step.get(step + offset) for offset in range(1, horizon + 1)]
        if any(item is None or item.get("action") is None for item in future):
            continue
        if any(not item.get("active", False) for item in future[:-1]):
            continue
        candidates.append((step, lane, [item["action"] for item in future]))
    candidates = candidates[::stride]
    if limit > 0 and len(candidates) > limit:
        indices = ([round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)]
                   if limit > 1 else [0])
        candidates = [candidates[index] for index in dict.fromkeys(indices)]
    return {step: (lane, actions) for step, lane, actions in candidates}


def _selection_key(trace: Path, group: int, lane: int, seed: int, step: int):
    return (str(trace), int(group), int(lane), int(seed), int(step))


def _row_key(row: dict[str, Any]):
    try:
        return _selection_key(
            Path(row["trace"]), row["group_index"], row["lane"], row["seed"], row["step"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed VLM likelihood row identity: {row!r}") from error


def _load_resume_rows(path: Path, args, valid_keys: set[tuple]):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"--resume requires a nonempty existing output: {path}")
    rows, seen = [], set()
    expected = {
        "temperature": float(args.temperature),
        "joint_delta_support": float(args.joint_delta),
        "quadrature_points": int(args.quadrature_points),
        "quadrature_scrambles": int(args.quadrature_scrambles),
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
                    f"Existing row {path}:{line_number} is outside the selected traces: {key}"
                )
            if key in seen:
                raise ValueError(f"Duplicate existing row in {path}:{line_number}: {key}")
            for field, value in expected.items():
                actual = row.get(field)
                if actual is None or not math.isclose(
                    float(actual), float(value), rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"Resume setting mismatch at {path}:{line_number}: "
                        f"{field}={actual!r}, expected {value}"
                    )
            seen.add(key)
            rows.append(row)
    return rows


def _write_summary(args, rows, summary_path: Path) -> None:
    finite = [
        float(row["log_likelihood_physical"])
        for row in rows
        if math.isfinite(float(row["log_likelihood_physical"]))
    ]
    mean = math.fsum(finite) / len(finite) if finite else None
    summary = {
        "sample_count": len(rows), "seed_count": len({row["seed"] for row in rows}),
        "support_feasible_count": len(finite),
        "mean_log_likelihood_physical": mean,
        "mean_nll_per_dim_physical": (-mean / (args.action_horizon * 8))
        if mean is not None else None,
        "density": "exp(-CompositeCost/temperature) normalized over executable physical action chunks",
        "grounding": {"source": args.vlm_cost, "state": "gt_replayed", "config": str(args.cost_config)},
        "selection": {"chunk_stride": args.chunk_stride,
                      "max_samples_per_seed": args.max_samples_per_seed,
                      "semantics": "every policy replan state with 15 subsequently executed actions"},
        "support": {"joint_delta": args.joint_delta, "joint_limits": "PANDA_JOINT_LIMITS",
                    "gripper": [0.0, 1.0]},
        "quadrature": {"kind": "owen_scrambled_sobol_autoregressive",
                       "points_per_scramble": args.quadrature_points,
                       "scrambles": args.quadrature_scrambles},
        "resumed_from_count": int(getattr(args, "resumed_from_count", 0)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _world_fk(fk, actions, context):
    from sim_free_mpc.fk import quat_mul_wxyz, transform_points_wxyz

    result = fk.forward(actions[..., :7])
    pos, quat = result.ee_pos, result.ee_quat
    root_pos = context["robot_root_pos"].to(device=pos.device, dtype=pos.dtype)
    root_quat = context["robot_root_quat"].to(device=quat.device, dtype=quat.dtype)
    return transform_points_wxyz(root_pos, root_quat, pos), quat_mul_wxyz(root_quat, quat)


def _support_diagnostics(actions, current, delta, limits):
    import numpy as np

    actions = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(current, dtype=np.float64)[:7]
    max_delta = 0.0
    feasible = True
    for row in actions:
        max_delta = max(max_delta, float(np.max(np.abs(row[:7] - previous))))
        feasible &= bool(np.all(np.abs(row[:7] - previous) <= delta + 1e-5))
        feasible &= bool(np.all(row[:7] >= limits[:, 0] - 1e-5))
        feasible &= bool(np.all(row[:7] <= limits[:, 1] + 1e-5))
        feasible &= bool(-1e-5 <= row[7] <= 1.0 + 1e-5)
        previous = row[:7]
    return feasible, max_delta


def main() -> None:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    args.enable_cameras = True

    allowed = ({int(value) for value in args.seeds.split(",") if value.strip()}
               if args.seeds else None)
    trajectories = _read_trajectories(args.traces, allowed)
    selected = {
        key: _select_starts(frames, args.action_horizon, args.chunk_stride,
                            args.max_samples_per_seed)
        for key, frames in trajectories.items()
    }
    count = sum(len(value) for value in selected.values())
    if not count:
        raise SystemExit("No complete action chunks matched the trace selection.")
    valid_keys = {
        _selection_key(trace, group, lane, int(frames[0][1]["seed"]), step)
        for (trace, group, lane), frames in trajectories.items()
        for step in selected[(trace, group, lane)]
    }
    existing_rows = _load_resume_rows(args.output, args, valid_keys) if args.resume else []
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_count = count - len(completed_keys)
    args.resumed_from_count = len(existing_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    print(
        f"Selected {count} chunks across {len(trajectories)} trajectories; "
        f"completed={len(existing_rows)} remaining={remaining_count}", flush=True,
    )
    if remaining_count == 0:
        _write_summary(args, existing_rows, summary_path)
        print(f"Already complete; rebuilt {summary_path}", flush=True)
        return
    progress_stream = (
        os.fdopen(args.progress_fd, "w", closefd=False) if args.progress_fd >= 0 else sys.stderr
    )
    progress = tqdm(
        total=count, initial=len(existing_rows), desc="VLM initialization", unit="chunk",
        position=args.progress_position, dynamic_ncols=True, file=progress_stream,
    )

    simulation_app = AppLauncher(args).app
    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from sim_free_mpc.fk import PANDA_JOINT_LIMITS, PandaFK
    from sim_free_mpc.quadrature_likelihood import (
        action_log_likelihood,
        estimate_log_partition_sobol,
    )
    from evaluate_pi05_trace_likelihood import _restore_state
    from vlm_dp.bridge import VlmDpBridge
    from vlm_dp.cost.base_cost import CompositeCost

    with (_REPO_DIR / "task_prompts.json").open(encoding="utf-8") as handle:
        entry = json.load(handle)[args.task_key]
    roles = {key: entry[key] for key in ("grasp_obj", "grasp_objs", "place_obj", "support")
             if key in entry}
    with args.cost_config.open(encoding="utf-8") as handle:
        cost_cfg = yaml.safe_load(handle)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = min(int(frames[0][1]["seed"]) for frames in trajectories.values())
    env_cfg.sim.physx.enable_enhanced_determinism = True
    if args.vlm_cost.startswith("rekep"):
        from rekep import isaaclab_helpers

        isaaclab_helpers.augment_table_cam_with_depth_and_seg(env_cfg)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    bridge = VlmDpBridge(
        args.vlm_cost,
        roles,
        cost_cfg,
        task_key=args.task_key,
        device=args.device,
        state="gt",
        vocab=entry.get("objects"),
        fixtures=entry.get("fixtures", ()),
    )
    cost = CompositeCost(cost_cfg["cost"]["terms"], cost_cfg["cost"]["geometry"])
    fk = PandaFK()
    limits = np.asarray(PANDA_JOINT_LIMITS, dtype=np.float64)

    rows = list(existing_rows)
    progress.set_description("VLM quadrature")
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as output:
        for trajectory_index, (key, frames) in enumerate(sorted(trajectories.items(), key=lambda x: int(x[1][0][1]["seed"]))):
            trace, group, lane = key
            wanted = selected[key]
            seed = int(frames[0][1]["seed"])
            pending_steps = {
                step for step in wanted
                if _selection_key(trace, group, lane, seed, step) not in completed_keys
            }
            if not pending_steps:
                continue
            env.reset(seed=seed)
            bridge.reset(env)
            for step, state in sorted(frames):
                _restore_state(env, state)
                env_obs = env.observation_manager.compute(update_history=False)
                action_entering_state = state.get("action")
                if step > 0 and action_entering_state is not None:
                    bridge.observe_step(np.asarray(action_entering_state, dtype=np.float32))
                if step > 0 and step % args.steps_per_inference == 0:
                    bridge.advance(state.get("subtasks", {}))
                if step not in wanted:
                    continue

                _, action_chunk = wanted[step]
                context = bridge.context(env, env_obs)
                if step not in pending_steps:
                    continue
                observed = torch.as_tensor(action_chunk, device=args.device, dtype=torch.float32).unsqueeze(0)

                def cost_fn(candidate_actions: torch.Tensor) -> torch.Tensor:
                    ee_pos, ee_quat = _world_fk(fk, candidate_actions, context)
                    return cost(real_actions=candidate_actions, ee_pos=ee_pos,
                                ee_quat=ee_quat, context=context)

                with torch.no_grad():
                    observed_cost = float(cost_fn(observed)[0].item())
                estimate = estimate_log_partition_sobol(
                    cost_fn,
                    context["joint_pos"],
                    horizon=args.action_horizon,
                    joint_delta=args.joint_delta,
                    temperature=args.temperature,
                    num_points=args.quadrature_points,
                    num_scrambles=args.quadrature_scrambles,
                    batch_size=args.cost_batch_size,
                    seed=args.quadrature_seed + seed * 100_000 + step,
                    device=args.device,
                )
                feasible, max_delta = _support_diagnostics(
                    action_chunk, context["joint_pos"].detach().cpu().numpy(),
                    args.joint_delta, limits,
                )
                log_likelihood = (action_log_likelihood(observed_cost, estimate.log_partition,
                                                        args.temperature)
                                  if feasible else -math.inf)
                row = {
                    "trace": str(trace), "group_index": group, "lane": lane,
                    "seed": seed, "step": step, "frame": step,
                    "task_phase": weight_phase(state.get("subtasks", {})),
                    "action_source": "next_15_executed_actions",
                    "stage_index": int(bridge.stage_idx),
                    "stage": bridge.stage().name, "observed_cost": observed_cost,
                    "temperature": args.temperature, "log_partition": estimate.log_partition,
                    "log_partition_std_error": estimate.log_partition_std_error,
                    "log_likelihood_physical": log_likelihood,
                    "nll_per_dim_physical": (-log_likelihood / (args.action_horizon * 8)
                                             if feasible else math.inf),
                    "support_feasible": feasible, "max_observed_joint_delta": max_delta,
                    "joint_delta_support": args.joint_delta,
                    "quadrature_points": args.quadrature_points,
                    "quadrature_scrambles": args.quadrature_scrambles,
                    "log_partition_by_scramble": estimate.log_partition_by_scramble,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                rows.append(row)
                progress.set_postfix(seed=seed, frame=step, stage=bridge.stage_idx)
                progress.update(1)
                print(f"[{len(rows)}/{count}] seed={seed} step={step} stage={bridge.stage_idx} "
                      f"cost={observed_cost:.4f} logp={log_likelihood:.4f} "
                      f"logZ_se={estimate.log_partition_std_error:.3g}", flush=True)
                pending_steps.remove(step)
                if not pending_steps:
                    break
    progress.close()

    _write_summary(args, rows, summary_path)
    print(f"Wrote {args.output} and {summary_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
