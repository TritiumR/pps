#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_free_mpc.costs_ref_style import RefStyleStateCost  # noqa: E402


OBJECT_NAMES = ("pear", "apple", "mango", "cabbage", "scale")
SUBTASK_SIGNAL_PATHS = (
    "obs/datagen_info/subtask_term_signals",
    "obs/subtask_terms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the weight-task grasp segments in expert demos with "
            "sim_free_mpc.costs_ref_style.RefStyleStateCost."
        )
    )
    parser.add_argument("--data_file", required=True, help="Path to annotated IsaacLab HDF5 dataset.")
    parser.add_argument(
        "--grasp_object",
        choices=("pear", "apple", "both"),
        default="both",
        help="Which weight-task grasp segment to score.",
    )
    parser.add_argument("--horizon", type=int, default=8, help="Sliding action window length.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between scored windows.")
    parser.add_argument("--max_demos", type=int, default=None, help="Limit number of demos.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--output_csv", default=None, help="Optional per-demo/per-subtask summary CSV path.")
    parser.add_argument(
        "--allow_missing_subtasks",
        action="store_true",
        help="Score the whole episode as pear grasp when subtask signals are missing.",
    )
    return parser.parse_args()


def episode_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (math.inf, name)


def as_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def read_actions(demo: h5py.Group) -> np.ndarray:
    if "obs" in demo and "joint_actions" in demo["obs"]:
        actions = np.asarray(demo["obs"]["joint_actions"], dtype=np.float32)
    elif "processed_actions" in demo:
        actions = np.asarray(demo["processed_actions"], dtype=np.float32)
    else:
        actions = np.asarray(demo["actions"], dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"Expected 2-D actions, got {actions.shape}.")

    if actions.shape[1] == 7:
        gripper = None
        if "actions" in demo:
            raw_actions = np.asarray(demo["actions"], dtype=np.float32)
            if raw_actions.ndim == 2 and raw_actions.shape[1] > 7:
                gripper = raw_actions[:, 7:8]
        if gripper is None and "obs" in demo and "gripper_pos" in demo["obs"]:
            gripper_obs = np.asarray(demo["obs"]["gripper_pos"], dtype=np.float32)
            gripper = gripper_obs[:, :1] if gripper_obs.ndim == 2 else gripper_obs.reshape(-1, 1)
        if gripper is not None:
            actions = np.concatenate([actions, gripper[: actions.shape[0]]], axis=1)
    return actions


def read_required_obs(demo: h5py.Group, key: str) -> np.ndarray:
    if "obs" not in demo or key not in demo["obs"]:
        raise KeyError(f"Missing obs/{key} in demo {demo.name}.")
    return np.asarray(demo["obs"][key], dtype=np.float32)


def first_existing_group(demo: h5py.Group, paths: tuple[str, ...]) -> h5py.Group | None:
    for path in paths:
        if path in demo:
            item = demo[path]
            if isinstance(item, h5py.Group):
                return item
    return None


def read_signal(group: h5py.Group | None, name: str, length: int) -> np.ndarray:
    if group is None or name not in group:
        return np.zeros(length, dtype=bool)
    values = np.asarray(group[name]).reshape(-1)
    return values[:length].astype(bool)


def first_true_index(signal: np.ndarray) -> int | None:
    indices = np.flatnonzero(signal)
    if indices.size == 0:
        return None
    return int(indices[0])


def object_pose_at(demo: h5py.Group, object_name: str, index: int) -> dict[str, torch.Tensor] | None:
    for root in ("rigid_object", "articulation"):
        path = f"states/{root}/{object_name}/root_pose"
        if path in demo:
            pose = np.asarray(demo[path][index], dtype=np.float32)
            return {
                "pos": torch.as_tensor(pose[:3], dtype=torch.float32),
                "quat": torch.as_tensor(pose[3:7], dtype=torch.float32),
            }
    for path in (
        f"obs/datagen_info/object_pose/{object_name}",
        f"obs/datagen_info/object_poses/{object_name}",
    ):
        if path in demo:
            pose = np.asarray(demo[path][index], dtype=np.float32)
            while pose.ndim > 2 and pose.shape[0] == 1:
                pose = pose[0]
            if pose.shape[-2:] == (4, 4):
                pos = pose[:3, 3]
                quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif pose.shape[-1] >= 7:
                pos = pose[:3]
                quat = pose[3:7]
            else:
                continue
            return {
                "pos": torch.as_tensor(pos, dtype=torch.float32),
                "quat": torch.as_tensor(quat, dtype=torch.float32),
            }
    return None


def subtask_terms_at(
    signals: h5py.Group | None,
    index: int,
    length: int,
    device: torch.device,
    overrides: dict[str, bool] | None = None,
) -> dict[str, torch.Tensor]:
    names = ("grasp_pear", "pear_on_scale", "grasp_apple")
    terms = {
        name: torch.as_tensor(read_signal(signals, name, length)[index], device=device)
        for name in names
    }
    if overrides:
        for name, value in overrides.items():
            terms[name] = torch.as_tensor(value, device=device)
    return terms


def context_at(
    demo: h5py.Group,
    *,
    index: int,
    length: int,
    signals: h5py.Group | None,
    device: torch.device,
    overrides: dict[str, bool] | None,
) -> dict[str, Any]:
    obs = demo["obs"]
    objects = {}
    for name in OBJECT_NAMES:
        item = object_pose_at(demo, name, index)
        if item is not None:
            objects[name] = {k: v.to(device=device) for k, v in item.items()}

    context: dict[str, Any] = {
        "subtasks": subtask_terms_at(signals, index, length, device, overrides),
        "objects": objects,
    }
    for key in ("joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"):
        if key in obs:
            context[key] = as_tensor(np.asarray(obs[key][index], dtype=np.float32), device)
    return context


def grasp_ranges(
    demo: h5py.Group,
    *,
    length: int,
    grasp_object: str,
    allow_missing_subtasks: bool,
) -> list[tuple[str, int, int, dict[str, bool] | None]]:
    signals = first_existing_group(demo, SUBTASK_SIGNAL_PATHS)
    if signals is None:
        if not allow_missing_subtasks:
            raise KeyError(
                f"{demo.name} is missing subtask signals. Run annotate first, or pass "
                "--allow_missing_subtasks for a rough whole-episode pear-grasp score."
            )
        return [("grasp_pear", 0, length, {"grasp_pear": False, "pear_on_scale": False, "grasp_apple": False})]

    grasp_pear = read_signal(signals, "grasp_pear", length)
    pear_on_scale = read_signal(signals, "pear_on_scale", length)
    grasp_apple = read_signal(signals, "grasp_apple", length)

    pear_end = first_true_index(grasp_pear)
    pear_place_end = first_true_index(pear_on_scale)
    apple_end = first_true_index(grasp_apple)

    ranges: list[tuple[str, int, int, dict[str, bool] | None]] = []
    if grasp_object in ("pear", "both") and pear_end is not None:
        ranges.append(("grasp_pear", 0, min(pear_end + 1, length), None))

    if grasp_object in ("apple", "both") and pear_place_end is not None and apple_end is not None:
        ranges.append(("grasp_apple", pear_place_end, min(apple_end + 1, length), None))

    return ranges


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "num_windows": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "terminal": float(arr[-1]),
    }


def score_range(
    demo: h5py.Group,
    *,
    subtask: str,
    start: int,
    end: int,
    overrides: dict[str, bool] | None,
    cost_fn: RefStyleStateCost,
    horizon: int,
    stride: int,
    device: torch.device,
) -> dict[str, float]:
    actions_np = read_actions(demo)
    eef_pos_np = read_required_obs(demo, "eef_pos")
    eef_quat_np = read_required_obs(demo, "eef_quat")
    length = min(actions_np.shape[0], eef_pos_np.shape[0], eef_quat_np.shape[0])
    signals = first_existing_group(demo, SUBTASK_SIGNAL_PATHS)

    end = min(end, length)
    if end - start < horizon:
        raise ValueError(
            f"{demo.name}/{subtask} has {end - start} samples, shorter than horizon={horizon}."
        )

    values = []
    for window_start in range(start, end - horizon + 1, stride):
        window_end = window_start + horizon
        actions = as_tensor(actions_np[window_start:window_end], device).unsqueeze(0)
        eef_pos = as_tensor(eef_pos_np[window_start:window_end], device).unsqueeze(0)
        eef_quat = as_tensor(eef_quat_np[window_start:window_end], device).unsqueeze(0)
        context = context_at(
            demo,
            index=window_start,
            length=length,
            signals=signals,
            device=device,
            overrides=overrides,
        )

        with torch.no_grad():
            cost = cost_fn(real_actions=actions, tcp_pos=eef_pos, tcp_quat=eef_quat, context=context)
        values.append(float(cost.detach().cpu().reshape(-1)[0]))

    stats = summarize(values)
    stats["start"] = int(start)
    stats["end"] = int(end)
    stats["num_samples"] = int(end - start)
    return stats


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    cost_fn = RefStyleStateCost("weight")
    rows = []

    with h5py.File(args.data_file, "r") as dataset:
        if "data" not in dataset:
            raise KeyError(f"{args.data_file} does not contain a top-level 'data' group.")
        demo_names = sorted(dataset["data"].keys(), key=episode_sort_key)
        if args.max_demos is not None:
            demo_names = demo_names[: args.max_demos]

        for demo_name in demo_names:
            demo = dataset["data"][demo_name]
            actions = read_actions(demo)
            eef_pos = read_required_obs(demo, "eef_pos")
            eef_quat = read_required_obs(demo, "eef_quat")
            length = min(actions.shape[0], eef_pos.shape[0], eef_quat.shape[0])
            ranges = grasp_ranges(
                demo,
                length=length,
                grasp_object=args.grasp_object,
                allow_missing_subtasks=args.allow_missing_subtasks,
            )
            if not ranges:
                print(
                    f"WARNING: {demo_name} has no complete requested grasp segment; skipping.",
                    file=sys.stderr,
                )
                continue

            for subtask, start, end, overrides in ranges:
                stats = score_range(
                    demo,
                    subtask=subtask,
                    start=start,
                    end=end,
                    overrides=overrides,
                    cost_fn=cost_fn,
                    horizon=args.horizon,
                    stride=args.stride,
                    device=device,
                )
                row = {
                    "demo": demo_name,
                    "task": "weight",
                    "cost": "ref_style",
                    "subtask": subtask,
                    **stats,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else [
            "demo",
            "task",
            "cost",
            "subtask",
            "start",
            "end",
            "num_samples",
            "num_windows",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "terminal",
        ]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
