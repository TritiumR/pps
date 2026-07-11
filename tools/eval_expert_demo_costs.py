#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_free_mpc.costs_grasp_flow import GraspFlowStateCost  # noqa: E402
from sim_free_mpc.costs_ref_style import RefStyleStateCost  # noqa: E402


OBJECT_NAMES = ("pear", "apple", "mango", "cabbage", "scale")
VIDEO_KEYS = ("table_cam", "wrist_cam")
SUBTASK_SIGNAL_PATHS = (
    "obs/datagen_info/subtask_term_signals",
    "obs/subtask_terms",
)
SUMMARY_FIELDS = (
    "demo",
    "object",
    "stage",
    "cost",
    "mean",
    "terminal",
    "max",
    "min",
    "std",
    "start",
    "end",
    "num_samples",
    "num_windows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate weight-task subtask segments in expert demos with "
            "sim_free_mpc state costs."
        )
    )
    parser.add_argument("--data_file", required=True, help="Path to annotated IsaacLab HDF5 dataset.")
    parser.add_argument(
        "--grasp_object",
        choices=("pear", "apple", "both"),
        default="both",
        help=(
            "Compatibility filter for weight-task segments. pear scores grasp_pear "
            "and place_pear, apple scores grasp_apple, both scores all three."
        ),
    )
    parser.add_argument("--horizon", type=int, default=8, help="Sliding action window length.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between scored windows.")
    parser.add_argument("--max_demos", type=int, default=None, help="Limit number of demos.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--cost",
        choices=("ref_style", "grasp_flow"),
        default="ref_style",
        help="Cost function used to score expert windows.",
    )
    parser.add_argument(
        "--grasp_flow_lift_height",
        type=float,
        default=None,
        help="Optional lift height used by grasp_flow before switching to place.",
    )
    parser.add_argument(
        "--grasp_flow_tail_cost",
        choices=("native", "place"),
        default="native",
        help="For grasp_flow, score lift/place stages with their native cost or force both to place cost.",
    )
    parser.add_argument("--output_csv", default=None, help="Optional per-demo/per-subtask summary CSV path.")
    parser.add_argument("--output_markdown", default=None, help="Optional human-readable Markdown table path.")
    parser.add_argument("--output_video_dir", default=None, help="Optional directory for cost-overlay videos.")
    parser.add_argument("--plot_demo", default=None, help="Optional demo name or index to plot as cost-frame curves.")
    parser.add_argument("--plot_path", default=None, help="Optional output path for the cost-frame plot.")
    parser.add_argument(
        "--plot_merge_tail",
        action="store_true",
        help="Plot lift and place windows together as one place stage.",
    )
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
        name: torch.as_tensor(bool(read_signal(signals, name, length)[index]), device=device)
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
    grasp_flow_lift_height: float | None = None,
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
    if grasp_flow_lift_height is not None:
        context["grasp_flow_lift_height"] = grasp_flow_lift_height
    for key in ("joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"):
        if key in obs:
            context[key] = as_tensor(np.asarray(obs[key][index], dtype=np.float32), device)
    return context


def subtask_ranges(
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

    if grasp_object in ("pear", "both") and pear_end is not None and pear_place_end is not None:
        ranges.append(("place_pear", pear_end, min(pear_place_end + 1, length), None))

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
        "max": float(arr.max()),
        "terminal": float(arr[-1]),
    }


def summarize_group(
    values: list[float],
    *,
    starts: list[int],
    horizon: int,
) -> dict[str, float]:
    stats = summarize(values)
    start = min(starts)
    end = max(starts) + horizon
    stats["start"] = int(start)
    stats["end"] = int(end)
    stats["num_samples"] = int(end - start)
    return stats


def make_cost_fn(name: str):
    if name == "ref_style":
        return RefStyleStateCost("weight")
    if name == "grasp_flow":
        return GraspFlowStateCost("weight")
    raise ValueError(f"Unsupported cost: {name}")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(SUMMARY_FIELDS)

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    with path.open("w") as f:
        f.write("# Weight Expert Demo Costs\n\n")
        f.write("Lower cost is better.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field, "")) for field in headers) + " |\n")


def parse_stage(stage_name: str) -> tuple[str, str]:
    if "_" not in stage_name:
        return "", stage_name
    stage, object_name = stage_name.split("_", 1)
    return object_name, stage


def compute_cost_trace(
    demo: h5py.Group,
    *,
    cost_fn: Any,
    horizon: int,
    stride: int,
    device: torch.device,
    grasp_flow_lift_height: float | None,
    grasp_flow_tail_cost: str = "native",
) -> list[dict[str, Any]]:
    actions_np = read_actions(demo)
    eef_pos_np = read_required_obs(demo, "eef_pos")
    eef_quat_np = read_required_obs(demo, "eef_quat")
    length = min(actions_np.shape[0], eef_pos_np.shape[0], eef_quat_np.shape[0])
    signals = first_existing_group(demo, SUBTASK_SIGNAL_PATHS)

    if length < horizon:
        raise ValueError(f"{demo.name} has {length} samples, shorter than horizon={horizon}.")

    trace = []
    for window_start in range(0, length - horizon + 1, stride):
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
            overrides=None,
            grasp_flow_lift_height=grasp_flow_lift_height,
        )

        with torch.no_grad():
            cost = cost_fn(real_actions=actions, tcp_pos=eef_pos, tcp_quat=eef_quat, context=context)

        stage_name = getattr(cost_fn, "last_stage", "unknown")
        if stage_name == "idle":
            continue
        object_name, stage = parse_stage(stage_name)
        cost_value = float(cost.detach().cpu().reshape(-1)[0])
        if grasp_flow_tail_cost == "place" and stage in ("lift", "place"):
            place_context = dict(context)
            place_context["grasp_flow_lift_height"] = 0.0
            object_pos = context.get("objects", {}).get(object_name, {}).get("pos")
            if object_pos is not None:
                place_context[f"{object_name}_lift_start_z"] = object_pos[2]
            place_cost_fn = GraspFlowStateCost("weight")
            with torch.no_grad():
                place_cost = place_cost_fn(
                    real_actions=actions,
                    tcp_pos=eef_pos,
                    tcp_quat=eef_quat,
                    context=place_context,
                )
            cost_value = float(place_cost.detach().cpu().reshape(-1)[0])
        trace.append(
            {
                "frame": window_start,
                "window_end": window_end,
                "object": object_name,
                "stage": stage,
                "cost": cost_value,
            }
        )

    return trace


def summarize_cost_trace(trace: list[dict[str, Any]], *, horizon: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for item in trace:
        key = (item["object"], item["stage"])
        group = groups.setdefault(key, {"values": [], "starts": []})
        group["values"].append(item["cost"])
        group["starts"].append(item["frame"])

    rows = []
    for (object_name, stage), group in groups.items():
        stats = summarize_group(group["values"], starts=group["starts"], horizon=horizon)
        rows.append({"object": object_name, "stage": stage, **stats})
    stage_order = {"grasp": 0, "lift": 1, "place": 2}
    object_order = {"pear": 0, "apple": 1}
    rows.sort(key=lambda row: (object_order.get(row["object"], 99), stage_order.get(row["stage"], 99)))
    return rows


def frame_trace(trace: list[dict[str, Any]], length: int) -> list[dict[str, Any] | None]:
    by_frame: list[dict[str, Any] | None] = [None] * length
    if not trace:
        return by_frame

    trace_by_start = {item["frame"]: item for item in trace}
    last = trace[0]
    for frame_idx in range(length):
        if frame_idx in trace_by_start:
            last = trace_by_start[frame_idx]
        by_frame[frame_idx] = last
    return by_frame


def stage_color(stage: str) -> tuple[int, int, int]:
    return {
        "grasp": (80, 220, 255),
        "lift": (80, 255, 120),
        "place": (255, 180, 80),
    }.get(stage, (255, 255, 255))


def draw_overlay(frame_rgb: np.ndarray, item: dict[str, Any] | None, frame_idx: int) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if item is None:
        lines = [f"frame {frame_idx}", "cost: n/a"]
        color = (255, 255, 255)
    else:
        color = stage_color(item["stage"])
        lines = [
            f"frame {frame_idx}  cost {item['cost']:.3f}",
            f"{item['object']} / {item['stage']}",
        ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y += 24
    return frame


def encode_h264(input_avi: Path, output_mp4: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_avi),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def write_overlay_videos(
    demo_name: str,
    demo: h5py.Group,
    trace: list[dict[str, Any]],
    output_dir: Path,
    *,
    framerate: int = 15,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = demo["obs"]
    available_keys = [key for key in VIDEO_KEYS if key in obs]
    if not available_keys:
        return
    length = min(len(obs[key]) for key in available_keys)
    per_frame = frame_trace(trace, length)

    for key in available_keys:
        frames = obs[key]
        height, width = frames.shape[1], frames.shape[2]
        avi_path = output_dir / f"{demo_name}_{key}_cost_overlay.avi"
        mp4_path = output_dir / f"{demo_name}_{key}_cost_overlay.mp4"
        writer = cv2.VideoWriter(
            str(avi_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            framerate,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {avi_path}")
        for frame_idx in range(length):
            writer.write(draw_overlay(np.asarray(frames[frame_idx]), per_frame[frame_idx], frame_idx))
        writer.release()
        if encode_h264(avi_path, mp4_path):
            avi_path.unlink()


def normalize_plot_demo(value: str | None) -> str | None:
    if value is None:
        return None
    if value.isdigit():
        return f"demo_{value}"
    return value


def plot_cost_trace(
    demo_name: str,
    trace: list[dict[str, Any]],
    output_path: Path,
    *,
    merge_tail: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_order = ("grasp", "place") if merge_tail else ("grasp", "lift", "place")
    fig, axes = plt.subplots(len(stage_order), 1, figsize=(11, 5 if merge_tail else 7), sharex=True)
    if len(stage_order) == 1:
        axes = [axes]
    colors = {"pear": "#2563eb", "apple": "#dc2626"}

    for ax, plot_stage in zip(axes, stage_order, strict=True):
        for object_name, color in colors.items():
            points = [
                item for item in trace
                if item["object"] == object_name
                and (
                    item["stage"] == plot_stage
                    or (merge_tail and plot_stage == "place" and item["stage"] == "lift")
                )
            ]
            if not points:
                continue
            ax.plot(
                [item["frame"] for item in points],
                [item["cost"] for item in points],
                label=object_name,
                color=color,
                linewidth=1.8,
            )
        ax.set_ylabel(plot_stage)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")
    suffix = " (lift+place as place)" if merge_tail else ""
    fig.suptitle(f"{demo_name} grasp_flow cost by frame{suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
    grasp_flow_lift_height: float | None = None,
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
            grasp_flow_lift_height=grasp_flow_lift_height,
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
    cost_fn = make_cost_fn(args.cost)
    rows = []
    traces_by_demo: dict[str, list[dict[str, Any]]] = {}

    with h5py.File(args.data_file, "r") as dataset:
        if "data" not in dataset:
            raise KeyError(f"{args.data_file} does not contain a top-level 'data' group.")
        demo_names = sorted(dataset["data"].keys(), key=episode_sort_key)
        if args.max_demos is not None:
            demo_names = demo_names[: args.max_demos]

        for demo_name in demo_names:
            demo = dataset["data"][demo_name]
            if args.cost == "grasp_flow":
                trace = compute_cost_trace(
                    demo,
                    cost_fn=cost_fn,
                    horizon=args.horizon,
                    stride=args.stride,
                    device=device,
                    grasp_flow_lift_height=args.grasp_flow_lift_height,
                    grasp_flow_tail_cost=args.grasp_flow_tail_cost,
                )
                traces_by_demo[demo_name] = trace
                stage_rows = summarize_cost_trace(trace, horizon=args.horizon)
                if args.grasp_object != "both":
                    stage_rows = [row for row in stage_rows if row["object"] == args.grasp_object]
                if not stage_rows:
                    print(f"WARNING: {demo_name} has no requested cost stages; skipping.", file=sys.stderr)
                    continue
                for stage_row in stage_rows:
                    row = {
                        "demo": demo_name,
                        "task": "weight",
                        "cost": args.cost,
                        **stage_row,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                if args.output_video_dir is not None:
                    write_overlay_videos(demo_name, demo, trace, Path(args.output_video_dir))
                continue

            actions = read_actions(demo)
            eef_pos = read_required_obs(demo, "eef_pos")
            eef_quat = read_required_obs(demo, "eef_quat")
            length = min(actions.shape[0], eef_pos.shape[0], eef_quat.shape[0])
            ranges = subtask_ranges(
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
                object_name, stage = parse_stage(subtask)
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
                    "cost": args.cost,
                    "object": object_name,
                    "stage": stage,
                    **stats,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(SUMMARY_FIELDS)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")

    if args.output_markdown is not None:
        output_path = Path(args.output_markdown)
        write_markdown(rows, output_path)
        print(f"Wrote {output_path}")

    plot_demo = normalize_plot_demo(args.plot_demo)
    if plot_demo is not None:
        if plot_demo not in traces_by_demo:
            print(f"WARNING: {plot_demo} has no trace to plot.", file=sys.stderr)
        else:
            if args.plot_path is None:
                raise ValueError("--plot_path is required when --plot_demo is set.")
            output_path = Path(args.plot_path)
            plot_cost_trace(plot_demo, traces_by_demo[plot_demo], output_path, merge_tail=args.plot_merge_tail)
            print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
