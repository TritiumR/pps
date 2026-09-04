#!/usr/bin/env python3
"""Render successful/failed pear-grasp event reels with pi0.5 likelihood plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

_REPO_DIR = Path(__file__).resolve().parents[1]
_ISAACLAB_DIR = _REPO_DIR / "IsaacLab"
for _package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _source = str(_ISAACLAB_DIR / "source" / _package)
    if _source not in sys.path:
        sys.path.insert(0, _source)

from isaaclab.app import AppLauncher


@dataclass
class Attempt:
    trace: str
    group: int
    lane: int
    seed: int
    event_step: int
    success: bool
    likelihood_rows: list[dict[str, Any]]
    end_step: int
    frames: list[tuple[int, dict[str, Any]]] = field(default_factory=list)

    @property
    def trajectory_key(self) -> tuple[str, int, int, int]:
        return (self.trace, self.group, self.lane, self.seed)

    @property
    def start_step(self) -> int:
        return int(self.likelihood_rows[0]["step"])

    @property
    def event_chunk_step(self) -> int:
        return int(self.likelihood_rows[-1]["step"])

    @property
    def mean_nll(self) -> float:
        return float(np.mean([float(row["nll_per_dim_normalized"]) for row in self.likelihood_rows]))


def build_parser() -> argparse.ArgumentParser:
    default_base = (
        _REPO_DIR / "results/Isaac-Weight-Droid-Visuomotor-v0/"
        "weight_task_eps_bidir_demo_meanstd_taskonly_100_batch25"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument("--count-per-outcome", type=int, default=20)
    parser.add_argument("--preceding-chunks", type=int, default=5)
    parser.add_argument("--post-event-frames", type=int, default=12)
    parser.add_argument("--title-frames", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--project", default="openpi")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="weight-pear-grasp-pi05-likelihood")
    parser.add_argument("--name", default="pear-grasp-pi05-likelihood-20-success-20-failure")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--no-upload", action="store_true")
    return parser


def _latest(parent: Path, name: str) -> Path:
    paths = list(parent.rglob(name))
    if not paths:
        raise FileNotFoundError(f"No {name} under {parent}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _load_likelihood(base: Path) -> dict[tuple[str, int, int, int], list[dict[str, Any]]]:
    rows_by_key: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for path in sorted((base / "pi05_hutchinson").glob("worker_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    str(row["trace"]), int(row["group_index"]), int(row["lane"]),
                    int(row["seed"]), int(row["step"]),
                )
                rows_by_key[key] = row
    trajectories: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for key, row in rows_by_key.items():
        trajectories[key[:4]].append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda row: int(row["step"]))
    return trajectories


def _detect_attempts(
    trajectories: dict[tuple[str, int, int, int], list[dict[str, Any]]],
    preceding_chunks: int,
    post_event_frames: int,
) -> list[Attempt]:
    compact_frames: dict[tuple[str, int, int, int], list[tuple[int, int, bool]]] = defaultdict(list)
    for trace_string in sorted({key[0] for key in trajectories}):
        with Path(trace_string).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") != "frame":
                    continue
                group = int(record["group_index"])
                step = int(record["step"])
                for lane_record in record["lanes"]:
                    if not lane_record.get("valid", False) or lane_record.get("action") is None:
                        continue
                    key = (trace_string, group, int(lane_record["lane"]), int(lane_record["seed"]))
                    if key not in trajectories:
                        continue
                    closed = int(float(lane_record["action"][7]) >= 0.5)
                    grasped = bool(lane_record.get("subtasks", {}).get("grasp_pear", False))
                    compact_frames[key].append((step, closed, grasped))

    attempts = []
    for key, frames in compact_frames.items():
        rows = trajectories[key]
        row_steps = np.asarray([int(row["step"]) for row in rows])
        previous = frames[0][1]
        for index in range(1, len(frames)):
            event_step, closed, _ = frames[index]
            if previous == 0 and closed == 1:
                row_index = int(np.searchsorted(row_steps, event_step, side="left") - 1)
                if row_index >= preceding_chunks and rows[row_index]["task_phase"] == "grasp_pear":
                    end = index
                    while end < len(frames) and frames[end][1] == 1:
                        end += 1
                    successful = any(item[2] for item in frames[index:end])
                    attempts.append(
                        Attempt(
                            trace=key[0], group=key[1], lane=key[2], seed=key[3],
                            event_step=event_step, success=successful,
                            likelihood_rows=rows[row_index - preceding_chunks : row_index + 1],
                            end_step=event_step + post_event_frames,
                        )
                    )
            previous = closed
    return attempts


def _select_distinct(attempts: list[Attempt], outcome: bool, count: int, rng: np.random.Generator) -> list[Attempt]:
    first_by_seed = {}
    for attempt in sorted(attempts, key=lambda item: (item.seed, item.event_step)):
        if attempt.success == outcome:
            first_by_seed.setdefault(attempt.seed, attempt)
    candidates = list(first_by_seed.values())
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct {'successful' if outcome else 'failed'} attempts; have {len(candidates)}")
    indices = rng.choice(len(candidates), size=count, replace=False)
    return sorted((candidates[int(index)] for index in indices), key=lambda item: item.seed)


def _load_render_frames(attempts: list[Attempt]) -> None:
    attempts_by_key: dict[tuple[str, int, int, int], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_key[attempt.trajectory_key].append(attempt)
    for trace_string in sorted({attempt.trace for attempt in attempts}):
        with Path(trace_string).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") != "frame":
                    continue
                step = int(record["step"])
                group = int(record["group_index"])
                for lane_record in record["lanes"]:
                    if not lane_record.get("valid", False):
                        continue
                    key = (trace_string, group, int(lane_record["lane"]), int(lane_record["seed"]))
                    for attempt in attempts_by_key.get(key, ()):
                        if attempt.start_step <= step <= attempt.end_step:
                            attempt.frames.append((step, lane_record))
    missing = [f"seed={item.seed},step={item.event_step}" for item in attempts if not item.frames]
    if missing:
        raise ValueError(f"No render frames for attempts: {missing}")


def _as_rgb(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected camera image, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        if image.min() < 0:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image[..., :3]


def _plot_panel(attempt: Attempt, frame_step: int, width: int, height: int) -> np.ndarray:
    import cv2

    canvas = np.full((height, width, 3), 248, np.uint8)
    rows = attempt.likelihood_rows
    values = np.asarray([float(row["log_prob_normalized"]) / float(row["dimensions"]) for row in rows])
    offsets = np.arange(-(len(rows) - 1), 1)
    left, right, top, bottom = 70, width - 24, 64, height - 62
    ymin, ymax = float(values.min()), float(values.max())
    if ymax - ymin < 1e-6:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    margin = 0.15 * (ymax - ymin)
    ymin, ymax = ymin - margin, ymax + margin

    def point(index: int) -> tuple[int, int]:
        x = left + int(round(index / max(len(rows) - 1, 1) * (right - left)))
        y = bottom - int(round((values[index] - ymin) / (ymax - ymin) * (bottom - top)))
        return x, y

    for fraction in np.linspace(0, 1, 5):
        y = top + int(round(fraction * (bottom - top)))
        value = ymax - fraction * (ymax - ymin)
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{value:.2f}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (60, 60, 60), 1)
    points = np.asarray([point(i) for i in range(len(rows))], np.int32)
    cv2.polylines(canvas, [points], False, (170, 170, 170), 2, cv2.LINE_AA)
    current = max((i for i, row in enumerate(rows) if int(row["step"]) <= frame_step), default=0)
    for index, xy in enumerate(points):
        color = (35, 155, 86) if index <= current else (185, 185, 185)
        cv2.circle(canvas, tuple(xy), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(int(offsets[index])), (int(xy[0]) - 7, bottom + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1)
    cv2.circle(canvas, tuple(points[current]), 8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "pi0.5 pear-grasp event likelihood", (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "normalized log p / action dim (higher is better)", (18, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "policy chunk relative to close event", (left + 45, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1)
    cv2.putText(canvas, f"current: {values[current]:.3f}", (left + 8, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _camera_annotation(image: np.ndarray, attempt: Attempt, clip_index: int, clip_count: int,
                       frame_step: int, grasped_now: bool) -> np.ndarray:
    import cv2

    result = image.copy()
    result[:76] = 0
    outcome = "SUCCESSFUL PEAR GRASP" if attempt.success else "FAILED PEAR GRASP"
    color = (50, 220, 90) if attempt.success else (240, 75, 65)
    cv2.putText(result, f"attempt {clip_index}/{clip_count} | seed {attempt.seed} | {outcome}",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.61, color, 2, cv2.LINE_AA)
    relative = frame_step - attempt.event_step
    live = "YES" if grasped_now else "no"
    cv2.putText(result, f"frame {frame_step} | close event {relative:+d} frames | grasp flag: {live}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def _title_frame(attempt: Attempt, clip_index: int, clip_count: int, width: int, height: int) -> np.ndarray:
    import cv2

    frame = np.full((height, width, 3), 20, np.uint8)
    outcome = "SUCCESSFUL PEAR GRASP" if attempt.success else "FAILED PEAR GRASP"
    color = (50, 220, 90) if attempt.success else (240, 75, 65)
    cv2.putText(frame, f"{outcome}  {clip_index}/{clip_count}", (70, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 1.35, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"seed {attempt.seed} | close frame {attempt.event_step} | six-chunk NLL/dim {attempt.mean_nll:.3f}",
                (70, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(frame, "Next: restored simulation frames + pi0.5 likelihood", (70, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (185, 185, 185), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if args.count_per_outcome < 1 or args.preceding_chunks < 0:
        parser.error("counts must be positive and --preceding-chunks must be nonnegative")

    base = args.base_dir.resolve()
    trajectories = _load_likelihood(base)
    attempts = _detect_attempts(trajectories, args.preceding_chunks, args.post_event_frames)
    rng = np.random.default_rng(args.selection_seed)
    successes = _select_distinct(attempts, True, args.count_per_outcome, rng)
    failures = _select_distinct(attempts, False, args.count_per_outcome, rng)
    selected = successes + failures
    _load_render_frames(selected)
    print("Selected successful attempts:", [(x.seed, x.event_step) for x in successes], flush=True)
    print("Selected failed attempts:", [(x.seed, x.event_step) for x in failures], flush=True)

    output_dir = (args.output_dir or base / "pi05_pear_grasp_event_reels").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_app = AppLauncher(args).app
    import cv2
    import gymnasium as gym
    import imageio.v2 as imageio
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from evaluate_pi05_trace_likelihood import _restore_state

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = selected[0].seed
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    total = sum(len(item.frames) + args.title_frames for item in selected)
    progress = tqdm(total=total, desc="render pear-grasp likelihood reels", unit="frame")
    reel_paths = {}
    manifest_records = []
    try:
        for outcome, reel_attempts in ((True, successes), (False, failures)):
            label = "success" if outcome else "failure"
            path = output_dir / f"pear_grasp_{label}_{len(reel_attempts)}_concat_pi05.mp4"
            writer = imageio.get_writer(
                str(path), fps=args.fps, codec="libx264", macro_block_size=1,
                output_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            try:
                for clip_index, attempt in enumerate(reel_attempts, start=1):
                    title = _title_frame(attempt, clip_index, len(reel_attempts), args.output_width, 360)
                    for _ in range(args.title_frames):
                        writer.append_data(title)
                        progress.update(1)
                    env.reset(seed=attempt.seed)
                    for frame_step, state in attempt.frames:
                        _restore_state(env, state)
                        observation = env.observation_manager.compute(update_history=False)["policy"]
                        camera = _as_rgb(observation["table_cam"])
                        panel_width = args.output_width // 2
                        output_height = int(round(panel_width * camera.shape[0] / camera.shape[1]))
                        camera = cv2.resize(camera, (panel_width, output_height), interpolation=cv2.INTER_AREA)
                        grasped_now = bool(state.get("subtasks", {}).get("grasp_pear", False))
                        camera = _camera_annotation(
                            camera, attempt, clip_index, len(reel_attempts), frame_step, grasped_now
                        )
                        panel = _plot_panel(attempt, frame_step, args.output_width - panel_width, output_height)
                        writer.append_data(np.concatenate((camera, panel), axis=1))
                        progress.update(1)
                        progress.set_postfix(outcome=label, clip=clip_index, seed=attempt.seed, frame=frame_step)
                    manifest_records.append({
                        "outcome": label, "seed": attempt.seed, "event_step": attempt.event_step,
                        "event_chunk_step": attempt.event_chunk_step, "start_step": attempt.start_step,
                        "end_step": attempt.end_step, "rendered_frames": len(attempt.frames),
                        "mean_nll_per_dim": attempt.mean_nll,
                        "chunk_nll_per_dim": [float(row["nll_per_dim_normalized"]) for row in attempt.likelihood_rows],
                    })
            finally:
                writer.close()
            reel_paths[label] = path
    finally:
        progress.close()
        env.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"records": manifest_records, "reels": {k: str(v) for k, v in reel_paths.items()}}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.no_upload:
        print(f"Wrote {manifest_path}; upload disabled", flush=True)
        simulation_app.close()
        return

    import wandb

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group, name=args.name, mode=args.wandb_mode,
        config={
            "task": args.task, "source": str(base), "count_per_outcome": args.count_per_outcome,
            "preceding_chunks": args.preceding_chunks, "post_event_frames": args.post_event_frames,
            "selection_seed": args.selection_seed,
            "success_definition": "grasp_pear becomes true during the closed interval after a 0->1 command crossing",
            "likelihood": "pi0.5 Hutchinson probability-flow ODE, one probe, normalized log p per action dimension",
        },
    )
    run.log({
        "reels/successful_pear_grasps": wandb.Video(str(reel_paths["success"]), format="mp4"),
        "reels/failed_pear_grasps": wandb.Video(str(reel_paths["failure"]), format="mp4"),
    })
    table = wandb.Table(columns=["outcome", "seed", "event_step", "mean_nll_per_dim"])
    for record in manifest_records:
        table.add_data(record["outcome"], record["seed"], record["event_step"], record["mean_nll_per_dim"])
    run.log({"attempts": table})
    artifact = wandb.Artifact(f"{run.name}-event-reels", type="pi05-likelihood-pear-grasp-reels")
    artifact.add_file(str(manifest_path), name="manifest.json")
    for path in reel_paths.values():
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run_url = run.url
    run.finish()
    print(f"WANDB_RUN_URL={run_url}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
