#!/usr/bin/env python3
"""Replay weight trajectories, overlay pi0.5 likelihood curves, and upload videos to W&B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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


def _latest(parent: Path, name: str) -> Path:
    candidates = list(parent.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"No {name} under {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def build_parser() -> argparse.ArgumentParser:
    default_base = (
        _REPO_DIR / "results/Isaac-Weight-Droid-Visuomotor-v0/"
        "weight_task_eps_bidir_demo_meanstd_taskonly_100_batch25"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument("--seeds", default=None, help="Explicit comma-separated seeds.")
    parser.add_argument("--per-outcome", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--project", default="openpi")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="weight-pi05-likelihood")
    parser.add_argument("--name", default="weight-task-pi05-likelihood-10-trajectories")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--no-upload", action="store_true")
    return parser


def _load_trace(paths: list[Path], wanted: set[int]) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    frames: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid trace JSON at {path}:{line_number}") from error
                if record.get("event") != "frame":
                    continue
                step = int(record["step"])
                for lane in record["lanes"]:
                    if lane.get("valid", False) and int(lane["seed"]) in wanted:
                        frames[int(lane["seed"])].append((step, lane))
    for seed in frames:
        frames[seed].sort(key=lambda item: item[0])
    return frames


def _load_likelihood(paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid likelihood JSON at {path}:{line_number}") from error
                rows[int(row["seed"])].append(row)
    for seed in rows:
        rows[seed].sort(key=lambda row: int(row["step"]))
    return rows


def _load_labels(paths: list[Path]) -> dict[int, bool]:
    labels = {}
    for path in paths:
        for episode in json.loads(path.read_text(encoding="utf-8"))["episodes"]:
            labels[int(episode["seed"])] = bool(episode["success"])
    return labels


def _choose_seeds(labels: dict[int, bool], likelihood, explicit: str | None, count: int) -> list[int]:
    if explicit:
        seeds = [int(value) for value in explicit.split(",") if value.strip()]
        if len(seeds) != len(set(seeds)):
            raise ValueError("--seeds contains duplicates")
        return seeds
    successes = sorted(seed for seed, success in labels.items() if success and seed in likelihood)
    failures = sorted(seed for seed, success in labels.items() if not success and seed in likelihood)
    if len(successes) < count or len(failures) < count:
        raise ValueError(f"Need {count} scored successes and failures; have {len(successes)}, {len(failures)}")
    chosen = []
    for success, failure in zip(successes[:count], failures[:count], strict=True):
        chosen.extend((success, failure))
    return chosen


def _as_rgb(value):
    import numpy as np
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


_PHASE_COLORS = {
    "grasp_pear": (35, 155, 86),
    "place_pear": (230, 126, 34),
    "grasp_apple": (196, 54, 65),
    "place_apple": (95, 70, 180),
}


def _likelihood_panel(rows, current_step: int, width: int, height: int):
    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 74, width - 24, 58, height - 62
    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.float64)
    values = np.asarray(
        [float(row["log_prob_normalized"]) / float(row["dimensions"]) for row in rows],
        dtype=np.float64,
    )
    xmin, xmax = 0.0, max(float(steps.max(initial=1)), float(current_step), 1.0)
    ymin, ymax = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    if ymax - ymin < 1e-6:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    margin = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - margin, ymax + margin

    def xy(step, value):
        x = left + int(round((float(step) - xmin) / (xmax - xmin) * (right - left)))
        y = bottom - int(round((float(value) - ymin) / (ymax - ymin) * (bottom - top)))
        return x, y

    for fraction in np.linspace(0, 1, 5):
        y = top + int(round(fraction * (bottom - top)))
        value = ymax - fraction * (ymax - ymin)
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{value:.2f}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (60, 60, 60), 1)
    points = np.asarray([xy(step, value) for step, value in zip(steps, values, strict=True)], np.int32)
    if len(points) > 1:
        cv2.polylines(canvas, [points], False, (190, 190, 190), 1, cv2.LINE_AA)
    history = np.flatnonzero(steps <= current_step)
    for index in history:
        color = _PHASE_COLORS.get(rows[int(index)]["task_phase"], (30, 110, 190))
        cv2.circle(canvas, tuple(points[index]), 2, color, -1, cv2.LINE_AA)
    current = int(history[-1]) if history.size else 0
    cx, cy = tuple(points[current])
    cv2.line(canvas, (cx, top), (cx, bottom), (35, 35, 35), 1)
    cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "pi0.5 executed-action log likelihood", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "normalized log p / action dimension (higher is better)", (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "simulation frame", ((left + right) // 2 - 60, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(canvas, f"current: {values[current]:.3f}", (left + 8, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas, rows[current]


def _annotate_camera(image, *, seed: int, success: bool, step: int, row):
    import cv2

    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 70), (0, 0, 0), -1)
    label = "SUCCESS" if success else "FAILURE"
    color = (50, 220, 90) if success else (240, 75, 65)
    cv2.putText(output, f"seed {seed} | {label} | frame {step}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(output, f"phase: {row['task_phase']} | NLL/dim: {row['nll_per_dim_normalized']:.3f}",
                (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if args.frame_stride < 1 or args.per_outcome < 1:
        parser.error("--frame-stride and --per-outcome must be positive")

    base = args.base_dir.resolve()
    trace_paths = [_latest(base / f"worker_0{i}_gpu_{i}", "state_trace.jsonl") for i in (0, 1)]
    result_paths = [_latest(base / f"worker_0{i}_gpu_{i}", "results.json") for i in (0, 1)]
    likelihood_paths = [base / "pi05_hutchinson" / f"worker_{i}.jsonl" for i in (0, 1)]
    for path in (*trace_paths, *result_paths, *likelihood_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    likelihood = _load_likelihood(likelihood_paths)
    labels = _load_labels(result_paths)
    seeds = _choose_seeds(labels, likelihood, args.seeds, args.per_outcome)
    traces = _load_trace(trace_paths, set(seeds))
    missing = [seed for seed in seeds if seed not in traces or seed not in likelihood or seed not in labels]
    if missing:
        raise ValueError(f"Missing trace, likelihood, or result labels for seeds: {missing}")
    output_dir = (args.output_dir or base / "pi05_likelihood_videos").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected seeds: {[(seed, labels[seed]) for seed in seeds]}", flush=True)

    simulation_app = AppLauncher(args).app
    import cv2
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from evaluate_pi05_trace_likelihood import _restore_state

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = seeds[0]
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    records = []
    total_frames = sum((len(traces[seed]) + args.frame_stride - 1) // args.frame_stride for seed in seeds)
    progress = tqdm(total=total_frames, desc="render likelihood videos", unit="frame")
    try:
        for seed in seeds:
            env.reset(seed=seed)
            rows = likelihood[seed]
            video_path = output_dir / f"seed_{seed:03d}_{'success' if labels[seed] else 'failure'}_pi05.mp4"
            writer = imageio.get_writer(
                str(video_path), fps=args.fps, codec="libx264", macro_block_size=1,
                output_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            rendered = 0
            try:
                for frame_index, (step, state) in enumerate(traces[seed]):
                    if frame_index % args.frame_stride:
                        continue
                    _restore_state(env, state)
                    obs = env.observation_manager.compute(update_history=False)["policy"]
                    camera = _as_rgb(obs["table_cam"])
                    panel_width = args.output_width // 2
                    output_height = int(round(panel_width * camera.shape[0] / camera.shape[1]))
                    camera = cv2.resize(camera, (panel_width, output_height), interpolation=cv2.INTER_AREA)
                    panel, current_row = _likelihood_panel(rows, step, args.output_width - panel_width, output_height)
                    camera = _annotate_camera(camera, seed=seed, success=labels[seed], step=step, row=current_row)
                    writer.append_data(np.concatenate((camera, panel), axis=1))
                    rendered += 1
                    progress.update(1)
                    progress.set_postfix(seed=seed, success=labels[seed], step=step)
            finally:
                writer.close()
            mean_nll = float(np.mean([float(row["nll_per_dim_normalized"]) for row in rows]))
            records.append({
                "seed": seed, "success": labels[seed], "frames": rendered,
                "likelihood_chunks": len(rows), "mean_nll_per_dim": mean_nll,
                "video": str(video_path), "size_bytes": video_path.stat().st_size,
            })
    finally:
        progress.close()
        env.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    if args.no_upload:
        print(f"Wrote {manifest_path}; upload disabled", flush=True)
        return

    import wandb

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group, name=args.name,
        mode=args.wandb_mode,
        config={
            "task": args.task, "source": str(base), "seeds": seeds,
            "selection": "first five scored successes and first five scored failures",
            "frame_stride": args.frame_stride, "fps": args.fps,
            "likelihood": "pi0.5 Hutchinson probability-flow ODE, normalized log p per action dimension",
        },
    )
    table = wandb.Table(columns=["seed", "success", "frames", "chunks", "mean_nll_per_dim", "video"])
    payload = {}
    for record in records:
        video = wandb.Video(record["video"], fps=args.fps, format="mp4")
        table.add_data(record["seed"], record["success"], record["frames"],
                       record["likelihood_chunks"], record["mean_nll_per_dim"], video)
        payload[f"trajectories/seed_{record['seed']:03d}"] = video
    payload["trajectories/table"] = table
    run.log(payload)
    artifact = wandb.Artifact(f"{run.name}-videos", type="pi05-likelihood-trajectory-videos")
    artifact.add_file(str(manifest_path), name="manifest.json")
    for record in records:
        artifact.add_file(record["video"], name=Path(record["video"]).name)
    run.log_artifact(artifact)
    run_url = run.url
    run.finish()
    print(f"WANDB_RUN_URL={run_url}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
