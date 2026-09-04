#!/usr/bin/env python3
"""Upload an eval_steering result directory and its rollout videos to W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Optional directory containing compressed videos; metadata still comes from results-dir.",
    )
    parser.add_argument("--project", default="openpi")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="eval-steering")
    parser.add_argument("--name", default=None)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    video_dir = args.video_dir.resolve() if args.video_dir is not None else results_dir
    results_path = results_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes found in {results_path}")

    records = []
    for episode in episodes:
        video_path = video_dir / str(episode["video"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing rollout video: {video_path}")
        records.append((episode, video_path))

    import wandb

    summary = payload.get("summary", {})
    config = dict(payload.get("config", {}))
    config.update(
        {
            "results_dir": str(results_dir),
            "video_dir": str(video_dir),
            "eval_run_id": payload.get("run_id"),
            "config_slug": payload.get("config_slug"),
            "num_episodes": summary.get("num_episodes", len(records)),
        }
    )
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=args.name or results_dir.name,
        mode=args.mode,
        config=config,
    )
    table = wandb.Table(
        columns=[
            "seed",
            "success",
            "steps",
            "inference_calls",
            "average_inference_ms",
            "video",
        ]
    )
    log_payload = {}
    for episode, video_path in records:
        video = wandb.Video(str(video_path), fps=args.fps, format="mp4")
        seed = int(episode["seed"])
        table.add_data(
            seed,
            bool(episode["success"]),
            int(episode["steps"]),
            int(episode.get("inference_calls", 0)),
            float(episode.get("average_inference_ms", 0.0)),
            video,
        )
        log_payload[f"rollouts/seed_{seed:04d}"] = video
    log_payload["rollouts/summary"] = table
    log_payload["eval/num_episodes"] = int(summary.get("num_episodes", len(records)))
    log_payload["eval/num_successes"] = int(summary.get("num_successes", 0))
    log_payload["eval/success_rate"] = float(summary.get("success_rate", 0.0))
    run.log(log_payload)

    artifact = wandb.Artifact(
        name=f"{run.name}-metadata",
        type="eval-results",
        metadata={
            "num_episodes": len(records),
            "success_rate": float(summary.get("success_rate", 0.0)),
        },
    )
    artifact.add_file(str(results_path), name="results.json")
    debug_path = results_dir / "mpc_debug.jsonl"
    if debug_path.is_file():
        artifact.add_file(str(debug_path), name="mpc_debug.jsonl")
    run.log_artifact(artifact)
    run_url = run.url
    run.finish()
    print(f"WANDB_RUN_URL={run_url}")
    print(f"UPLOADED_EPISODES={len(records)}")


if __name__ == "__main__":
    main()
