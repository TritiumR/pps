#!/usr/bin/env python3
"""Flatten multi-worker eval outputs into seed-indexed videos and one results.json."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import shutil
import tempfile


def parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", 1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=pathlib.Path)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--expected-seeds", default="1-20")
    parser.add_argument("--archive-root", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    expected = parse_seeds(args.expected_seeds)
    selected: dict[int, tuple[tuple[int, float], dict, pathlib.Path, pathlib.Path]] = {}
    source_results: list[pathlib.Path] = []
    for source_index, source in enumerate(args.sources):
        if not source.is_dir():
            raise FileNotFoundError(source)
        for result_path in sorted(source.rglob("results.json")):
            if result_path == args.target / "results.json":
                continue
            payload = json.loads(result_path.read_text())
            source_results.append(result_path)
            for episode in payload.get("episodes", []):
                seed = int(episode["seed"])
                video_name = episode.get("video")
                if not video_name:
                    raise RuntimeError(f"seed {seed} has no video in {result_path}")
                video_path = result_path.parent / video_name
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                priority = (source_index, result_path.stat().st_mtime)
                if seed not in selected or priority > selected[seed][0]:
                    selected[seed] = (priority, copy.deepcopy(episode), video_path, result_path)

    actual = sorted(selected)
    if actual != expected:
        raise RuntimeError(f"seed coverage mismatch: expected={expected}, actual={actual}")

    print(f"target={args.target}")
    for seed in expected:
        _, episode, video_path, result_path = selected[seed]
        print(f"  seed={seed:02d} success={bool(episode.get('success'))} source={result_path} video={video_path.name}")
    if args.dry_run:
        return

    args.target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".consolidate-", dir=args.target) as temp_name:
        temp_dir = pathlib.Path(temp_name)
        episodes = []
        for seed in expected:
            _, episode, video_path, _ = selected[seed]
            outcome = "success" if bool(episode.get("success")) else "fail"
            output_video = temp_dir / f"{seed}_{outcome}.mp4"
            try:
                os.link(video_path, output_video)
            except OSError:
                shutil.copy2(video_path, output_video)
            episode["video"] = output_video.name
            episodes.append(episode)

        first_result = selected[expected[0]][3]
        merged = json.loads(first_result.read_text())
        successes = sum(bool(episode.get("success")) for episode in episodes)
        errored = sum(bool(episode.get("error")) for episode in episodes)
        merged["run_id"] = "consolidated"
        merged["episodes"] = episodes
        merged["summary"] = {
            "num_episodes": len(episodes),
            "num_successes": successes,
            "success_rate": successes / len(episodes),
            "num_errored": errored,
            "num_requested": len(expected),
            "success_rate_including_errored": successes / len(expected),
        }
        merged["total_inference_calls"] = sum(int(episode.get("inference_calls", 0)) for episode in episodes)
        merged["source_results"] = [str(path) for path in source_results]
        (temp_dir / "results.json").write_text(json.dumps(merged, indent=2) + "\n")
        manifest = {
            "expected_seeds": expected,
            "selection": {
                str(seed): {
                    "source_results": str(selected[seed][3]),
                    "source_video": str(selected[seed][2]),
                }
                for seed in expected
            },
        }
        (temp_dir / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")

        for path in sorted(temp_dir.iterdir()):
            os.replace(path, args.target / path.name)

    if args.archive_root is not None:
        archive_dir = args.archive_root / args.target.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        for child in sorted(args.target.iterdir()):
            if child.is_dir() and (child.name == "_workers" or child.name.startswith("worker_")):
                destination = archive_dir / child.name
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(child), destination)

    videos = sorted(args.target.glob("*.mp4"), key=lambda path: int(path.stem.split("_", 1)[0]))
    if [int(path.stem.split("_", 1)[0]) for path in videos] != expected:
        raise RuntimeError("post-consolidation video verification failed")
    merged = json.loads((args.target / "results.json").read_text())
    if sorted(int(episode["seed"]) for episode in merged["episodes"]) != expected:
        raise RuntimeError("post-consolidation results verification failed")
    print(f"consolidated {len(expected)} seeds into {args.target}")


if __name__ == "__main__":
    main()
