#!/usr/bin/env python3
"""Visualize runtime ref-cache observations as per-episode contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def _episode_number(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[-1]), name
    except ValueError:
        return 0, name


def _sample_evenly(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count)
    return indices[np.rint(positions).astype(np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument("--episodes-per-page", type=int, default=5)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional comma-separated seed numbers or full episode names.",
    )
    args = parser.parse_args()

    cache = np.load(args.cache_path, allow_pickle=False)
    metadata = json.loads(str(cache["metadata_json"].item()))
    k = int(metadata["trajectories_per_observation"])
    levels = int(metadata["labels_per_trajectory"])
    n_obs = int(metadata["num_observations"])
    group = k * levels
    if len(cache["demo_name"]) != n_obs * group:
        raise ValueError("Cache labels do not match observation grouping metadata.")

    demo_grid = cache["demo_name"].reshape(n_obs, k, levels)
    step_grid = cache["step_index"].reshape(n_obs, k, levels)
    if not np.all(demo_grid == demo_grid[:, :1, :1]):
        raise ValueError("demo_name changes within an observation group.")
    if not np.all(step_grid == step_grid[:, :1, :1]):
        raise ValueError("step_index changes within an observation group.")
    demos = demo_grid[:, 0, 0]
    steps = step_grid[:, 0, 0]

    observation_dir = pathlib.Path(f"{args.cache_path}.observations")
    observation_metadata = json.loads((observation_dir / "metadata.json").read_text())
    images = np.load(observation_dir / "images.npy", mmap_mode="r")
    if images.shape[0] != n_obs:
        raise ValueError("Observation image count does not match cache metadata.")
    image_keys = list(observation_metadata["image_keys"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_names = sorted(set(demos.tolist()), key=_episode_number)
    if args.episodes:
        requested = {
            item if item.startswith("seed_") else f"seed_{int(item):06d}"
            for item in args.episodes.split(",")
        }
        missing = requested.difference(episode_names)
        if missing:
            raise ValueError(
                f"Requested episodes are absent from the cache: {sorted(missing)}"
            )
        episode_names = [name for name in episode_names if name in requested]
    episode_indices = {
        name: np.flatnonzero(demos == name)[np.argsort(steps[demos == name])]
        for name in episode_names
    }

    with (args.output_dir / "episode_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "num_observations", "first_step", "last_step"])
        for name in episode_names:
            indices = episode_indices[name]
            writer.writerow([name, len(indices), int(steps[indices[0]]), int(steps[indices[-1]])])

    cell_width = int(images.shape[3])
    image_height = int(images.shape[2])
    caption_height = 20
    label_width = 118
    cell_height = image_height + caption_height
    pages = math.ceil(len(episode_names) / args.episodes_per_page)
    manifest: dict[str, object] = {
        "cache_path": str(args.cache_path),
        "num_observations": n_obs,
        "num_episodes": len(episode_names),
        "frames_per_episode": args.frames_per_episode,
        "image_keys": image_keys,
        "pages": [],
    }

    for camera_index, camera_name in enumerate(image_keys):
        for page in range(pages):
            names = episode_names[
                page * args.episodes_per_page : (page + 1) * args.episodes_per_page
            ]
            canvas = Image.new(
                "RGB",
                (
                    label_width + args.frames_per_episode * cell_width,
                    len(names) * cell_height,
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for row, name in enumerate(names):
                indices = _sample_evenly(episode_indices[name], args.frames_per_episode)
                y = row * cell_height
                draw.text((6, y + 6), name, fill="black")
                draw.text((6, y + 24), f"n={len(episode_indices[name])}", fill="black")
                for col, obs_index in enumerate(indices):
                    x = label_width + col * cell_width
                    frame = Image.fromarray(np.asarray(images[obs_index, camera_index]))
                    canvas.paste(frame, (x, y))
                    draw.text((x + 3, y + image_height + 2), f"step {int(steps[obs_index])}", fill="black")
            output = args.output_dir / f"{camera_name}_page_{page + 1:02d}.jpg"
            canvas.save(output, quality=92, subsampling=0)
            manifest["pages"].append(str(output))

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
