#!/usr/bin/env python3
"""Visualize the exact observation set used by the 50-demo ref cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import h5py
import numpy as np
from PIL import Image, ImageDraw

from vlm_dp.offline_context import weight_episode_signals


STAGE_NAMES = (
    "approach pear",
    "lift pear",
    "carry pear",
    "place pear",
    "pear released / approach apple",
    "lift apple",
    "carry apple",
    "place apple",
    "apple released / done",
)
STAGE_COLORS = (
    "#4e79a7",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#f28e2b",
    "#e15759",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
)


def _demo_number(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def _representatives(edges: np.ndarray) -> np.ndarray:
    indices = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end <= start:
            indices.append(int(np.clip(start, 0, edges[-1] - 1)))
        else:
            indices.append(int((start + end - 1) // 2))
    return np.asarray(indices, dtype=np.int64)


def _uniform_indices(length: int, count: int) -> np.ndarray:
    """Uniform temporal representatives from an exhaustively sampled demo."""
    return np.rint(np.linspace(0, length - 1, min(length, count))).astype(np.int64)


def _uniform_observation_pages(
    source: h5py.File,
    records: list[dict[str, object]],
    camera: str,
    output_dir: pathlib.Path,
    demos_per_page: int,
    cell_size: int,
    samples_per_demo: int,
) -> list[str]:
    outputs = []
    label_width = 118
    caption_height = 20
    row_height = cell_size + caption_height
    for page in range(math.ceil(len(records) / demos_per_page)):
        page_records = records[page * demos_per_page : (page + 1) * demos_per_page]
        canvas = Image.new(
            "RGB",
            (label_width + samples_per_demo * cell_size, len(page_records) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row, record in enumerate(page_records):
            name = str(record["demo"])
            length = int(record["length"])
            y = row * row_height
            draw.text((5, y + 7), name, fill="black")
            draw.text((5, y + 24), f"cache obs={length}", fill="black")
            frames = source["data"][name]["obs"][camera]
            indices = _uniform_indices(length, samples_per_demo)
            for col, index in enumerate(indices):
                x = label_width + col * cell_size
                frame = Image.fromarray(np.asarray(frames[int(index)]))
                if frame.size != (cell_size, cell_size):
                    frame = frame.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                canvas.paste(frame, (x, y))
                draw.text((x + 3, y + cell_size + 2), f"obs {int(index)}", fill="black")
        path = output_dir / f"cache_{camera}_uniform_page_{page + 1:02d}.jpg"
        canvas.save(path, quality=91, subsampling=0)
        outputs.append(str(path))
    return outputs


def _stage_distribution(records: list[dict[str, object]], output: pathlib.Path) -> list[int]:
    counts = np.asarray([record["durations"] for record in records], dtype=np.int64).sum(axis=0)
    total = int(counts.sum())
    width, height = 1000, 70 + 52 * len(STAGE_NAMES)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"Exact cache observations by stage (total={total}; K=8 trajectories/obs)", fill="black")
    max_count = int(counts.max())
    for row, (name, color, count) in enumerate(zip(STAGE_NAMES, STAGE_COLORS, counts, strict=True)):
        y = 48 + row * 52
        bar_width = round(590 * int(count) / max_count)
        draw.text((12, y + 8), name, fill="black")
        draw.rectangle((290, y, 290 + bar_width, y + 28), fill=color)
        draw.text((895, y + 8), f"{int(count)} ({100 * int(count) / total:.1f}%)", fill="black")
    canvas.save(output)
    return counts.tolist()


def _contact_pages(
    source: h5py.File,
    records: list[dict[str, object]],
    camera: str,
    output_dir: pathlib.Path,
    demos_per_page: int,
    cell_size: int,
) -> list[str]:
    outputs = []
    label_width = 118
    caption_height = 36
    row_height = cell_size + caption_height
    for page in range(math.ceil(len(records) / demos_per_page)):
        page_records = records[page * demos_per_page : (page + 1) * demos_per_page]
        canvas = Image.new(
            "RGB",
            (label_width + len(STAGE_NAMES) * cell_size, len(page_records) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row, record in enumerate(page_records):
            name = str(record["demo"])
            y = row * row_height
            draw.text((5, y + 7), name, fill="black")
            draw.text((5, y + 24), f'n={record["length"]}', fill="black")
            frames = source["data"][name]["obs"][camera]
            for col, (stage, index) in enumerate(
                zip(STAGE_NAMES, record["representatives"], strict=True)
            ):
                x = label_width + col * cell_size
                frame = Image.fromarray(np.asarray(frames[int(index)]))
                if frame.size != (cell_size, cell_size):
                    frame = frame.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                canvas.paste(frame, (x, y))
                draw.rectangle((x, y, x + cell_size - 1, y + 4), fill=STAGE_COLORS[col])
                draw.text((x + 3, y + cell_size + 2), stage, fill="black")
                draw.text((x + 3, y + cell_size + 18), f"frame {int(index)}", fill="black")
        path = output_dir / f"{camera}_page_{page + 1:02d}.jpg"
        canvas.save(path, quality=91, subsampling=0)
        outputs.append(str(path))
    return outputs


def _timeline(records: list[dict[str, object]], output: pathlib.Path) -> None:
    label_width = 108
    bar_width = 1200
    row_height = 25
    legend_height = 92
    canvas = Image.new(
        "RGB",
        (label_width + bar_width + 10, legend_height + len(records) * row_height + 8),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS, strict=True)):
        col = index % 3
        row = index // 3
        x = 8 + col * 390
        y = 7 + row * 25
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 20, y), name, fill="black")
    for row, record in enumerate(records):
        y = legend_height + row * row_height
        name = str(record["demo"])
        length = int(record["length"])
        edges = np.asarray(record["edges"], dtype=np.int64)
        draw.text((5, y + 5), name, fill="black")
        for stage, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            x0 = label_width + round(bar_width * int(start) / length)
            x1 = label_width + round(bar_width * int(end) / length)
            draw.rectangle((x0, y + 3, max(x0 + 1, x1), y + row_height - 4), fill=STAGE_COLORS[stage])
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--demos-per-page", type=int, default=10)
    parser.add_argument("--cell-size", type=int, default=160)
    parser.add_argument("--uniform-samples", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with h5py.File(args.hdf5_path, "r") as source:
        names = sorted(source["data"].keys(), key=_demo_number)
        for name in names:
            demo = source["data"][name]
            length = int(demo.attrs["num_samples"])
            signals = weight_episode_signals(demo)
            bounds = np.asarray(signals["bounds"], dtype=np.int64)
            edges = np.concatenate(([0], bounds, [length]))
            if np.any(np.diff(edges) < 0) or edges[0] != 0 or edges[-1] != length:
                raise ValueError(f"Invalid stage boundaries for {name}: {edges.tolist()}")
            records.append(
                {
                    "demo": name,
                    "length": length,
                    "success": bool(demo.attrs.get("success", False)),
                    "bounds": bounds.tolist(),
                    "edges": edges.tolist(),
                    "representatives": _representatives(edges).tolist(),
                    "durations": np.diff(edges).tolist(),
                }
            )

        pages = {}
        uniform_pages = {}
        for camera in ("table_cam", "wrist_cam"):
            pages[camera] = _contact_pages(
                source,
                records,
                camera,
                args.output_dir,
                args.demos_per_page,
                args.cell_size,
            )
            uniform_pages[camera] = _uniform_observation_pages(
                source,
                records,
                camera,
                args.output_dir,
                args.demos_per_page,
                args.cell_size,
                args.uniform_samples,
            )

    _timeline(records, args.output_dir / "stage_timeline.png")
    stage_counts = _stage_distribution(records, args.output_dir / "cache_stage_distribution.png")
    durations = np.asarray([record["durations"] for record in records], dtype=np.int64)
    summary = {
        "source": str(args.hdf5_path),
        "num_demos": len(records),
        "num_observations": int(sum(int(record["length"]) for record in records)),
        "observation_sampling": "all frames, stride=1; no max_observations",
        "trajectories_per_observation": 8,
        "num_successful_demos": int(sum(bool(record["success"]) for record in records)),
        "all_stage_boundaries_valid": True,
        "stage_names": list(STAGE_NAMES),
        "stage_duration_frames": {
            name: {
                "min": int(durations[:, index].min()),
                "median": float(np.median(durations[:, index])),
                "max": int(durations[:, index].max()),
            }
            for index, name in enumerate(STAGE_NAMES)
        },
        "contact_pages": pages,
        "uniform_cache_observation_pages": uniform_pages,
        "stage_observation_counts": dict(zip(STAGE_NAMES, stage_counts, strict=True)),
        "stage_distribution": str(args.output_dir / "cache_stage_distribution.png"),
        "timeline": str(args.output_dir / "stage_timeline.png"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "stage_boundaries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["demo", "length", "success", *STAGE_NAMES])
        for record in records:
            writer.writerow(
                [record["demo"], record["length"], int(bool(record["success"])), *record["durations"]]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
