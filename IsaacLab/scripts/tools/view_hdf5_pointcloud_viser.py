#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View IsaacLab point-cloud HDF5 demos in a Viser web client.

The expected dataset layout matches the point-cloud IsaacLab converter:

* ``data/demo_*/obs/point_positions``: ``(T, N, 3)`` float XYZ points.
* ``data/demo_*/obs/point_color``: ``(T, N, 3)`` RGB colors.
* (optional) ``data/demo_*/obs/<image_key>``: ``(T, H, W, 3)`` uint8 RGB frames.

Example:
    python scripts/tools/view_hdf5_pointcloud_viser.py \
        ../diffusion_policy/data/tea_wood/generated_dataset_pointcloud.hdf5
"""

from __future__ import annotations

import argparse
import contextlib
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class DemoInfo:
    """Small metadata record for one HDF5 demo group."""

    name: str
    num_frames: int
    num_points: int
    success: bool | None


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"demo_(\d+)", name)
    if match is None:
        return (10**12, name)
    return (int(match.group(1)), name)


def _read_demo_infos(file: h5py.File) -> list[DemoInfo]:
    if "data" not in file:
        raise KeyError("Expected top-level group 'data'.")

    infos: list[DemoInfo] = []
    for demo_name in sorted(file["data"].keys(), key=_demo_sort_key):
        demo = file["data"][demo_name]
        if "obs/point_positions" not in demo or "obs/point_color" not in demo:
            continue

        positions = demo["obs/point_positions"]
        colors = demo["obs/point_color"]
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"{demo_name}/obs/point_positions must have shape (T, N, 3), got {positions.shape}.")
        if colors.shape != positions.shape:
            raise ValueError(
                f"{demo_name} point position/color shape mismatch: {positions.shape} vs {colors.shape}."
            )

        raw_success = demo.attrs.get("success")
        success = bool(raw_success) if raw_success is not None else None
        infos.append(
            DemoInfo(
                name=demo_name,
                num_frames=int(positions.shape[0]),
                num_points=int(positions.shape[1]),
                success=success,
            )
        )

    if not infos:
        raise ValueError("No demos with 'obs/point_positions' and 'obs/point_color' were found.")
    return infos


def _format_status(demo: DemoInfo, frame_index: int, visible_points: int) -> str:
    success_text = "unknown" if demo.success is None else str(demo.success)
    return (
        f"**Dataset frame**  \n"
        f"Demo: `{demo.name}`  \n"
        f"Frame: `{frame_index}` / `{demo.num_frames - 1}`  \n"
        f"Visible points: `{visible_points}` / `{demo.num_points}`  \n"
        f"Success: `{success_text}`"
    )


def _process_point_cloud(
    point_positions: np.ndarray,
    point_color: np.ndarray,
    *,
    drop_zero_points: bool,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same core validity assumptions as the LeRobot converter."""
    points = np.asarray(point_positions, dtype=np.float32)
    colors = np.asarray(point_color, dtype=np.float32)

    if points.shape != colors.shape:
        raise ValueError(f"Point position/color shape mismatch: {points.shape} vs {colors.shape}.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}.")

    valid_mask = np.isfinite(points).all(axis=1)
    if drop_zero_points:
        valid_mask &= (np.abs(points).sum(axis=1) > 0) | (np.abs(colors).sum(axis=1) > 0)

    if valid_mask.any():
        points = points[valid_mask]
        colors = colors[valid_mask]
    else:
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.zeros((1, 3), dtype=np.float32)

    colors = np.nan_to_num(colors, nan=0.0, posinf=255.0, neginf=0.0)
    if colors.size > 0 and float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    if max_points is not None and points.shape[0] > max_points:
        sample_indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[sample_indices]
        colors = colors[sample_indices]

    return points, colors


def _detect_image_keys(file: h5py.File, demo_names: list[str], candidate_keys: list[str]) -> list[str]:
    """Return only the candidate keys that exist as (T, H, W, 3) uint8 datasets in every demo."""
    available: list[str] = []
    for key in candidate_keys:
        ok = True
        for demo_name in demo_names:
            obs = file["data"][demo_name].get("obs")
            if obs is None or key not in obs:
                ok = False
                break
            ds = obs[key]
            if ds.ndim != 4 or ds.shape[-1] != 3:
                ok = False
                break
        if ok:
            available.append(key)
    return available


def _load_image(file: h5py.File, demo_name: str, key: str, frame_index: int, max_width: int | None) -> np.ndarray:
    """Read a single RGB frame from the HDF5 dataset, optionally downsampled to ``max_width``."""
    img = np.asarray(file["data"][demo_name]["obs"][key][frame_index])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if max_width is not None and img.shape[1] > max_width:
        stride = int(np.ceil(img.shape[1] / max_width))
        img = img[::stride, ::stride]
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="View IsaacLab HDF5 point clouds in Viser.")
    parser.add_argument(
        "input",
        nargs="?",
        default="../diffusion_policy/data/tea_wood/generated_dataset_pointcloud.hdf5",
        help="Path to the IsaacLab point-cloud HDF5 dataset.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--demo", type=str, default="demo_0", help="Initial demo name.")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame index.")
    parser.add_argument("--fps", type=float, default=15.0, help="Initial playback FPS.")
    parser.add_argument("--point-size", type=float, default=0.006, help="Rendered point size.")
    parser.add_argument(
        "--drop-zero-points",
        action="store_true",
        help="Drop all-zero placeholder points in addition to non-finite positions.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Uniformly subsample to at most this many visible points per frame.",
    )
    parser.add_argument(
        "--image-keys",
        type=str,
        nargs="*",
        default=["table_cam", "wrist_cam"],
        help="HDF5 obs keys to display as RGB images alongside the point cloud. Pass nothing to disable.",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=320,
        help="Downsample image previews so width does not exceed this many pixels (set 0 to disable).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.input).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    import viser

    h5_file = h5py.File(dataset_path, "r")
    demo_infos = _read_demo_infos(h5_file)
    demo_by_name = {demo.name: demo for demo in demo_infos}
    demo_names = [demo.name for demo in demo_infos]
    initial_demo_name = args.demo if args.demo in demo_by_name else demo_names[0]

    image_keys = _detect_image_keys(h5_file, demo_names, list(args.image_keys or []))
    if args.image_keys and not image_keys:
        print(f"[WARN] Requested image keys {args.image_keys} not found in dataset; skipping image preview.")
    image_max_width = args.image_max_width if args.image_max_width and args.image_max_width > 0 else None

    data_lock = threading.Lock()

    def load_frame(demo_name: str, frame_index: int) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray]]:
        with data_lock:
            demo = demo_by_name[demo_name]
            clamped_frame = min(max(int(frame_index), 0), demo.num_frames - 1)
            obs = h5_file["data"][demo_name]["obs"]
            points, colors = _process_point_cloud(
                obs["point_positions"][clamped_frame],
                obs["point_color"][clamped_frame],
                drop_zero_points=args.drop_zero_points,
                max_points=args.max_points,
            )
            images = {
                key: _load_image(h5_file, demo_name, key, clamped_frame, image_max_width) for key in image_keys
            }
            return points, colors, clamped_frame, images

    initial_points, initial_colors, initial_frame, initial_images = load_frame(initial_demo_name, args.frame)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction((0.0, 0.0, 1.0))

    center = initial_points.mean(axis=0)
    extent = np.maximum(initial_points.max(axis=0) - initial_points.min(axis=0), 0.1)
    camera_distance = float(max(extent.max() * 3.0, 0.75))
    server.initial_camera.look_at = tuple(float(x) for x in center)
    server.initial_camera.position = (
        float(center[0] + camera_distance),
        float(center[1] - camera_distance),
        float(center[2] + camera_distance * 0.7),
    )

    point_cloud_handle = server.scene.add_point_cloud(
        name="/pointcloud",
        points=initial_points,
        colors=initial_colors,
        point_size=args.point_size,
        point_shape="circle",
    )

    max_frame_count = max(demo.num_frames for demo in demo_infos)
    with server.gui.add_folder("Dataset", expand_by_default=True):
        server.gui.add_markdown(f"File: `{dataset_path}`  \nDemos: `{len(demo_infos)}`")
        demo_dropdown = server.gui.add_dropdown("Demo", options=demo_names, initial_value=initial_demo_name)
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(max_frame_count - 1, 0),
            step=1,
            initial_value=initial_frame,
        )
        status_markdown = server.gui.add_markdown(
            _format_status(demo_by_name[initial_demo_name], initial_frame, int(initial_points.shape[0]))
        )

    with server.gui.add_folder("Playback", expand_by_default=True):
        playing_checkbox = server.gui.add_checkbox("Play", initial_value=False)
        fps_input = server.gui.add_number("FPS", initial_value=args.fps, min=0.1, step=1.0)
        point_size_input = server.gui.add_number("Point Size", initial_value=args.point_size, min=0.0005, step=0.0005)
        prev_button = server.gui.add_button("Previous", icon="player-skip-back")
        next_button = server.gui.add_button("Next", icon="player-skip-forward")

    image_handles: dict[str, object] = {}
    if image_keys:
        with server.gui.add_folder("Cameras", expand_by_default=True):
            for key in image_keys:
                image_handles[key] = server.gui.add_image(initial_images[key], label=key)

    def update_frame() -> None:
        demo_name = str(demo_dropdown.value)
        demo = demo_by_name[demo_name]
        requested_frame = int(frame_slider.value)
        if requested_frame >= demo.num_frames:
            frame_slider.value = demo.num_frames - 1
            return

        points, colors, clamped_frame, images = load_frame(demo_name, requested_frame)
        point_cloud_handle.points = points
        point_cloud_handle.colors = colors
        for key, handle in image_handles.items():
            handle.image = images[key]
        status_markdown.content = _format_status(demo, clamped_frame, int(points.shape[0]))

    @demo_dropdown.on_update
    def _(_) -> None:
        frame_slider.value = 0
        update_frame()

    @frame_slider.on_update
    def _(_) -> None:
        update_frame()

    @point_size_input.on_update
    def _(_) -> None:
        point_cloud_handle.point_size = float(point_size_input.value)

    @prev_button.on_click
    def _(_) -> None:
        demo = demo_by_name[str(demo_dropdown.value)]
        frame_slider.value = (int(frame_slider.value) - 1) % demo.num_frames

    @next_button.on_click
    def _(_) -> None:
        demo = demo_by_name[str(demo_dropdown.value)]
        frame_slider.value = (int(frame_slider.value) + 1) % demo.num_frames

    print(f"[INFO] Loaded {len(demo_infos)} demos from {dataset_path}")
    print(f"[INFO] Showing {initial_demo_name} frame {initial_frame}")
    print(f"[INFO] Open Viser at http://localhost:{args.port}")

    try:
        while True:
            if playing_checkbox.value:
                demo = demo_by_name[str(demo_dropdown.value)]
                frame_slider.value = (int(frame_slider.value) + 1) % demo.num_frames
            time.sleep(1.0 / max(float(fps_input.value), 0.1))
    finally:
        with contextlib.suppress(Exception):
            h5_file.close()


if __name__ == "__main__":
    main()
