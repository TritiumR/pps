#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play back a saved pointcloud sequence in Viser."""

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a saved pointcloud video in Viser.")
    parser.add_argument("input", type=str, help="Path to pointcloud_video_viser.npz")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--point_size", type=float, default=0.003, help="Rendered point size.")
    args = parser.parse_args()

    import viser

    data = np.load(args.input)
    points_video = data["points"]
    colors_video = data["colors"]
    mask_video = data["mask"]
    step_indices = data["step_indices"]
    playback_fps = int(data["playback_fps"]) if "playback_fps" in data.files else 10

    num_frames = int(points_video.shape[0])
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction((0.0, 0.0, 1.0))

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(num_frames - 1, 0),
            step=1,
            initial_value=0,
        )
        fps_input = server.gui.add_number("FPS", initial_value=playback_fps, min=1, step=1)
        playing_checkbox = server.gui.add_checkbox("Play", initial_value=True)
        point_size_input = server.gui.add_number(
            "Point Size",
            initial_value=args.point_size,
            min=0.0005,
            step=0.0005,
        )

    initial_mask = mask_video[0]
    point_cloud_handle = server.scene.add_point_cloud(
        name="/pointcloud",
        points=points_video[0][initial_mask],
        colors=colors_video[0][initial_mask],
        point_size=args.point_size,
    )

    def update_frame(frame_index: int) -> None:
        mask = mask_video[frame_index]
        point_cloud_handle.points = points_video[frame_index][mask]
        point_cloud_handle.colors = colors_video[frame_index][mask]

    @frame_slider.on_update
    def _(_) -> None:
        update_frame(int(frame_slider.value))

    @point_size_input.on_update
    def _(_) -> None:
        point_cloud_handle.point_size = float(point_size_input.value)

    print(f"[INFO] Loaded {num_frames} frames from {args.input}")
    print(f"[INFO] Step indices: {step_indices[0]} -> {step_indices[-1]}")
    print(f"[INFO] Open Viser at http://{args.host}:{args.port}")

    while True:
        if playing_checkbox.value and num_frames > 1:
            frame_slider.value = (int(frame_slider.value) + 1) % num_frames
        time.sleep(1.0 / max(float(fps_input.value), 1.0))


if __name__ == "__main__":
    main()
