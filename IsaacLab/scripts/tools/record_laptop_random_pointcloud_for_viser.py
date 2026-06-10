#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create the laptop Droid IK pointcloud task, apply random actions, and export the observed point cloud sequence.

The script saves:
1. ``pointcloud_video_viser.npz`` with ``points``, ``colors``, ``mask`` and ``step_indices`` arrays
   for direct loading in a Viser playback script.
2. ``pointcloud_viser.npz`` containing the final frame only.
3. ``pointcloud.ply`` for inspection of the final frame in standard point-cloud tools.

Example:
    ./isaaclab.sh -p scripts/tools/record_laptop_random_pointcloud_for_viser.py --headless
"""

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a random-rollout laptop point cloud for Viser.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Laptop-Droid-PointCloud-IK-Rel-v0",
    help="Task name.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=100, help="Number of random-action steps to run.")
parser.add_argument("--capture_env", type=int, default=0, help="Environment index to export from.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.25,
    help="Uniform random action scale. Actions are sampled from [-action_scale, action_scale].",
)
parser.add_argument(
    "--export_stride",
    type=int,
    default=1,
    help="Export one point-cloud frame every N environment steps.",
)
parser.add_argument(
    "--playback_fps",
    type=int,
    default=10,
    help="Suggested playback FPS stored in the exported sequence file.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./pointcloud_outputs",
    help="Directory where outputs are written.",
)
parser.add_argument(
    "--keep_zero_points",
    action="store_true",
    default=False,
    help="Keep zeroed placeholder points instead of filtering them out.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _sample_random_actions(env, action_scale: float) -> torch.Tensor:
    """Sample random actions with a robust shape across env wrappers/configs."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "total_action_dim"):
        action_dim = int(action_manager.total_action_dim)
        return (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0) * action_scale

    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    if len(action_shape) == 1:
        return (2.0 * torch.rand(env.num_envs, action_shape[0], device=env.device) - 1.0) * action_scale
    return (2.0 * torch.rand(*action_shape, device=env.device) - 1.0) * action_scale


def _extract_point_cloud_frame(
    obs: dict, env_index: int, keep_zero_points: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a single environment point cloud frame from policy observations."""
    if "policy" not in obs:
        raise KeyError(f"Expected 'policy' observations. Available keys: {list(obs.keys())}")

    policy_obs = obs["policy"]
    if "point_positions" not in policy_obs or "point_color" not in policy_obs:
        raise KeyError(
            "Expected 'point_positions' and 'point_color' in policy observations. "
            f"Available keys: {list(policy_obs.keys())}"
        )

    points = policy_obs["point_positions"][env_index].detach().cpu()
    colors = policy_obs["point_color"][env_index].detach().cpu()

    if colors.dtype != torch.uint8:
        if torch.max(colors) <= 1.0:
            colors = (colors * 255.0).round()
        colors = colors.clamp(0, 255).to(torch.uint8)

    valid_mask = torch.isfinite(points).all(dim=-1) & torch.isfinite(colors.float()).all(dim=-1)
    if not keep_zero_points:
        valid_mask &= (points.abs().sum(dim=-1) > 0) | (colors.to(torch.float32).sum(dim=-1) > 0)

    return (
        points.numpy().astype(np.float32),
        colors.numpy().astype(np.uint8),
        valid_mask.numpy().astype(bool),
    )


def _save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Save a colored point cloud as an ASCII PLY file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset(seed=args_cli.seed)

    latest_obs = obs
    frame_points = []
    frame_colors = []
    frame_masks = []
    frame_steps = []

    def record_frame(observations: dict, step_index: int) -> None:
        points, colors, mask = _extract_point_cloud_frame(
            observations,
            env_index=args_cli.capture_env,
            keep_zero_points=args_cli.keep_zero_points,
        )
        frame_points.append(points)
        frame_colors.append(colors)
        frame_masks.append(mask)
        frame_steps.append(step_index)

    record_frame(latest_obs, step_index=0)

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = _sample_random_actions(env, args_cli.action_scale)
            latest_obs, _, _, _, _ = env.step(actions)

        if (step + 1) % args_cli.export_stride == 0 or step == args_cli.num_steps - 1:
            record_frame(latest_obs, step_index=step + 1)

        if (step + 1) % 25 == 0 or step == args_cli.num_steps - 1:
            print(f"[INFO] Step {step + 1}/{args_cli.num_steps}")

    points_video = np.stack(frame_points, axis=0)
    colors_video = np.stack(frame_colors, axis=0)
    mask_video = np.stack(frame_masks, axis=0)
    step_indices = np.asarray(frame_steps, dtype=np.int32)

    final_points = points_video[-1][mask_video[-1]]
    final_colors = colors_video[-1][mask_video[-1]]

    video_npz_path = os.path.join(output_dir, "pointcloud_video_viser.npz")
    npz_path = os.path.join(output_dir, "pointcloud_viser.npz")
    ply_path = os.path.join(output_dir, "pointcloud.ply")
    meta_path = os.path.join(output_dir, "metadata.json")

    np.savez_compressed(
        video_npz_path,
        points=points_video,
        colors=colors_video,
        mask=mask_video,
        step_indices=step_indices,
        playback_fps=np.asarray(args_cli.playback_fps, dtype=np.int32),
    )
    np.savez_compressed(npz_path, points=final_points, colors=final_colors)
    _save_ply(ply_path, final_points, final_colors)

    metadata = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "capture_env": args_cli.capture_env,
        "num_steps": args_cli.num_steps,
        "action_scale": args_cli.action_scale,
        "export_stride": args_cli.export_stride,
        "playback_fps": args_cli.playback_fps,
        "seed": args_cli.seed,
        "num_frames_saved": int(points_video.shape[0]),
        "num_points_per_frame": int(points_video.shape[1]),
        "num_points_saved_last_frame": int(final_points.shape[0]),
        "sequence_npz_file": os.path.basename(video_npz_path),
        "npz_file": os.path.basename(npz_path),
        "ply_file": os.path.basename(ply_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    print(f"[INFO] Saved Viser point-cloud video to: {video_npz_path}")
    print(f"[INFO] Saved Viser-friendly point cloud to: {npz_path}")
    print(f"[INFO] Saved PLY point cloud to: {ply_path}")
    print(f"[INFO] Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
