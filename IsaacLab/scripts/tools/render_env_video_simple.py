#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple script to render video from environment cameras.

This script renders videos from the table camera and wrist camera of the laptop environment.

Usage:
    ./isaaclab.sh --python scripts/tools/render_env_video_simple.py --task Isaac-Laptop-Droid-Visuomotor-v0 --num_steps 200
"""

import argparse
import cv2
import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Render video from environment cameras.")
parser.add_argument("--task", type=str, default="Isaac-Laptop-Droid-Visuomotor-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to render.")
parser.add_argument("--output_dir", type=str, default="./videos", help="Output directory for videos.")
parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# Add app launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def create_video_writer(output_path: str, fps: int, resolution: tuple):
    """Create a video writer."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, resolution)


def process_image(image: np.ndarray) -> np.ndarray:
    """Process image for video writing (convert to uint8 BGR)."""
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Ensure image is (H, W, 3)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def sample_random_actions(env) -> torch.Tensor:
    """Sample random actions with the expected shape (num_envs, action_dim)."""
    action_manager = getattr(getattr(env, "unwrapped", None), "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "total_action_dim"):
        action_dim = int(action_manager.total_action_dim)
        return torch.randn(env.num_envs, action_dim, device=env.device)

    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    if len(action_shape) == 1:
        return torch.randn(env.num_envs, action_shape[0], device=env.device)
    return torch.randn(*action_shape, device=env.device)


def main():
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    # Disable randomization for consistent rendering
    if hasattr(env_cfg, "events"):
        env_cfg.events = None

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Reset environment
    obs, info = env.reset(seed=args_cli.seed)

    # Storage for frames
    table_frames = []
    wrist_frames = []

    print(f"[INFO] Recording {args_cli.num_steps} steps...")

    # Run simulation
    for step in range(args_cli.num_steps):
        # Random actions with robust shape handling across wrapped/vectorized env spaces
        actions = sample_random_actions(env)

        # Step
        obs, reward, terminated, truncated, info = env.step(actions)

        # Extract camera observations
        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"]

            # Table camera
            if "table_cam" in policy_obs:
                img = policy_obs["table_cam"][0].cpu().numpy()
                if img.shape[0] < 10:  # If channels first, transpose
                    img = np.transpose(img, (1, 2, 0))
                table_frames.append(process_image(img.copy()))

            # Wrist camera
            if "wrist_cam" in policy_obs:
                img = policy_obs["wrist_cam"][0].cpu().numpy()
                if img.shape[0] < 10:  # If channels first, transpose
                    img = np.transpose(img, (1, 2, 0))
                wrist_frames.append(process_image(img.copy()))

        # Reset if terminated
        if terminated.any() or truncated.any():
            obs, info = env.reset()

        # Progress
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{args_cli.num_steps}")

    # Save videos
    print(f"[INFO] Saving videos...")

    if len(table_frames) > 0:
        height, width = table_frames[0].shape[:2]
        table_path = os.path.join(output_dir, "table_cam.mp4")
        writer = create_video_writer(table_path, args_cli.fps, (width, height))
        for frame in table_frames:
            writer.write(frame)
        writer.release()
        print(f"  Table cam: {table_path} ({len(table_frames)} frames)")

    if len(wrist_frames) > 0:
        height, width = wrist_frames[0].shape[:2]
        wrist_path = os.path.join(output_dir, "wrist_cam.mp4")
        writer = create_video_writer(wrist_path, args_cli.fps, (width, height))
        for frame in wrist_frames:
            writer.write(frame)
        writer.release()
        print(f"  Wrist cam: {wrist_path} ({len(wrist_frames)} frames)")

    # Cleanup
    env.close()
    print(f"[INFO] Done! Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
