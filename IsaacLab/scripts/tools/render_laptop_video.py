# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to render a video from the laptop environment cameras."""

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Render video from laptop environment cameras.")
parser.add_argument("--task", type=str, default="Isaac-Laptop-Droid-Visuomotor-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to render.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--output_dir", type=str, default="./videos", help="Output directory for videos.")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode.")
parser.add_argument("--save_images", action="store_true", default=False, help="Save individual frames as images.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def save_video(frames, output_path, fps=30):
    """Save frames as a video file using OpenCV."""
    import cv2

    if len(frames) == 0:
        print(f"[WARNING] No frames to save for {output_path}")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

    out.release()
    print(f"[INFO] Saved video to: {output_path}")


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
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=True,
        num_envs=args_cli.num_envs,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Set seed
    env.reset(seed=args_cli.seed)

    # Storage for frames
    table_cam_frames = []
    wrist_cam_frames = []

    print(f"[INFO] Recording {args_cli.num_steps} steps...")

    # Run simulation and collect frames
    for step in range(args_cli.num_steps):
        # Sample random actions with robust shape handling across wrapped/vectorized env spaces
        actions = sample_random_actions(env)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Extract camera observations
        if "policy" in obs:
            policy_obs = obs["policy"]

            # Table camera (assuming it's named "table_cam" in observations)
            if "table_cam" in policy_obs:
                table_img = policy_obs["table_cam"][0].cpu().numpy()
                # Convert from (H, W, C) format
                if table_img.dtype == np.uint8:
                    table_cam_frames.append(table_img)
                else:
                    # Normalize if needed
                    table_img = (table_img * 255).astype(np.uint8) if table_img.max() <= 1.0 else table_img.astype(np.uint8)
                    table_cam_frames.append(table_img)

            # Wrist camera (assuming it's named "wrist_cam" in observations)
            if "wrist_cam" in policy_obs:
                wrist_img = policy_obs["wrist_cam"][0].cpu().numpy()
                if wrist_img.dtype == np.uint8:
                    wrist_cam_frames.append(wrist_img)
                else:
                    wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
                    wrist_cam_frames.append(wrist_img)

        # Save individual frames if requested
        if args_cli.save_images and step % 10 == 0:
            if len(table_cam_frames) > 0:
                cv2.imwrite(
                    os.path.join(output_dir, f"table_cam_step_{step:04d}.png"),
                    cv2.cvtColor(table_cam_frames[-1], cv2.COLOR_RGB2BGR)
                )
            if len(wrist_cam_frames) > 0:
                cv2.imwrite(
                    os.path.join(output_dir, f"wrist_cam_step_{step:04d}.png"),
                    cv2.cvtColor(wrist_cam_frames[-1], cv2.COLOR_RGB2BGR)
                )

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"[INFO] Step {step + 1}/{args_cli.num_steps}")

    # Save videos
    print(f"[INFO] Saving videos to {output_dir}...")

    if len(table_cam_frames) > 0:
        table_video_path = os.path.join(output_dir, "table_cam.mp4")
        save_video(table_cam_frames, table_video_path)

    if len(wrist_cam_frames) > 0:
        wrist_video_path = os.path.join(output_dir, "wrist_cam.mp4")
        save_video(wrist_cam_frames, wrist_video_path)

    # Close environment
    env.close()

    print(f"[INFO] Done! Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
