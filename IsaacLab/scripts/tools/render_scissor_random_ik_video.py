#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the scissor IK visuomotor env with random actions and save table camera video."""

import argparse
import json
import os
import subprocess
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run the scissor IK visuomotor env with random actions and save table_cam video."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Scissor-Droid-Visuomotor-IK-Rel-v0",
    help="Task name.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps_per_reset", type=int, default=30, help="Number of random-action steps before resetting.")
parser.add_argument("--num_resets", type=int, default=4, help="Number of reset-rollout cycles to record.")
parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
parser.add_argument("--action_scale", type=float, default=0.25, help="Scale for random arm actions.")
parser.add_argument("--output_dir", type=str, default="./videos", help="Directory for output video and metadata.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert an observation image into uint8 RGB."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 4:
        image = image[0]

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}.")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected last image dim to be 1, 3, or 4. Got {image.shape}.")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return image


def extract_table_cam(obs: dict) -> np.ndarray:
    """Extract the first environment's table camera image from observations."""
    if not isinstance(obs, dict):
        raise TypeError(f"Expected dict observations, got {type(obs)}.")

    policy_obs = obs.get("policy", obs)
    if not isinstance(policy_obs, dict):
        raise TypeError(f"Expected policy observations dict, got {type(policy_obs)}.")

    if "table_cam" not in policy_obs:
        raise KeyError(f"'table_cam' not found in observations. Keys: {list(policy_obs.keys())}")

    return to_uint8_rgb(policy_obs["table_cam"][0])


def write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Write RGB frames to MP4."""
    if not frames:
        raise RuntimeError(f"No frames collected for {output_path}.")

    frames_np = np.stack(frames, axis=0)
    if frames_np.shape[-1] == 4:
        frames_np = frames_np[..., :3]

    if frames_np.dtype != np.uint8:
        frames_np = frames_np.astype(np.float32)
        if frames_np.max() <= 1.0:
            frames_np = frames_np * 255.0
        frames_np = np.clip(frames_np, 0.0, 255.0).astype(np.uint8)

    num_frames, height, width, _ = frames_np.shape
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        frames_np = np.pad(frames_np, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        _, height, width, _ = frames_np.shape

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames_np.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.communicate()[1]
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.decode()}")
    print(f"Saved video: {output_path} ({num_frames} frames)")


def sample_random_actions(env, action_scale: float) -> torch.Tensor:
    """Sample random actions term-by-term to keep gripper actions valid."""
    action_manager = env.action_manager
    chunks = []

    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        action_dim = int(term.action_dim)

        if "gripper" in term_name:
            chunk = torch.randint(
                low=0,
                high=2,
                size=(env.num_envs, action_dim),
                device=env.device,
                dtype=torch.int64,
            ).to(dtype=torch.float32)
        else:
            chunk = (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0) * action_scale

        chunks.append(chunk)

    return torch.cat(chunks, dim=-1)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    frames: list[np.ndarray] = []
    reset_log: list[dict[str, int]] = []

    obs, _ = env.reset(seed=args_cli.seed)
    frames.append(extract_table_cam(obs))
    reset_log.append({"reset_index": 0, "global_step": 0})

    global_step = 0
    with torch.inference_mode():
        for reset_idx in range(args_cli.num_resets):
            for _ in range(args_cli.steps_per_reset):
                actions = sample_random_actions(env, args_cli.action_scale)
                obs, _, terminated, truncated, _ = env.step(actions)
                frames.append(extract_table_cam(obs))
                global_step += 1

                if torch.any(terminated).item() or torch.any(truncated).item():
                    obs, _ = env.reset()
                    frames.append(extract_table_cam(obs))
                    reset_log.append({"reset_index": reset_idx + 1, "global_step": global_step})

            if reset_idx < args_cli.num_resets - 1:
                obs, _ = env.reset()
                frames.append(extract_table_cam(obs))
                reset_log.append({"reset_index": reset_idx + 1, "global_step": global_step})

    video_path = os.path.join(output_dir, "table_cam.mp4")
    write_video(frames, video_path, args_cli.fps)

    metadata = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps_per_reset": args_cli.steps_per_reset,
        "num_resets": args_cli.num_resets,
        "fps": args_cli.fps,
        "action_scale": args_cli.action_scale,
        "seed": args_cli.seed,
        "num_frames": len(frames),
        "video_path": video_path,
        "resets": reset_log,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    print(output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
