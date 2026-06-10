#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the Tianji laptop IK env with random EE targets and save camera videos."""

import argparse
import json
import os
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Tianji laptop IK with random EE targets.")
parser.add_argument("--task", type=str, default="Isaac-Laptop-Tianji-Visuomotor-IK-Rel-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_targets", type=int, default=6, help="Number of random targets to sample.")
parser.add_argument("--settle_steps", type=int, default=25, help="Controller steps per sampled target.")
parser.add_argument("--output_dir", type=str, default="./videos", help="Directory for output videos.")
parser.add_argument("--fps", type=int, default=20, help="Output video frame rate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--pos_delta", type=float, default=0.04, help="Max xyz delta in meters for target sampling.")
parser.add_argument("--rot_delta", type=float, default=0.18, help="Max axis-angle delta in radians for target sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = (image * 255.0).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    return image


def _write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    if not frames:
        raise RuntimeError(f"No frames collected for {output_path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _camera_rgb(env, camera_name: str) -> np.ndarray:
    camera = env.unwrapped.scene[camera_name]
    if "rgb" in camera.data.output:
        return _to_uint8_rgb(camera.data.output["rgb"])
    if "rgba" in camera.data.output:
        return _to_uint8_rgb(camera.data.output["rgba"])
    raise KeyError(f"Camera '{camera_name}' does not provide rgb/rgba output. Keys: {list(camera.data.output.keys())}")


def _current_pose(env, frame_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    frame = env.unwrapped.scene[frame_name]
    pos = frame.data.target_pos_w[:, 0, :].clone()
    quat = frame.data.target_quat_w[:, 0, :].clone()
    return pos, quat


def _sample_target(curr_pos: torch.Tensor, curr_quat: torch.Tensor, pos_delta: float, rot_delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    pos_noise = (2.0 * torch.rand_like(curr_pos) - 1.0) * pos_delta
    rot_noise = (2.0 * torch.rand(curr_pos.shape[0], 3, device=curr_pos.device) - 1.0) * rot_delta
    target_pos, target_quat = math_utils.apply_delta_pose(curr_pos, curr_quat, torch.cat((pos_noise, rot_noise), dim=-1))
    return target_pos, target_quat


def _relative_pose_command(curr_pos: torch.Tensor, curr_quat: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    pos_err, rot_err = math_utils.compute_pose_error(
        curr_pos, curr_quat, target_pos, target_quat, rot_error_type="axis_angle"
    )
    command = torch.cat((pos_err, rot_err), dim=-1)
    command[:, :3] = torch.clamp(command[:, :3], min=-0.02, max=0.02)
    command[:, 3:] = torch.clamp(command[:, 3:], min=-0.12, max=0.12)
    return command


def _compose_action(env, left_arm_cmd: torch.Tensor, right_arm_cmd: torch.Tensor, left_gripper: float, right_gripper: float) -> torch.Tensor:
    action_chunks = []
    for term_name in env.unwrapped.action_manager.active_terms:
        term = env.unwrapped.action_manager.get_term(term_name)
        if term_name == "left_arm_action":
            action_chunks.append(left_arm_cmd)
        elif term_name == "right_arm_action":
            action_chunks.append(right_arm_cmd)
        elif term_name == "left_gripper_action":
            action_chunks.append(torch.full((env.num_envs, term.action_dim), left_gripper, device=env.unwrapped.device))
        elif term_name == "right_gripper_action":
            action_chunks.append(torch.full((env.num_envs, term.action_dim), right_gripper, device=env.unwrapped.device))
        else:
            action_chunks.append(torch.zeros((env.num_envs, term.action_dim), device=env.unwrapped.device))
    return torch.cat(action_chunks, dim=-1)


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset(seed=args_cli.seed)

    frames = {
        "table_cam": [],
        "left_wrist_cam": [],
        "right_wrist_cam": [],
    }
    target_log = []

    left_gripper_state = 1.0
    right_gripper_state = 1.0

    for target_idx in range(args_cli.num_targets):
        left_pos, left_quat = _current_pose(env, "left_ee_frame")
        right_pos, right_quat = _current_pose(env, "right_ee_frame")

        left_target_pos, left_target_quat = _sample_target(
            left_pos, left_quat, args_cli.pos_delta, args_cli.rot_delta
        )
        right_target_pos, right_target_quat = _sample_target(
            right_pos, right_quat, args_cli.pos_delta, args_cli.rot_delta
        )

        left_gripper_state = 1.0 if target_idx % 2 == 0 else 0.0
        right_gripper_state = 0.0 if target_idx % 2 == 0 else 1.0

        target_log.append(
            {
                "target_index": target_idx,
                "left_target_pos": left_target_pos[0].detach().cpu().tolist(),
                "left_target_quat": left_target_quat[0].detach().cpu().tolist(),
                "right_target_pos": right_target_pos[0].detach().cpu().tolist(),
                "right_target_quat": right_target_quat[0].detach().cpu().tolist(),
                "left_gripper": left_gripper_state,
                "right_gripper": right_gripper_state,
            }
        )

        for _ in range(args_cli.settle_steps):
            left_pos, left_quat = _current_pose(env, "left_ee_frame")
            right_pos, right_quat = _current_pose(env, "right_ee_frame")
            left_cmd = _relative_pose_command(left_pos, left_quat, left_target_pos, left_target_quat)
            right_cmd = _relative_pose_command(right_pos, right_quat, right_target_pos, right_target_quat)

            actions = _compose_action(env, left_cmd, right_cmd, left_gripper_state, right_gripper_state)
            env.step(actions)

            frames["table_cam"].append(_camera_rgb(env, "table_cam"))
            frames["left_wrist_cam"].append(_camera_rgb(env, "left_wrist_cam"))
            frames["right_wrist_cam"].append(_camera_rgb(env, "right_wrist_cam"))

    for camera_name, camera_frames in frames.items():
        _write_video(camera_frames, os.path.join(output_dir, f"{camera_name}.mp4"), args_cli.fps)

    with open(os.path.join(output_dir, "targets.json"), "w", encoding="utf-8") as f:
        json.dump(target_log, f, indent=2)

    env.close()
    print(output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
