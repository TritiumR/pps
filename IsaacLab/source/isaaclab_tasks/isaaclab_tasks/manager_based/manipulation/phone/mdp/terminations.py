# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the phone task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if the asset falls below a minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def _gripper_is_closed(
    env: ManagerBasedRLEnv,
    robot: Articulation,
) -> torch.Tensor:
    """Check if both gripper joints are away from the configured open value."""
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    gripper_1_closed = (
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - gripper_open_val) > env.cfg.gripper_threshold
    )
    gripper_2_closed = (
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - gripper_open_val) > env.cfg.gripper_threshold
    )
    return torch.logical_and(gripper_1_closed, gripper_2_closed)


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def phone_grasped_and_lifted(
    env: ManagerBasedRLEnv,
    phone_cfg: SceneEntityCfg = SceneEntityCfg("phone_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.9,
    diff_threshold: float = 0.08,
) -> torch.Tensor:
    """Check if phone_1 is grasped and lifted above the height threshold."""
    phone: RigidObject = env.scene[phone_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    phone_pos = phone.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(phone_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pose_diff < diff_threshold, _gripper_is_closed(env, robot))
    lifted = phone_pos[:, 2] > height_threshold
    return torch.logical_and(grasped, lifted)


def task_done_phone(
    env: ManagerBasedRLEnv,
    phone_cfg: SceneEntityCfg = SceneEntityCfg("phone_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.8,
    diff_threshold: float = 0.08,
) -> torch.Tensor:
    """Success when phone_1 is grasped and lifted."""
    return phone_grasped_and_lifted(
        env,
        phone_cfg=phone_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        height_threshold=height_threshold,
        diff_threshold=diff_threshold,
    )
