# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions used for wash-task subtasks and termination checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if the asset falls below a minimum height.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        minimum_height: Minimum height threshold.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    done = height < minimum_height
    return done


def _get_joint_position(asset: Articulation, joint_name: str) -> torch.Tensor:
    """Return joint position tensor for a named joint."""
    joint_ids, _ = asset.find_joints([joint_name])
    assert len(joint_ids) == 1, f"Expected exactly one joint named '{joint_name}', got {len(joint_ids)}"
    return asset.data.joint_pos[:, joint_ids[0]]


def _gripper_is_open(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if both gripper joints are close to the configured open value."""
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    gripper_1_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    gripper_2_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[1]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    return torch.logical_and(gripper_1_open, gripper_2_open)


def task_done_plate(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rack_xy_threshold: float = 0.15,
    desired_z: float = 0.40,
    z_threshold: float = 0.05,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Subtask 3: plate is placed on plate rack and released."""
    plate: RigidObject = env.scene[plate_cfg.name]
    rack: RigidObject = env.scene[rack_cfg.name]

    pos_diff = plate.data.root_pos_w - rack.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(plate.data.root_pos_w[:, 2] - desired_z)
    # print(f"xy_dist: {xy_dist}, z_height: {plate.data.root_pos_w[:, 2]}")
    on_rack = torch.logical_and(xy_dist <= rack_xy_threshold, z_dist <= z_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        on_rack = torch.logical_and(on_rack, gripper_open)

    return on_rack

