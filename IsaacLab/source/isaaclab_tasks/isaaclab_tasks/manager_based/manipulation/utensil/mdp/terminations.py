# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from isaaclab_tasks.manager_based.manipulation.plate.mdp.terminations import root_height_below_minimum  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def task_done_utensil_inserted(
    env: ManagerBasedRLEnv,
    utensil_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    insert_xy_threshold: float = 0.06,
    min_utensil_height: float = 0.24,
    max_utensil_height: float = 0.42,
    min_gripper_utensil_distance: float = 0.12,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Stateless success check: the utensil is seated in the holder and released.

    Seated means the utensil root is within ``insert_xy_threshold`` of the holder center (over the crock
    opening) and within a plausible standing-height band; released means the gripper is open and has
    withdrawn from the utensil. The height band and xy threshold depend on the crock/utensil geometry, so
    retune them from the rendered scene.
    """
    utensil: RigidObject = env.scene[utensil_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]

    pos_diff = utensil.data.root_pos_w - holder.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    over_opening = xy_dist <= insert_xy_threshold

    height = utensil.data.root_pos_w[:, 2]
    standing = torch.logical_and(height >= min_utensil_height, height <= max_utensil_height)
    success = torch.logical_and(over_opening, standing)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        success = torch.logical_and(success, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_to_utensil_dist = torch.linalg.vector_norm(ee_pos_w - utensil.data.root_pos_w, dim=1)
    success = torch.logical_and(success, ee_to_utensil_dist >= min_gripper_utensil_distance)

    return success
