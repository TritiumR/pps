# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the ice task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def _asset_root_position_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def _gripper_is_open(
    env,
    robot: Articulation,
    atol: float | None = None,
    rtol: float | None = None,
) -> torch.Tensor:
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(gripper_joint_ids) == 0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    threshold = env.cfg.gripper_threshold if atol is None else atol
    relative_tol = 0.0 if rtol is None else rtol
    target = torch.full(
        (env.num_envs,),
        float(env.cfg.gripper_open_val),
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )

    open_mask = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        target,
        atol=threshold,
        rtol=relative_tol,
    )
    if len(gripper_joint_ids) > 1:
        open_mask = torch.logical_and(
            open_mask,
            torch.isclose(
                robot.data.joint_pos[:, gripper_joint_ids[1]],
                target,
                atol=threshold,
                rtol=relative_tol,
            ),
        )
    return open_mask


def ice_in_cup(
    env,
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    ice_cfg: SceneEntityCfg = SceneEntityCfg("ice_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.055,
    z_min_threshold: float = 0.035,
    z_max_threshold: float = 0.16,
    require_gripper_open: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
) -> torch.Tensor:
    """Check if the ice cube is inside the cup and released."""
    cup_pos_w = _asset_root_position_w(env, cup_cfg)
    ice_pos_w = _asset_root_position_w(env, ice_cfg)
    pos_diff = ice_pos_w - cup_pos_w

    placed = torch.linalg.norm(pos_diff[:, :2], dim=1) < xy_threshold
    placed = torch.logical_and(placed, pos_diff[:, 2] > z_min_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] < z_max_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        placed = torch.logical_and(placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    return placed
