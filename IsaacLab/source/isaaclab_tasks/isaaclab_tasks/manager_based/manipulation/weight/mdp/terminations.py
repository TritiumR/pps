# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the weight task."""

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
    """Terminate if the asset falls below a minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


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


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def _object_close_to_scale_xy(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    scale_cfg: SceneEntityCfg,
    xy_threshold: float,
    y_offset: float = 0.0,
) -> torch.Tensor:
    """Check if an object root is close enough to the scale in XY."""
    object_pos_w = _asset_root_position_w(env, object_cfg)
    # print(f"object_pos_w: {object_pos_w}")
    scale_pos_w = _asset_root_position_w(env, scale_cfg)
    # print(f"scale_pos_w: {scale_pos_w}")
    xy_dist = torch.linalg.vector_norm(object_pos_w[:, :2] - (scale_pos_w[:, :2] + torch.tensor([0.0, y_offset], device=env.device, dtype=torch.float32)), dim=1)
    # print(f"xy_dist: {xy_dist}")
    # print(f"xy_threshold: {xy_threshold}")
    return xy_dist <= xy_threshold


def apple_on_scale(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    scale_cfg: SceneEntityCfg = SceneEntityCfg("scale"),
    y_offset: float = 0.0,
    xy_threshold: float = 0.12,
) -> torch.Tensor:
    """Check if the apple is close enough to the scale in XY."""
    return _object_close_to_scale_xy(
        env,
        object_cfg=apple_cfg,
        scale_cfg=scale_cfg,
        y_offset=y_offset,
        xy_threshold=xy_threshold,
    )


def pear_on_scale(
    env: ManagerBasedRLEnv,
    pear_cfg: SceneEntityCfg = SceneEntityCfg("pear"),
    scale_cfg: SceneEntityCfg = SceneEntityCfg("scale"),
    y_offset: float = 0.0,
    xy_threshold: float = 0.12,
) -> torch.Tensor:
    """Check if the pear is close enough to the scale in XY."""
    return _object_close_to_scale_xy(
        env,
        object_cfg=pear_cfg,
        scale_cfg=scale_cfg,
        y_offset=y_offset,
        xy_threshold=xy_threshold,
    )


def task_done_weight(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    pear_cfg: SceneEntityCfg = SceneEntityCfg("pear"),
    scale_cfg: SceneEntityCfg = SceneEntityCfg("scale"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.12,
    y_offset: float = 0.0,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when both fruits are on the scale and the gripper is open."""
    apple_placed = apple_on_scale(
        env,
        apple_cfg=apple_cfg,
        scale_cfg=scale_cfg,
        xy_threshold=xy_threshold,
        y_offset=y_offset,
    )
    pear_placed = pear_on_scale(
        env,
        pear_cfg=pear_cfg,
        scale_cfg=scale_cfg,
        xy_threshold=xy_threshold,
        y_offset=y_offset,
    )
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
    return torch.logical_and(torch.logical_and(apple_placed, pear_placed), gripper_open)


def placeholder_task_term(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Unused placeholder retained for compatibility with copied configs."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
