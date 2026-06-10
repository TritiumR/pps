# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the tea task."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
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


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def _asset_root_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root orientation in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


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


def _teapot_mouth_position_w(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg,
    mouth_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Return the teapot mouth position in world frame using a fixed local offset."""
    teapot_pos_w = _asset_root_position_w(env, teapot_cfg)
    teapot_quat_w = _asset_root_quat_w(env, teapot_cfg)
    mouth_offset_t = torch.tensor(mouth_offset, dtype=teapot_pos_w.dtype, device=env.device).unsqueeze(0)
    mouth_offset_t = mouth_offset_t.repeat(teapot_pos_w.shape[0], 1)
    return teapot_pos_w + math_utils.quat_apply(teapot_quat_w, mouth_offset_t)


def _teapot_relative_roll(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg = SceneEntityCfg("teapot"),
) -> torch.Tensor:
    """Return the teapot roll relative to pi/2 in radians."""
    teapot_quat_w = _asset_root_quat_w(env, teapot_cfg)
    roll, _, _ = math_utils.euler_xyz_from_quat(teapot_quat_w)
    return roll - torch.pi / 2.0


def teapot_mouth_near_teacup_xy(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg = SceneEntityCfg("teapot"),
    teacup_cfg: SceneEntityCfg = SceneEntityCfg("teacup"),
    mouth_offset: tuple[float, float, float] = (0.0, 0.05847, -0.06146),
    xy_threshold: float = 0.10,
) -> torch.Tensor:
    """Check if the teapot mouth is positioned close to the teacup in XY."""
    teapot_pos_w = _teapot_mouth_position_w(
        env,
        teapot_cfg=teapot_cfg,
        mouth_offset=mouth_offset,
    )
    teacup_pos_w = _asset_root_position_w(env, teacup_cfg)
    pos_diff = teacup_pos_w - teapot_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    # print(f"xy_dist: {xy_dist}")
    return xy_dist <= xy_threshold


def teapot_rolled(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg = SceneEntityCfg("teapot"),
    min_roll_rad: float = math.pi / 6.0,
) -> torch.Tensor:
    """Check if the teapot roll magnitude exceeds the threshold."""
    relative_roll = _teapot_relative_roll(env, teapot_cfg=teapot_cfg)
    # print(f"relative_roll: {relative_roll}")
    return relative_roll >= min_roll_rad


def teapot_relative_roll_exceeds_max(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg = SceneEntityCfg("teapot"),
    max_relative_roll_rad: float = math.pi / 4.0,
) -> torch.Tensor:
    """Check if the teapot roll relative to pi/2 exceeds the allowed maximum."""
    relative_roll = _teapot_relative_roll(env, teapot_cfg=teapot_cfg)
    return relative_roll > max_relative_roll_rad


def task_done_tea(
    env: ManagerBasedRLEnv,
    teapot_cfg: SceneEntityCfg = SceneEntityCfg("teapot"),
    teacup_cfg: SceneEntityCfg = SceneEntityCfg("teacup"),
    mouth_offset: tuple[float, float, float] = (0.0, 0.05847, -0.06146),
    xy_threshold: float = 0.10,
    min_roll_rad: float = math.pi / 6.0,
) -> torch.Tensor:
    """Success when the teapot mouth is over the cup and the teapot is tilted for pouring."""
    mouth_close = teapot_mouth_near_teacup_xy(
        env,
        teapot_cfg=teapot_cfg,
        teacup_cfg=teacup_cfg,
        mouth_offset=mouth_offset,
        xy_threshold=xy_threshold,
    )
    poured = teapot_rolled(
        env,
        teapot_cfg=teapot_cfg,
        min_roll_rad=min_roll_rad,
    )
    return torch.logical_and(mouth_close, poured)
