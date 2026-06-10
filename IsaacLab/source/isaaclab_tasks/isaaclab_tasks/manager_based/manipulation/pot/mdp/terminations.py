# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the pot task."""

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


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


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


def lid_removed_from_pot(
    env: ManagerBasedRLEnv,
    pot_cfg: SceneEntityCfg = SceneEntityCfg("pot"),
    cover_cfg: SceneEntityCfg = SceneEntityCfg("cover"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if the pot lid has been moved away from the pot."""
    pot_pos_w = _asset_root_position_w(env, pot_cfg)
    cover_pos_w = _asset_root_position_w(env, cover_cfg)
    pos_diff = cover_pos_w - pot_pos_w

    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    # print(f"xy_dist: {xy_dist}")
    placed = xy_dist >= xy_threshold

    robot: Articulation = env.scene[robot_cfg.name]
    placed = torch.logical_and(placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))
    # height_offset = pos_diff[:, 2]
    return placed


def egg_in_pot(
    env: ManagerBasedRLEnv,
    pot_cfg: SceneEntityCfg = SceneEntityCfg("pot"),
    egg_cfg: SceneEntityCfg = SceneEntityCfg("egg"),
    xy_threshold: float = 0.10,
    z_min_threshold: float = 0.00,
    z_max_threshold: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    require_gripper_open: bool = False,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if the egg is within the pot area."""
    pot_pos_w = _asset_root_position_w(env, pot_cfg)
    egg_pos_w = _asset_root_position_w(env, egg_cfg)
    pos_diff = egg_pos_w - pot_pos_w

    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2])
    # print(f"xy_dist: {xy_dist}, z_dist: {z_dist}")
    placed = torch.logical_and(xy_dist <= xy_threshold, z_dist >= z_min_threshold)
    placed = torch.logical_and(placed, z_dist <= z_max_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        placed = torch.logical_and(placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    return placed


def task_done_pot(
    env: ManagerBasedRLEnv,
    pot_cfg: SceneEntityCfg = SceneEntityCfg("pot"),
    cover_cfg: SceneEntityCfg = SceneEntityCfg("cover"),
    egg_cfg: SceneEntityCfg = SceneEntityCfg("egg"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lid_xy_threshold: float = 0.08,
    lid_height_threshold: float = 0.06,
    egg_xy_threshold: float = 0.10,
    egg_z_min_threshold: float = 0.00,
    egg_z_max_threshold: float = 0.05,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when the lid is removed and the egg is placed in the pot and released."""
    lid_removed = lid_removed_from_pot(
        env,
        pot_cfg=pot_cfg,
        cover_cfg=cover_cfg,
        xy_threshold=lid_xy_threshold,
        height_threshold=lid_height_threshold,
    )
    egg_placed = egg_in_pot(
        env,
        pot_cfg=pot_cfg,
        egg_cfg=egg_cfg,
        xy_threshold=egg_xy_threshold,
        z_min_threshold=egg_z_min_threshold,
        z_max_threshold=egg_z_max_threshold,
        robot_cfg=robot_cfg,
        require_gripper_open=True,
        atol=atol,
        rtol=rtol,
    )
    return torch.logical_and(lid_removed, egg_placed)


def pot_cover_separated(
    env: ManagerBasedRLEnv,
    pot_cfg: SceneEntityCfg = SceneEntityCfg("pot"),
    cover_cfg: SceneEntityCfg = SceneEntityCfg("cover"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.06,
) -> torch.Tensor:
    """Backward-compatible alias for lid removal."""
    return lid_removed_from_pot(
        env,
        pot_cfg=pot_cfg,
        cover_cfg=cover_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
    )
