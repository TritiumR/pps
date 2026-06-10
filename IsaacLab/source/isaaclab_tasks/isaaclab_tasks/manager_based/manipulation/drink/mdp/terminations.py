# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the drink task."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if the asset falls below a minimum height."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def drink_lid_removed(
    env: ManagerBasedRLEnv,
    drink_cfg: SceneEntityCfg = SceneEntityCfg("drink"),
    lid_cfg: SceneEntityCfg = SceneEntityCfg("drink_lid"),
    body_top_z_offset: float = 0.21825,
    xy_threshold: float = 0.05,
    extra_height_threshold: float = 0.04,
) -> torch.Tensor:
    """Check if the lid has been removed from the drink body."""
    drink: RigidObject = env.scene[drink_cfg.name]
    lid: RigidObject = env.scene[lid_cfg.name]

    pos_diff = lid.data.root_pos_w - drink.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    # height_offset = pos_diff[:, 2]

    # lifted_clear = height_offset >= (body_top_z_offset + extra_height_threshold)
    moved_aside = xy_dist >= xy_threshold
    return moved_aside


def _drink_mouth_position_w(
    env: ManagerBasedRLEnv,
    drink_cfg: SceneEntityCfg,
    body_top_z_offset: float,
) -> torch.Tensor:
    """Return the drink mouth position in world frame from a local z-offset."""
    drink: RigidObject = env.scene[drink_cfg.name]

    drink_pos_w = drink.data.root_pos_w
    drink_quat_w = drink.data.root_quat_w
    mouth_offset_t = torch.tensor(
        (0.0, 0.0, body_top_z_offset),
        dtype=drink_pos_w.dtype,
        device=env.device,
    ).unsqueeze(0)
    mouth_offset_t = mouth_offset_t.repeat(drink_pos_w.shape[0], 1)
    return drink_pos_w + math_utils.quat_apply(drink_quat_w, mouth_offset_t)


def asset_rotated_from_z_axis(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold_rad: float = torch.pi / 3,
) -> torch.Tensor:
    """Check if an asset's local z-axis deviates from the world z-axis by at least the threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]

    local_z_axis = torch.tensor(
        (0.0, 0.0, 1.0),
        dtype=asset.data.root_quat_w.dtype,
        device=env.device,
    ).unsqueeze(0)
    local_z_axis = local_z_axis.repeat(asset.data.root_quat_w.shape[0], 1)
    z_axis_w = math_utils.quat_apply(asset.data.root_quat_w, local_z_axis)
    angle_with_z_axis = torch.arccos(torch.clamp(z_axis_w[:, 2], -1.0, 1.0))
    # print(f"angle_with_z_axis: {angle_with_z_axis}")
    return angle_with_z_axis >= threshold_rad


def drink_poured_into_cup(
    env: ManagerBasedRLEnv,
    drink_cfg: SceneEntityCfg = SceneEntityCfg("drink"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    body_top_z_offset: float = 0.21825,
    xy_threshold: float = 0.15,
    height_threshold: float = 0.10,
    tilt_threshold: float = torch.pi / 4,
) -> torch.Tensor:
    """Check if the drink is positioned over the cup and tilted enough to pour."""
    drink: RigidObject = env.scene[drink_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]

    drink_mouth_pos_w = _drink_mouth_position_w(
        env,
        drink_cfg=drink_cfg,
        body_top_z_offset=body_top_z_offset,
    )
    pos_diff = drink_mouth_pos_w - cup.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    # print(f"xy_dist: {xy_dist}")
    height_dist = pos_diff[:, 2]
    # print(f"height_dist: {height_dist}")
    positioned_to_pour = torch.logical_and(
        xy_dist <= xy_threshold,
        height_dist >= height_threshold,
    )

    tilted_enough = asset_rotated_from_z_axis(
        env,
        asset_cfg=drink_cfg,
        threshold_rad=tilt_threshold,
    )

    return torch.logical_and(positioned_to_pour, tilted_enough)


def task_done_drink(
    env: ManagerBasedRLEnv,
    drink_cfg: SceneEntityCfg = SceneEntityCfg("drink"),
    lid_cfg: SceneEntityCfg = SceneEntityCfg("drink_lid"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    body_top_z_offset: float = 0.21825,
    lid_remove_xy_threshold: float = 0.05,
    lid_remove_height_margin: float = 0.04,
    pour_xy_threshold: float = 0.15,
    pour_height_threshold: float = 0.05,
    pour_tilt_threshold: float = torch.pi / 4,
) -> torch.Tensor:
    """Success when the lid is removed and the drink is in a pouring pose over the cup."""
    lid_removed = drink_lid_removed(
        env,
        drink_cfg=drink_cfg,
        lid_cfg=lid_cfg,
        body_top_z_offset=body_top_z_offset,
        xy_threshold=lid_remove_xy_threshold,
        extra_height_threshold=lid_remove_height_margin,
    )
    poured = drink_poured_into_cup(
        env,
        drink_cfg=drink_cfg,
        cup_cfg=cup_cfg,
        body_top_z_offset=body_top_z_offset,
        xy_threshold=pour_xy_threshold,
        height_threshold=pour_height_threshold,
        tilt_threshold=pour_tilt_threshold,
    )
    return torch.logical_and(lid_removed, poured)


def placeholder_task_term(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Placeholder task term until drink-specific success logic is defined."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
