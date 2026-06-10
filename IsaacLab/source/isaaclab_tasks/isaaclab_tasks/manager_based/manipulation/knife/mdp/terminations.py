# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the knife task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
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
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def knife_blade_touching_apple(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    blade_local_points: tuple[tuple[float, float, float], ...] = (
        (0.08, -0.01, 0.0),
        (0.16, -0.01, 0.0),
        (0.24, -0.01, 0.0),
    ),
    touch_threshold: float = 0.07,
) -> torch.Tensor:
    """Check if any sampled knife blade point is close enough to the apple root."""
    apple_pos_w = _asset_root_position_w(env, apple_cfg)
    knife: RigidObject = env.scene[knife_cfg.name]

    blade_points_b = torch.tensor(
        blade_local_points,
        device=env.device,
        dtype=knife.data.root_pos_w.dtype,
    )
    blade_points_b = blade_points_b.unsqueeze(0).repeat(env.num_envs, 1, 1)
    knife_quat_w = knife.data.root_quat_w.unsqueeze(1).repeat(1, blade_points_b.shape[1], 1)
    blade_points_w = knife.data.root_pos_w.unsqueeze(1) + math_utils.quat_apply(
        knife_quat_w.reshape(-1, 4),
        blade_points_b.reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)

    blade_to_apple_dist = torch.linalg.vector_norm(
        blade_points_w - apple_pos_w.unsqueeze(1),
        dim=2,
    )
    min_blade_to_apple_dist = torch.min(blade_to_apple_dist, dim=1).values
    return min_blade_to_apple_dist <= touch_threshold


def task_done_knife(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    blade_local_points: tuple[tuple[float, float, float], ...] = (
        (0.08, -0.01, 0.0),
        (0.16, -0.01, 0.0),
        (0.24, -0.01, 0.0),
    ),
    touch_threshold: float = 0.07,
) -> torch.Tensor:
    """Success when the knife blade touches the apple."""
    return knife_blade_touching_apple(
        env,
        apple_cfg=apple_cfg,
        knife_cfg=knife_cfg,
        blade_local_points=blade_local_points,
        touch_threshold=touch_threshold,
    )


def placeholder_task_term(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Unused placeholder retained for compatibility with copied configs."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
