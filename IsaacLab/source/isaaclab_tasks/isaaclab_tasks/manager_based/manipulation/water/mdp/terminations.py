# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the angry bird task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_rotation_exceeds_threshold(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    threshold: float = 4 * torch.pi / 7,
):
    asset: RigidObject = env.scene[asset_cfg.name]

    asset_rot = asset.data.root_quat_w

    angle_with_z_axis = torch.arccos(
        1 - 2 * (asset_rot[:, 1] ** 2 + asset_rot[:, 2] ** 2)
    )  # angle between asset and z axis
    done = angle_with_z_axis > threshold

    return done


def root_velocity_exceeds_threshold(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.5,
):
    asset: RigidObject = env.scene[asset_cfg.name]

    asset_lin_vel = asset.data.root_lin_vel_w
    asset_ang_vel = asset.data.root_ang_vel_w
    # print(f"asset_lin_vel: {asset_lin_vel}")
    # print(f"asset_ang_vel: {asset_ang_vel}")

    asset_lin_vel_norm = torch.norm(asset_lin_vel, dim=-1)
    asset_ang_vel_norm = torch.norm(asset_ang_vel, dim=-1)

    done = torch.logical_or(
        asset_lin_vel_norm > lin_vel_threshold, asset_ang_vel_norm > ang_vel_threshold
    )

    # for i in range(done.shape[0]):
    #     if done[i]:
    #         print(f"env {i} done")
    #         print(f"asset_lin_vel_norm: {asset_lin_vel_norm[i]}")
    #         print(f"asset_ang_vel_norm: {asset_ang_vel_norm[i]}")
    #         print(f"lin_vel_threshold: {lin_vel_threshold}")
    #         print(f"ang_vel_threshold: {ang_vel_threshold}")

    return done


def task_done_pour(
    env: ManagerBasedRLEnv,
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    plant_cfg: SceneEntityCfg = SceneEntityCfg("plant"),
    xy_threshold: float = 0.15,
    height_threshold: float = 0.2,
):
    cup: RigidObject = env.scene[cup_cfg.name]
    plant: RigidObject = env.scene[plant_cfg.name]

    pos_diff = cup.data.root_pos_w - plant.data.root_pos_w
    height_dist = pos_diff[:, 2]
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    done = torch.logical_and(xy_dist < xy_threshold, height_dist - height_threshold > 0)

    # Check cup rotation
    cup_rot = cup.data.root_quat_w

    angle_with_z_axis = torch.arccos(1 - 2 * (cup_rot[:, 1] ** 2 + cup_rot[:, 2] ** 2))
    # print(f"angle_with_z_axis: {angle_with_z_axis}")

    done = torch.logical_and(done, angle_with_z_axis > torch.pi / 4)

    return done
