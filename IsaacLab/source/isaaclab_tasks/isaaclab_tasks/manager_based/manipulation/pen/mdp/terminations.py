# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the pen task."""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
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


def _asset_root_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root orientation in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def pen_in_holder(
    env: ManagerBasedRLEnv,
    pen_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder001"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    center_xy_threshold: float = 0.035,
    insertion_probe_depth: float = 0.11,
    insertion_min_height: float = 0.0,
    insertion_max_height: float = 0.05,
    max_axis_deviation_rad: float = math.radians(35.0),
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if the pen is inserted into the holder and released.

    The holder root is near the cavity centerline, so success is evaluated in the holder's local frame.
    A valid placement must keep the pen center near the holder axis, align the pen shaft with the holder axis,
    and place a lower shaft probe point below the holder rim.
    """
    pen_pos_w = _asset_root_position_w(env, pen_cfg)
    pen_quat_w = _asset_root_quat_w(env, pen_cfg)
    holder_pos_w = _asset_root_position_w(env, holder_cfg)
    holder_quat_w = _asset_root_quat_w(env, holder_cfg)

    pen_pos_holder = math_utils.quat_apply_inverse(holder_quat_w, pen_pos_w - holder_pos_w)
    center_xy_dist = torch.linalg.vector_norm(pen_pos_holder[:, :2], dim=1)

    pen_axis = torch.tensor((1.0, 0.0, 0.0), dtype=pen_pos_holder.dtype, device=env.device).unsqueeze(0)
    pen_axis = pen_axis.repeat(pen_pos_holder.shape[0], 1)
    pen_axis_w = math_utils.quat_apply(pen_quat_w, pen_axis)
    pen_axis_holder = math_utils.quat_apply_inverse(holder_quat_w, pen_axis_w)

    probe_a_holder = pen_pos_holder + pen_axis_holder * insertion_probe_depth
    probe_b_holder = pen_pos_holder - pen_axis_holder * insertion_probe_depth
    lower_probe_holder = torch.where(
        (probe_a_holder[:, 2] <= probe_b_holder[:, 2]).unsqueeze(1),
        probe_a_holder,
        probe_b_holder,
    )

    placed = torch.logical_and(
        center_xy_dist <= center_xy_threshold,
        torch.logical_and(
            torch.abs(pen_axis_holder[:, 2]) >= math.cos(max_axis_deviation_rad),
            torch.logical_and(
                lower_probe_holder[:, 2] >= insertion_min_height,
                lower_probe_holder[:, 2] <= insertion_max_height,
            ),
        ),
    )

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        placed = torch.logical_and(placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    return placed


def task_done_pen(
    env: ManagerBasedRLEnv,
    pen_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder001"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    center_xy_threshold: float = 0.035,
    insertion_probe_depth: float = 0.11,
    insertion_min_height: float = 0.0,
    insertion_max_height: float = 0.05,
    max_axis_deviation_rad: float = math.radians(35.0),
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when the pen is in the holder and the gripper is open."""
    return pen_in_holder(
        env,
        pen_cfg=pen_cfg,
        holder_cfg=holder_cfg,
        robot_cfg=robot_cfg,
        center_xy_threshold=center_xy_threshold,
        insertion_probe_depth=insertion_probe_depth,
        insertion_min_height=insertion_min_height,
        insertion_max_height=insertion_max_height,
        max_axis_deviation_rad=max_axis_deviation_rad,
        require_gripper_open=True,
        atol=atol,
        rtol=rtol,
    )
