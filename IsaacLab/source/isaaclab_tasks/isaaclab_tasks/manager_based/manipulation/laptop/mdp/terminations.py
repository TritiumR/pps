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
from isaaclab.sensors import FrameTransformer

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


def _gripper_is_closed(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    close_threshold: float | None = None,
) -> torch.Tensor:
    """Check if both gripper joints have moved away from the configured open value."""
    if close_threshold is None:
        close_threshold = env.cfg.gripper_threshold

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    gripper_1_closed = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - gripper_open_val) > close_threshold
    gripper_2_closed = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - gripper_open_val) > close_threshold
    return torch.logical_and(gripper_1_closed, gripper_2_closed)


# def _laptop_pos_in_table_frame(
#     laptop: Articulation,
#     table: Articulation,
# ) -> torch.Tensor:
#     """Return laptop root position represented in the table root frame."""
#     laptop_pos_table, _ = math_utils.subtract_frame_transforms(
#         table.data.root_pos_w,
#         table.data.root_quat_w,
#         laptop.data.root_pos_w,
#         laptop.data.root_quat_w,
#     )
#     return laptop_pos_table


def laptop_is_closed(
    env: ManagerBasedRLEnv,
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    lid_joint_name: str = "RevoluteJoint_computer_9_up",
    closed_threshold: float = 0.01,
) -> torch.Tensor:
    """Subtask 1: laptop lid is closed."""
    laptop: Articulation = env.scene[laptop_cfg.name]
    lid_joint_pos = _get_joint_position(laptop, lid_joint_name)
    # print(f"lid_joint_pos: {lid_joint_pos}")
    return lid_joint_pos <= closed_threshold


# def drawer_is_opened(
#     env: ManagerBasedRLEnv,
#     table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
#     drawer_joint_name: str = "PrismaticJoint_table_3_right1",
#     drawer_closed_pos: float = 0.1,
#     open_displacement_threshold: float = 0.05,
# ) -> torch.Tensor:
#     """Subtask 2: drawer joint moved away from its closed reference position."""
#     table: Articulation = env.scene[table_cfg.name]
#     drawer_joint_pos = _get_joint_position(table, drawer_joint_name)
#     return torch.abs(drawer_joint_pos - drawer_closed_pos) >= open_displacement_threshold


def laptop_is_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    diff_threshold: float = 0.25,
    gripper_close_threshold: float | None = None,
) -> torch.Tensor:
    """Subtask 3: laptop is near end-effector while the gripper is closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    laptop: Articulation = env.scene[laptop_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    laptop_pos_w = laptop.data.root_pos_w
    ee_to_laptop_dist = torch.linalg.vector_norm(ee_pos_w - laptop_pos_w, dim=1)

    gripper_closed = _gripper_is_closed(env, robot, close_threshold=gripper_close_threshold)
    return torch.logical_and(ee_to_laptop_dist <= diff_threshold, gripper_closed)


def laptop_is_in_drawer(
    env: ManagerBasedRLEnv,
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # desired_x: float = 0.42,
    # desired_y: float = -0.42,
    # xy_threshold: float = 0.15,
    desired_z: float = -0.01,
    z_threshold: float = 0.01,
    require_laptop_open: bool = True,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Subtask 4: laptop is placed in the open drawer and released."""
    laptop: Articulation = env.scene[laptop_cfg.name]
    table: Articulation = env.scene[table_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    laptop_pos_w = laptop.data.root_pos_w
    # print(f"laptop_pos_w: {laptop_pos_w}")

    # laptop_pos_table = _laptop_pos_in_table_frame(laptop, table)
    # laptop_xy_dist = torch.linalg.vector_norm(laptop_pos_table[:, :2], dim=1)
    # laptop_in_xy = laptop_xy_dist <= drawer_xy_threshold

    # laptop_height = laptop_pos_table[:, 2]
    # laptop_in_height = torch.logical_and(
    #     laptop_height >= drawer_min_height_in_table_frame,
    #     laptop_height <= drawer_max_height_in_table_frame,
    # )

    # laptop_in_xy = torch.linalg.vector_norm(laptop_pos_w[:, :2] - torch.tensor([desired_x, desired_y], device=env.device), dim=1) <= xy_threshold
    laptop_in_height = torch.abs(laptop_pos_w[:, 2] - desired_z) <= z_threshold
    # print(f"laptop_in_xy: {laptop_in_xy}, laptop_in_height: {laptop_in_height}")
    # placed = torch.logical_and(laptop_in_xy, laptop_in_height)
    placed = laptop_in_height

    if require_gripper_open:
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        placed = torch.logical_and(placed, gripper_open)

    return placed
    # return False


def drawer_is_closed(
    env: ManagerBasedRLEnv,
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    drawer_joint_name: str = "PrismaticJoint_table_3_right1",
    drawer_closed_pos: float = 0.0,
    close_tolerance: float = 0.02,
) -> torch.Tensor:
    """Subtask 5: drawer joint returns close to the closed reference position."""
    table: Articulation = env.scene[table_cfg.name]
    drawer_joint_pos = _get_joint_position(table, drawer_joint_name)
    # print(f"drawer_joint_pos: {drawer_joint_pos}")
    return torch.abs(drawer_joint_pos - drawer_closed_pos) <= close_tolerance


def task_done_laptop(
    env: ManagerBasedRLEnv,
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lid_joint_name: str = "RevoluteJoint_computer_9_up",
    laptop_closed_threshold: float = 0.01,
    drawer_joint_name: str = "PrismaticJoint_table_3_right1",
    drawer_closed_pos: float = 0.0,
    drawer_close_tolerance: float = 0.02,
    # desired_x: float = 0.68,
    # desired_y: float = -0.43,
    desired_z: float = -0.01,
    z_threshold: float = 0.01,
    # xy_threshold: float = 0.15,
    atol: float = 0.01,
    rtol: float = 0.01,
):
    """Success when laptop is closed, placed in drawer, and drawer is closed."""
    lid_closed = laptop_is_closed(
        env,
        laptop_cfg=laptop_cfg,
        lid_joint_name=lid_joint_name,
        closed_threshold=laptop_closed_threshold,
    )
    # print(f"lid_closed: {lid_closed}")
    laptop_in_drawer = laptop_is_in_drawer(
        env,
        laptop_cfg=laptop_cfg,
        table_cfg=table_cfg,
        robot_cfg=robot_cfg,
        # desired_x=desired_x,
        # desired_y=desired_y,
        desired_z=desired_z,
        z_threshold=z_threshold,
        # xy_threshold=xy_threshold,
        require_gripper_open=False,
        atol=atol,
        rtol=rtol,
    )
    # print(f"laptop_in_drawer: {laptop_in_drawer}")
    drawer_closed = drawer_is_closed(
        env,
        table_cfg=table_cfg,
        drawer_joint_name=drawer_joint_name,
        drawer_closed_pos=drawer_closed_pos,
        close_tolerance=drawer_close_tolerance,
    )
    # print(f"drawer_closed: {drawer_closed}")
    return torch.logical_and(lid_closed, torch.logical_and(laptop_in_drawer, drawer_closed))
