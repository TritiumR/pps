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
from isaaclab.sensors import ContactSensor

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


def task_done_cylinder_contact(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cylinder_cfg: SceneEntityCfg = SceneEntityCfg("cylinder"),
    triangle_cfg: SceneEntityCfg = SceneEntityCfg("triangle"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("triangle_contact"),
    cylinder_desired_height: float = 0.015,
    triangle_desired_height: float = 0.09,
    contact_threshold: float = 0.001,
    atol=0.0001,
    rtol=0.0001,
):
    """Uses contact sensor to check if triangle is on the cylinder. Kept for reference."""
    robot: Articulation = env.scene[robot_cfg.name]
    cylinder: RigidObject = env.scene[cylinder_cfg.name]
    triangle: RigidObject = env.scene[triangle_cfg.name]

    cylinder_height = cylinder.data.root_pos_w[:, 2]
    triangle_height = triangle.data.root_pos_w[:, 2]

    done = cylinder_height > cylinder_desired_height
    done = torch.logical_and(done, triangle_height > triangle_desired_height)

    # force_matrix_w shape: (num_envs, num_bodies, num_filter_shapes, 3)
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w
    force_magnitudes = torch.norm(force_matrix, dim=-1)
    force_per_filter = force_magnitudes.sum(dim=1)
    contact_with_cylinder = force_per_filter[:, 0] > contact_threshold
    done = torch.logical_and(done, contact_with_cylinder)

    # Check gripper positions
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

    done = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        done,
    )
    done = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        done,
    )

    return done


def task_done_block(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cylinder_cfg: SceneEntityCfg = SceneEntityCfg("cylinder"),
    triangle_cfg: SceneEntityCfg = SceneEntityCfg("triangle"),
    cylinder_desired_height: float = 0.015,
    triangle_desired_height: float = 0.095,
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    atol=0.01,
    rtol=0.01,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cylinder: RigidObject = env.scene[cylinder_cfg.name]
    triangle: RigidObject = env.scene[triangle_cfg.name]

    cylinder_pos = cylinder.data.root_pos_w
    triangle_pos = triangle.data.root_pos_w

    cylinder_height = cylinder_pos[:, 2]
    triangle_height = triangle_pos[:, 2]

    # print(f"cylinder_height: {cylinder_height}")
    # print(f"triangle_height: {triangle_height}")

    done = cylinder_height > cylinder_desired_height
    done = torch.logical_and(done, triangle_height > triangle_desired_height)

    # Check xy distance between triangle and cylinder
    # xy_dist = torch.norm(triangle_pos[:, :2] - cylinder_pos[:, :2], dim=1)
    # done = torch.logical_and(done, xy_dist < xy_threshold)

    # Check that triangle is on the cylinder
    z_dist = triangle_height - cylinder_height
    # print(f"z_dist: {z_dist}")
    
    desired_z_dist = triangle_desired_height - cylinder_desired_height
    done = torch.logical_and(done, abs(z_dist - desired_z_dist) < z_threshold)

    xy_dist = torch.norm(triangle_pos[:, :2] - cylinder_pos[:, :2], dim=1)
    # print(f"xy_dist: {xy_dist}")

    # check if the block is within the xy threshold
    done = torch.logical_and(done, xy_dist < xy_threshold)

    # Check gripper positions
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

    done = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        done,
    )
    done = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        done,
    )

    return done
