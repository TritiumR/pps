# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the wash plate task.

TODO: Define observation functions for wash task subtasks:
- object_grasped: check if plate is grasped
- plate_at_sink: check if plate is at sink area
- plate_at_rack: check if plate is placed on rack
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The position of the object in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_pos_w


def object_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The orientation of the object in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_quat_w


def ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert (
                len(gripper_joint_ids) == 2
            ), "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = (
                robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            )
            finger_joint_2 = -1 * robot.data.joint_pos[
                :, gripper_joint_ids[1]
            ].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError(
                "[Error] Cannot find gripper_joint_names in the environment config"
            )


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    # print(f"pose_diff: {pose_diff}")

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(
            -1, 1
        )  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert (
                len(gripper_joint_ids) == 2
            ), "Observations only support parallel gripper for now"

            gripper_1 = torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                    env.device
                )
            )
            gripper_2 = torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                    env.device
                )
            )

            # print(f"gripper_1: {gripper_1}")
            # print(f"gripper_2: {gripper_2}")
            # print(f"env.cfg.gripper_threshold: {env.cfg.gripper_threshold}")

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    )
                )
                > env.cfg.gripper_threshold,
            )
            # print(f"grasped: {grasped}")
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    )
                )
                > env.cfg.gripper_threshold,
            )
            # print(f"grasped: {grasped}")

    return grasped


# TODO: Define plate_at_sink function for wash task
# def plate_at_sink(
#     env: ManagerBasedRLEnv,
#     plate_cfg: SceneEntityCfg,
#     sink_pos: tuple[float, float, float],
#     threshold: float = 0.1,
# ) -> torch.Tensor:
#     """Check if the plate is at the sink area (washed).
#
#     Args:
#         env: The environment.
#         plate_cfg: Plate configuration.
#         sink_pos: Position of the sink area.
#         threshold: Distance threshold for considering plate at sink.
#
#     Returns:
#         Boolean tensor indicating if plate is at sink.
#     """
#     plate: RigidObject = env.scene[plate_cfg.name]
#     # TODO: Implement sink area check
#     pass
#

# TODO: Define plate_at_rack function for wash task
# def plate_at_rack(
#     env: ManagerBasedRLEnv,
#     plate_cfg: SceneEntityCfg,
#     rack_cfg: SceneEntityCfg,
#     threshold: float = 0.05,
# ) -> torch.Tensor:
#     """Check if the plate is placed on the rack.
#
#     Args:
#         env: The environment.
#         plate_cfg: Plate configuration.
#         rack_cfg: Plate rack configuration.
#         threshold: Distance threshold for considering plate on rack.
#
#     Returns:
#         Boolean tensor indicating if plate is on rack.
#     """
#     plate: RigidObject = env.scene[plate_cfg.name]
#     rack: RigidObject = env.scene[rack_cfg.name]
#     # TODO: Implement rack placement check
#     pass


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    elif return_key is None:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
