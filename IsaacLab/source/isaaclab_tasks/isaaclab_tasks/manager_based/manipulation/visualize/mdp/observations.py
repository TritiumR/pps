# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
    """The position of the bird in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_pos_w


def object_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The orientation of the bird in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_quat_w


def pig_position_in_world_frame(
    env: ManagerBasedRLEnv,
    pig_cfg: SceneEntityCfg = SceneEntityCfg("pig"),
) -> torch.Tensor:
    """The position of the pig in the world frame."""
    pig: RigidObject = env.scene[pig_cfg.name]

    return pig.data.root_pos_w


def vase_position_in_world_frame(
    env: ManagerBasedRLEnv,
    vase_cfg: SceneEntityCfg = SceneEntityCfg("vase"),
) -> torch.Tensor:
    """The position of the vase in the world frame."""
    vase: RigidObject = env.scene[vase_cfg.name]

    return vase.data.root_pos_w


def instance_randomize_bird_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    bird_cfg: SceneEntityCfg = SceneEntityCfg("bird"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    bird: RigidObjectCollection = env.scene[bird_cfg.name]
    bird_pos_w = []
    for env_id in range(env.num_envs):
        bird_pos_w.append(
            bird.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
        )
    bird_pos_w = torch.stack(bird_pos_w)

    return bird_pos_w


def instance_randomize_pig_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    pig_cfg: SceneEntityCfg = SceneEntityCfg("pig"),
) -> torch.Tensor:
    """The position of the pig in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    pig: RigidObjectCollection = env.scene[pig_cfg.name]
    pig_pos_w = []
    for env_id in range(env.num_envs):
        pig_pos_w.append(
            pig.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
        )
    pig_pos_w = torch.stack(pig_pos_w)

    return pig_pos_w


def instance_randomize_vase_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    vase_cfg: SceneEntityCfg = SceneEntityCfg("vase"),
) -> torch.Tensor:
    """The position of the vase in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    vase: RigidObjectCollection = env.scene[vase_cfg.name]
    vase_pos_w = []
    for env_id in range(env.num_envs):
        vase_pos_w.append(
            vase.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
        )
    vase_pos_w = torch.stack(vase_pos_w)

    return vase_pos_w


def bird_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    bird_cfg: SceneEntityCfg = SceneEntityCfg("bird"),
):
    """The orientation of the bird in the world frame."""
    bird: RigidObject = env.scene[bird_cfg.name]

    return bird.data.root_quat_w


def instance_randomize_bird_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(
            cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4]
        )
        cube_2_quat_w.append(
            cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4]
        )
        cube_3_quat_w.append(
            cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4]
        )
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


# def object_obs(
#     env: ManagerBasedRLEnv,
#     bird_cfg: SceneEntityCfg = SceneEntityCfg("bird"),
#     pig_cfg: SceneEntityCfg = SceneEntityCfg("pig"),
#     vase_cfg: SceneEntityCfg = SceneEntityCfg("vase"),
#     block_2_cfg: SceneEntityCfg = SceneEntityCfg("block_2"),
#     block_3_cfg: SceneEntityCfg = SceneEntityCfg("block_3"),
#     block_4_cfg: SceneEntityCfg = SceneEntityCfg("block_4"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ):
#     """
#     Object observations (in world frame):
#         cube_1 pos,
#         cube_1 quat,
#         cube_2 pos,
#         cube_2 quat,
#         cube_3 pos,
#         cube_3 quat,
#         gripper to cube_1,
#         gripper to cube_2,
#         gripper to cube_3,
#         cube_1 to cube_2,
#         cube_2 to cube_3,
#         cube_1 to cube_3,
#     """
#     bird: RigidObject = env.scene[bird_cfg.name]
#     pig: RigidObject = env.scene[pig_cfg.name]
#     vase: RigidObject = env.scene[vase_cfg.name]
#     block_2: RigidObject = env.scene[block_2_cfg.name]
#     block_3: RigidObject = env.scene[block_3_cfg.name]
#     block_4: RigidObject = env.scene[block_4_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     bird_pos_w = bird.data.root_pos_w
#     bird_quat_w = bird.data.root_quat_w

#     pig_pos_w = pig.data.root_pos_w
#     pig_quat_w = pig.data.root_quat_w

#     vase_pos_w = vase.data.root_pos_w
#     vase_quat_w = vase.data.root_quat_w

#     ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
#     gripper_to_bird = bird_pos_w - ee_pos_w
#     gripper_to_pig = pig_pos_w - ee_pos_w
#     gripper_to_vase = vase_pos_w - ee_pos_w

#     bird_to_pig = bird_pos_w - pig_pos_w
#     bird_to_vase = bird_pos_w - vase_pos_w
#     pig_to_vase = pig_pos_w - vase_pos_w

#     return torch.cat(
#         (
#             bird_pos_w - env.scene.env_origins,
#             bird_quat_w,
#             pig_pos_w - env.scene.env_origins,
#             pig_quat_w,
#             vase_pos_w - env.scene.env_origins,
#             vase_quat_w,
#             gripper_to_bird,
#             gripper_to_pig,
#             gripper_to_vase,
#             bird_to_pig,
#             bird_to_vase,
#             pig_to_vase,
#         ),
#         dim=1,
#     )


# def instance_randomize_object_obs(
#     env: ManagerBasedRLEnv,
#     cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
#     cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ):
#     """
#     Object observations (in world frame):
#         cube_1 pos,
#         cube_1 quat,
#         cube_2 pos,
#         cube_2 quat,
#         cube_3 pos,
#         cube_3 quat,
#         gripper to cube_1,
#         gripper to cube_2,
#         gripper to cube_3,
#         cube_1 to cube_2,
#         cube_2 to cube_3,
#         cube_1 to cube_3,
#     """
#     if not hasattr(env, "rigid_objects_in_focus"):
#         return torch.full((env.num_envs, 9), fill_value=-1)

#     cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
#     cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
#     cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     cube_1_pos_w = []
#     cube_2_pos_w = []
#     cube_3_pos_w = []
#     cube_1_quat_w = []
#     cube_2_quat_w = []
#     cube_3_quat_w = []
#     for env_id in range(env.num_envs):
#         cube_1_pos_w.append(
#             cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
#         )
#         cube_2_pos_w.append(
#             cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3]
#         )
#         cube_3_pos_w.append(
#             cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3]
#         )
#         cube_1_quat_w.append(
#             cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4]
#         )
#         cube_2_quat_w.append(
#             cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4]
#         )
#         cube_3_quat_w.append(
#             cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4]
#         )
#     cube_1_pos_w = torch.stack(cube_1_pos_w)
#     cube_2_pos_w = torch.stack(cube_2_pos_w)
#     cube_3_pos_w = torch.stack(cube_3_pos_w)
#     cube_1_quat_w = torch.stack(cube_1_quat_w)
#     cube_2_quat_w = torch.stack(cube_2_quat_w)
#     cube_3_quat_w = torch.stack(cube_3_quat_w)

#     ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
#     gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
#     gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
#     gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

#     cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
#     cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
#     cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

#     return torch.cat(
#         (
#             cube_1_pos_w - env.scene.env_origins,
#             cube_1_quat_w,
#             cube_2_pos_w - env.scene.env_origins,
#             cube_2_quat_w,
#             cube_3_pos_w - env.scene.env_origins,
#             cube_3_quat_w,
#             gripper_to_cube_1,
#             gripper_to_cube_2,
#             gripper_to_cube_3,
#             cube_1_to_2,
#             cube_2_to_3,
#             cube_1_to_3,
#         ),
#         dim=1,
#     )


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


def object_moved(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_height: float = 0.2,
    xy_threshold: float = 0.15,
) -> torch.Tensor:
    """Check if an object is moved to the target position."""

    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_cfg.name]

    pos_diff = object.data.root_pos_w - target_object.data.root_pos_w
    height_dist = pos_diff[:, 2]
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    moved = torch.logical_and(xy_dist < xy_threshold, height_dist - target_height > 0)

    return moved


# def cube_poses_in_base_frame(
#     env: ManagerBasedRLEnv,
#     cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
#     cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     return_key: Literal["pos", "quat", None] = None,
# ) -> torch.Tensor:
#     """The position and orientation of the cubes in the robot base frame."""

#     cube_1: RigidObject = env.scene[cube_1_cfg.name]
#     cube_2: RigidObject = env.scene[cube_2_cfg.name]
#     cube_3: RigidObject = env.scene[cube_3_cfg.name]

#     pos_cube_1_world = cube_1.data.root_pos_w
#     pos_cube_2_world = cube_2.data.root_pos_w
#     pos_cube_3_world = cube_3.data.root_pos_w

#     quat_cube_1_world = cube_1.data.root_quat_w
#     quat_cube_2_world = cube_2.data.root_quat_w
#     quat_cube_3_world = cube_3.data.root_quat_w

#     robot: Articulation = env.scene[robot_cfg.name]
#     root_pos_w = robot.data.root_pos_w
#     root_quat_w = robot.data.root_quat_w

#     pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, pos_cube_1_world, quat_cube_1_world
#     )
#     pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, pos_cube_2_world, quat_cube_2_world
#     )
#     pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, pos_cube_3_world, quat_cube_3_world
#     )

#     pos_cubes_base = torch.cat(
#         (pos_cube_1_base, pos_cube_2_base, pos_cube_3_base), dim=1
#     )
#     quat_cubes_base = torch.cat(
#         (quat_cube_1_base, quat_cube_2_base, quat_cube_3_base), dim=1
#     )

#     if return_key == "pos":
#         return pos_cubes_base
#     elif return_key == "quat":
#         return quat_cubes_base
#     elif return_key is None:
#         return torch.cat((pos_cubes_base, quat_cubes_base), dim=1)


# def object_abs_obs_in_base_frame(
#     env: ManagerBasedRLEnv,
#     cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
#     cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """
#     Object Abs observations (in base frame): remove the relative observations, and add abs gripper pos and quat in robot base frame
#         cube_1 pos,
#         cube_1 quat,
#         cube_2 pos,
#         cube_2 quat,
#         cube_3 pos,
#         cube_3 quat,
#         gripper pos,
#         gripper quat,
#     """
#     cube_1: RigidObject = env.scene[cube_1_cfg.name]
#     cube_2: RigidObject = env.scene[cube_2_cfg.name]
#     cube_3: RigidObject = env.scene[cube_3_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot: Articulation = env.scene[robot_cfg.name]

#     root_pos_w = robot.data.root_pos_w
#     root_quat_w = robot.data.root_quat_w

#     cube_1_pos_w = cube_1.data.root_pos_w
#     cube_1_quat_w = cube_1.data.root_quat_w

#     cube_2_pos_w = cube_2.data.root_pos_w
#     cube_2_quat_w = cube_2.data.root_quat_w

#     cube_3_pos_w = cube_3.data.root_pos_w
#     cube_3_quat_w = cube_3.data.root_quat_w

#     pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, cube_1_pos_w, cube_1_quat_w
#     )
#     pos_cube_2_base, quat_cube_2_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, cube_2_pos_w, cube_2_quat_w
#     )
#     pos_cube_3_base, quat_cube_3_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, cube_3_pos_w, cube_3_quat_w
#     )

#     ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
#     ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
#     ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(
#         root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
#     )

#     return torch.cat(
#         (
#             pos_cube_1_base,
#             quat_cube_1_base,
#             pos_cube_2_base,
#             quat_cube_2_base,
#             pos_cube_3_base,
#             quat_cube_3_base,
#             ee_pos_base,
#             ee_quat_base,
#         ),
#         dim=1,
#     )


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
