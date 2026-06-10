# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the oven plate task.

TODO: Define observation functions for oven task subtasks:
- object_grasped: check if plate is grasped
- plate_in_oven: check if plate is inside oven
- oven_door_closed: check if the oven door is closed
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_POINT_CLOUD_PIXELS_CACHE: dict[tuple[int, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_sampled_camera_pixels(
    height: int, width: int, num_points: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return evenly sampled flattened pixel indices and homogeneous pixel coordinates."""
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")

    cache_key = (height, width, num_points, str(device))
    cached = _POINT_CLOUD_PIXELS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    total_pixels = height * width
    if num_points >= total_pixels:
        flat_indices = torch.arange(total_pixels, device=device, dtype=torch.long)
    else:
        flat_indices = torch.floor(
            torch.arange(num_points, device=device, dtype=torch.float32) * total_pixels / num_points
        ).to(dtype=torch.long)

    u_coords = torch.div(flat_indices, height, rounding_mode="floor").to(dtype=torch.float32)
    v_coords = torch.remainder(flat_indices, height).to(dtype=torch.float32)
    homogeneous_pixels = torch.stack((u_coords, v_coords, torch.ones_like(u_coords)), dim=0)

    _POINT_CLOUD_PIXELS_CACHE[cache_key] = (flat_indices, homogeneous_pixels)
    return flat_indices, homogeneous_pixels


def _sample_rgbd_camera_point_cloud(
    camera: Camera,
    num_points: int,
    normalize_color: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed-size point cloud from an RGB-D camera in the world frame."""
    depth = camera.data.output["distance_to_image_plane"]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    rgb = camera.data.output["rgb"][..., :3].float()
    if normalize_color:
        rgb = rgb / 255.0

    num_envs, height, width = depth.shape
    flat_indices, homogeneous_pixels = _get_sampled_camera_pixels(height, width, num_points, depth.device)

    flat_depth = depth.transpose(1, 2).reshape(num_envs, -1)
    flat_rgb = rgb.permute(0, 2, 1, 3).reshape(num_envs, -1, 3)

    sampled_depth = flat_depth.index_select(1, flat_indices)
    sampled_rgb = flat_rgb.index_select(1, flat_indices)
    valid_mask = torch.isfinite(sampled_depth) & (sampled_depth > 0.0)
    safe_depth = torch.where(valid_mask, sampled_depth, torch.zeros_like(sampled_depth))

    pixel_rays = torch.matmul(
        torch.linalg.inv(camera.data.intrinsic_matrices),
        homogeneous_pixels.unsqueeze(0).expand(num_envs, -1, -1),
    )
    pixel_rays = pixel_rays / pixel_rays[:, 2:3, :]
    points_camera = pixel_rays.transpose(1, 2) * safe_depth.unsqueeze(-1)
    points_world = math_utils.transform_points(points_camera, camera.data.pos_w, camera.data.quat_w_ros)

    points_world = torch.where(valid_mask.unsqueeze(-1), points_world, torch.zeros_like(points_world))
    sampled_rgb = torch.where(valid_mask.unsqueeze(-1), sampled_rgb, torch.zeros_like(sampled_rgb))
    return points_world, sampled_rgb


def _get_merged_rgbd_point_cloud(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    num_points: int,
    normalize_color: bool = False,
) -> dict[str, torch.Tensor]:
    """Create and cache a merged point cloud from multiple RGB-D cameras for the current env step."""
    cache_name = "_oven_merged_rgbd_point_cloud_cache"
    cache = getattr(env, cache_name, {})
    cache_key = (sensor_names, num_points, normalize_color)
    step_count = env.common_step_counter

    if cache_key in cache and cache[cache_key]["step"] == step_count:
        return cache[cache_key]

    points_per_camera = [num_points // len(sensor_names)] * len(sensor_names)
    for index in range(num_points % len(sensor_names)):
        points_per_camera[index] += 1

    point_positions = []
    point_colors = []
    for sensor_name, sensor_num_points in zip(sensor_names, points_per_camera, strict=True):
        camera: Camera = env.scene[sensor_name]
        positions_w, colors = _sample_rgbd_camera_point_cloud(
            camera=camera,
            num_points=sensor_num_points,
            normalize_color=normalize_color,
        )
        point_positions.append(positions_w - env.scene.env_origins[:, None, :])
        point_colors.append(colors)

    merged_point_cloud = {
        "step": step_count,
        "point_positions": torch.cat(point_positions, dim=1),
        "point_color": torch.cat(point_colors, dim=1),
    }
    cache[cache_key] = merged_point_cloud
    setattr(env, cache_name, cache)
    return merged_point_cloud


def merged_rgbd_point_cloud_positions(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("table_cam", "table_cam_mirror"),
    num_points: int = 2048,
    normalize_color: bool = False,
) -> torch.Tensor:
    """Merged RGB-D point positions in the environment frame."""
    return _get_merged_rgbd_point_cloud(env, sensor_names, num_points, normalize_color)["point_positions"]


def merged_rgbd_point_cloud_color(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("table_cam", "table_cam_mirror"),
    num_points: int = 2048,
    normalize_color: bool = False,
) -> torch.Tensor:
    """Merged RGB-D point colors aligned with :func:`merged_rgbd_point_cloud_positions`."""
    return _get_merged_rgbd_point_cloud(env, sensor_names, num_points, normalize_color)["point_color"]


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


# TODO: Define plate_in_oven function for oven task
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

# TODO: Define oven_door_closed function for oven task
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
