# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import os
import random
import json
import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../../diffusion_policy/reconstruction/asset/water_world",
)


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(
        env.num_envs, 1
    )


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(
        mean, std, joint_pos.shape, joint_pos.device
    )
    # print(joint_pos)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    # print(joint_pos_limits)
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_random_color(base=(0.75, 0.75, 0.75), variation=0.1):
    """
    Generates a randomized color that stays close to the base color while preserving overall brightness.
    The relative balance between the R, G, and B components is maintained by ensuring that
    the sum of random offsets is zero.

    Parameters:
        base (tuple): The base RGB color with each component between 0 and 1.
        variation (float): Maximum deviation to sample for each channel before balancing.

    Returns:
        tuple: A new RGB color with balanced random variation.
    """
    # Generate random offsets for each channel in the range [-variation, variation]
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    # Compute the average offset
    avg_offset = sum(offsets) / 3
    # Adjust offsets so their sum is zero (maintaining brightness)
    balanced_offsets = [offset - avg_offset for offset in offsets]

    # Apply the balanced offsets to the base color and clamp each channel between 0 and 1
    new_color = tuple(
        max(0, min(1, base_component + offset))
        for base_component, offset in zip(base, balanced_offsets)
    )

    return new_color


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    color_attr = light_prim.GetAttribute("inputs:color")
    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])
    # Set light intensity to light prim
    intensity_attr.Set(new_intensity)

    # Sample new light color
    new_color = sample_random_color(base=default_color, variation=color_variation)
    # Set light color to light prim
    color_attr.Set(new_color)

    # Sample new light texture (background)
    new_texture = random.sample(textures, 1)[0]
    # Set light texture to light prim
    texture_file_attr.Set(new_texture)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [
                math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list
            ]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


def randomize_table_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("table"),
):
    """Randomize the pose of a kinematic rigid body asset (e.g., table).

    This function works with RigidObjectCfg assets that have kinematic_enabled=True.

    Args:
        env: The environment object.
        env_ids: The indices of the environments to randomize.
        pose_range: Dictionary with keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
            specifying (min, max) ranges for pose sampling.
        asset_cfg: The scene entity configuration for the asset. Defaults to "table".
    """
    if env_ids is None:
        return

    # Get the RigidObject asset from the scene
    asset = env.scene[asset_cfg.name]

    # Sample poses for all environments at once
    num_envs = len(env_ids)
    positions = torch.zeros(num_envs, 3, device=env.device)
    euler_angles = torch.zeros(num_envs, 3, device=env.device)

    for i, axis in enumerate(["x", "y", "z"]):
        if axis in pose_range:
            min_val, max_val = pose_range[axis]
            positions[:, i] = (
                torch.rand(num_envs, device=env.device) * (max_val - min_val) + min_val
            )

    for i, axis in enumerate(["roll", "pitch", "yaw"]):
        if axis in pose_range:
            min_val, max_val = pose_range[axis]
            euler_angles[:, i] = (
                torch.rand(num_envs, device=env.device) * (max_val - min_val) + min_val
            )

    # Add environment origins to positions
    positions = positions + env.scene.env_origins[env_ids, :3]

    # Convert euler angles to quaternions
    orientations = math_utils.quat_from_euler_xyz(
        euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    )

    # Write pose to simulation (works for RigidObject with kinematic_enabled=True)
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )


def randomize_table_and_objects_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_pose_range: dict[str, tuple[float, float]],
    object_pose_range: dict[str, tuple[float, float]],
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    object_cfgs: list[SceneEntityCfg] = [],
    min_separation: float = 0.0,
    max_sample_tries: int = 5000,
):
    """Randomize table pose and object poses together, ensuring objects share the table's z value.

    This function first samples the table pose (including z), then places objects on the table
    with the same z value as the sampled table position.

    Args:
        env: The environment object.
        env_ids: The indices of the environments to randomize.
        table_pose_range: Dictionary with keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
            specifying (min, max) ranges for table pose sampling.
        object_pose_range: Dictionary with keys 'x', 'y', 'yaw' specifying (min, max)
            ranges for object pose sampling. The 'z' value will be inherited from the table.
        table_cfg: The scene entity configuration for the table.
        object_cfgs: List of scene entity configurations for objects to place on the table.
        min_separation: Minimum separation distance between objects.
        max_sample_tries: Maximum attempts to sample valid object positions.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        # === Sample table pose ===
        table_pose = [
            random.uniform(
                table_pose_range.get(key, (0.0, 0.0))[0],
                table_pose_range.get(key, (0.0, 0.0))[1],
            )
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]

        # Get the sampled table z value (this will be used for all objects)
        table_z = table_pose[2]

        # Set table pose
        table_asset = env.scene[table_cfg.name]
        table_pos = torch.tensor([table_pose[:3]], device=env.device)
        table_pos = table_pos + env.scene.env_origins[cur_env, :3]
        table_orient = math_utils.quat_from_euler_xyz(
            torch.tensor([table_pose[3]], device=env.device),
            torch.tensor([table_pose[4]], device=env.device),
            torch.tensor([table_pose[5]], device=env.device),
        )
        table_asset.write_root_pose_to_sim(
            torch.cat([table_pos, table_orient], dim=-1),
            env_ids=torch.tensor([cur_env], device=env.device),
        )

        # === Sample object poses with same z as table ===
        # Override object z range to use the sampled table z
        object_pose_range_with_table_z = object_pose_range.copy()
        object_pose_range_with_table_z["z"] = (table_z, table_z)

        pose_list = sample_object_poses(
            num_objects=len(object_cfgs),
            min_separation=min_separation,
            pose_range=object_pose_range_with_table_z,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(object_cfgs)):
            asset_cfg = object_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


def randomize_rigid_objects_in_focus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    out_focus_state: torch.Tensor,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])
    env.rigid_objects_in_focus = []

    for cur_env in env_ids.tolist():
        # Sample in focus object poses
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        selected_ids = []
        for asset_idx in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[asset_idx]
            asset = env.scene[asset_cfg.name]

            # Randomly select an object to bring into focus
            object_id = random.randint(0, asset.num_objects - 1)
            selected_ids.append(object_id)

            # Create object state tensor
            object_states = torch.stack([out_focus_state] * asset.num_objects).to(
                device=env.device
            )
            pose_tensor = torch.tensor([pose_list[asset_idx]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            asset.write_object_state_to_sim(
                object_state=object_states,
                env_ids=torch.tensor([cur_env], device=env.device),
            )

        env.rigid_objects_in_focus.append(selected_ids)


def randomize_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    rotation_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("table_cam"),
):
    """Randomize camera offset by applying small perturbations to position and rotation.

    This event adds random offsets to the camera's world pose. The offsets are sampled
    uniformly from the specified ranges for each axis.

    Args:
        env: The environment object.
        env_ids: The indices of the environments to randomize.
        position_range: Dictionary with keys 'x', 'y', 'z' specifying (min, max) offset ranges in meters.
        rotation_range: Optional dictionary with keys 'roll', 'pitch', 'yaw' specifying (min, max)
            offset ranges in radians. Defaults to None (no rotation randomization).
        asset_cfg: The scene entity configuration for the camera. Defaults to "table_cam".
    """
    if env_ids is None:
        return

    # Get camera sensor from scene
    camera = env.scene[asset_cfg.name]

    # Get current camera world poses
    current_pos = camera.data.pos_w.clone()
    current_quat = camera.data.quat_w_world.clone()

    # Sample position offsets
    pos_offset = torch.zeros_like(current_pos)
    for i, axis in enumerate(["x", "y", "z"]):
        if axis in position_range:
            min_val, max_val = position_range[axis]
            pos_offset[env_ids, i] = (
                torch.rand(len(env_ids), device=env.device) * (max_val - min_val)
                + min_val
            )

    # Apply position offset
    new_pos = current_pos.clone()
    new_pos[env_ids] = current_pos[env_ids] + pos_offset[env_ids]

    # Sample rotation offsets if specified
    new_quat = current_quat.clone()
    if rotation_range is not None:
        roll_offset = torch.zeros(len(env_ids), device=env.device)
        pitch_offset = torch.zeros(len(env_ids), device=env.device)
        yaw_offset = torch.zeros(len(env_ids), device=env.device)

        if "roll" in rotation_range:
            min_val, max_val = rotation_range["roll"]
            roll_offset = (
                torch.rand(len(env_ids), device=env.device) * (max_val - min_val)
                + min_val
            )
        if "pitch" in rotation_range:
            min_val, max_val = rotation_range["pitch"]
            pitch_offset = (
                torch.rand(len(env_ids), device=env.device) * (max_val - min_val)
                + min_val
            )
        if "yaw" in rotation_range:
            min_val, max_val = rotation_range["yaw"]
            yaw_offset = (
                torch.rand(len(env_ids), device=env.device) * (max_val - min_val)
                + min_val
            )

        # Convert euler offsets to quaternion
        offset_quat = math_utils.quat_from_euler_xyz(
            roll_offset, pitch_offset, yaw_offset
        )

        # Apply rotation offset (multiply quaternions)
        new_quat[env_ids] = math_utils.quat_mul(current_quat[env_ids], offset_quat)

    # Set new camera poses
    camera.set_world_poses(
        positions=new_pos[env_ids],
        orientations=new_quat[env_ids],
        env_ids=env_ids.tolist(),
        convention="world",
    )
