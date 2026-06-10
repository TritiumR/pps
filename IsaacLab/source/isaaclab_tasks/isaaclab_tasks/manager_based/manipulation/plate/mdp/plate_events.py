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
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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


def set_rigid_body_dynamic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Enable dynamics and gravity on an existing rigid body prim in the scene."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sim_utils.modify_rigid_body_properties(
        asset.cfg.prim_path,
        sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
        ),
    )


def apply_scale_from_spawn_cfg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Apply the configured spawn scale to existing USD prims.

    For assets that already exist in a referenced scene (e.g. kitchen child prims),
    ``UsdFileCfg.scale`` is not applied by the file spawner because no new prim is created.
    This utility enforces the configured scale at prestartup time.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if asset.cfg.spawn is None or asset.cfg.spawn.scale is None:
        return

    sx, sy, sz = asset.cfg.spawn.scale
    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xformable = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

        scale_op.Set(Gf.Vec3d(sx, sy, sz))


def apply_mass_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mass: float | None = None,
    density: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Apply mass properties to rigid bodies under an existing asset prim path.

    This is intended for assets that already exist inside a referenced scene where
    ``spawn.mass_props`` cannot be reliably applied during file spawning.
    """
    if mass is None and density is None:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(asset.cfg.prim_path)
    if not root_prim.IsValid():
        return

    prim_stack = [root_prim]
    while prim_stack:
        prim = prim_stack.pop()
        if prim.IsInstance():
            continue

        if UsdPhysics.RigidBodyAPI(prim):
            mass_api = UsdPhysics.MassAPI(prim)
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            if mass is not None:
                mass_api.CreateMassAttr().Set(float(mass))
            if density is not None:
                mass_api.CreateDensityAttr().Set(float(density))

        prim_stack.extend(prim.GetChildren())


def set_plate_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Set the plate to its default pose as defined in the USD file.

    Since the plate is loaded from Interactive_kitchen.usd, we use the
    default_root_state which contains the pose from the USD file.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Use the default root state from the USD file
    default_root_state = asset.data.default_root_state.clone()

    # Write to simulation
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)


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
    default_intensity: float = 3000.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(default_intensity)

    color_attr = light_prim.GetAttribute("inputs:color")
    color_attr.Set(default_color)

    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
    texture_file_attr.Set(default_texture)

    if not hasattr(env.cfg, "eval_mode") or not env.cfg.eval_mode:
        return

    if env.cfg.eval_type in ["light_intensity", "all"]:
        # Sample new light intensity
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        # Set light intensity to light prim
        intensity_attr.Set(new_intensity)

    if env.cfg.eval_type in ["light_color", "all"]:
        # Sample new light color
        new_color = sample_random_color(base=default_color, variation=color_variation)
        # Set light color to light prim
        color_attr.Set(new_color)

    if env.cfg.eval_type in ["light_texture", "all"]:
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
