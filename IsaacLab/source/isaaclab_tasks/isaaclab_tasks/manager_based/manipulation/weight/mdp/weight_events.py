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


def apply_scale_from_spawn_cfg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("can"),
):
    """Apply the configured spawn scale to existing USD prims.

    For assets that already exist in a referenced scene (e.g. kitchen child prims),
    ``UsdFileCfg.scale`` is not applied by the file spawner because no new prim is created.
    This utility applies the configured scale as a multiplier at prestartup time.
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
            current_scale = Gf.Vec3d(1.0, 1.0, 1.0)
        else:
            current_scale = scale_op.Get()
            if current_scale is None:
                current_scale = Gf.Vec3d(1.0, 1.0, 1.0)

        scale_op.Set(
            Gf.Vec3d(
                current_scale[0] * sx,
                current_scale[1] * sy,
                current_scale[2] * sz,
            )
        )


def apply_mass_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mass: float | None = None,
    density: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("can"),
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


def _resolve_first_joint_id(
    asset: Articulation, joint_name_candidates: tuple[str, ...]
) -> tuple[int | None, str | None]:
    """Resolve the first matching joint id from a list of candidate names."""
    for joint_name in joint_name_candidates:
        joint_ids, _ = asset.find_joints([joint_name])
        if len(joint_ids) == 1:
            return int(joint_ids[0]), joint_name
    return None, None


def sync_oven_button_to_door(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    oven_cfg: SceneEntityCfg = SceneEntityCfg("oven"),
    button_joint_name_candidates: tuple[str, ...] = (
        "PrismaticJoint_oven_button",
        # "PrismaticJoint_oven_down",
    ),
    door_joint_name_candidates: tuple[str, ...] = (
        "RevoluteJoint_oven_door",
        "RevoluteJoint_oven_up",
    ),
    button_press_threshold: float = -1.0e-3,
    door_intermediate_threshold: float = -0.5235987756,  # -30 deg
    door_open_intermediate_target: float = -0.8726646259,  # -50 deg
    door_open_target: float = -1.3613568166,  # -78 deg
    door_close_latch_threshold: float = -0.3490658504,  # -20 deg
):
    """Drive the oven door from button presses.

    This mirrors the scripted oven behavior in the interactive kitchen assets and is used as a
    runtime fallback for kitchen USD variants where embedded scripting may be missing or mismatched.
    """
    oven: Articulation = env.scene[oven_cfg.name]

    cache = getattr(env, "_oven_button_to_door_joint_cache", None)
    if cache is None:
        button_joint_id, button_joint_name = _resolve_first_joint_id(
            oven, button_joint_name_candidates
        )
        door_joint_id, door_joint_name = _resolve_first_joint_id(
            oven, door_joint_name_candidates
        )
        if button_joint_id is None or door_joint_id is None:
            if not getattr(env, "_oven_button_to_door_warned", False):
                print(
                    "[WARN] Failed to resolve oven button/door joints. "
                    f"button candidates={button_joint_name_candidates}, "
                    f"door candidates={door_joint_name_candidates}"
                )
                env._oven_button_to_door_warned = True
            return

        cache = {
            "button_joint_id": button_joint_id,
            "door_joint_id": door_joint_id,
            "button_joint_name": button_joint_name,
            "door_joint_name": door_joint_name,
        }
        env._oven_button_to_door_joint_cache = cache

    button_joint_id = cache["button_joint_id"]
    door_joint_id = cache["door_joint_id"]

    if not hasattr(env, "_oven_door_locked"):
        env._oven_door_locked = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    if env_ids is None:
        env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[
            env_ids
        ]
    elif isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)
    else:
        env_ids_t = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if env_ids_t.numel() == 0:
        return

    button_pos = oven.data.joint_pos[env_ids_t, button_joint_id]
    door_pos = oven.data.joint_pos[env_ids_t, door_joint_id]
    door_locked = env._oven_door_locked[env_ids_t]

    # Pressed button unlocks and opens door.
    should_open = (button_pos < button_press_threshold) & door_locked
    if should_open.any():
        open_env_ids = env_ids_t[should_open]
        open_target = torch.full(
            (len(open_env_ids), 1),
            door_open_intermediate_target,
            dtype=oven.data.joint_pos.dtype,
            device=env.device,
        )
        oven.set_joint_position_target(
            open_target, joint_ids=[door_joint_id], env_ids=open_env_ids
        )

    # Latch open once past a partial-open threshold.
    newly_unlocked = (door_pos < door_intermediate_threshold) & door_locked
    if newly_unlocked.any():
        door_locked = door_locked.clone()
        door_locked[newly_unlocked] = False

    # Continue opening to the fully open target once near intermediate target.
    near_intermediate = (
        torch.abs(door_pos - door_open_intermediate_target) < math.radians(0.5)
    )
    if near_intermediate.any():
        full_open_env_ids = env_ids_t[near_intermediate]
        full_open_target = torch.full(
            (len(full_open_env_ids), 1),
            door_open_target,
            dtype=oven.data.joint_pos.dtype,
            device=env.device,
        )
        oven.set_joint_position_target(
            full_open_target, joint_ids=[door_joint_id], env_ids=full_open_env_ids
        )

    # Re-lock once manually closed near the shut angle.
    should_relock = (door_pos > door_close_latch_threshold) & (~door_locked)
    if should_relock.any():
        relock_env_ids = env_ids_t[should_relock]
        closed_target = torch.zeros(
            (len(relock_env_ids), 1),
            dtype=oven.data.joint_pos.dtype,
            device=env.device,
        )
        oven.set_joint_position_target(
            closed_target, joint_ids=[door_joint_id], env_ids=relock_env_ids
        )
        door_locked = door_locked.clone()
        door_locked[should_relock] = True

    env._oven_door_locked[env_ids_t] = door_locked


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


def randomize_grouped_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    anchor_pose_range: dict[str, tuple[float, float]],
    anchor_asset_cfg: SceneEntityCfg,
    attached_asset_cfgs: list[SceneEntityCfg],
    attached_pos_offsets: list[tuple[float, float, float]],
    attached_quat_offsets: list[tuple[float, float, float, float]],
):
    """Randomize an anchor rigid object and place attached objects at fixed relative transforms."""
    if env_ids is None:
        return

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if env_ids.numel() == 0:
        return

    if len(attached_asset_cfgs) != len(attached_pos_offsets) or len(attached_asset_cfgs) != len(attached_quat_offsets):
        raise ValueError("Attached asset configs, position offsets, and quaternion offsets must have the same length.")

    anchor_asset = env.scene[anchor_asset_cfg.name]

    pose_list = sample_object_poses(
        num_objects=len(env_ids),
        pose_range=anchor_pose_range,
    )
    pose_tensor = torch.tensor(pose_list, device=env.device, dtype=torch.float32)

    anchor_positions = pose_tensor[:, 0:3] + env.scene.env_origins[env_ids, 0:3]
    anchor_orientations = math_utils.quat_from_euler_xyz(
        pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
    )

    anchor_asset.write_root_pose_to_sim(
        torch.cat([anchor_positions, anchor_orientations], dim=-1),
        env_ids=env_ids,
    )
    anchor_asset.write_root_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device, dtype=anchor_positions.dtype),
        env_ids=env_ids,
    )

    for asset_cfg, pos_offset, quat_offset in zip(
        attached_asset_cfgs, attached_pos_offsets, attached_quat_offsets, strict=True
    ):
        asset = env.scene[asset_cfg.name]
        pos_offset_t = torch.tensor(
            pos_offset, device=env.device, dtype=anchor_positions.dtype
        ).unsqueeze(0).repeat(len(env_ids), 1)
        quat_offset_t = torch.tensor(
            quat_offset, device=env.device, dtype=anchor_orientations.dtype
        ).unsqueeze(0).repeat(len(env_ids), 1)

        positions, orientations = math_utils.combine_frame_transforms(
            anchor_positions,
            anchor_orientations,
            pos_offset_t,
            quat_offset_t,
        )
        asset.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=env_ids,
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=env.device, dtype=anchor_positions.dtype),
            env_ids=env_ids,
        )


def deactivate_prim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
):
    """Deactivate matching prims on stage."""
    del env, env_ids

    stage = get_current_stage()
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.SetActive(False)


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
