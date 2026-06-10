# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from isaaclab_tasks.manager_based.manipulation.plate.mdp.terminations import root_height_below_minimum  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_MUG_HANDLE_PROBE_OFFSETS = (
    # Scaled mug-local target near the center of the handle opening.
    (0.060, 0.080, 0.043),
)

_DEFAULT_HOLDER_TOP_STICK_SEGMENTS = (
    # Scaled holder-local centerline for the single valid highest peg.
    ((-0.0116, 0.0, 0.3283), (-0.1053, 0.0, 0.3665)),
)


def _local_offsets_to_world(asset: RigidObject, local_offsets: tuple[tuple[float, float, float], ...]) -> torch.Tensor:
    """Transform local asset offsets into world-frame points."""
    offsets = torch.tensor(local_offsets, dtype=asset.data.root_pos_w.dtype, device=asset.data.root_pos_w.device)
    num_envs = asset.data.root_pos_w.shape[0]
    num_offsets = offsets.shape[0]

    offsets = offsets.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_offsets, 3)
    quats = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_offsets, -1).reshape(num_envs * num_offsets, 4)
    offsets_w = math_utils.quat_apply(quats, offsets).reshape(num_envs, num_offsets, 3)
    return asset.data.root_pos_w.unsqueeze(1) + offsets_w


def _local_segments_to_world(
    asset: RigidObject,
    local_segments: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform local asset line segments into world-frame start and end points."""
    segment_starts = tuple(segment[0] for segment in local_segments)
    segment_ends = tuple(segment[1] for segment in local_segments)
    segment_starts_w = _local_offsets_to_world(asset, segment_starts)
    segment_ends_w = _local_offsets_to_world(asset, segment_ends)
    return segment_starts_w, segment_ends_w


def _min_point_to_segment_distance(
    points_w: torch.Tensor,
    segment_starts_w: torch.Tensor,
    segment_ends_w: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum distance from any point to any segment for each environment."""
    points = points_w[:, :, None, :]
    starts = segment_starts_w[:, None, :, :]
    segment_vecs = (segment_ends_w - segment_starts_w)[:, None, :, :]

    point_vecs = points - starts
    segment_len_sq = torch.sum(segment_vecs * segment_vecs, dim=-1).clamp_min(1.0e-8)
    segment_t = torch.sum(point_vecs * segment_vecs, dim=-1) / segment_len_sq
    closest_points = starts + torch.clamp(segment_t, 0.0, 1.0).unsqueeze(-1) * segment_vecs
    distances = torch.linalg.vector_norm(points - closest_points, dim=-1)
    return distances.flatten(start_dim=1).min(dim=1).values


def _mug_handle_to_holder_stick_distance(
    mug: RigidObject,
    holder: RigidObject,
    mug_handle_probe_offsets: tuple[tuple[float, float, float], ...],
    holder_stick_segments: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> torch.Tensor:
    """Compute the closest distance between the mug handle probe and any holder stick segment."""
    handle_points_w = _local_offsets_to_world(mug, mug_handle_probe_offsets)
    stick_starts_w, stick_ends_w = _local_segments_to_world(holder, holder_stick_segments)
    return _min_point_to_segment_distance(handle_points_w, stick_starts_w, stick_ends_w)


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


def task_done_holder(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    holder_xy_threshold: float = 0.15,
    desired_z: float = 0.40,
    z_threshold: float = 0.05,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Cup is placed on the holder and released."""
    mug: RigidObject = env.scene[mug_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]

    pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(mug.data.root_pos_w[:, 2] - desired_z)
    on_holder = torch.logical_and(xy_dist <= holder_xy_threshold, z_dist <= z_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        on_holder = torch.logical_and(on_holder, gripper_open)

    return on_holder


def task_done_holder_stable(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    holder_xy_threshold: float | None = None,
    min_mug_height: float = 0.36,
    stable_duration_s: float = 1.0,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Return true after the mug is high enough and released for a consecutive duration."""
    counter_name = "_task_done_holder_stable_success_steps"
    last_step_name = "_task_done_holder_stable_last_step"
    last_result_name = "_task_done_holder_stable_last_result"

    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        setattr(env, last_step_name, -1)
        setattr(env, last_result_name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    current_step = getattr(env, "common_step_counter", -1)
    if getattr(env, last_step_name) == current_step:
        return getattr(env, last_result_name).clone()

    mug: RigidObject = env.scene[mug_cfg.name]
    hanging = mug.data.root_pos_w[:, 2] >= min_mug_height

    if holder_xy_threshold is not None:
        holder: RigidObject = env.scene[holder_cfg.name]
        pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
        xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
        hanging = torch.logical_and(hanging, xy_dist <= holder_xy_threshold)

    if max_linear_velocity is not None:
        lin_vel = torch.linalg.vector_norm(mug.data.root_lin_vel_w, dim=1)
        hanging = torch.logical_and(hanging, lin_vel <= max_linear_velocity)
    if max_angular_velocity is not None:
        ang_vel = torch.linalg.vector_norm(mug.data.root_ang_vel_w, dim=1)
        hanging = torch.logical_and(hanging, ang_vel <= max_angular_velocity)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        hanging = torch.logical_and(hanging, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    success_steps = getattr(env, counter_name)
    episode_start = env.episode_length_buf <= 1
    success_steps = torch.where(episode_start, torch.zeros_like(success_steps), success_steps)
    success_steps = torch.where(hanging, success_steps + 1, torch.zeros_like(success_steps))
    setattr(env, counter_name, success_steps)

    required_steps = max(1, math.ceil(stable_duration_s / env.step_dt))
    success = success_steps >= required_steps
    setattr(env, last_step_name, current_step)
    setattr(env, last_result_name, success.clone())
    return success


def task_done_holder_released(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    min_mug_height: float = 0.42,
    holder_xy_threshold: float | None = 0.18,
    max_handle_stick_distance: float | None = 0.012,
    mug_handle_probe_offsets: tuple[tuple[float, float, float], ...] = _DEFAULT_MUG_HANDLE_PROBE_OFFSETS,
    holder_stick_segments: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
        _DEFAULT_HOLDER_TOP_STICK_SEGMENTS
    ),
    min_gripper_mug_distance: float = 0.12,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Stateless success check for annotation: mug is hanging and released."""
    mug: RigidObject = env.scene[mug_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]
    success = mug.data.root_pos_w[:, 2] >= min_mug_height

    if holder_xy_threshold is not None:
        pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
        xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
        success = torch.logical_and(success, xy_dist <= holder_xy_threshold)

    if max_handle_stick_distance is not None:
        handle_to_stick_dist = _mug_handle_to_holder_stick_distance(
            mug,
            holder,
            mug_handle_probe_offsets,
            holder_stick_segments,
        )
        success = torch.logical_and(success, handle_to_stick_dist <= max_handle_stick_distance)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        success = torch.logical_and(success, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_to_mug_dist = torch.linalg.vector_norm(ee_pos_w - mug.data.root_pos_w, dim=1)
    success = torch.logical_and(success, ee_to_mug_dist >= min_gripper_mug_distance)

    return success
