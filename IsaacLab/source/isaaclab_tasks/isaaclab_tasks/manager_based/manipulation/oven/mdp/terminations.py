# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the oven task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
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


def _oven_slot_position_w(
    env: ManagerBasedRLEnv,
    oven: Articulation,
    slot_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Compute oven slot target position in world frame from oven root pose."""
    slot_offset_b = torch.tensor(
        slot_offset,
        device=env.device,
        dtype=oven.data.root_pos_w.dtype,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    return oven.data.root_pos_w + math_utils.quat_apply(oven.data.root_quat_w, slot_offset_b)


def oven_opened(
    env: ManagerBasedRLEnv,
    oven_cfg: SceneEntityCfg = SceneEntityCfg("oven"),
    door_joint_name: str = "RevoluteJoint_oven_door",
    door_open_threshold: float = -0.1,
) -> torch.Tensor:
    """Subtask 1: oven door has been opened."""
    oven: Articulation = env.scene[oven_cfg.name]
    door_joint_pos = _get_joint_position(oven, door_joint_name)
    # print(f"door_joint_pos: {door_joint_pos}")
    return door_joint_pos <= door_open_threshold


def can_in_oven(
    env: ManagerBasedRLEnv,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    oven_cfg: SceneEntityCfg = SceneEntityCfg("oven"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_name: str = "RevoluteJoint_oven_door",
    slot_offset: tuple[float, float, float] = (0.06, 0.09, -0.09),
    can_in_oven_threshold: float = 0.12,
    door_open_threshold: float = 0.25,
    require_oven_open: bool = True,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Subtask 2: can is inside oven slot and released by gripper."""
    can: RigidObject = env.scene[can_cfg.name]
    oven: Articulation = env.scene[oven_cfg.name]

    # slot_pos_w = _oven_slot_position_w(env, oven, slot_offset)
    can_pos_w = can.data.root_pos_w
    oven_pos_w = oven.data.root_pos_w
    # print(f"can_pos_w: {can_pos_w}, oven_pos_w: {oven_pos_w}")
    can_oven_offset = can_pos_w - oven_pos_w
    # print(f"can_oven_offset: {can_oven_offset}")
    can_to_slot_dist = torch.linalg.vector_norm(can_pos_w - torch.tensor(slot_offset, device=env.device) - oven_pos_w, dim=1)
    # in_oven = can_to_slot_dist <= can_in_oven_threshold
    # print(f"can_to_slot_dist: {can_to_slot_dist}")
    in_oven = can_to_slot_dist <= can_in_oven_threshold

    if require_oven_open:
        door_is_open = oven_opened(
            env,
            oven_cfg=oven_cfg,
            door_joint_name=door_joint_name,
            door_open_threshold=door_open_threshold,
        )
        in_oven = torch.logical_and(in_oven, door_is_open)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        in_oven = torch.logical_and(in_oven, gripper_open)

    return in_oven


def oven_closed(
    env: ManagerBasedRLEnv,
    oven_cfg: SceneEntityCfg = SceneEntityCfg("oven"),
    door_joint_name: str = "RevoluteJoint_oven_door",
    door_closed_threshold: float = -0.01,
) -> torch.Tensor:
    """Subtask 3: oven door is closed."""
    oven: Articulation = env.scene[oven_cfg.name]
    door_joint_pos = _get_joint_position(oven, door_joint_name)
    # print(f"door_joint_pos: {door_joint_pos}")
    return door_joint_pos >= door_closed_threshold


def task_done_oven(
    env: ManagerBasedRLEnv,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    oven_cfg: SceneEntityCfg = SceneEntityCfg("oven"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_name: str = "RevoluteJoint_oven_door",
    slot_offset: tuple[float, float, float] = (0.06, 0.09, -0.09),
    can_in_oven_threshold: float = 0.12,
    door_open_threshold: float = -0.6,
    door_closed_threshold: float = -0.01,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when can is in oven and oven door is closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    can_placed = can_in_oven(
        env,
        can_cfg=can_cfg,
        oven_cfg=oven_cfg,
        robot_cfg=robot_cfg,
        door_joint_name=door_joint_name,
        slot_offset=slot_offset,
        can_in_oven_threshold=can_in_oven_threshold,
        door_open_threshold=door_open_threshold,
        require_oven_open=False,
        require_gripper_open=False,
        atol=atol,
        rtol=rtol,
    )

    oven_open = oven_opened(
        env,
        oven_cfg=oven_cfg,
        door_joint_name=door_joint_name,
        door_open_threshold=door_open_threshold,
    )

    gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)

    return torch.logical_and(can_placed, torch.logical_and(oven_open, gripper_open))
    # door_is_closed = oven_closed(
    #     env,
    #     oven_cfg=oven_cfg,
    #     door_joint_name=door_joint_name,
    #     door_closed_threshold=door_closed_threshold,
    # )
    # return can_placed
