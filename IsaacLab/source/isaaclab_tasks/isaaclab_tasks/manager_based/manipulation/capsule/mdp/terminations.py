# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination and subtask terms for the capsule coffee machine task."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def coffee_lid_opened(
    env: ManagerBasedRLEnv,
    capsule_cfg: SceneEntityCfg = SceneEntityCfg("capsule"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lid_joint_name: str = "RevoluteJoint_capsule_coffee_maker_3_up",
    open_threshold: float = -0.5,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if the coffee maker lid has been opened and the gripper is open."""
    capsule: Articulation = env.scene[capsule_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    lid_joint_pos = _get_joint_position(capsule, lid_joint_name)
    lid_opened = lid_joint_pos <= open_threshold
    return torch.logical_and(lid_opened, _gripper_is_open(env, robot, atol=atol, rtol=rtol))


def pod_in_coffee_maker(
    env: ManagerBasedRLEnv,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    capsule_cfg: SceneEntityCfg = SceneEntityCfg("capsule"),
    target_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.27),
    xy_threshold: float = 0.10,
    min_local_z: float = 0.25,
    max_local_z: float = 0.33,
) -> torch.Tensor:
    """Check if the pod can is in the coffee maker's capsule bay."""
    can: RigidObject = env.scene[can_cfg.name]
    capsule: Articulation = env.scene[capsule_cfg.name]

    can_pos_capsule = math_utils.quat_apply_inverse(
        capsule.data.root_quat_w,
        can.data.root_pos_w - capsule.data.root_pos_w,
    )
    target_xy = torch.tensor(target_local_pos[:2], dtype=can_pos_capsule.dtype, device=env.device)
    # print(f"can_pos_capsule: {can_pos_capsule}, target_xy: {target_xy}")
    xy_dist = torch.linalg.vector_norm(can_pos_capsule[:, :2] - target_xy.unsqueeze(0), dim=1)
    z_in_range = torch.logical_and(can_pos_capsule[:, 2] >= min_local_z, can_pos_capsule[:, 2] <= max_local_z)
    return torch.logical_and(xy_dist <= xy_threshold, z_in_range)


def task_done_capsule(
    env: ManagerBasedRLEnv,
    capsule_cfg: SceneEntityCfg = SceneEntityCfg("capsule"),
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.27),
    xy_threshold: float = 0.10,
    min_local_z: float = 0.25,
    max_local_z: float = 0.33,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when the pod is in the coffee maker and the gripper is open."""
    robot: Articulation = env.scene[robot_cfg.name]
    pod_placed = pod_in_coffee_maker(
        env,
        can_cfg=can_cfg,
        capsule_cfg=capsule_cfg,
        target_local_pos=target_local_pos,
        xy_threshold=xy_threshold,
        min_local_z=min_local_z,
        max_local_z=max_local_z,
    )
    return torch.logical_and(pod_placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))
