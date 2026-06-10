# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions used for wash-task subtasks and termination checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if the asset falls below a minimum height.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        minimum_height: Minimum height threshold.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    done = height < minimum_height
    return done


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


def tap_opened(
    env: ManagerBasedRLEnv,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("large_cabinet"),
    tap_left_joint_name: str = "RevoluteJoint_largecabinet_down12",
    tap_right_joint_name: str = "RevoluteJoint_largecabinet_down13",
    tap_left_closed_pos: float = -1.3962634,
    tap_right_closed_pos: float = 0.0,
    open_displacement_threshold: float = 0.20,
    require_both_knobs: bool = False,
) -> torch.Tensor:
    """Subtask 1: faucet/tap has been opened by rotating one or both knobs."""
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    left_joint_pos = _get_joint_position(cabinet, tap_left_joint_name)
    right_joint_pos = _get_joint_position(cabinet, tap_right_joint_name)

    left_opened = torch.abs(left_joint_pos - tap_left_closed_pos) >= open_displacement_threshold
    right_opened = torch.abs(right_joint_pos - tap_right_closed_pos) >= open_displacement_threshold

    if require_both_knobs:
        return torch.logical_and(left_opened, right_opened)
    return torch.logical_or(left_opened, right_opened)


def plate_washed(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("large_cabinet"),
    sink_pos: tuple[float, float, float] = (1.85, 4.90, 0.26),
    wash_distance_threshold: float = 0.22,
    tap_left_joint_name: str = "RevoluteJoint_largecabinet_down12",
    tap_right_joint_name: str = "RevoluteJoint_largecabinet_down13",
    tap_left_closed_pos: float = -1.3962634,
    tap_right_closed_pos: float = 0.0,
    open_displacement_threshold: float = 0.20,
    require_tap_open: bool = True,
) -> torch.Tensor:
    """Subtask 2: plate is brought to sink area while tap is open."""
    plate: RigidObject = env.scene[plate_cfg.name]
    sink_pos_tensor = torch.tensor(sink_pos, dtype=plate.data.root_pos_w.dtype, device=env.device).unsqueeze(0)
    plate_to_sink_dist = torch.linalg.vector_norm(plate.data.root_pos_w - sink_pos_tensor, dim=1)
    washed = plate_to_sink_dist <= wash_distance_threshold

    if require_tap_open:
        is_tap_opened = tap_opened(
            env,
            cabinet_cfg=cabinet_cfg,
            tap_left_joint_name=tap_left_joint_name,
            tap_right_joint_name=tap_right_joint_name,
            tap_left_closed_pos=tap_left_closed_pos,
            tap_right_closed_pos=tap_right_closed_pos,
            open_displacement_threshold=open_displacement_threshold,
            require_both_knobs=False,
        )
        washed = torch.logical_and(washed, is_tap_opened)

    return washed


def plate_on_rack(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rack_xy_threshold: float = 0.15,
    rack_z_threshold: float = 0.12,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Subtask 3: plate is placed on plate rack and released."""
    plate: RigidObject = env.scene[plate_cfg.name]
    rack: RigidObject = env.scene[rack_cfg.name]

    pos_diff = plate.data.root_pos_w - rack.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(pos_diff[:, 2])
    on_rack = torch.logical_and(xy_dist <= rack_xy_threshold, z_dist <= rack_z_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        on_rack = torch.logical_and(on_rack, gripper_open)

    return on_rack


def task_done_wash(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("large_cabinet"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tap_left_joint_name: str = "RevoluteJoint_largecabinet_down12",
    tap_right_joint_name: str = "RevoluteJoint_largecabinet_down13",
    tap_left_closed_pos: float = -1.3962634,
    tap_right_closed_pos: float = 0.0,
    open_displacement_threshold: float = 0.20,
    rack_xy_threshold: float = 0.15,
    rack_z_threshold: float = 0.12,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when tap is opened and plate is placed on rack."""
    is_tap_opened = tap_opened(
        env,
        cabinet_cfg=cabinet_cfg,
        tap_left_joint_name=tap_left_joint_name,
        tap_right_joint_name=tap_right_joint_name,
        tap_left_closed_pos=tap_left_closed_pos,
        tap_right_closed_pos=tap_right_closed_pos,
        open_displacement_threshold=open_displacement_threshold,
        require_both_knobs=False,
    )
    is_plate_on_rack = plate_on_rack(
        env,
        plate_cfg=plate_cfg,
        rack_cfg=rack_cfg,
        robot_cfg=robot_cfg,
        rack_xy_threshold=rack_xy_threshold,
        rack_z_threshold=rack_z_threshold,
        require_gripper_open=True,
        atol=atol,
        rtol=rtol,
    )
    return torch.logical_and(is_tap_opened, is_plate_on_rack)
