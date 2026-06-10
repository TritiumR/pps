# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the angry bird task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pig_hit(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pig_cfg: SceneEntityCfg = SceneEntityCfg("pig"),
    bird_cfg: SceneEntityCfg = SceneEntityCfg("bird"),
    vase_cfg: SceneEntityCfg = SceneEntityCfg("vase"),
    atol=0.0001,
    rtol=0.0001,
    height_threshold=-0.1,
):
    robot: Articulation = env.scene[robot_cfg.name]
    pig: RigidObject = env.scene[pig_cfg.name]
    bird: RigidObject = env.scene[bird_cfg.name]
    vase: RigidObject = env.scene[vase_cfg.name]

    # check if the pig is knocked over (changed position)
    h_pig = pig.data.root_pos_w[:, 2]
    hit = h_pig < height_threshold

    # check if the vase is knocked over
    h_vase = vase.data.root_pos_w[:, 2]
    hit = torch.logical_and(hit, h_vase < height_threshold)

    # check if the bird changed position
    h_bird = bird.data.root_pos_w[:, 2]
    hit = torch.logical_and(hit, h_bird < height_threshold)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(
            -1, 1
        )  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        hit = torch.logical_and(suction_cup_is_open, hit)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert (
                len(gripper_joint_ids) == 2
            ), "Terminations only support parallel gripper for now"

            hit = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    ),
                    atol=atol,
                    rtol=rtol,
                ),
                hit,
            )
            hit = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    ),
                    atol=atol,
                    rtol=rtol,
                ),
                hit,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return hit
