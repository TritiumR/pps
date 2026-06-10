# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import (
    block_joint_pos_visuomotor_env_cfg,
    block_ik_rel_visuomotor_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control with Visual Observations
##

gym.register(
    id="Isaac-Block-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": block_joint_pos_visuomotor_env_cfg.DroidBlockJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


# #
# Inverse Kinematics - Relative Pose Control
# #

gym.register(
    id="Isaac-Block-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": block_ik_rel_visuomotor_env_cfg.DroidBlockIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)
