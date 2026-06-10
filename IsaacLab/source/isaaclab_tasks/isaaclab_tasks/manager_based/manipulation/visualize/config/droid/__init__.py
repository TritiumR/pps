# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import (
    visualize_joint_pos_visuomotor_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control with Visual Observations
##

gym.register(
    id="Isaac-Visualize-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": visualize_joint_pos_visuomotor_env_cfg.DroidVisualizeJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)
