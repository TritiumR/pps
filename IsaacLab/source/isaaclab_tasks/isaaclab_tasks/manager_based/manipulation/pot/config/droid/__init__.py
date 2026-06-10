# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    pot_ik_rel_pointcloud_env_cfg,
    pot_ik_rel_visuomotor_env_cfg,
    pot_joint_pos_pointcloud_env_cfg,
    pot_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Pot-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pot_joint_pos_visuomotor_env_cfg.DroidPotJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pot-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pot_ik_rel_visuomotor_env_cfg.DroidPotIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pot-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pot_joint_pos_pointcloud_env_cfg.DroidPotJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pot-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pot_ik_rel_pointcloud_env_cfg.DroidPotIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
