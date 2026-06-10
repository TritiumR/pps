# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import tea_wood_droid_env_cfg


gym.register(
    id="Isaac-Tea-Wood-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-PointCloud-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodJointPosPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-PointCloud-Masked-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_wood_droid_env_cfg.DroidTeaWoodIkRelPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)

