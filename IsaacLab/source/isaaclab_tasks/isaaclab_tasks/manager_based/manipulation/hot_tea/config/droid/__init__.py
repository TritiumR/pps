# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import hot_tea_droid_env_cfg, hot_tea_droid_graybody_thermal_env_cfg, hot_tea_droid_thermal_env_cfg


gym.register(
    id="Isaac-Hot-Tea-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-Thermal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_thermal_env_cfg.DroidHotTeaJointPosThermalEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-Thermal-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_thermal_env_cfg.DroidHotTeaIkRelThermalEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-Thermal-Graybody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_graybody_thermal_env_cfg.DroidHotTeaJointPosGraybodyThermalEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-Thermal-Graybody-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_graybody_thermal_env_cfg.DroidHotTeaIkRelGraybodyThermalEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-PointCloud-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaJointPosPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-PointCloud-Masked-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": hot_tea_droid_env_cfg.DroidHotTeaIkRelPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)
