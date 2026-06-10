# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    tea_ik_rel_pointcloud_masked_env_cfg,
    tea_ik_rel_pointcloud_env_cfg,
    tea_ik_rel_visuomotor_env_cfg,
    tea_joint_pos_pointcloud_masked_env_cfg,
    tea_joint_pos_pointcloud_env_cfg,
    tea_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Tea-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_joint_pos_visuomotor_env_cfg.DroidTeaJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_ik_rel_visuomotor_env_cfg.DroidTeaIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_joint_pos_pointcloud_env_cfg.DroidTeaJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_ik_rel_pointcloud_env_cfg.DroidTeaIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Droid-PointCloud-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_joint_pos_pointcloud_masked_env_cfg.DroidTeaJointPosPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Droid-PointCloud-Masked-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tea_ik_rel_pointcloud_masked_env_cfg.DroidTeaIkRelPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)
