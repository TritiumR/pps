# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    knife_ik_rel_pointcloud_env_cfg,
    knife_ik_rel_pointcloud_masked_env_cfg,
    knife_ik_rel_visuomotor_env_cfg,
    knife_joint_pos_pointcloud_masked_env_cfg,
    knife_joint_pos_pointcloud_env_cfg,
    knife_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Knife-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_joint_pos_visuomotor_env_cfg.DroidKnifeJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Knife-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_ik_rel_visuomotor_env_cfg.DroidKnifeIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Knife-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_joint_pos_pointcloud_env_cfg.DroidKnifeJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Knife-Droid-PointCloud-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_joint_pos_pointcloud_masked_env_cfg.DroidKnifeJointPosPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Knife-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_ik_rel_pointcloud_env_cfg.DroidKnifeIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Knife-Droid-PointCloud-IK-Rel-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": knife_ik_rel_pointcloud_masked_env_cfg.DroidKnifeIkRelPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)
