# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    weight_ik_rel_pointcloud_env_cfg,
    weight_ik_rel_pointcloud_masked_env_cfg,
    weight_ik_rel_visuomotor_env_cfg,
    weight_joint_pos_pointcloud_masked_env_cfg,
    weight_joint_pos_pointcloud_env_cfg,
    weight_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Weight-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_joint_pos_visuomotor_env_cfg.DroidWeightJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Weight-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_ik_rel_visuomotor_env_cfg.DroidWeightIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Weight-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_joint_pos_pointcloud_env_cfg.DroidWeightJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Weight-Droid-PointCloud-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_joint_pos_pointcloud_masked_env_cfg.DroidWeightJointPosPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Weight-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_ik_rel_pointcloud_env_cfg.DroidWeightIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Weight-Droid-PointCloud-IK-Rel-Masked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": weight_ik_rel_pointcloud_masked_env_cfg.DroidWeightIkRelPointCloudMaskedEnvCfg,
    },
    disable_env_checker=True,
)
