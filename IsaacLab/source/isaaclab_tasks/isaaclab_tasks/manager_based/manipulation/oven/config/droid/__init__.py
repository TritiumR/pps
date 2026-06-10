# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    oven_ik_rel_pointcloud_env_cfg,
    oven_ik_rel_visuomotor_env_cfg,
    oven_joint_pos_pointcloud_env_cfg,
    oven_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Oven-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": oven_joint_pos_visuomotor_env_cfg.DroidOvenJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Oven-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": oven_ik_rel_visuomotor_env_cfg.DroidOvenIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Oven-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": oven_joint_pos_pointcloud_env_cfg.DroidOvenJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Oven-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": oven_ik_rel_pointcloud_env_cfg.DroidOvenIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
