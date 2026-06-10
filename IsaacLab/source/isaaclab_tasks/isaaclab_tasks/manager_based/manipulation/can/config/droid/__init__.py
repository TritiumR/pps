# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    can_ik_rel_pointcloud_env_cfg,
    can_ik_rel_visuomotor_env_cfg,
    can_joint_pos_pointcloud_env_cfg,
    can_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Can-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": can_joint_pos_visuomotor_env_cfg.DroidCanJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Can-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": can_ik_rel_visuomotor_env_cfg.DroidCanIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Can-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": can_joint_pos_pointcloud_env_cfg.DroidCanJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Can-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": can_ik_rel_pointcloud_env_cfg.DroidCanIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
