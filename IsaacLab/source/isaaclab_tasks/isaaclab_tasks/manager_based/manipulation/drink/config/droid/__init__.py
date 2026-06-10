# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    drink_ik_rel_pointcloud_env_cfg,
    drink_ik_rel_visuomotor_env_cfg,
    drink_joint_pos_pointcloud_env_cfg,
    drink_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Drink-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drink_joint_pos_visuomotor_env_cfg.DroidDrinkJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Drink-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drink_ik_rel_visuomotor_env_cfg.DroidDrinkIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Drink-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drink_joint_pos_pointcloud_env_cfg.DroidDrinkJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Drink-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drink_ik_rel_pointcloud_env_cfg.DroidDrinkIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
