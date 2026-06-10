# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    ice_ik_rel_pointcloud_env_cfg,
    ice_ik_rel_visuomotor_env_cfg,
    ice_joint_pos_pointcloud_env_cfg,
    ice_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Ice-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ice_joint_pos_visuomotor_env_cfg.DroidIceJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Ice-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ice_ik_rel_visuomotor_env_cfg.DroidIceIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Ice-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ice_joint_pos_pointcloud_env_cfg.DroidIceJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Ice-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ice_ik_rel_pointcloud_env_cfg.DroidIceIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
