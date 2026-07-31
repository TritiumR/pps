# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    utensil_ik_rel_pointcloud_env_cfg,
    utensil_ik_rel_visuomotor_env_cfg,
    utensil_joint_pos_pointcloud_env_cfg,
    utensil_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Utensil-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": utensil_joint_pos_visuomotor_env_cfg.DroidUtensilJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Utensil-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": utensil_ik_rel_visuomotor_env_cfg.DroidUtensilIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Utensil-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": utensil_joint_pos_pointcloud_env_cfg.DroidUtensilJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Utensil-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": utensil_ik_rel_pointcloud_env_cfg.DroidUtensilIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
