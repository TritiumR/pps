# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    pen_ik_rel_pointcloud_env_cfg,
    pen_ik_rel_visuomotor_env_cfg,
    pen_joint_pos_pointcloud_env_cfg,
    pen_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Pen-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pen_joint_pos_visuomotor_env_cfg.DroidPenJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pen-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pen_ik_rel_visuomotor_env_cfg.DroidPenIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pen-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pen_joint_pos_pointcloud_env_cfg.DroidPenJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Pen-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pen_ik_rel_pointcloud_env_cfg.DroidPenIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
