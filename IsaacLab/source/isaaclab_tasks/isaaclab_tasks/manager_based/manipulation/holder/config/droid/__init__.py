# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    holder_ik_rel_pointcloud_env_cfg,
    holder_ik_rel_visuomotor_env_cfg,
    holder_joint_pos_pointcloud_env_cfg,
    holder_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Holder-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": holder_joint_pos_visuomotor_env_cfg.DroidHolderJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Holder-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": holder_ik_rel_visuomotor_env_cfg.DroidHolderIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Holder-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": holder_joint_pos_pointcloud_env_cfg.DroidHolderJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Holder-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": holder_ik_rel_pointcloud_env_cfg.DroidHolderIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
