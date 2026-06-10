# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    capsule_ik_rel_pointcloud_env_cfg,
    capsule_ik_rel_visuomotor_env_cfg,
    capsule_joint_pos_pointcloud_env_cfg,
    capsule_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Capsule-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": capsule_joint_pos_visuomotor_env_cfg.DroidCapsuleJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Capsule-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": capsule_ik_rel_visuomotor_env_cfg.DroidCapsuleIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Capsule-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": capsule_joint_pos_pointcloud_env_cfg.DroidCapsuleJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Capsule-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": capsule_ik_rel_pointcloud_env_cfg.DroidCapsuleIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
