# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import (
    plate_joint_pos_pointcloud_env_cfg,
    plate_joint_pos_visuomotor_env_cfg,
    plate_ik_rel_pointcloud_env_cfg,
    plate_ik_rel_visuomotor_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control with Visual Observations
##

gym.register(
    id="Isaac-Plate-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": plate_joint_pos_visuomotor_env_cfg.DroidPlateJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


# #
# Inverse Kinematics - Relative Pose Control
# #

gym.register(
    id="Isaac-Plate-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": plate_ik_rel_visuomotor_env_cfg.DroidPlateIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Plate-Droid-PointCloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": plate_joint_pos_pointcloud_env_cfg.DroidPlateJointPosPointCloudEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Plate-Droid-PointCloud-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": plate_ik_rel_pointcloud_env_cfg.DroidPlateIkRelPointCloudEnvCfg,
    },
    disable_env_checker=True,
)
