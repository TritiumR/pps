# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    # book_ik_rel_pointcloud_env_cfg,
    scissor_ik_rel_visuomotor_env_cfg,
    # book_joint_pos_pointcloud_env_cfg,
    scissor_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Scissor-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": scissor_joint_pos_visuomotor_env_cfg.DroidScissorJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Scissor-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": scissor_ik_rel_visuomotor_env_cfg.DroidScissorIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


# gym.register(
#     id="Isaac-Book-Droid-PointCloud-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": book_joint_pos_pointcloud_env_cfg.DroidBookJointPosPointCloudEnvCfg,
#     },
#     disable_env_checker=True,
# )


# gym.register(
#     id="Isaac-Book-Droid-PointCloud-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": book_ik_rel_pointcloud_env_cfg.DroidBookIkRelPointCloudEnvCfg,
#     },
#     disable_env_checker=True,
# )
