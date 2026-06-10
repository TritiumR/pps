# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    phone_ik_rel_sound_env_cfg,
    phone_ik_rel_visuomotor_env_cfg,
    phone_joint_pos_sound_env_cfg,
    phone_joint_pos_visuomotor_env_cfg,
)


gym.register(
    id="Isaac-Phone-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": phone_joint_pos_visuomotor_env_cfg.DroidPhoneJointPosVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Phone-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": phone_ik_rel_visuomotor_env_cfg.DroidPhoneIkRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Phone-Droid-Sound-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": phone_joint_pos_sound_env_cfg.DroidPhoneJointPosSoundEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Phone-Droid-Sound-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": phone_ik_rel_sound_env_cfg.DroidPhoneIkRelSoundEnvCfg,
    },
    disable_env_checker=True,
)
