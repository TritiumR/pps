# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    stack_joint_pos_env_cfg,
    stack_joint_pos_sim_env_cfg,
    stack_joint_pos_visuomotor_env_cfg,
    stack_joint_pos_visuomotor_ood_env_cfg,
    stack_joint_pos_visuomotor_sim_env_cfg,
    stack_joint_pos_visuomotor_sim_id_env_cfg,
    # stack_ik_env_cfg,
    stack_ik_rel_visuomotor_env_cfg,
    stack_ik_rel_visuomotor_ood_env_cfg,
    stack_ik_rel_visuomotor_sim_env_cfg,
    stack_ik_rel_visuomotor_sim_id_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Droid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_env_cfg.DroidCubeStackEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_sim_env_cfg.DroidCubeStackSimEnvCfg,
    },
    disable_env_checker=True,
)

##
# Joint Position Control with Visual Observations
##

gym.register(
    id="Isaac-Stack-Cube-Droid-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_visuomotor_env_cfg.DroidCubeStackVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-Visuomotor-OOD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_visuomotor_ood_env_cfg.DroidCubeStackVisuomotorOODEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-Visuomotor-Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_visuomotor_sim_env_cfg.DroidCubeStackVisuomotorSimEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-Visuomotor-Sim-ID-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_visuomotor_sim_id_env_cfg.DroidCubeStackVisuomotorSimIDEnvCfg,
    },
    disable_env_checker=True,
)

# #
# Inverse Kinematics - Relative Pose Control
# #

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_env_cfg.DroidIkRelCubeStackVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-OOD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_ood_env_cfg.DroidIkRelCubeStackVisuomotorOODEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_sim_env_cfg.DroidIkRelCubeStackVisuomotorSimEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Sim-ID-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_sim_id_env_cfg.DroidIkRelCubeStackVisuomotorSimIDEnvCfg,
    },
    disable_env_checker=True,
)
