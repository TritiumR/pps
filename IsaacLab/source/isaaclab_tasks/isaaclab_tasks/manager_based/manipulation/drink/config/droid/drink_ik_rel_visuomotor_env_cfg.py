# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import drink_joint_pos_visuomotor_env_cfg


@configclass
class DroidDrinkIkRelVisuomotorEnvCfg(
    drink_joint_pos_visuomotor_env_cfg.DroidDrinkJointPosVisuomotorEnvCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )

        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.wrist_cam.height = 720
        self.scene.wrist_cam.width = 1280
