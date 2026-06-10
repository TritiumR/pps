# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import block_joint_pos_visuomotor_env_cfg


@configclass
class DroidBlockIkRelVisuomotorEnvCfg(
    block_joint_pos_visuomotor_env_cfg.DroidBlockJointPosVisuomotorEnvCfg
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set IK controller for the robot
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

        # change camera resolutions to save memory
        self.scene.table_cam.height = 720 / 4
        self.scene.table_cam.width = 1280 / 4
        self.scene.wrist_cam.height = 720 / 4
        self.scene.wrist_cam.width = 1280 / 4
