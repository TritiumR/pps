# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import laptop_joint_pos_visuomotor_env_cfg

EE_FRAME_EQUIVALENT_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(
    pos=[0.0, 0.0, 0.2414],
    rot=[0.0, 0.0, 0.0, 1.0],
)


@configclass
class TianjiLaptopIkRelVisuomotorEnvCfg(
    laptop_joint_pos_visuomotor_env_cfg.TianjiLaptopJointPosVisuomotorEnvCfg
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set IK controller for the robot
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_joint.*"],
            body_name="left_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=EE_FRAME_EQUIVALENT_OFFSET,
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_joint.*"],
            body_name="right_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=EE_FRAME_EQUIVALENT_OFFSET,
        )

        # change camera resolutions to save memory
        # self.scene.table_cam.height = 720 // 4
        # self.scene.table_cam.width = 1280 // 4
        # self.scene.left_wrist_cam.height = 720 // 4
        # self.scene.left_wrist_cam.width = 1280 // 4
        # self.scene.right_wrist_cam.height = 720 // 4
        # self.scene.right_wrist_cam.width = 1280 // 4

        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.left_wrist_cam.height = 720
        self.scene.left_wrist_cam.width = 1280
        self.scene.right_wrist_cam.height = 720
        self.scene.right_wrist_cam.width = 1280
