# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import (
    DroidStackEnvCfg,
)
import numpy as np

##
# Pre-defined configs
##
from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events (custom version for bigger randomization range)."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0,
                -1 / 5 * np.pi,
                0.0,
                -4 / 5 * np.pi,
                0.0,
                3 / 5 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                # 0.0,
                # 0.0,
            ],
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.65),
                "y": (-0.15, 0.15),
                "z": (0.0203, 0.0203),
                "yaw": (-1.0, 1, 0),
            },
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
        },
    )


@configclass
class DroidCubeStackSimEnvCfg(DroidStackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        self.scene.robot = DROID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )

        self.actions.gripper_action = mdp.BinaryZeroOneJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": np.pi / 4},
        )

        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "finger_joint",
        #         "right_outer_knuckle_joint",
        #         "right_outer_finger_joint",
        #         "right_inner_finger_joint",
        #         "right_inner_finger_knuckle_joint",
        #         "left_outer_finger_joint",
        #         "left_inner_finger_knuckle_joint",
        #         "left_inner_finger_joint",
        #     ],
        #     open_command_expr={
        #         "finger_joint": 0.0,
        #         "right_outer_knuckle_joint": 0.0,
        #         "right_outer_finger_joint": 0.0,
        #         "right_inner_finger_joint": 0.0,
        #         "right_inner_finger_knuckle_joint": 0.0,
        #         "left_outer_finger_joint": 0.0,
        #         "left_inner_finger_knuckle_joint": 0.0,
        #         "left_inner_finger_joint": 0.0,
        #     },
        #     close_command_expr={
        #         "finger_joint": 0.785398163,
        #         "right_outer_knuckle_joint": 0.785398163,
        #         "right_outer_finger_joint": 0.0,
        #         "right_inner_finger_joint": 0.785398163,
        #         "right_inner_finger_knuckle_joint": -0.785398163,
        #         "left_outer_finger_joint": 0.0,
        #         "left_inner_finger_knuckle_joint": -0.785398163,
        #         "left_inner_finger_joint": -0.785398163,
        #     },
        # )

        # self.actions.gripper_action = mdp.ProgressiveJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "finger_joint",
        #         "right_outer_knuckle_joint",
        #         "right_outer_finger_joint",
        #         "right_inner_finger_joint",
        #         "right_inner_finger_knuckle_joint",
        #         "left_outer_finger_joint",
        #         "left_inner_finger_knuckle_joint",
        #         "left_inner_finger_joint",
        #     ],
        #     open_command_expr={
        #         "finger_joint": 0.0,
        #         "right_outer_knuckle_joint": 0.0,
        #         "right_outer_finger_joint": 0.0,
        #         "right_inner_finger_joint": 0.0,
        #         "right_inner_finger_knuckle_joint": 0.0,
        #         "left_outer_finger_joint": 0.0,
        #         "left_inner_finger_knuckle_joint": 0.0,
        #         "left_inner_finger_joint": 0.0,
        #     },
        #     close_command_expr={
        #         "finger_joint": 0.785398163,
        #         "right_outer_knuckle_joint": 0.785398163,
        #         "right_outer_finger_joint": 0.0,
        #         "right_inner_finger_joint": 0.785398163,
        #         "right_inner_finger_knuckle_joint": -0.785398163,
        #         "left_outer_finger_joint": 0.0,
        #         "left_inner_finger_knuckle_joint": -0.785398163,
        #         "left_inner_finger_joint": -0.785398163,
        #     },
        #     total_progress=8,
        # )

        # utilities for gripper status check
        self.gripper_joint_names = ["right_outer_knuckle_joint", "finger_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Mass properties for cubes (you can change these values)
        cube_mass_properties = MassPropertiesCfg(
            mass=0.001,  # Mass in kg - change this value to modify cube mass
            # Alternative: use density instead of mass
            # density=1000.0,  # Density in kg/m³
        )

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                mass_props=cube_mass_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/base_link",
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.1534, 0.0, 0.0),
                        rot=(0.0, 0.7071068, 0.0, 0.7071068),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger",
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger",
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
