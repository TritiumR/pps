# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.laptop import mdp
from isaaclab_tasks.manager_based.manipulation.laptop.mdp import laptop_events
from isaaclab_tasks.manager_based.manipulation.laptop.laptop_env_cfg import EventCfg as BaseEventCfg
from isaaclab_tasks.manager_based.manipulation.laptop.laptop_env_cfg import LaptopEnvCfg

from isaaclab_assets.robots.tianji import TIANJI_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.laptop.laptop_env_cfg import ASSET_INIT_POS

ARM_JOINT_NAMES = ["left_joint.*", "right_joint.*"]
LEFT_GRIPPER_JOINT = "finger_joint"
RIGHT_GRIPPER_JOINT = "finger_joint_0"


@configclass
class ActionsCfg:
    """Action specification for the Tianji bimanual robot."""

    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg(BaseEventCfg):
    """Configuration for events."""

    init_tianji_arm_pose = EventTerm(
        func=laptop_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0] * 14,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )

    randomize_tianji_joint_state = EventTerm(
        func=laptop_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )

    randomize_object_positions = EventTerm(
        func=laptop_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.55, 0.62),
                "y": (-0.25, 0.18),
                "z": (ASSET_INIT_POS[2], ASSET_INIT_POS[2]),
                "yaw": (-0.5, 0.5),
            },
            "min_separation": 0.25,
            "asset_cfgs": [
                SceneEntityCfg("laptop"),
            ],
        },
    )

    reset_laptop_joint_state = EventTerm(
        func=laptop_events.set_laptop_default_joint_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("laptop"),
        },
    )

    reset_table_joint_state = EventTerm(
        func=laptop_events.set_table_default_joint_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    randomize_light = EventTerm(
        func=laptop_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1500.0, 10000.0),
            "color_variation": 0.4,
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            ],
            "default_intensity": 1500.0,
            "default_color": (0.75, 0.75, 0.75),
            # "default_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_actions = ObsTerm(func=mdp.last_joint_action)

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        table_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
            # noise=GaussianNoiseCfg(mean=0.0, std=15.0, operation="add"),
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
            # noise=GaussianNoiseCfg(mean=0.0, std=15.0, operation="add"),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""


        laptop_grasped = ObsTerm(
            func=mdp.laptop_is_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "laptop_cfg": SceneEntityCfg("laptop"),
                "diff_threshold": 0.25,
            },
        )

        laptop_placed_in_drawer = ObsTerm(
            func=mdp.laptop_is_in_drawer,
            params={
                "laptop_cfg": SceneEntityCfg("laptop"),
                "table_cfg": SceneEntityCfg("table"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        laptop_closed = ObsTerm(
            func=mdp.laptop_is_closed,
            params={
                "laptop_cfg": SceneEntityCfg("laptop"),
                "lid_joint_name": "RevoluteJoint_computer_9_up",
                # "closed_threshold": -1.2,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TianjiLaptopJointPosVisuomotorEnvCfg(LaptopEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        robot_y_offset = 0.6
        self.scene.robot_table.init_state.pos = (-0.40, robot_y_offset, 0.0)
        self.scene.robot = TIANJI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0.0, robot_y_offset, 0.0)

        # Bimanual action representation: one action term per arm and gripper.
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )

        self.actions.left_gripper_action = mdp.BinaryZeroOneJointPositionActionCfg(
            asset_name="robot",
            joint_names=[LEFT_GRIPPER_JOINT],
            open_command_expr={LEFT_GRIPPER_JOINT: 0.0},
            close_command_expr={LEFT_GRIPPER_JOINT: np.pi / 4},
        )

        self.actions.right_gripper_action = mdp.BinaryZeroOneJointPositionActionCfg(
            asset_name="robot",
            joint_names=[RIGHT_GRIPPER_JOINT],
            open_command_expr={RIGHT_GRIPPER_JOINT: 0.0},
            close_command_expr={RIGHT_GRIPPER_JOINT: np.pi / 4},
        )

        # Keep the left gripper as the primary task gripper for laptop-specific observations.
        self.gripper_joint_names = ["right_outer_knuckle_joint", "finger_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.005

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.1534, 0.0, 0.0),
                        rot=(0.0, 0.7071068, 0.0, 0.7071068),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_gripper/Robotiq_2F_85/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_gripper/Robotiq_2F_85/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.1534, 0.0, 0.0),
                        rot=(0.0, 0.7071068, 0.0, 0.7071068),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_gripper/Robotiq_2F_85/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_gripper/Robotiq_2F_85/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        self.scene.ee_frame = self.scene.left_ee_frame

        # # Set table camera as the real-world camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            # spawn=sim_utils.PinholeCameraCfg(
            #     focal_length=2.1,
            #     focus_distance=28.0,
            #     horizontal_aperture=5.376,
            #     vertical_aperture=3.024,
            #     clipping_range=(1e-4, 5),
            # ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0476,
                horizontal_aperture=2.5452,
                vertical_aperture=1.4721,
                clipping_range=(1e-4, 5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.404620336834421451, -0.4088594867462788, 0.754018368138419),
                # rot=(0.2595868830, 0.3175587775, 0.7575422903, 0.5078392969),
                rot=(-0.3012355305, 0.8678666196, -0.3638063066, 0.1539794043),
                convention="ros",
            ),
        )

        # Set wrist camera
        self.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_gripper/Robotiq_2F_85/base_link/wrist_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.011, -0.031, -0.074),
                rot=(-0.420, 0.570, 0.576, -0.409),
                convention="opengl",
            ),
        )

        # Set wrist camera
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_gripper/Robotiq_2F_85/base_link/wrist_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.011, -0.031, -0.074),
                rot=(-0.420, 0.570, 0.576, -0.409),
                convention="opengl",
            ),
        )
        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # change camera resolutions to save memory
        self.scene.table_cam.height = 720 // 4
        self.scene.table_cam.width = 1280 // 4
        self.scene.left_wrist_cam.height = 720 // 4
        self.scene.left_wrist_cam.width = 1280 // 4
        self.scene.right_wrist_cam.height = 720 // 4
        self.scene.right_wrist_cam.width = 1280 // 4

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
