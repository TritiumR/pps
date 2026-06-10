# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.bird import mdp
from isaaclab_tasks.manager_based.manipulation.bird.mdp import bird_events
from isaaclab_tasks.manager_based.manipulation.bird.bird_env_cfg import (
    BirdEnvCfg,
)

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../../diffusion_policy/reconstruction/asset/angry_world",
)


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=bird_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0444,
                -0.1894,
                -0.1107,
                -2.5148,
                0.0044,
                2.3775,
                0.6952,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=bird_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_object_positions = EventTerm(
        func=bird_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.6),
                "y": (-0.3, 0.3),
                "z": (0.003, 0.003),
                "yaw": (-1.0, 1, 0),
            },
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("bird"),
                SceneEntityCfg("pig"),
                SceneEntityCfg("vase"),
                SceneEntityCfg("block"),
            ],
        },
    )

    randomize_light = EventTerm(
        func=bird_events.randomize_scene_lighting_domelight,
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
            "default_intensity": 2000.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            # "default_texture": "",
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        # object = ObsTerm(func=mdp.object_obs)
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
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("bird"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidBirdJointPosVisuomotorEnvCfg(BirdEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        self.scene.robot = DROID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # # Add semantics to table
        # self.scene.table.spawn.semantic_tags = [("class", "table")]

        # # Add semantics to ground
        # self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
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

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "finger_joint",
                "right_outer_knuckle_joint",
                "right_outer_finger_joint",
                "right_inner_finger_joint",
                "right_inner_finger_knuckle_joint",
                "left_outer_finger_joint",
                "left_inner_finger_knuckle_joint",
                "left_inner_finger_joint",
            ],
            scale=1.0,
            use_default_offset=False,
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["right_outer_knuckle_joint", "finger_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.005

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
                    prim_path="{ENV_REGEX_NS}/Robot/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.1534, 0.0, 0.0),
                        rot=(0.0, 0.7071068, 0.0, 0.7071068),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.046, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.046, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/wrist_cam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                # pos=(-0.03, -0.03, -0.09), rot=(-0.56472, -0.42555, -0.42555, -0.56472), convention="ros"
                pos=(0.005, -0.03, -0.07),
                rot=(-0.56472, -0.42555, -0.42555, -0.56472),
                convention="ros",
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
