# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg

from isaaclab_tasks.manager_based.manipulation.water_align import mdp
from isaaclab_tasks.manager_based.manipulation.water_align.mdp import water_events
from isaaclab_tasks.manager_based.manipulation.water_align.water_align_env_cfg import (
    WaterAlignEnvCfg,
    ASSET_INIT_POS,
    TABLE_INIT_POS,
)

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=water_events.set_default_joint_pose,
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
            ],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=water_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_table_and_objects = EventTerm(
        func=water_events.randomize_table_and_objects_pose,
        mode="reset",
        params={
            "table_pose_range": {
                "x": (TABLE_INIT_POS[0] - 0.025, TABLE_INIT_POS[0] + 0.025),
                "y": (TABLE_INIT_POS[1] - 0.025, TABLE_INIT_POS[1] + 0.025),
                "z": (TABLE_INIT_POS[2] - 0.025, TABLE_INIT_POS[2] + 0.025),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.01, 0.01),
            },
            "object_pose_range": {
                "x": (0.38, 0.52),
                "y": (-0.38, 0.30),
                "yaw": (-0.5, 0.5),
            },
            "table_cfg": SceneEntityCfg("table"),
            "object_cfgs": [
                SceneEntityCfg("cup"),
                SceneEntityCfg("plant"),
                SceneEntityCfg("bowl"),
            ],
            "min_separation": 0.25,
        },
    )

    randomize_light = EventTerm(
        func=water_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1000.0, 2000.0),
            "color_variation": 0.07,
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
            "default_color": (0.75, 0.75, 0.75),
        },
    )

    randomize_table_cam_offset = EventTerm(
        func=water_events.randomize_camera_offset,
        mode="reset",
        params={
            "position_range": {
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (-0.015, 0.015),
            },
            "rotation_range": {
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
            "asset_cfg": SceneEntityCfg("table_cam"),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_actions = ObsTerm(func=mdp.last_droid_action)

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
            noise=GaussianNoiseCfg(mean=0.0, std=15.0, operation="add"),
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
            noise=GaussianNoiseCfg(mean=0.0, std=15.0, operation="add"),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cup"),
                "diff_threshold": 0.1,
            },
        )

        move_1 = ObsTerm(
            func=mdp.object_moved,
            params={
                "object_cfg": SceneEntityCfg("cup"),
                "target_cfg": SceneEntityCfg("plant"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidWaterAlignJointPosVisuomotorEnvCfg(WaterAlignEnvCfg):
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
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.1534, 0.0, 0.0),
                        rot=(0.0, 0.7071068, 0.0, 0.7071068),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # Set table camera as the real-world camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.1,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
                clipping_range=(1e-4, 5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.004620336834421451, -0.5388594867462788, 0.454018368138419),
                # rot=(0.2595868830, 0.3175587775, 0.7575422903, 0.5078392969),
                rot=(-0.5078392969, 0.7575422903, -0.3175587775, 0.2595868830),
                convention="ros",
            ),
        )

        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
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
        self.scene.table_cam.height = 720 / 4
        self.scene.table_cam.width = 1280 / 4
        self.scene.wrist_cam.height = 720 / 4
        self.scene.wrist_cam.width = 1280 / 4

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
