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
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_tasks.manager_based.manipulation.weight import mdp
from isaaclab_tasks.manager_based.manipulation.weight.mdp import weight_events
from isaaclab_tasks.manager_based.manipulation.weight.weight_env_cfg import EventCfg as BaseEventCfg
from isaaclab_tasks.manager_based.manipulation.weight.weight_env_cfg import WeightEnvCfg
from isaaclab_tasks.manager_based.manipulation.weight.weight_env_cfg import (
    APPLE_GRASP_DIFF_THRESHOLD,
    PEAR_GRASP_DIFF_THRESHOLD,
    SCALE_XY_THRESHOLD,
)

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

SHIFT_X = -2.5

ROBOT_INIT_POS = (3.0 + SHIFT_X, 1.86, 0.12)
ROBOT_INIT_ROT = (0.7071, 0.0, 0.0, -0.7071)

ROBOT_TABLE_INIT_POS = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1], ROBOT_INIT_POS[2])
ROBOT_TABLE_INIT_YAW_DEG = 0.0
ROBOT_TABLE_INIT_ROT = (
    float(np.cos(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
)

SCALE_DEFAULT_POS = (3.0 + SHIFT_X, 1.3, 0.23)
SCALE_RANDOMIZE_POSE_RANGE = {
    "x": (SCALE_DEFAULT_POS[0] - 0.0, SCALE_DEFAULT_POS[0] + 0.25),
    "y": (SCALE_DEFAULT_POS[1] - 0.1, SCALE_DEFAULT_POS[1] + 0.2),
    "z": (SCALE_DEFAULT_POS[2], SCALE_DEFAULT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (np.pi, np.pi),
}

BOARD_DEFAULT_POS = (2.6 + SHIFT_X, 1.3, 0.23)
BOARD_RANDOMIZE_POSE_RANGE = {
    "x": (BOARD_DEFAULT_POS[0] - 0.05, BOARD_DEFAULT_POS[0] + 0.2),
    "y": (BOARD_DEFAULT_POS[1] - 0.0, BOARD_DEFAULT_POS[1] + 0.2),
    "z": (BOARD_DEFAULT_POS[2], BOARD_DEFAULT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

APPLE_REL_POS_TO_BOARD = (-0.08238987332084574, -0.04169753589476233, 0.0724487)
APPLE_REL_QUAT_TO_BOARD = (0.9999937957700049, 0.0002367066033963648, -0.00039975546005791907, 0.003491788517939705)

PEAR_REL_POS_TO_BOARD = (0.0083409162111532, 0.003640049095619693, 0.0987787)
PEAR_REL_QUAT_TO_BOARD = (-0.0446399287174948, 0.9936850534673, -0.063057673728588, 0.08136965689816525)

MANGO_REL_POS_TO_BOARD = (-0.09154030000000002, 0.07006140000000001, 0.1383396)
MANGO_REL_QUAT_TO_BOARD = (0.09672192178722909, 0.9514776447224945, -0.26097066916851586, -0.13126108057172448)

CABBAGE_REL_POS_TO_BOARD = (0.10755740000000001, -0.0947881, 0.0635123)
CABBAGE_REL_QUAT_TO_BOARD = (-0.3624254678505684, 0.60247217023873, -0.23433703706677222, 0.6713875314599591)

@configclass
class EventCfg(BaseEventCfg):
    """Configuration for events."""

    apply_apple_scale = EventTerm(
        func=weight_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("apple")},
    )

    apply_pear_scale = EventTerm(
        func=weight_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("pear")},
    )

    apply_mango_scale = EventTerm(
        func=weight_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("mango")},
    )

    apply_cabbage_scale = EventTerm(
        func=weight_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("cabbage")},
    )

    apply_board_scale = EventTerm(
        func=weight_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("board")},
    )

    remove_unused_board_components = EventTerm(
        func=weight_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": "/World/envs/env_.*/chopping_board_set/E_Component154_04",
        },
    )

    remove_scale_buttons = EventTerm(
        func=weight_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": "/World/envs/env_.*/scale_set/(E_button_01_38|E_button_02_41|E_button_03_42)",
        },
    )

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    init_franka_arm_pose = EventTerm(
        func=weight_events.set_default_joint_pose,
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
        func=weight_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # make_plate_dynamic = EventTerm(
    #     func=oven_events.set_rigid_body_dynamic,
    #     mode="prestartup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("plate"),
    #     },
    # )

    randomize_board_with_fruits = EventTerm(
        func=weight_events.randomize_grouped_object_pose,
        mode="reset",
        params={
            "anchor_pose_range": BOARD_RANDOMIZE_POSE_RANGE,
            "anchor_asset_cfg": SceneEntityCfg("board"),
            "attached_asset_cfgs": [
                SceneEntityCfg("apple"),
                SceneEntityCfg("pear"),
                SceneEntityCfg("mango"),
                SceneEntityCfg("cabbage"),
            ],
            "attached_pos_offsets": [
                APPLE_REL_POS_TO_BOARD,
                PEAR_REL_POS_TO_BOARD,
                MANGO_REL_POS_TO_BOARD,
                CABBAGE_REL_POS_TO_BOARD,
            ],
            "attached_quat_offsets": [
                APPLE_REL_QUAT_TO_BOARD,
                PEAR_REL_QUAT_TO_BOARD,
                MANGO_REL_QUAT_TO_BOARD,
                CABBAGE_REL_QUAT_TO_BOARD,
            ],
        },
    )

    randomize_scale_positions = EventTerm(
        func=weight_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": SCALE_RANDOMIZE_POSE_RANGE,
            "asset_cfgs": [SceneEntityCfg("scale")],
        },
    )

    # randomize_light = EventTerm(
    #     func=weight_events.randomize_scene_lighting_domelight,
    #     mode="reset",
    #     params={
    #         "intensity_range": (1500.0, 10000.0),
    #         "color_variation": 0.4,
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    #         ],
    #         "default_intensity": 1500.0,
    #         "default_color": (0.75, 0.75, 0.75),
    #         # "default_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    #     },
    # )

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
            # noise=GaussianNoiseCfg(mean=0.0, std=15.0, operation="add"),
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
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

        grasp_pear = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pear"),
                "diff_threshold": PEAR_GRASP_DIFF_THRESHOLD,
            },
        )

        pear_on_scale = ObsTerm(
            func=mdp.pear_on_scale,
            params={
                "pear_cfg": SceneEntityCfg("pear"),
                "scale_cfg": SceneEntityCfg("scale"),
                "y_offset": -0.05,
                "xy_threshold": SCALE_XY_THRESHOLD,
            },
        )

        grasp_apple = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("apple"),
                "diff_threshold": APPLE_GRASP_DIFF_THRESHOLD,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidWeightJointPosVisuomotorEnvCfg(WeightEnvCfg):
    """Configuration for weight task with Droid robot using joint position control."""

    observations: ObservationsCfg = ObservationsCfg()

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Robot Table
        self.scene.robot_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RobotTable",
            init_state=AssetBaseCfg.InitialStateCfg(pos=ROBOT_TABLE_INIT_POS, rot=ROBOT_TABLE_INIT_ROT),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(0.6, 0.3, 1.0),
            ),
        )

        self.scene.robot = DROID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = ROBOT_INIT_POS
        self.scene.robot.init_state.rot = ROBOT_INIT_ROT
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
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0/table_cam",
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

        # # change camera resolutions to save memory
        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.wrist_cam.height = 720
        self.scene.wrist_cam.width = 1280

        # self.scene.table_cam.height = 720
        # self.scene.table_cam.width = 1280
        # self.scene.wrist_cam.height = 720
        # self.scene.wrist_cam.width = 1280

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
