# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.tea import mdp
from isaaclab_tasks.manager_based.manipulation.tea.mdp import tea_events
from isaaclab_tasks.manager_based.manipulation.tea.tea_env_cfg import (
    EventCfg as BaseEventCfg,
    TEAPOT_GRASP_DIFF_THRESHOLD,
    TEAPOT_MOUTH_LOCAL_OFFSET,
    TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
    TEAPOT_ROLL_THRESHOLD_RAD,
    TeaEnvCfg,
)

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

ROBOT_INIT_POS = (2.5, 5.6, 0.7)
ROBOT_INIT_YAW_DEG = 0.0
ROBOT_INIT_ROT = (
    float(np.cos(np.deg2rad(ROBOT_INIT_YAW_DEG) / 2.0)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(ROBOT_INIT_YAW_DEG) / 2.0)),
)

ROBOT_TABLE_INIT_POS = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1], ROBOT_INIT_POS[2])
ROBOT_TABLE_INIT_YAW_DEG = ROBOT_INIT_YAW_DEG + 90.0
ROBOT_TABLE_INIT_ROT = (
    float(np.cos(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
)

TEA_OBJECT_RESET_CENTER = (3.0, 5.5, 0.86)
TEA_OBJECT_X_RANGE = 0.17
TEA_OBJECT_Y_RANGE = 0.4


@configclass
class EventCfg(BaseEventCfg):
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_tea_objects = EventTerm(
        func=tea_events.randomize_object_pose,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("teapot"), SceneEntityCfg("teacup")],
            "pose_range": {
                "x": (
                    TEA_OBJECT_RESET_CENTER[0] - TEA_OBJECT_X_RANGE / 2.0,
                    TEA_OBJECT_RESET_CENTER[0] + TEA_OBJECT_X_RANGE / 2.0,
                ),
                "y": (
                    TEA_OBJECT_RESET_CENTER[1] - TEA_OBJECT_Y_RANGE / 2.0,
                    TEA_OBJECT_RESET_CENTER[1] + TEA_OBJECT_Y_RANGE / 2.0,
                ),
                "z": (TEA_OBJECT_RESET_CENTER[2], TEA_OBJECT_RESET_CENTER[2]),
                "roll": (np.pi / 2.0, np.pi / 2.0),
                "pitch": (0.0, 0.0),
                "yaw": (np.pi / 2.0 - 0.5, np.pi / 2.0 + 0.5),
            },
            "min_separation": 0.3,
        },
    )

    init_franka_arm_pose = EventTerm(
        func=tea_events.set_default_joint_pose,
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
        func=tea_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
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
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_teapot = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("teapot"),
                "diff_threshold": TEAPOT_GRASP_DIFF_THRESHOLD,
            },
        )

        # teapot_mouth_near_teacup = ObsTerm(
        #     func=mdp.teapot_mouth_near_teacup_xy,
        #     params={
        #         "teapot_cfg": SceneEntityCfg("teapot"),
        #         "teacup_cfg": SceneEntityCfg("teacup"),
        #         "mouth_offset": TEAPOT_MOUTH_LOCAL_OFFSET,
        #         "xy_threshold": TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
        #     },
        # )

        # teapot_rolled = ObsTerm(
        #     func=mdp.teapot_rolled,
        #     params={
        #         "teapot_cfg": SceneEntityCfg("teapot"),
        #         "min_roll_rad": TEAPOT_ROLL_THRESHOLD_RAD,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidTeaJointPosVisuomotorEnvCfg(TeaEnvCfg):
    """Configuration for tea task with Droid robot using joint position control."""

    observations: ObservationsCfg = ObservationsCfg()

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
                pos=(0.004620336834421451, -0.5388594867462788, 0.504018368138419),
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
        # self.scene.table_cam.height = 720 // 2
        # self.scene.table_cam.width = 1280 // 2
        # self.scene.wrist_cam.height = 720 // 2
        # self.scene.wrist_cam.width = 1280 // 2

        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.wrist_cam.height = 720
        self.scene.wrist_cam.width = 1280

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
