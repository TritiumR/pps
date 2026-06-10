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

from isaaclab_tasks.manager_based.manipulation.scissor import mdp
from isaaclab_tasks.manager_based.manipulation.scissor.mdp import scissor_events
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import EventCfg as BaseEventCfg
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_HOLDER001_INIT_POS
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_INIT_POS
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_GRASP_DIFF_THRESHOLD
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_HOLDER_MAX_HEIGHT_OFFSET
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_HOLDER_MIN_HEIGHT_OFFSET
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import PEN_HOLDER_XY_THRESHOLD
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import SCISSOR_GRASP_DIFF_THRESHOLD
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import SCISSOR_HOLDER_MIN_XY_DISTANCE
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import SCISSORS010_INIT_POS
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import ScissorEnvCfg
from isaaclab_tasks.manager_based.manipulation.scissor.scissor_env_cfg import holder_scissor_contact_material

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

ROBOT_INIT_POS = (7.2, 0.0, 0.7)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

ROBOT_TABLE_INIT_POS = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1], ROBOT_INIT_POS[2])
ROBOT_TABLE_INIT_YAW_DEG = 90.0
ROBOT_TABLE_INIT_ROT = (
    float(np.cos(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
)

OVEN_DEFAULT_POS = (3.0, 1.1, 0.35)
OVEN_RANDOMIZE_POSE_RANGE = {
    "x": (OVEN_DEFAULT_POS[0] - 0.1, OVEN_DEFAULT_POS[0] + 0.1),
    "y": (OVEN_DEFAULT_POS[1] - 0.1, OVEN_DEFAULT_POS[1] + 0.1),
    "z": (OVEN_DEFAULT_POS[2], OVEN_DEFAULT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (np.pi, np.pi),
}

CAN_DEFAULT_POS = (2.6, 1.35, 0.22)
CAN_RANDOMIZE_POSE_RANGE = {
    "x": (CAN_DEFAULT_POS[0] - 0.1, CAN_DEFAULT_POS[0] + 0.2),
    "y": (CAN_DEFAULT_POS[1] - 0.1, CAN_DEFAULT_POS[1] + 0.2),
    "z": (CAN_DEFAULT_POS[2], CAN_DEFAULT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

PEN_HOLDER_RANDOMIZE_POSE_RANGE = {
    "x": (PEN_HOLDER001_INIT_POS[0] - 0.12, PEN_HOLDER001_INIT_POS[0] + 0.12),
    "y": (PEN_HOLDER001_INIT_POS[1] - 0.12, PEN_HOLDER001_INIT_POS[1] + 0.12),
    "z": (PEN_HOLDER001_INIT_POS[2], PEN_HOLDER001_INIT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.7, 0.7),
}

PEN_RANDOMIZE_POSE_RANGE = {
    "x": (PEN_INIT_POS[0] - 0.12, PEN_INIT_POS[0] + 0.12),
    "y": (PEN_INIT_POS[1] - 0.20, PEN_INIT_POS[1] + 0.20),
    "z": (PEN_INIT_POS[2], PEN_INIT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.7, 0.7),
}

SCISSORS008_IN_HOLDER_OFFSET = (0.0, 0.0, 0.12)
SCISSORS008_IN_HOLDER_ROT_EULER = (np.pi / 2.0, 0.0, 0.0)

@configclass
class EventCfg(BaseEventCfg):
    """Configuration for events."""

    remove_office_chair = EventTerm(
        func=scissor_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": "/World/envs/env_.*/interactive_smalllivingroom/model_officechair_3",
        },
    )

    remove_laptop = EventTerm(
        func=scissor_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": "/World/envs/env_.*/interactive_smalllivingroom/model_computer_9",
        },
    )

    remove_light1 = EventTerm(
        func=scissor_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": "/World/envs/env_.*/interactive_smalllivingroom/light1",
        },
    )

    holder_contact_material = EventTerm(
        func=scissor_events.bind_rigid_body_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("pen_holder001"),
            "material_cfg": holder_scissor_contact_material,
        },
    )

    scissors008_contact_material = EventTerm(
        func=scissor_events.bind_rigid_body_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("scissors008"),
            "material_cfg": holder_scissor_contact_material,
        },
    )

    # improve_pen_holder_collision = EventTerm(
    #     func=scissor_events.set_asset_mesh_collision_to_convex_decomposition,
    #     mode="prestartup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("pen_holder001"),
    #         "hull_vertex_limit": 64,
    #         "max_convex_hulls": 64,
    #         "min_thickness": 0.001,
    #         "voxel_resolution": 1_000_000,
    #         "error_percentage": 2.0,
    #         "shrink_wrap": True,
    #         "contact_offset": 0.001,
    #         "rest_offset": 0.0,
    #     },
    # )

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    init_franka_arm_pose = EventTerm(
        func=scissor_events.set_default_joint_pose,
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
        func=scissor_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_pen_holder_with_scissors008 = EventTerm(
        func=scissor_events.randomize_attached_object_pose,
        mode="reset",
        params={
            "anchor_pose_range": PEN_HOLDER_RANDOMIZE_POSE_RANGE,
            "anchor_asset_cfg": SceneEntityCfg("pen_holder001"),
            "attached_asset_cfg": SceneEntityCfg("scissors008"),
            "attached_pos_offset": SCISSORS008_IN_HOLDER_OFFSET,
            "attached_euler_offset": SCISSORS008_IN_HOLDER_ROT_EULER,
        },
    )

    randomize_objects_pose = EventTerm(
        func=scissor_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": PEN_RANDOMIZE_POSE_RANGE,
            "asset_cfgs": [SceneEntityCfg("pen"), SceneEntityCfg("scissors010")],
            "min_separation": 0.17,
        },
    )


    # make_plate_dynamic = EventTerm(
    #     func=oven_events.set_rigid_body_dynamic,
    #     mode="prestartup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("plate"),
    #     },
    # )

    # randomize_can_positions = EventTerm(
    #     func=oven_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": CAN_RANDOMIZE_POSE_RANGE,
    #         "asset_cfgs": [SceneEntityCfg("can")],
    #     },
    # )

    # randomize_oven_positions = EventTerm(
    #     func=oven_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": OVEN_RANDOMIZE_POSE_RANGE,
    #         "asset_cfgs": [SceneEntityCfg("oven")],
    #     },
    # )

    # randomize_light = EventTerm(
    #     func=oven_events.randomize_scene_lighting_domelight,
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

    # sync_oven_button_to_door = EventTerm(
    #     func=oven_events.sync_oven_button_to_door,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "oven_cfg": SceneEntityCfg("oven"),
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

        grasp_scissors008 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("scissors008"),
                "diff_threshold": SCISSOR_GRASP_DIFF_THRESHOLD,
            },
        )

        scissors008_away_from_holder = ObsTerm(
            func=mdp.scissor_away_from_holder,
            params={
                "scissor_cfg": SceneEntityCfg("scissors008"),
                "holder_cfg": SceneEntityCfg("pen_holder001"),
                "robot_cfg": SceneEntityCfg("robot"),
                "min_xy_distance": SCISSOR_HOLDER_MIN_XY_DISTANCE,
            },
        )

        grasp_pen = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pen"),
                "diff_threshold": PEN_GRASP_DIFF_THRESHOLD,
            },
        )

        pen_placed_in_holder = ObsTerm(
            func=mdp.task_done_scissor,
            params={
                "pen_cfg": SceneEntityCfg("pen"),
                "holder_cfg": SceneEntityCfg("pen_holder001"),
                "scissor_cfg": SceneEntityCfg("scissors008"),
                "robot_cfg": SceneEntityCfg("robot"),
                "pen_holder_xy_threshold": PEN_HOLDER_XY_THRESHOLD,
                "pen_holder_min_height_offset": PEN_HOLDER_MIN_HEIGHT_OFFSET,
                "pen_holder_max_height_offset": PEN_HOLDER_MAX_HEIGHT_OFFSET,
                "scissor_holder_min_xy_distance": SCISSOR_HOLDER_MIN_XY_DISTANCE,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidScissorJointPosVisuomotorEnvCfg(ScissorEnvCfg):
    """Configuration for scissor task with Droid robot using joint position control."""

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
                pos=(0.054620336834421451, -0.4388594867462788, 0.454018368138419),
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
        self.scene.table_cam.height = 720 / 4
        self.scene.table_cam.width = 1280 / 4
        self.scene.wrist_cam.height = 720 / 4
        self.scene.wrist_cam.width = 1280 / 4

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
