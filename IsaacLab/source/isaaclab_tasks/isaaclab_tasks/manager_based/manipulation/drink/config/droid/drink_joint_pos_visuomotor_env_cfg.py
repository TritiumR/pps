# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.drink import mdp
from isaaclab_tasks.manager_based.manipulation.drink.mdp import drink_events
from isaaclab_tasks.manager_based.manipulation.drink.drink_env_cfg import (
    DRINK_GRASP_DIFF_THRESHOLD,
    DRINK_GRASP_DIFF_Z,
    DRINK_BODY_TOP_Z_OFFSET,
    DRINK_LID_GRASP_DIFF_THRESHOLD,
    DRINK_LID_REMOVE_HEIGHT_MARGIN,
    DRINK_LID_REMOVE_XY_THRESHOLD,
    DRINK_POUR_HEIGHT_THRESHOLD,
    DRINK_POUR_TILT_THRESHOLD,
    DRINK_POUR_XY_THRESHOLD,
    BIG_TABLE_CENTER_POS,
    DrinkEnvCfg,
    DRINK_SET_INIT_POS,
    EventCfg as BaseEventCfg,
)

from isaaclab_assets.robots.droid import DROID_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

ROBOT_INIT_POS = (3.1, 3.6, 0.5)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

ROBOT_TABLE_INIT_POS = (ROBOT_INIT_POS[0], ROBOT_INIT_POS[1], ROBOT_INIT_POS[2])
ROBOT_TABLE_INIT_YAW_DEG = 90.0
ROBOT_TABLE_INIT_ROT = (
    float(np.cos(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(ROBOT_TABLE_INIT_YAW_DEG) / 2.0)),
)

TABLE_OBJECT_RANDOMIZE_POSE_RANGE = {
    "x": (BIG_TABLE_CENTER_POS[0] - 0.25, BIG_TABLE_CENTER_POS[0] + 0.1),
    "y": (BIG_TABLE_CENTER_POS[1] - 0.5, BIG_TABLE_CENTER_POS[1] - 0.10),
    "z": (DRINK_SET_INIT_POS[2], DRINK_SET_INIT_POS[2]),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.5, 0.5),
}


@configclass
class EventCfg(BaseEventCfg):
    """Configuration for events."""

    refine_drink_collision = EventTerm(
        func=drink_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("drink"),
            "hull_vertex_limit": 128,
            "max_convex_hulls": 32,
            "min_thickness": 0.001,
            "voxel_resolution": 2_000_000,
            "error_percentage": 1.0,
            "shrink_wrap": True,
        },
    )

    apply_drink_mass = EventTerm(
        func=drink_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("drink"), "mass": 0.05},
    )

    apply_drink_lid_mass = EventTerm(
        func=drink_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("drink_lid"), "mass": 0.01},
    )

    apply_drink_lid_scale = EventTerm(
        func=drink_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("drink_lid")},
    )

    # reset_all = EventTerm(
    #     func=mdp.reset_scene_to_default,
    #     mode="reset",
    #     params={"reset_joint_targets": True},
    # )

    init_franka_arm_pose = EventTerm(
        func=drink_events.set_default_joint_pose,
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
        func=drink_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_objects_pose = EventTerm(
        func=drink_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": TABLE_OBJECT_RANDOMIZE_POSE_RANGE,
            "asset_cfgs": [SceneEntityCfg("drink"), SceneEntityCfg("cup")],
            "min_separation": 0.14,
        },
    )

    place_lid_on_drink = EventTerm(
        func=drink_events.align_attached_object_to_anchor,
        mode="reset",
        params={
            "anchor_asset_cfg": SceneEntityCfg("drink"),
            "attached_asset_cfg": SceneEntityCfg("drink_lid"),
            "attached_pos_offset": (0.0, 0.0, DRINK_BODY_TOP_Z_OFFSET),
            "attached_euler_offset": (0.0, 0.0, 0.0),
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

        grasp_drink_lid = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("drink_lid"),
                "diff_threshold": DRINK_LID_GRASP_DIFF_THRESHOLD,
            },
        )

        drink_lid_removed = ObsTerm(
            func=mdp.drink_lid_removed,
            params={
                "drink_cfg": SceneEntityCfg("drink"),
                "lid_cfg": SceneEntityCfg("drink_lid"),
                "body_top_z_offset": DRINK_BODY_TOP_Z_OFFSET,
                "xy_threshold": DRINK_LID_REMOVE_XY_THRESHOLD,
                "extra_height_threshold": DRINK_LID_REMOVE_HEIGHT_MARGIN,
            },
        )

        grasp_drink = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("drink"),
                "diff_threshold": DRINK_GRASP_DIFF_THRESHOLD,
                "diff_z": DRINK_GRASP_DIFF_Z,
            },
        )

        # drink_poured_into_cup = ObsTerm(
        #     func=mdp.task_done_drink,
        #     params={
        #         "drink_cfg": SceneEntityCfg("drink"),
        #         "lid_cfg": SceneEntityCfg("drink_lid"),
        #         "cup_cfg": SceneEntityCfg("cup"),
        #         "body_top_z_offset": DRINK_BODY_TOP_Z_OFFSET,
        #         "lid_remove_xy_threshold": DRINK_LID_REMOVE_XY_THRESHOLD,
        #         "lid_remove_height_margin": DRINK_LID_REMOVE_HEIGHT_MARGIN,
        #         "pour_xy_threshold": DRINK_POUR_XY_THRESHOLD,
        #         "pour_height_threshold": DRINK_POUR_HEIGHT_THRESHOLD,
        #         "pour_tilt_threshold": DRINK_POUR_TILT_THRESHOLD,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidDrinkJointPosVisuomotorEnvCfg(DrinkEnvCfg):
    """Configuration for drink task with Droid robot using joint position control."""

    observations: ObservationsCfg = ObservationsCfg()

    eval_mode = False
    eval_type = None

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

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

        self.gripper_joint_names = ["right_outer_knuckle_joint", "finger_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.005

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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )

        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0/table_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0476,
                horizontal_aperture=2.5452,
                vertical_aperture=1.4721,
                clipping_range=(1e-4, 5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.054620336834421451, -0.4388594867462788, 0.454018368138419),
                rot=(-0.5078392969, 0.7575422903, -0.3175587775, 0.2595868830),
                convention="ros",
            ),
        )

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

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.wrist_cam.height = 720
        self.scene.wrist_cam.width = 1280

        # self.scene.table_cam.height = 720
        # self.scene.table_cam.width = 1280
        # self.scene.wrist_cam.height = 720
        # self.scene.wrist_cam.width = 1280

        self.image_obs_list = ["table_cam", "wrist_cam"]
