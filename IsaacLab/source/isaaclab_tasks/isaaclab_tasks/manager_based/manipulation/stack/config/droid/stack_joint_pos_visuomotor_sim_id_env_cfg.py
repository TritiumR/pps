# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import stack_joint_pos_env_cfg
from . import stack_joint_pos_sim_env_cfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class EventCfg(stack_joint_pos_env_cfg.EventCfg):  # real-world poses
    """Configuration for events (custom version for different textures and lighting)."""

    # sim lighting

    randomize_light = EventTerm(
        func=franka_stack_events.randomize_scene_lighting_domelight,
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
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )

    randomize_table_visual_material = EventTerm(
        func=franka_stack_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
            "default_texture": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
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
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
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
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidCubeStackVisuomotorSimIDEnvCfg(
    stack_joint_pos_sim_env_cfg.DroidCubeStackSimEnvCfg
):  # sim dynamics
    observations: ObservationsCfg = ObservationsCfg()

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set cameras
        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/wrist_cam",
            update_period=0.0,
            height=144,
            width=256,
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
            height=144,
            width=256,
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
