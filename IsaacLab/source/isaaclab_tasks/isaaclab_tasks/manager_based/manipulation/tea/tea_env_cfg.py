# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import tea_events

SCENE_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/diningroom",
)

DININGROOM_INIT_POS = (0.0, 0.0, 0.0)
DININGROOM_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
TEA_OBJECT_SCALE = (1.4, 1.4, 1.4)
TEAPOT_OBJECT_SCALE = (1.0, 1.0, 1.0)
TEAPOT_HANDLE_MESH_PRIM_PATH = (
    "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teapot_5/P_c680523765f8dbb9"
)
TEAPOT_HANDLE_LOCAL_BOUNDS_MIN = (-0.02, 0.0, -0.08)
TEAPOT_HANDLE_LOCAL_BOUNDS_MAX = (0.02, 0.065, -0.025)
TEAPOT_HANDLE_SCALE = (1.5, 1.0, 1.5)
TEAPOT_HANDLE_SCALE_CENTER = (0.0, 0.0378, -0.0584)
TEAPOT_CENTER_OF_MASS = (0.0, 0.005, -0.01)

TEAPOT_GRASP_DIFF_THRESHOLD = 0.10
TEAPOT_MOUTH_LOCAL_OFFSET_BASE = (0.0, 0.05, 0.062)
TEAPOT_MOUTH_LOCAL_OFFSET = tuple(
    base * scale for base, scale in zip(TEAPOT_MOUTH_LOCAL_OFFSET_BASE, TEAPOT_OBJECT_SCALE, strict=True)
)
TEAPOT_MOUTH_TEACUP_XY_THRESHOLD = 0.10
TEAPOT_ROLL_THRESHOLD_RAD = math.radians(30.0)
TEAPOT_MAX_RELATIVE_ROLL_RAD = math.pi / 4.0

TEAPOT_MASS_PROPERTIES = MassPropertiesCfg(mass=0.01)
TEACUP_MASS_PROPERTIES = MassPropertiesCfg(mass=0.12)
TEAPOT_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=0.5,
)


@configclass
class TeaSceneCfg(InteractiveSceneCfg):
    """Configuration for the tea task scene in the dining room."""

    interactive_diningroom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_diningroom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(DININGROOM_INIT_POS),
            rot=list(DININGROOM_INIT_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(SCENE_ASSET_DIR, "Interactive_diningroom.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    teapot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teapot_5",
        spawn=UsdFileCfg(
            usd_path="",
            scale=TEAPOT_OBJECT_SCALE,
            rigid_props=rigid_body_properties,
            mass_props=TEAPOT_MASS_PROPERTIES,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    teacup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teacup005_20",
        spawn=UsdFileCfg(
            usd_path="",
            scale=TEA_OBJECT_SCALE,
            rigid_props=rigid_body_properties,
            mass_props=TEACUP_MASS_PROPERTIES,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""

    scale_teapot = EventTerm(
        func=tea_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("teapot")},
    )

    scale_teacup = EventTerm(
        func=tea_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("teacup")},
    )

    scale_teapot_handle = EventTerm(
        func=tea_events.scale_mesh_points_in_local_bounds,
        mode="prestartup",
        params={
            "mesh_prim_path_regex": TEAPOT_HANDLE_MESH_PRIM_PATH,
            "bounds_min": TEAPOT_HANDLE_LOCAL_BOUNDS_MIN,
            "bounds_max": TEAPOT_HANDLE_LOCAL_BOUNDS_MAX,
            "scale": TEAPOT_HANDLE_SCALE,
            "center": TEAPOT_HANDLE_SCALE_CENTER,
        },
    )

    teapot_convex_decomposition_collision = EventTerm(
        func=tea_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("teapot"),
            "max_convex_hulls": 64,
            "hull_vertex_limit": 64,
            "voxel_resolution": 1_000_000,
        },
    )

    teapot_physics_material = EventTerm(
        func=tea_events.bind_rigid_body_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("teapot"),
            "material_cfg": TEAPOT_PHYSICS_MATERIAL,
            "material_name": "teaPotPhysicsMaterial",
        },
    )

    teapot_center_of_mass = EventTerm(
        func=tea_events.set_center_of_mass,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("teapot"),
            "center_of_mass": TEAPOT_CENTER_OF_MASS,
        },
    )

    deactivate_other_teacups = EventTerm(
        func=tea_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": (
                "/World/envs/env_.*/interactive_diningroom/model_TeaTable/"
                "E_teacup(_13|001_16|002_17|003_18|004_19)"
            ),
        },
    )

    deactivate_teatable_chairs = EventTerm(
        func=tea_events.deactivate_prim,
        mode="prestartup",
        params={
            "prim_path_regex": (
                "/World/envs/env_.*/interactive_diningroom/model_TeaTableChair(_01)?"
            ),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_action = ObsTerm(func=mdp.last_droid_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask terms for the tea task."""

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
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the tea task."""

    teapot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("teapot"),
        },
    )

    teacup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("teacup"),
        },
    )

    teapot_over_rolled = DoneTerm(
        func=mdp.teapot_relative_roll_exceeds_max,
        params={
            "teapot_cfg": SceneEntityCfg("teapot"),
            "max_relative_roll_rad": TEAPOT_MAX_RELATIVE_ROLL_RAD,
        },
    )

    success = DoneTerm(
        func=mdp.task_done_tea,
        params={
            "teapot_cfg": SceneEntityCfg("teapot"),
            "teacup_cfg": SceneEntityCfg("teacup"),
            "mouth_offset": TEAPOT_MOUTH_LOCAL_OFFSET,
            "xy_threshold": TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
            "min_roll_rad": TEAPOT_ROLL_THRESHOLD_RAD,
        },
    )


@configclass
class TeaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the tea task environment."""

    scene: TeaSceneCfg = TeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 8
        self.episode_length_s = 30.0

        self.sim.dt = 1 / (self.decimation * 15)
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
