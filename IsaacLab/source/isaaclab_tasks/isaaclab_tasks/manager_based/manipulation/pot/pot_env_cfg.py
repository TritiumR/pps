# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
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
from .mdp import pot_events

SCENE_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen_with_parlor",
)
CUSTOM_ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../../../../../assets")

PEN_TASK_TARGET_POT_POS = (7.556153774261475, 0.15213478952646255, 0.8191297936439514)
AUTHORED_POT_POS = (1.937098979966911, 5.049524784088135, 1.016234040260315)

ROOM_INIT_POS = (
    PEN_TASK_TARGET_POT_POS[0] - AUTHORED_POT_POS[0],
    PEN_TASK_TARGET_POT_POS[1] - AUTHORED_POT_POS[1],
    PEN_TASK_TARGET_POT_POS[2] - AUTHORED_POT_POS[2],
)
ROOM_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

POT_INIT_POS = PEN_TASK_TARGET_POT_POS
POT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
EGG_INIT_POS = (POT_INIT_POS[0] + 0.85, POT_INIT_POS[1] - 0.05, POT_INIT_POS[2] - 0.03)
EGG_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

POT_GRASP_DIFF_THRESHOLD = 0.10
COVER_GRASP_DIFF_THRESHOLD = 0.25
EGG_GRASP_DIFF_THRESHOLD = 0.08
POT_COVER_REMOVE_XY_THRESHOLD = 0.05
POT_COVER_REMOVE_HEIGHT_THRESHOLD = 0.06
EGG_POT_XY_THRESHOLD = 0.12
EGG_POT_Z_MIN_THRESHOLD = 0.00
EGG_POT_Z_MAX_THRESHOLD = 0.05
COVER_SCALE = (1.05, 1.05, 1.4)

pot_mass_properties = MassPropertiesCfg(mass=0.6)
cover_mass_properties = MassPropertiesCfg(mass=0.01)
egg_mass_properties = MassPropertiesCfg(mass=0.05)

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=0.5,
)

egg_rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=10.0,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
)

kinematic_rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=False,
)

@configclass
class PotSceneCfg(InteractiveSceneCfg):
    """Configuration for the pot task scene in the kitchen with parlor."""

    interactive_kitchen_with_parlor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen_with_parlor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(ROOM_INIT_POS),
            rot=list(ROOM_INIT_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(SCENE_ASSET_DIR, "Interactive_kitchen_with_parlor.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    pot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen_with_parlor/model_kitchenware006/E_pot1_1",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=kinematic_rigid_body_properties,
            mass_props=pot_mass_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )

    cover = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen_with_parlor/model_kitchenware006/E_cover_2",
        spawn=UsdFileCfg(
            usd_path="",
            scale=COVER_SCALE,
            rigid_props=rigid_body_properties,
            mass_props=cover_mass_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    egg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/egg",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "egg", "model_egg.usd")
            ),
            rigid_props=egg_rigid_body_properties,
            mass_props=egg_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(EGG_INIT_POS), rot=list(EGG_INIT_ROT)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""

    scale_cover = EventTerm(
        func=pot_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("cover")},
    )

    refine_pot_collision = EventTerm(
        func=pot_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("pot"),
            "hull_vertex_limit": 128,
            "max_convex_hulls": 128,
            "min_thickness": 0.002,
            "voxel_resolution": 2_000_000,
            "error_percentage": 1.0,
            "shrink_wrap": True,
        },
    )

    refine_egg_collision = EventTerm(
        func=pot_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("egg"),
            "hull_vertex_limit": 64,
            "max_convex_hulls": 16,
            "min_thickness": 0.001,
            "voxel_resolution": 1_000_000,
            "error_percentage": 1.0,
            "shrink_wrap": True,
            "contact_offset": 0.01,
            "rest_offset": 0.0,
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
        """Subtask terms for the pot task."""

        grasp_cover = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cover"),
                "diff_threshold": COVER_GRASP_DIFF_THRESHOLD,
            },
        )

        lid_removed = ObsTerm(
            func=mdp.lid_removed_from_pot,
            params={
                "pot_cfg": SceneEntityCfg("pot"),
                "cover_cfg": SceneEntityCfg("cover"),
                "robot_cfg": SceneEntityCfg("robot"),
                "xy_threshold": POT_COVER_REMOVE_XY_THRESHOLD,
                "height_threshold": POT_COVER_REMOVE_HEIGHT_THRESHOLD,
            },
        )

        grasp_egg = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("egg"),
                "diff_threshold": EGG_GRASP_DIFF_THRESHOLD,
            },
        )

        # egg_in_pot = ObsTerm(
        #     func=mdp.egg_in_pot,
        #     params={
        #         "pot_cfg": SceneEntityCfg("pot"),
        #         "egg_cfg": SceneEntityCfg("egg"),
        #         "xy_threshold": EGG_POT_XY_THRESHOLD,
        #         "z_threshold": EGG_POT_Z_THRESHOLD,
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
    """Termination terms for the pot task."""

    pot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("pot"),
        },
    )

    # cover_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.5,
    #         "asset_cfg": SceneEntityCfg("cover"),
    #     },
    # )

    egg_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("egg"),
        },
    )

    success = DoneTerm(
        func=mdp.task_done_pot,
        params={
            "pot_cfg": SceneEntityCfg("pot"),
            "cover_cfg": SceneEntityCfg("cover"),
            "egg_cfg": SceneEntityCfg("egg"),
            "robot_cfg": SceneEntityCfg("robot"),
            "lid_xy_threshold": POT_COVER_REMOVE_XY_THRESHOLD,
            "lid_height_threshold": POT_COVER_REMOVE_HEIGHT_THRESHOLD,
            "egg_xy_threshold": EGG_POT_XY_THRESHOLD,
            "egg_z_min_threshold": EGG_POT_Z_MIN_THRESHOLD,
            "egg_z_max_threshold": EGG_POT_Z_MAX_THRESHOLD,
        },
    )


@configclass
class PotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pot task environment."""

    scene: PotSceneCfg = PotSceneCfg(
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

        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
