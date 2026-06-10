# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import oven_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen",
)

ROOM_INIT_POS = [-4.3, -0.8, -0.6]
OVEN_INIT_POS = [2.3, 1.3, 0.1]

ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
OVEN_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
# Offset from oven articulation root (/oven/oven_E_body_7) to inner placement target
# (/oven/oven_E_body_7/E_plate_31) in source USD.
OVEN_SLOT_OFFSET = (0.0681985, 0.0134229, -0.1067788)

mass_properties = MassPropertiesCfg(
    mass=0.03,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
)


@configclass
class OvenSceneCfg(InteractiveSceneCfg):
    """Configuration for the oven task scene in the kitchen."""

    # Full kitchen as static background geometry.
    interactive_kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROOM_INIT_POS,
            rot=ROOM_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(KITCHEN_ASSET_DIR, "kitchen.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen/model_kitchenware003/E_can3_04",
        spawn=UsdFileCfg(
            usd_path="",  # Prim already exists, but usd_path is required by config
            rigid_props=rigid_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.004,
                # rest_offset=0.001,
            ),
            scale=(0.7, 0.7, 0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[3.0, 1.5, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    # USDA inspection:
    #   articulation root: /World/oven/oven_E_body_7
    #   oven door joint:   RevoluteJoint_oven_door
    oven = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen/oven",
        articulation_root_prim_path="/oven_E_body_7",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            # interactive_kitchen is kinematic; override oven subtree to dynamic.
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROOM_INIT_POS,
            rot=ROOM_INIT_ROT,
            joint_pos={
                # Closed by default (joint limits in USD: [-90 deg, 0 deg]).
                "RevoluteJoint_oven_door": 0.0,
                # Keep both names for compatibility across kitchen USD variants.
                "PrismaticJoint_oven_button": 0.0,
                # "PrismaticJoint_oven_down": 0.0,
            },
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint_oven_door"],
                effort_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "button": ImplicitActuatorCfg(
                # joint_names_expr=["PrismaticJoint_oven_button", "PrismaticJoint_oven_down"],
                joint_names_expr=["PrismaticJoint_oven_button"],
                effort_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
        },
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""

    apply_can_scale = EventTerm(
        func=oven_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("can")},
    )


    apply_can_mass_props = EventTerm(
        func=oven_events.apply_mass_props,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("can"),
            "mass": mass_properties.mass,
            "density": mass_properties.density,
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
        """Subtask terms for the oven task."""

        oven_opened = ObsTerm(
            func=mdp.oven_opened,
            params={
                "oven_cfg": SceneEntityCfg("oven"),
                "door_joint_name": "RevoluteJoint_oven_door",
                "door_open_threshold": -0.01,
            },
        )

        grasp_can = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("can"),
                "diff_threshold": 0.1,
            },
        )

        # can_in_oven = ObsTerm(
        #     func=mdp.can_in_oven,
        #     params={
        #         "can_cfg": SceneEntityCfg("can"),
        #         "oven_cfg": SceneEntityCfg("oven"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "door_joint_name": "RevoluteJoint_oven_door",
        #         "slot_offset": OVEN_SLOT_OFFSET,
        #         "can_in_oven_threshold": 0.12,
        #         "door_open_threshold": 0.25,
        #         "require_oven_open": True,
        #         "require_gripper_open": True,
        #     },
        # )

        # oven_closed = ObsTerm(
        #     func=mdp.oven_closed,
        #     params={
        #         "oven_cfg": SceneEntityCfg("oven"),
        #         "door_joint_name": "RevoluteJoint_oven_door",
        #         "door_closed_threshold": 0.10,
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
    """Termination terms for the oven task."""

    can_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.0,
            "asset_cfg": SceneEntityCfg("can"),
        },
    )

    success = DoneTerm(
        func=mdp.task_done_oven,
        params={
            "can_cfg": SceneEntityCfg("can"),
            "oven_cfg": SceneEntityCfg("oven"),
            "slot_offset": OVEN_SLOT_OFFSET,
            "can_in_oven_threshold": 0.12,
            "door_closed_threshold": 0.10,
        },
    )


@configclass
class OvenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the oven task environment."""

    scene: OvenSceneCfg = OvenSceneCfg(
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
        self.decimation = 6
        self.episode_length_s = 30.0

        self.sim.dt = 1 / (6 * 15)
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
