# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
from .mdp import pen_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/smalllivingroom",
)
CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)

DESK_LAPTOP_POS = (7.556153774261475, 0.052134789526462555, 0.7891297936439514)

PEN_HOLDER001_INIT_POS = (DESK_LAPTOP_POS[0] + 0.12, DESK_LAPTOP_POS[1] - 0.28, DESK_LAPTOP_POS[2])
PEN_HOLDER001_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

PEN_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1] + 0.10, DESK_LAPTOP_POS[2] + 0.03)
PEN_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

SCISSORS010_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1], DESK_LAPTOP_POS[2] + 0.03)
SCISSORS010_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

PEN_GRASP_DIFF_THRESHOLD = 0.08
PEN_HOLDER_CENTER_XY_THRESHOLD = 0.07
PEN_HOLDER_INSERTION_PROBE_DEPTH = 0.11
PEN_HOLDER_INSERTION_MIN_HEIGHT = 0.0
PEN_HOLDER_INSERTION_MAX_HEIGHT = 0.05
PEN_HOLDER_MAX_AXIS_DEVIATION_RAD = math.radians(35.0)

# Backward-compatible aliases for configs that still import the previous success-threshold names.
PEN_HOLDER_XY_THRESHOLD = PEN_HOLDER_CENTER_XY_THRESHOLD
PEN_HOLDER_MIN_HEIGHT_OFFSET = PEN_HOLDER_INSERTION_MIN_HEIGHT
PEN_HOLDER_MAX_HEIGHT_OFFSET = PEN_HOLDER_INSERTION_MAX_HEIGHT

pen_mass_properties = MassPropertiesCfg(
    mass=0.01,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)

PEN_CONTACT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)

mass_properties = MassPropertiesCfg(
    mass=0.1,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)

SCISSOR_JOINT_DAMPING = 10.0

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=0.5,
)

pen_holder_rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)


@configclass
class PenSceneCfg(InteractiveSceneCfg):
    """Configuration for the pen task scene in the living room."""

    # Full room as static background geometry.
    interactive_smalllivingroom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_smalllivingroom",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(KITCHEN_ASSET_DIR, "Interactive_smalllivingroom.usd")
            ),
            rigid_props=kinematic_body_properties,
        ),
    )

    pen_holder001 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pen_holder001",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "pen holder001", "model_pen holder001_0.usd")
            ),
            scale=(1.0, 1.0, 1.0),
            rigid_props=pen_holder_rigid_body_properties,
            # mass_props=pen_holder_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PEN_HOLDER001_INIT_POS),
            rot=list(PEN_HOLDER001_INIT_ROT),
        ),
    )

    pen = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pen",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "pen", "model_pen_0.usd")
            ),
            scale=(1.3, 1.8, 1.8),
            rigid_props=rigid_body_properties,
            mass_props=pen_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PEN_INIT_POS),
            rot=list(PEN_INIT_ROT),
        ),
    )

    scissors010 = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/scissors010",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "scissors010", "scissors_backup.usd")
            ),
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=list(SCISSORS010_INIT_POS),
            rot=list(SCISSORS010_INIT_ROT),
            joint_pos={},
            joint_vel={},
        ),
        actuators={
            "scissor_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=SCISSOR_JOINT_DAMPING,
            ),
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""
    pen_contact_material = EventTerm(
        func=pen_events.bind_rigid_body_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("pen"),
            "material_cfg": PEN_CONTACT_MATERIAL,
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
        """Subtask terms for the pen task."""

        grasp_pen = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pen"),
                "diff_threshold": PEN_GRASP_DIFF_THRESHOLD,
            },
        )

        # pen_placed_in_holder = ObsTerm(
        #     func=mdp.task_done_pen,
        #     params={
        #         "pen_cfg": SceneEntityCfg("pen"),
        #         "holder_cfg": SceneEntityCfg("pen_holder001"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "center_xy_threshold": PEN_HOLDER_CENTER_XY_THRESHOLD,
        #         "insertion_probe_depth": PEN_HOLDER_INSERTION_PROBE_DEPTH,
        #         "insertion_min_height": PEN_HOLDER_INSERTION_MIN_HEIGHT,
        #         "insertion_max_height": PEN_HOLDER_INSERTION_MAX_HEIGHT,
        #         "max_axis_deviation_rad": PEN_HOLDER_MAX_AXIS_DEVIATION_RAD,
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
    """Termination terms for the pen task."""

    pen_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )

    scissors010_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("scissors010"),
        },
    )

    success = DoneTerm(
        func=mdp.task_done_pen,
        params={
            "pen_cfg": SceneEntityCfg("pen"),
            "holder_cfg": SceneEntityCfg("pen_holder001"),
            "robot_cfg": SceneEntityCfg("robot"),
            "center_xy_threshold": PEN_HOLDER_CENTER_XY_THRESHOLD,
            "insertion_probe_depth": PEN_HOLDER_INSERTION_PROBE_DEPTH,
            "insertion_min_height": PEN_HOLDER_INSERTION_MIN_HEIGHT,
            "insertion_max_height": PEN_HOLDER_INSERTION_MAX_HEIGHT,
            "max_axis_deviation_rad": PEN_HOLDER_MAX_AXIS_DEVIATION_RAD,
        },
    )


@configclass
class PenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pen task environment."""

    scene: PenSceneCfg = PenSceneCfg(
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
