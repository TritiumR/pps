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
from .mdp import scissor_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/smalllivingroom",
)
CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)

DESK_LAPTOP_POS = (7.556153774261475, 0.052134789526462555, 0.7891297936439514)

PEN_HOLDER001_INIT_POS = (DESK_LAPTOP_POS[0] + 0.12, DESK_LAPTOP_POS[1] - 0.28, DESK_LAPTOP_POS[2] + 0.03)
PEN_HOLDER001_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

PEN_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1] + 0.10, DESK_LAPTOP_POS[2] + 0.03)
PEN_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

SCISSORS008_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1], DESK_LAPTOP_POS[2] + 0.03)
SCISSORS008_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

SCISSORS010_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1], DESK_LAPTOP_POS[2] + 0.03)
SCISSORS010_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

SCISSOR_GRASP_DIFF_THRESHOLD = 0.10
PEN_GRASP_DIFF_THRESHOLD = 0.08
PEN_HOLDER_XY_THRESHOLD = 0.05
PEN_HOLDER_MIN_HEIGHT_OFFSET = 0.06
PEN_HOLDER_MAX_HEIGHT_OFFSET = 0.24
SCISSOR_HOLDER_MIN_XY_DISTANCE = 0.18

mass_properties = MassPropertiesCfg(
    mass=0.1,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)

pen_holder_mass_properties = MassPropertiesCfg(
    mass=2.0,
)

holder_scissor_contact_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
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
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=0.5,
)


@configclass
class ScissorSceneCfg(InteractiveSceneCfg):
    """Configuration for the scissor task scene in the living room."""

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
            rigid_props=pen_holder_rigid_body_properties,
            mass_props=pen_holder_mass_properties,
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
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PEN_INIT_POS),
            rot=list(PEN_INIT_ROT),
        ),
    )

    scissors008 = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/scissors008",
        articulation_root_prim_path="/E_body_1",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "scissors008", "model_scissors_22.usd")
            ),
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=list(SCISSORS008_INIT_POS),
            rot=list(SCISSORS008_INIT_ROT),
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
    pass

    # apply_scissors_mass = EventTerm(
    #     func=scissor_events.apply_mass_props,
    #     mode="prestartup",
    #     params={
    #         "mass": 0.03,
    #         "asset_cfg": SceneEntityCfg("scissors"),
    #     },
    # )

    # randomize_box_pose = EventTerm(
    #     func=scissor_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": BOX_RANDOMIZE_POSE_RANGE,
    #         "asset_cfgs": [SceneEntityCfg("box")],
    #     },
    # )

    # randomize_scissors_pose = EventTerm(
    #     func=scissor_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": SCISSORS_RANDOMIZE_POSE_RANGE,
    #         "asset_cfgs": [SceneEntityCfg("scissors")],
    #     },
    # )


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
        """Subtask terms for the scissor task."""

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

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the scissor task."""

    pen_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )

    scissors008_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("scissors008"),
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


@configclass
class ScissorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the scissor task environment."""

    scene: ScissorSceneCfg = ScissorSceneCfg(
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
        self.decimation = 14
        self.episode_length_s = 30.0

        self.sim.dt = 1 / (self.decimation * 15)
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
