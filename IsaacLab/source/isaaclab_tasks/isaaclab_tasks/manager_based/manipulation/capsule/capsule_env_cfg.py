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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import capsule_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen",
)
CAPSULE_ASSET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/capsule coffee machine002/model_capsule_coffee_maker_3.usd",
)

ROOM_INIT_POS = [-4.3, -0.8, -0.6]
ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

CAPSULE_INIT_POS = [3.0, 1.2, 0.2]
CAPSULE_INIT_ROT = [0.0, 0.0, 0.0, 1.0]
CAPSULE_SHELL_JOINT = "RevoluteJoint_capsule_coffee_maker_3_up"
CAPSULE_LEFT_BUTTON_JOINT = "PrismaticJoint_capsule_coffee_maker_3_left"
CAPSULE_RIGHT_BUTTON_JOINT = "PrismaticJoint_capsule_coffee_maker_3_right"
CAPSULE_SHELL_LINK_Y_SCALE = 1.25

mass_properties = MassPropertiesCfg(
    mass=0.03,
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
class CapsuleSceneCfg(InteractiveSceneCfg):
    """Configuration for the capsule coffee machine task scene in the kitchen."""

    interactive_kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROOM_INIT_POS,
            rot=ROOM_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(KITCHEN_ASSET_DIR, "kitchen.usd")),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen/model_kitchenware003/E_can3_04",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=(0.55, 0.55, 0.55),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[3.0, 1.5, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    capsule = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/capsule",
        articulation_root_prim_path="/E_body_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(CAPSULE_ASSET_PATH),
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
            semantic_tags=[("class", "capsule")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=CAPSULE_INIT_POS,
            rot=CAPSULE_INIT_ROT,
            joint_pos={
                CAPSULE_SHELL_JOINT: 0.0,
                CAPSULE_LEFT_BUTTON_JOINT: 0.0,
                CAPSULE_RIGHT_BUTTON_JOINT: 0.0,
            },
        ),
        actuators={
            "shell": ImplicitActuatorCfg(
                joint_names_expr=[CAPSULE_SHELL_JOINT],
                effort_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "buttons": ImplicitActuatorCfg(
                joint_names_expr=[CAPSULE_LEFT_BUTTON_JOINT, CAPSULE_RIGHT_BUTTON_JOINT],
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

    deactivate_kitchen_oven = EventTerm(
        func=capsule_events.deactivate_prim,
        mode="prestartup",
        params={"prim_path_regex": "{ENV_REGEX_NS}/interactive_kitchen/oven"},
    )

    apply_can_scale = EventTerm(
        func=capsule_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("can")},
    )

    apply_can_mass_props = EventTerm(
        func=capsule_events.apply_mass_props,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("can"),
            "mass": mass_properties.mass,
            "density": mass_properties.density,
        },
    )

    scale_capsule_shell_link = EventTerm(
        func=capsule_events.set_prim_local_scale,
        mode="prestartup",
        params={
            "prim_path_regex": "{ENV_REGEX_NS}/capsule/E_shell_8",
            "scale": (1.0, CAPSULE_SHELL_LINK_Y_SCALE, 1.0),
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
        """Observations for subtask group."""

        open_coffee_lid = ObsTerm(
            func=mdp.coffee_lid_opened,
            params={
                "capsule_cfg": SceneEntityCfg("capsule"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        grasp_pod = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("can"),
                "diff_threshold": 0.1,
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
    """Termination terms for the capsule coffee machine task."""

    success = DoneTerm(
        func=mdp.task_done_capsule,
        params={
            "capsule_cfg": SceneEntityCfg("capsule"),
            "can_cfg": SceneEntityCfg("can"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CapsuleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the capsule coffee machine task environment."""

    scene: CapsuleSceneCfg = CapsuleSceneCfg(
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
