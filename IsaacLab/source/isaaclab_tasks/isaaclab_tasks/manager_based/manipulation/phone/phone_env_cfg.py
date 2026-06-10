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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/smalllivingroom",
)
CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)

DESK_LAPTOP_POS = (7.556153774261475, 0.052134789526462555, 0.7891297936439514)

PHONE_1_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1] + 0.10, DESK_LAPTOP_POS[2] + 0.01)
PHONE_1_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

PHONE_2_INIT_POS = (DESK_LAPTOP_POS[0], DESK_LAPTOP_POS[1], DESK_LAPTOP_POS[2] + 0.03)
PHONE_2_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

PHONE_GRASP_DIFF_THRESHOLD = 0.08
PHONE_PICKUP_HEIGHT_THRESHOLD = PHONE_1_INIT_POS[2] + 0.05

phone_mass_properties = MassPropertiesCfg(
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
    max_depenetration_velocity=0.5,
)


@configclass
class PhoneSceneCfg(InteractiveSceneCfg):
    """Configuration for the phone task scene in the living room."""

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

    phone_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/phone_1",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "phone", "phone.usd")),
            scale=(0.12, 0.45, 0.12),
            rigid_props=rigid_body_properties,
            mass_props=phone_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PHONE_1_INIT_POS),
            rot=list(PHONE_1_INIT_ROT),
        ),
    )

    phone_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/phone_2",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "phone", "phone.usd")),
            scale=(0.12, 0.45, 0.12),
            rigid_props=rigid_body_properties,
            mass_props=phone_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(PHONE_2_INIT_POS),
            rot=list(PHONE_2_INIT_ROT),
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
        """Subtask terms for the phone task."""

        phone_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("phone_1"),
                "diff_threshold": PHONE_GRASP_DIFF_THRESHOLD,
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
    """Termination terms for the phone task."""

    phone_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("phone_1"),
        },
    )

    phone_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("phone_2"),
        },
    )

    success = DoneTerm(
        func=mdp.task_done_phone,
        params={
            "phone_cfg": SceneEntityCfg("phone_1"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "height_threshold": PHONE_PICKUP_HEIGHT_THRESHOLD,
            "diff_threshold": PHONE_GRASP_DIFF_THRESHOLD,
        },
    )


@configclass
class PhoneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the phone task environment."""

    scene: PhoneSceneCfg = PhoneSceneCfg(
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
