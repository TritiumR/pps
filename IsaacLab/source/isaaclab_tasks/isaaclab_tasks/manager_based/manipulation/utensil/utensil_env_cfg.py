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
from .mdp import utensil_events

CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)
KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen",
)

# Reuse the capsule task's kitchen counter as the work surface. The robot mounts at the counter facing
# -y, so relative to it: front = -y, right = -x, left = +x. The holder sits directly ahead of the robot,
# the knife to its right, the spatula to its left, all on the counter.
ROOM_INIT_POS = [-4.3, -0.8, -0.6]
ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

HOLDER_INIT_POS = [2.9, 1.35, 0.24]
KNIFE_INIT_POS = [2.7, 1.35, 0.24]
SPATULA_INIT_POS = [3.1, 1.35, 0.24]
ASSET_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
)

utensil_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    linear_damping=0.0,
    angular_damping=0.0,
    sleep_threshold=0.0,
)

utensil_mass_properties = MassPropertiesCfg(
    mass=0.05,  # ~realistic utensil weight; the previous 10 g fled the gripper on contact
)


@configclass
class UtensilSceneCfg(InteractiveSceneCfg):
    """Configuration for the utensil insertion scene on a plain table."""

    interactive_kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROOM_INIT_POS, rot=ROOM_INIT_ROT),
        spawn=UsdFileCfg(usd_path=os.path.abspath(os.path.join(KITCHEN_ASSET_DIR, "kitchen.usd"))),
    )

    knife = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/knife",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "knife", "knife.usd")),
            rigid_props=utensil_body_properties,
            mass_props=utensil_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=(1.5, 1.5, 1.5),
            semantic_tags=[("class", "knife")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=KNIFE_INIT_POS,
            rot=ASSET_INIT_ROT,
        ),
    )

    spatula = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/spatula",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "spatula", "spatula_physics.usd")),
            rigid_props=utensil_body_properties,
            mass_props=utensil_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=(0.004, 0.004, 0.004),  # CC0 Kenney spatula: raw ~68 units long -> ~0.27 m
            semantic_tags=[("class", "spatula")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SPATULA_INIT_POS,
            rot=ASSET_INIT_ROT,
        ),
    )

    holder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/holder",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "pen holder001", "model_pen holder001_0.usd")),
            rigid_props=rigid_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "holder")],
            scale=(1.5, 1.5, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HOLDER_INIT_POS,
            rot=ASSET_INIT_ROT,
        ),
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
        func=utensil_events.deactivate_prim,
        mode="prestartup",
        params={"prim_path_regex": "{ENV_REGEX_NS}/interactive_kitchen/oven"},
    )

    knife_convex_decomposition_collision = EventTerm(
        func=utensil_events.apply_convex_decomposition_collision,
        mode="prestartup",
        params={
            "prim_path_regex": "{ENV_REGEX_NS}/knife/geometry/mesh",
            "hull_vertex_limit": 128,
            "max_convex_hulls": 128,
            "voxel_resolution": 1_000_000,
            "error_percentage": 2.5,
            "shrink_wrap": True,
        },
    )

    spatula_convex_decomposition_collision = EventTerm(
        func=utensil_events.apply_convex_decomposition_collision,
        mode="prestartup",
        params={
            "prim_path_regex": "{ENV_REGEX_NS}/spatula/geometry/mesh",
            "hull_vertex_limit": 128,
            "max_convex_hulls": 128,
            "voxel_resolution": 1_000_000,
            "error_percentage": 2.5,
            "shrink_wrap": True,
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

        knife_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("knife"),
                "diff_threshold": 0.15,
            },
        )

        spatula_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("spatula"),
                "diff_threshold": 0.15,
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
    """Termination terms for the MDP."""

    knife_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.0,
            "asset_cfg": SceneEntityCfg("knife"),
        },
    )

    success = DoneTerm(
        func=mdp.task_done_utensil_inserted,
        params={
            "utensil_cfg": SceneEntityCfg("knife"),
            "holder_cfg": SceneEntityCfg("holder"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            # placeholder thresholds carried over from the holder scene scale; retune from the rendered
            # pen-holder crock + knife geometry (see the utensil_scene render step).
            "insert_xy_threshold": 0.06,
            "min_utensil_height": 0.24,
            "max_utensil_height": 0.42,
            "min_gripper_utensil_distance": 0.12,
        },
    )


@configclass
class UtensilEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the utensil insertion environment."""

    scene: UtensilSceneCfg = UtensilSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
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
