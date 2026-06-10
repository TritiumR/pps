# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg  # Uncomment when adding actuators
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .mdp import plate_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen",
)

ROOM_INIT_POS = [-4.3, -0.8, -0.6]
ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

# Sink area position (where washing happens)
SINK_POS = [1.85, 4.90, 0.26]

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
)

plate_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    linear_damping=0.0,
    angular_damping=0.0,
    sleep_threshold=0.0,
)

mass_properties = MassPropertiesCfg(
    mass=0.01,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)


##
# Scene definition
##
@configclass
class PlateSceneCfg(InteractiveSceneCfg):
    """Configuration for the plate scene in the kitchen."""

    # # Robot Table
    # robot_table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/RobotTable",
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=[-0.40, 0, 0], rot=[0.707, 0, 0, 0.707]
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(0.3, 1.0, 1.0),
    #     ),
    # )

    # Full Interactive Kitchen - loads the entire kitchen scene
    # This includes walls, floor, cabinets, fridge, oven, and various objects
    interactive_kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROOM_INIT_POS,
            rot=ROOM_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(KITCHEN_ASSET_DIR, "Interactive_kitchen.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Plate rack - reference from within the loaded kitchen USD
    # The plate rack is inside Interactive_kitchen.usd at path /interactive_kitchen/plateRack
    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen/platerack",
        spawn=UsdFileCfg(
            usd_path="",  # Prim already exists, but usd_path is required by config
            rigid_props=rigid_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.002,
                # rest_offset=0.00,
            ),
            scale=(0.8, 0.8, 0.8),
        ),
    )

    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen/model_kitchenware005/E_plate_22/E_plate2_30",
        spawn=UsdFileCfg(
            usd_path="",  # Prim already exists, but usd_path is required by config
            rigid_props=plate_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.002,
                # rest_offset=0.00,
            ),
            scale=(0.6, 0.6, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.6, 5.0, 0.4],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""

    apply_plate_scale = EventTerm(
        func=plate_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("plate")},
    )

    apply_rack_scale = EventTerm(
        func=plate_events.apply_scale_from_spawn_cfg,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("rack")},
    )

    apply_plate_mass_props = EventTerm(
        func=plate_events.apply_mass_props,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
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
        """Observations for subtask group."""

        plate_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("plate"),
                "diff_threshold": 0.1,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    plate_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.0,
            "asset_cfg": SceneEntityCfg("plate"),
        },
    )

    success = DoneTerm(func=mdp.task_done_plate)


@configclass
class PlateEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the plate environment."""

    # Scene settings
    scene: PlateSceneCfg = PlateSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 1 / (6 * 15)  # control frequency: 15Hz, decimation: 6
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
