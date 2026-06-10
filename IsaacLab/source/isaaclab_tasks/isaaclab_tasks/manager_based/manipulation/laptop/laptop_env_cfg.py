# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
from math import sqrt
from typing import Any
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .mdp import laptop_events

CHILDRENROOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/childrenroom",
)

ROOM_INIT_POS = [-4.3, -0.7, -0.6]
ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
# ROOM_INIT_ROT = [0.7071068, 0.0, 0.0, -0.7071068]

TABLE_INIT_POS = [0.80, 0.2, -0.6]
# TABLE_INIT_POS = [0.0, 0.0, 0.0]
TABLE_INIT_ROT = [0.7071068, 0.0, 0.0, -0.7071068]
# TABLE_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
ASSET_INIT_POS = [0.7, 0.1, 0.2]
ASSET_INIT_ROT = [0.7071068, 0.0, 0.0, -0.7071068]

rigid_body_properties = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

mass_properties = MassPropertiesCfg(
    mass=0.0001,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)


##
# Scene definition
##
@configclass
class LaptopSceneCfg(InteractiveSceneCfg):
    """Configuration for the water plant scene aligned with the real-world camera."""

    # Robot Table
    robot_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.40, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.6, 1.0, 1.0),
        ),
    )

    # table (articulated – 8 prismatic drawer/leaf joints)
    table = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CHILDRENROOM_ASSET_DIR, "table_3/model_table_3.usd")
            ),
            scale=(0.9, 1.0, 1.0),
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            # activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TABLE_INIT_POS,
            rot=TABLE_INIT_ROT,
            joint_pos={
                "PrismaticJoint_table_3_left1": 0.0,
                "PrismaticJoint_table_3_left2": 0.0,
                "PrismaticJoint_table_3_left3": 0.0,
                "PrismaticJoint_table_3_left4": 0.0,
                "PrismaticJoint_table_3_right1": 0.25,
                "PrismaticJoint_table_3_right2": 0.0,
                "PrismaticJoint_table_3_right3": 0.0,
                "PrismaticJoint_table_3_right4": 0.0,
            },
        ),
        actuators={
            "table_joints": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint_table_3_.*"],
                effort_limit_sim=None,
                stiffness=0.0,
                damping=None,
            ),
        },
    )

    # laptop (articulated – 1 revolute joint for the lid)
    laptop = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/laptop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CHILDRENROOM_ASSET_DIR, "computer_9/model_computer_9.usd")
            ),
            scale=(0.6, 0.6, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ASSET_INIT_POS,
            rot=ASSET_INIT_ROT,
            joint_pos={
                "RevoluteJoint_computer_9_up": -0.85,
            },
        ),
        actuators={
            "lid": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint_computer_9_up"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=900.0,
            ),
        },
        # actuators={
        #     "lid": ImplicitActuatorCfg(
        #         joint_names_expr=["RevoluteJoint_computer_9_up"],
        #         effort_limit_sim=None,
        #         stiffness=None,
        #         damping=None,
        #     ),
        # },
    )

    # Childrenroom walls and floor
    childrenroom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/childrenroom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROOM_INIT_POS,
            rot=ROOM_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CHILDRENROOM_ASSET_DIR, "childrenroom/model_childrenroom.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # # Invisible ground below the table to prevent objects from falling infinitely.
    # # Shared across all sub-environments (global prim path, not per-env).
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(2000.0, 2000.0, 0.02),
    #         visible=False,
    #         collision_props=sim_utils.schemas.CollisionPropertiesCfg(),
    #     ),
    # )

    # potted plant (static decoration)
    plant = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/plant",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.65, -0.45, 0.15], rot=[0.7071068, 0.0, 0.0, -0.7071068]
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CHILDRENROOM_ASSET_DIR, "item/pottedplant/model_pottedplant.usd")
            ),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # distant_light = AssetBaseCfg(
    # prim_path="/World/distant_light",
    # spawn=sim_utils.DistantLightCfg(
    #     color=(1.0, 1.0, 1.0),
    #     intensity=2000.0,
    #     angle=0.53,
    # ),
    # init_state=AssetBaseCfg.InitialStateCfg(
    #     rot=(0.866, 0.5, 0.0, 0.0),  # tilted 60 degrees from zenith
    # ),
    # )


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

    apply_laptop_mass_props = EventTerm(
        func=laptop_events.apply_mass_props,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("laptop"),
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

        laptop_grasped = ObsTerm(
            func=mdp.laptop_is_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "laptop_cfg": SceneEntityCfg("laptop"),
                "diff_threshold": 0.25,
            },
        )

        laptop_placed_in_drawer = ObsTerm(
            func=mdp.laptop_is_in_drawer,
            params={
                "laptop_cfg": SceneEntityCfg("laptop"),
                "table_cfg": SceneEntityCfg("table"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        laptop_closed = ObsTerm(
            func=mdp.laptop_is_closed,
            params={
                "laptop_cfg": SceneEntityCfg("laptop"),
                "lid_joint_name": "RevoluteJoint_computer_9_up",
                # "closed_threshold": -1.2,
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

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)

    laptop_dropping_off_table = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": ASSET_INIT_POS[2] - 0.5,
            "asset_cfg": SceneEntityCfg("laptop"),
        },
    )

    success = DoneTerm(func=mdp.task_done_laptop)


@configclass
class LaptopEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the water plant environment aligned with the real-world layout."""

    # Scene settings
    scene: LaptopSceneCfg = LaptopSceneCfg(
        num_envs=4096, env_spacing=20, replicate_physics=False
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
        # self.sim.render_interval = self.decimation
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
