# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
from math import sqrt
from typing import Any
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../../diffusion_policy/reconstruction/asset/water_world",
)

TABLE_INIT_POS = [0.38, -0.12, -0.10]
ASSET_INIT_POS = [0.25, 0.0, -0.05]
ASSET_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

TABLE_INIT_STATE = RigidObjectCfg.InitialStateCfg(
    pos=TABLE_INIT_POS,
    rot=ASSET_INIT_ROT,
)

ASSET_INIT_STATE = RigidObjectCfg.InitialStateCfg(
    pos=ASSET_INIT_POS,
    # rot=[0, 0, -sqrt(2) / 2, sqrt(2) / 2],
    rot=ASSET_INIT_ROT,
)

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
    mass=0.001,  # Mass in kg
    # Alternative: use density instead of mass
    # density=1000.0,  # Density in kg/m³
)


##
# Scene definition
##
@configclass
class WaterAlignSceneCfg(InteractiveSceneCfg):
    """Configuration for the water plant scene aligned with the real-world camera."""

    # Robot Table
    robot_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.3, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.6, 1.0, 1.0),
        ),
    )

    # table (kinematic rigid body to allow pose randomization with collision)
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        init_state=TABLE_INIT_STATE,
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(ASSET_DIR.rsplit("/", 1)[0], "table_move.usd")
            ),
            rigid_props=kinematic_body_properties,
        ),
    )

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cup",
        init_state=ASSET_INIT_STATE,
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(ASSET_DIR, "cup.usd")),
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            semantic_tags=[("class", "cup")],
        ),
    )

    plant = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/plant",
        init_state=ASSET_INIT_STATE,
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(ASSET_DIR, "plant.usd")),
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            semantic_tags=[("class", "plant")],
        ),
    )

    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bowl",
        init_state=ASSET_INIT_STATE,
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(ASSET_DIR, "bowl.usd")),
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_body_properties,
            mass_props=mass_properties,
            semantic_tags=[("class", "bowl")],
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

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cup"),
                "diff_threshold": 0.1,
            },
        )

        # move
        move_1 = ObsTerm(
            func=mdp.object_moved,
            params={
                "object_cfg": SceneEntityCfg("cup"),
                "target_cfg": SceneEntityCfg("plant"),
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

    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": ASSET_INIT_POS[2] - 0.1,
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    # bowl_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={
    #         "minimum_height": ASSET_INIT_POS[2] - 0.1,
    #         "asset_cfg": SceneEntityCfg("bowl"),
    #     },
    # )

    # plant_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={
    #         "minimum_height": ASSET_INIT_POS[2] - 0.1,
    #         "asset_cfg": SceneEntityCfg("plant"),
    #     },
    # )

    # bowl_falling = DoneTerm(
    #     func=mdp.root_rotation_exceeds_threshold,
    #     params={"asset_cfg": SceneEntityCfg("bowl"), "threshold": 0.2},
    # )

    # plant_falling = DoneTerm(
    #     func=mdp.root_rotation_exceeds_threshold,
    #     params={"asset_cfg": SceneEntityCfg("plant"), "threshold": 0.2},
    # )

    bowl_moving = DoneTerm(
        func=mdp.root_velocity_exceeds_threshold,
        params={
            "asset_cfg": SceneEntityCfg("bowl"),
        },
    )

    plant_moving = DoneTerm(
        func=mdp.root_velocity_exceeds_threshold,
        params={
            "asset_cfg": SceneEntityCfg("plant"),
        },
    )

    cup_pouring = DoneTerm(
        func=mdp.root_rotation_exceeds_threshold,
        params={"asset_cfg": SceneEntityCfg("cup")},
    )

    success = DoneTerm(func=mdp.task_done_pour)


@configclass
class WaterAlignEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the water plant environment aligned with the real-world layout."""

    # Scene settings
    scene: WaterAlignSceneCfg = WaterAlignSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
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
