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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp

LIVINGROOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/largelivingroom",
)
CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)

BIG_TABLE_CENTER_POS = (3.6874645197200904, 3.846993282921349, 0.5005000531284776)
DRINK_SET_INIT_POS = (BIG_TABLE_CENTER_POS[0] - 0.10, BIG_TABLE_CENTER_POS[1], BIG_TABLE_CENTER_POS[2])
DRINK_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
DRINK_BODY_TOP_Z_OFFSET = 0.24250000715255737 * 0.82
DRINK_LID_INIT_POS = (
    DRINK_SET_INIT_POS[0],
    DRINK_SET_INIT_POS[1],
    DRINK_SET_INIT_POS[2] + DRINK_BODY_TOP_Z_OFFSET,
)
DRINK_LID_INIT_ROT = DRINK_INIT_ROT
CUP_INIT_POS = (BIG_TABLE_CENTER_POS[0] + 0.10, BIG_TABLE_CENTER_POS[1], BIG_TABLE_CENTER_POS[2])
CUP_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

DRINK_GRASP_DIFF_THRESHOLD = 0.06
DRINK_GRASP_DIFF_Z = 0.10
DRINK_LID_GRASP_DIFF_THRESHOLD = 0.03
DRINK_LID_REMOVE_XY_THRESHOLD = 0.05
DRINK_LID_REMOVE_HEIGHT_MARGIN = 0.04
DRINK_POUR_XY_THRESHOLD = 0.1
DRINK_POUR_HEIGHT_THRESHOLD = 0.2
DRINK_POUR_TILT_THRESHOLD = 0.7853981633974483
ROTATION_FROM_Z_AXIS_THRESHOLD = 1.4

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
class DrinkSceneCfg(InteractiveSceneCfg):
    """Configuration for the drink task scene in the large living room."""

    interactive_largelivingroom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_largelivingroom",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(LIVINGROOM_ASSET_DIR, "Interactive_largelivingroom.usd")
            ),
            rigid_props=kinematic_body_properties,
        ),
    )

    drink_set = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/drink_set",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(DRINK_SET_INIT_POS),
            rot=list(DRINK_INIT_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "drink087", "model_drink087.usd")
            ),
            scale=(0.8, 0.8, 0.8),
            rigid_props=rigid_body_properties,
        ),
    )

    drink = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/drink_set/E_Component86_1",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(DRINK_SET_INIT_POS),
            rot=list(DRINK_INIT_ROT),
        ),
    )

    drink_lid = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/drink_set/lid_2",
        spawn=UsdFileCfg(
            usd_path="",
            scale=(1.0, 1.0, 4.0),
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(DRINK_LID_INIT_POS),
            rot=list(DRINK_LID_INIT_ROT),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cup",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "cup", "model_papercup.usd")
            ),
            rigid_props=rigid_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(CUP_INIT_POS),
            rot=list(CUP_INIT_ROT),
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

    pass


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
        """Subtask terms for the drink task."""

        grasp_drink_lid = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("drink_lid"),
                "diff_threshold": DRINK_LID_GRASP_DIFF_THRESHOLD,
            },
        )

        drink_lid_removed = ObsTerm(
            func=mdp.drink_lid_removed,
            params={
                "drink_cfg": SceneEntityCfg("drink"),
                "lid_cfg": SceneEntityCfg("drink_lid"),
                "body_top_z_offset": DRINK_BODY_TOP_Z_OFFSET,
                "xy_threshold": DRINK_LID_REMOVE_XY_THRESHOLD,
                "extra_height_threshold": DRINK_LID_REMOVE_HEIGHT_MARGIN,
            },
        )

        grasp_drink = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("drink"),
                "diff_threshold": DRINK_GRASP_DIFF_THRESHOLD,
                "diff_z": DRINK_GRASP_DIFF_Z,
            },
        )

        # drink_poured_into_cup = ObsTerm(
        #     func=mdp.task_done_drink,
        #     params={
        #         "drink_cfg": SceneEntityCfg("drink"),
        #         "lid_cfg": SceneEntityCfg("drink_lid"),
        #         "cup_cfg": SceneEntityCfg("cup"),
        #         "body_top_z_offset": DRINK_BODY_TOP_Z_OFFSET,
        #         "lid_remove_xy_threshold": DRINK_LID_REMOVE_XY_THRESHOLD,
        #         "lid_remove_height_margin": DRINK_LID_REMOVE_HEIGHT_MARGIN,
        #         "pour_xy_threshold": DRINK_POUR_XY_THRESHOLD,
        #         "pour_height_threshold": DRINK_POUR_HEIGHT_THRESHOLD,
        #         "pour_tilt_threshold": DRINK_POUR_TILT_THRESHOLD,
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
    """Termination terms for the drink task."""

    drink_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("drink"),
        },
    )

    # drink_lid_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.2,
    #         "asset_cfg": SceneEntityCfg("drink_lid"),
    #     },
    # )

    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    drink_rotated_from_z_axis = DoneTerm(
        func=mdp.asset_rotated_from_z_axis,
        params={
            "asset_cfg": SceneEntityCfg("drink"),
            "threshold_rad": ROTATION_FROM_Z_AXIS_THRESHOLD,
        },
    )

    cup_rotated_from_z_axis = DoneTerm(
        func=mdp.asset_rotated_from_z_axis,
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "threshold_rad": ROTATION_FROM_Z_AXIS_THRESHOLD,
        },
    )

    success = DoneTerm(
        func=mdp.task_done_drink,
        params={
            "drink_cfg": SceneEntityCfg("drink"),
            "lid_cfg": SceneEntityCfg("drink_lid"),
            "cup_cfg": SceneEntityCfg("cup"),
            "body_top_z_offset": DRINK_BODY_TOP_Z_OFFSET,
            "lid_remove_xy_threshold": DRINK_LID_REMOVE_XY_THRESHOLD,
            "lid_remove_height_margin": DRINK_LID_REMOVE_HEIGHT_MARGIN,
            "pour_xy_threshold": DRINK_POUR_XY_THRESHOLD,
            "pour_height_threshold": DRINK_POUR_HEIGHT_THRESHOLD,
            "pour_tilt_threshold": DRINK_POUR_TILT_THRESHOLD,
        },
    )


@configclass
class DrinkEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drink task environment."""

    scene: DrinkSceneCfg = DrinkSceneCfg(
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
