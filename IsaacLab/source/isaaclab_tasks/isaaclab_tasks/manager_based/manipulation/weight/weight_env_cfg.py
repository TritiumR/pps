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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import weight_events

KITCHEN_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen",
)
CUSTOM_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets",
)

ROOM_INIT_POS = [-4.3, -0.8, -0.6]
ROOM_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

SHIFT_X = -2.5

SCALE_SET_INIT_POS = [3.0 + SHIFT_X, 1.3, 0.2]
SCALE_SET_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

BOARD_SET_INIT_POS = [2.6 + SHIFT_X, 1.35, 0.22]
BOARD_SET_INIT_ROT = [1.0, 0.0, 0.0, 0.0]

BOARD_LOCAL_POS = (0.0302234, -0.01644, -0.01)
BOARD_LOCAL_ROT = (0.00027266516203578475, 0.0, 0.0, 0.999999962826854)

APPLE_LOCAL_POS = (0.112636, 0.025212600000000005, 0.0624487)
APPLE_LOCAL_ROT = (-0.0032191249177805324, 0.000399819986842121, 0.00023659759520994425, 0.9999947106861715)

PEAR_LOCAL_POS = (0.021880499999999997, -0.020075499999999996, 0.08877870000000002)
PEAR_LOCAL_ROT = (-0.08138182562679615, 0.0633286146806521, 0.9936678228980758, -0.044617740387405284)

MANGO_LOCAL_POS = (0.12176370000000002, -0.0865014, 0.1283396)
MANGO_LOCAL_ROT = (0.9514776447224945, -0.26097066916851586, -0.13126108057172448, 0.09672192178722909)

CABBAGE_LOCAL_POS = (-0.07733400000000001, 0.0783481, 0.05351229999999999)
CABBAGE_LOCAL_ROT = (0.60247217023873, -0.23433703706677222, 0.6713875314599591, -0.3624254678505684)

APPLE_GRASP_DIFF_THRESHOLD = 0.08
PEAR_GRASP_DIFF_THRESHOLD = 0.08
SCALE_XY_THRESHOLD = 0.12

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
)

scale_rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
)

mass_properties = MassPropertiesCfg(
    mass=0.01,
)


@configclass
class WeightSceneCfg(InteractiveSceneCfg):
    """Configuration for the weight task scene in the kitchen."""

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

    chopping_board_set = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BOARD_SET_INIT_POS,
            rot=BOARD_SET_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "chopping board001", "model_chopping_board.usd")
            ),
        ),
    )

    scale_set = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scale_set",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SCALE_SET_INIT_POS,
            rot=SCALE_SET_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(CUSTOM_ASSET_DIR, "electronic scales", "model_electronicscales.usd")
            ),
            scale=(1.8, 1.8, 1.8),
        ),
    )

    board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set/E_Component96_1",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(0.7, 0.7, 0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                BOARD_SET_INIT_POS[0] + BOARD_LOCAL_POS[0],
                BOARD_SET_INIT_POS[1] + BOARD_LOCAL_POS[1],
                BOARD_SET_INIT_POS[2] + BOARD_LOCAL_POS[2],
            ],
            rot=list(BOARD_LOCAL_ROT),
        ),
    )

    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set/E_Component18_05",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(0.7, 0.7, 0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=mass_properties,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                BOARD_SET_INIT_POS[0] + APPLE_LOCAL_POS[0],
                BOARD_SET_INIT_POS[1] + APPLE_LOCAL_POS[1],
                BOARD_SET_INIT_POS[2] + APPLE_LOCAL_POS[2],
            ],
            rot=list(APPLE_LOCAL_ROT),
        ),
    )

    pear = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set/E_Component113_06",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(0.7, 0.7, 0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                BOARD_SET_INIT_POS[0] + PEAR_LOCAL_POS[0],
                BOARD_SET_INIT_POS[1] + PEAR_LOCAL_POS[1],
                BOARD_SET_INIT_POS[2] + PEAR_LOCAL_POS[2],
            ],
            rot=list(PEAR_LOCAL_ROT),
        ),
    )

    mango = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set/E_Component131_07",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(0.7, 0.7, 0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                BOARD_SET_INIT_POS[0] + MANGO_LOCAL_POS[0],
                BOARD_SET_INIT_POS[1] + MANGO_LOCAL_POS[1],
                BOARD_SET_INIT_POS[2] + MANGO_LOCAL_POS[2],
            ],
            rot=list(MANGO_LOCAL_ROT),
        ),
    )

    cabbage = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chopping_board_set/E_Component17_08",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(0.7, 0.7, 0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                BOARD_SET_INIT_POS[0] + CABBAGE_LOCAL_POS[0],
                BOARD_SET_INIT_POS[1] + CABBAGE_LOCAL_POS[1],
                BOARD_SET_INIT_POS[2] + CABBAGE_LOCAL_POS[2],
            ],
            rot=list(CABBAGE_LOCAL_ROT),
        ),
    )

    scale = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scale_set/E_body_28",
        spawn=UsdFileCfg(
            usd_path="",
            rigid_props=scale_rigid_body_properties,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(SCALE_SET_INIT_POS),
            rot=list(SCALE_SET_INIT_ROT),
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

    apply_apple_mass_props = EventTerm(
        func=weight_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("apple"), "mass": mass_properties.mass, "density": mass_properties.density},
    )

    apply_pear_mass_props = EventTerm(
        func=weight_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("pear"), "mass": mass_properties.mass, "density": mass_properties.density},
    )

    apply_mango_mass_props = EventTerm(
        func=weight_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("mango"), "mass": mass_properties.mass, "density": mass_properties.density},
    )

    apply_cabbage_mass_props = EventTerm(
        func=weight_events.apply_mass_props,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("cabbage"), "mass": mass_properties.mass, "density": mass_properties.density},
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
        """Subtask terms for the weight task."""
        grasp_pear = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("pear"),
                "diff_threshold": PEAR_GRASP_DIFF_THRESHOLD,
            },
        )

        pear_on_scale = ObsTerm(
            func=mdp.pear_on_scale,
            params={
                "pear_cfg": SceneEntityCfg("pear"),
                "scale_cfg": SceneEntityCfg("scale"),
                "y_offset": -0.05,
                "xy_threshold": SCALE_XY_THRESHOLD,
            },
        )

        grasp_apple = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("apple"),
                "diff_threshold": APPLE_GRASP_DIFF_THRESHOLD,
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
    """Termination terms for the weight task."""

    apple_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("apple")},
    )

    pear_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("pear")},
    )

    success = DoneTerm(
        func=mdp.task_done_weight,
        params={
            "apple_cfg": SceneEntityCfg("apple"),
            "pear_cfg": SceneEntityCfg("pear"),
            "scale_cfg": SceneEntityCfg("scale"),
            "robot_cfg": SceneEntityCfg("robot"),
            "xy_threshold": SCALE_XY_THRESHOLD,
            "y_offset": -0.05,
        },
    )


@configclass
class WeightEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the weight task environment."""

    scene: WeightSceneCfg = WeightSceneCfg(
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
