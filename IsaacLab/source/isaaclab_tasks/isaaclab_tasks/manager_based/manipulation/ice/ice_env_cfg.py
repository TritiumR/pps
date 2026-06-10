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
from .mdp import ice_events

SCENE_ASSET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../assets/ArtVIP/Interactive_scene/kitchen_with_parlor",
)
CUSTOM_ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../../../../../assets")

ICE_TASK_TABLE_POS = (7.556153774261475, 0.15213478952646255, 0.8191297936439514)
AUTHORED_TABLE_POS = (1.937098979966911, 5.049524784088135, 1.016234040260315)

ROOM_INIT_POS = (
    ICE_TASK_TABLE_POS[0] - AUTHORED_TABLE_POS[0],
    ICE_TASK_TABLE_POS[1] - AUTHORED_TABLE_POS[1],
    ICE_TASK_TABLE_POS[2] - AUTHORED_TABLE_POS[2],
)
ROOM_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

CUP_INIT_POS = (ICE_TASK_TABLE_POS[0] + 1.72, ICE_TASK_TABLE_POS[1] - 4.0, ICE_TASK_TABLE_POS[2] - 0.2)
CUP_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ICE_CUBE_INIT_POS = (CUP_INIT_POS[0] + 0.22, CUP_INIT_POS[1] - 0.08, CUP_INIT_POS[2])
ICE_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

CUP_SCALE = (0.02, 0.02, 0.02)
ICE_CUBE_SCALE = (0.02, 0.02, 0.02)

ICE_GRASP_DIFF_THRESHOLD = 0.08
ICE_CUP_XY_THRESHOLD = 0.055
ICE_CUP_Z_MIN_THRESHOLD = 0.035
ICE_CUP_Z_MAX_THRESHOLD = 0.16

cup_mass_properties = MassPropertiesCfg(mass=0.25)
ice_cube_mass_properties = MassPropertiesCfg(mass=0.03)

kinematic_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)

rigid_body_properties = RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    max_depenetration_velocity=1.0,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
)


@configclass
class IceSceneCfg(InteractiveSceneCfg):
    """Configuration for the ice task scene in the kitchen with parlor."""

    interactive_kitchen_with_parlor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/interactive_kitchen_with_parlor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(ROOM_INIT_POS),
            rot=list(ROOM_INIT_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(SCENE_ASSET_DIR, "Interactive_kitchen_with_parlor.usd")),
            rigid_props=kinematic_body_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/glass_cup",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "glass_cup", "glass_cup.usd")),
            scale=CUP_SCALE,
            rigid_props=rigid_body_properties,
            mass_props=cup_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(CUP_INIT_POS), rot=list(CUP_INIT_ROT)),
    )

    ice_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ice_cube",
        spawn=UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(CUSTOM_ASSET_DIR, "ice_cube", "ice_cube.usd")),
            scale=ICE_CUBE_SCALE,
            rigid_props=rigid_body_properties,
            mass_props=ice_cube_mass_properties,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(ICE_CUBE_INIT_POS), rot=list(ICE_CUBE_INIT_ROT)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCfg:
    """Configuration for startup events."""

    trim_cup_mesh = EventTerm(
        func=ice_events.trim_mesh_faces_outside_bounds,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "max_abs_x": 20.0,
            "max_abs_z": 20.0,
        },
    )

    cup_visible_material = EventTerm(
        func=ice_events.bind_visual_material,
        mode="prestartup",
        params={
            "prim_path_regex": "{ENV_REGEX_NS}/glass_cup/geometry/mesh",
            "material_cfg": sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.85, 1.0),
                roughness=0.35,
                metallic=0.0,
                opacity=0.1,
            ),
            "material_name": "transparentCupMaterial",
        },
    )

    ice_cube_visible_material = EventTerm(
        func=ice_events.bind_visual_material,
        mode="prestartup",
        params={
            "prim_path_regex": "{ENV_REGEX_NS}/ice_cube/geometry/mesh",
            "material_cfg": sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.85, 1.0),
                roughness=0.35,
                metallic=0.0,
                opacity=0.1,
            ),
            "material_name": "transparentIceCubeMaterial",
        },
    )

    refine_cup_collision = EventTerm(
        func=ice_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "hull_vertex_limit": 64,
            "max_convex_hulls": 128,
            "min_thickness": 0.001,
            "voxel_resolution": 1_000_000,
            "error_percentage": 1.0,
            "shrink_wrap": True,
            "contact_offset": 0.002,
            "rest_offset": 0.0,
        },
    )

    refine_ice_cube_collision = EventTerm(
        func=ice_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("ice_cube"),
            "hull_vertex_limit": 64,
            "max_convex_hulls": 16,
            "min_thickness": 0.001,
            "voxel_resolution": 1_000_000,
            "error_percentage": 1.0,
            "shrink_wrap": True,
            "contact_offset": 0.002,
            "rest_offset": 0.0,
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
        """Subtask terms for the ice task."""

        grasp_ice = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("ice_cube"),
                "diff_threshold": ICE_GRASP_DIFF_THRESHOLD,
            },
        )

        ice_in_cup = ObsTerm(
            func=mdp.ice_in_cup,
            params={
                "cup_cfg": SceneEntityCfg("cup"),
                "ice_cfg": SceneEntityCfg("ice_cube"),
                "robot_cfg": SceneEntityCfg("robot"),
                "xy_threshold": ICE_CUP_XY_THRESHOLD,
                "z_min_threshold": ICE_CUP_Z_MIN_THRESHOLD,
                "z_max_threshold": ICE_CUP_Z_MAX_THRESHOLD,
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
    """Termination terms for the ice task."""

    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    ice_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("ice_cube"),
        },
    )

    success = DoneTerm(
        func=mdp.ice_in_cup,
        params={
            "cup_cfg": SceneEntityCfg("cup"),
            "ice_cfg": SceneEntityCfg("ice_cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "xy_threshold": ICE_CUP_XY_THRESHOLD,
            "z_min_threshold": ICE_CUP_Z_MIN_THRESHOLD,
            "z_max_threshold": ICE_CUP_Z_MAX_THRESHOLD,
        },
    )


@configclass
class IceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ice task environment."""

    scene: IceSceneCfg = IceSceneCfg(num_envs=4096, env_spacing=25, replicate_physics=False)
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
        self.sim.render.rendering_mode = "performance"
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = False
        self.sim.render.enable_global_illumination = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.samples_per_pixel = 1
        self.sim.render.antialiasing_mode = None
        self.sim.render.carb_settings = {
            **(self.sim.render.carb_settings or {}),
            "rtx.rendermode": "PathTracing",
            "rtx.pathtracing.spp": 8,
            "rtx.pathtracing.totalSpp": 8,
            "rtx.pathtracing.maxBounces": 2,
            "rtx.pathtracing.maxSpecularAndTransmissionBounces": 2,
            "rtx.pathtracing.maxVolumeBounces": 0,
            "rtx.pathtracing.ptfog.maxBounces": 1,
            "rtx.pathtracing.adaptiveSampling.enabled": False,
            "rtx.pathtracing.cached.enabled": True,
            "rtx.pathtracing.fractionalCutoutOpacity": True,
            "rtx.reflections.enabled": True,
            "rtx.indirectDiffuse.enabled": True,
            "rtx.ambientOcclusion.enabled": False,
            "rtx-transient.dldenoiser.enabled": False,
        }

        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
