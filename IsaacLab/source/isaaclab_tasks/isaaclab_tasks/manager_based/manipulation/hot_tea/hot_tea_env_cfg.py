# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.tea import tea_env_cfg
from isaaclab_tasks.manager_based.manipulation.tea.mdp import tea_events

HOT_TEA_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
HOT_TEAPOT_USD = os.path.abspath(os.path.join(HOT_TEA_ASSET_DIR, "hot_teapot.usda"))
HOT_TEAPOT_PRIM_NAME = "E_hot_teapot_5"
HOT_TEAPOT_PRIM_PATH = "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_hot_teapot_5"
HOT_TEAPOT_HANDLE_MESH_PRIM_PATH = f"{HOT_TEAPOT_PRIM_PATH}/P_c680523765f8dbb9"


@configclass
class HotTeaSceneCfg(tea_env_cfg.TeaSceneCfg):
    """Tea scene with one additional teapot on the tea table."""

    hot_teapot = RigidObjectCfg(
        prim_path=HOT_TEAPOT_PRIM_PATH,
        spawn=UsdFileCfg(
            usd_path=HOT_TEAPOT_USD,
            scale=tea_env_cfg.TEAPOT_OBJECT_SCALE,
            rigid_props=tea_env_cfg.rigid_body_properties,
            mass_props=tea_env_cfg.TEAPOT_MASS_PROPERTIES,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class EventCfg(tea_env_cfg.EventCfg):
    """Startup events for the hot tea variant."""

    scale_hot_teapot_handle = EventTerm(
        func=tea_events.scale_mesh_points_in_local_bounds,
        mode="prestartup",
        params={
            "mesh_prim_path_regex": HOT_TEAPOT_HANDLE_MESH_PRIM_PATH,
            "bounds_min": tea_env_cfg.TEAPOT_HANDLE_LOCAL_BOUNDS_MIN,
            "bounds_max": tea_env_cfg.TEAPOT_HANDLE_LOCAL_BOUNDS_MAX,
            "scale": tea_env_cfg.TEAPOT_HANDLE_SCALE,
            "center": tea_env_cfg.TEAPOT_HANDLE_SCALE_CENTER,
        },
    )

    hot_teapot_convex_decomposition_collision = EventTerm(
        func=tea_events.set_asset_mesh_collision_to_convex_decomposition,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("hot_teapot"),
            "max_convex_hulls": 64,
            "hull_vertex_limit": 64,
            "voxel_resolution": 1_000_000,
        },
    )

    hot_teapot_physics_material = EventTerm(
        func=tea_events.bind_rigid_body_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("hot_teapot"),
            "material_cfg": tea_env_cfg.TEAPOT_PHYSICS_MATERIAL,
            "material_name": "hotTeaPotPhysicsMaterial",
        },
    )

    hot_teapot_center_of_mass = EventTerm(
        func=tea_events.set_center_of_mass,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("hot_teapot"),
            "center_of_mass": tea_env_cfg.TEAPOT_CENTER_OF_MASS,
        },
    )


@configclass
class HotTeaEnvCfg(tea_env_cfg.TeaEnvCfg):
    """Tea task variant with an additional teapot distractor."""

    scene: HotTeaSceneCfg = HotTeaSceneCfg(num_envs=4096, env_spacing=25, replicate_physics=False)
    events: EventCfg = EventCfg()
