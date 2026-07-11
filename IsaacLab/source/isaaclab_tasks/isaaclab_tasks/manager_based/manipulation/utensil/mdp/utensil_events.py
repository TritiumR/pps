# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.manipulation.plate.mdp.plate_events import (  # noqa: F401
    apply_mass_props,
    apply_scale_from_spawn_cfg,
    randomize_joint_by_gaussian_offset,
    randomize_object_pose,
    randomize_scene_lighting_domelight,
    set_default_joint_pose,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def deactivate_prim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
):
    """Deactivate matching prims on stage."""
    del env_ids

    if "{ENV_REGEX_NS}" in prim_path_regex:
        prim_path_regex = prim_path_regex.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    stage = get_current_stage()
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.SetActive(False)


def bind_visual_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
    material_cfg: sim_utils.VisualMaterialCfg,
    material_name: str = "visualMaterial",
):
    """Create and bind a visual material for all matching prims on stage."""
    del env_ids

    if "{ENV_REGEX_NS}" in prim_path_regex:
        prim_path_regex = prim_path_regex.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    stage = get_current_stage()
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        material_path = f"{prim_path}/{material_name}"
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_visual_material(prim_path, material_path, stage=stage)


def apply_convex_decomposition_collision(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
    hull_vertex_limit: int | None = 128,
    max_convex_hulls: int | None = 128,
    voxel_resolution: int | None = 1_000_000,
    error_percentage: float | None = 2.5,
    shrink_wrap: bool | None = True,
):
    """Use tuned convex decomposition on matching mesh prims."""
    del env_ids

    if "{ENV_REGEX_NS}" in prim_path_regex:
        prim_path_regex = prim_path_regex.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    stage = get_current_stage()
    convex_decomposition_cfg = sim_utils.ConvexDecompositionPropertiesCfg(
        hull_vertex_limit=hull_vertex_limit,
        max_convex_hulls=max_convex_hulls,
        voxel_resolution=voxel_resolution,
        error_percentage=error_percentage,
        shrink_wrap=shrink_wrap,
    )
    collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        sim_utils.define_collision_properties(prim_path, collision_cfg, stage=stage)
        sim_utils.define_mesh_collision_properties(prim_path, convex_decomposition_cfg, stage=stage)
