# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the ice task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.pot.mdp.pot_events import (  # noqa: F401
    deactivate_prim,
    randomize_joint_by_gaussian_offset,
    randomize_object_pose,
    set_asset_mesh_collision_to_convex_decomposition,
    set_default_joint_pose,
)

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


def trim_mesh_faces_outside_bounds(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_abs_x: float = 20.0,
    max_abs_z: float = 20.0,
):
    """Remove oversized mesh faces from imported assets before physics cooks collisions."""
    del env_ids

    asset: AssetBase = env.scene[asset_cfg.name]
    stage = get_current_stage()

    for root_path in sim_utils.find_matching_prim_paths(asset.cfg.prim_path, stage):
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue

        prim_stack = [root_prim]
        while prim_stack:
            prim = prim_stack.pop()
            if prim.IsA(UsdGeom.Mesh):
                _trim_mesh_prim(UsdGeom.Mesh(prim), max_abs_x=max_abs_x, max_abs_z=max_abs_z)
            prim_stack.extend(prim.GetChildren())


def _trim_mesh_prim(mesh: UsdGeom.Mesh, max_abs_x: float, max_abs_z: float):
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        return

    kept_counts = []
    kept_old_indices = []
    kept_flat_positions = []
    used_point_indices = set()
    cursor = 0

    for count in counts:
        face_indices = list(indices[cursor : cursor + count])
        keep_face = True
        for point_id in face_indices:
            point = points[point_id]
            if abs(point[0]) > max_abs_x or abs(point[2]) > max_abs_z:
                keep_face = False
                break

        if keep_face:
            kept_counts.append(count)
            kept_old_indices.extend(face_indices)
            kept_flat_positions.extend(range(cursor, cursor + count))
            used_point_indices.update(face_indices)
        cursor += count

    if len(kept_counts) == len(counts) or len(kept_counts) == 0:
        return

    sorted_point_indices = sorted(used_point_indices)
    index_map = {old_index: new_index for new_index, old_index in enumerate(sorted_point_indices)}
    new_points = [points[old_index] for old_index in sorted_point_indices]
    new_indices = [index_map[old_index] for old_index in kept_old_indices]

    old_flat_count = len(indices)
    _filter_face_varying_attributes(mesh, kept_flat_positions, old_flat_count)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(new_points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(kept_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(new_indices))


def _filter_face_varying_attributes(mesh: UsdGeom.Mesh, kept_flat_positions: list[int], old_flat_count: int):
    for attr in mesh.GetPrim().GetAttributes():
        values = attr.Get()
        if values is None:
            continue

        try:
            value_count = len(values)
        except TypeError:
            continue

        if value_count != old_flat_count:
            continue

        attr.Set(type(values)([values[index] for index in kept_flat_positions]))


def bind_visual_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
    material_cfg: sim_utils.VisualMaterialCfg,
    material_name: str = "visibleMaterial",
):
    """Create and bind a visual material to matching prims."""
    del env_ids

    if "{ENV_REGEX_NS}" in prim_path_regex:
        prim_path_regex = prim_path_regex.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    stage = get_current_stage()
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        material_path = f"{prim_path}/{material_name}"
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_visual_material(prim_path, material_path, stage=stage)
