# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.manipulation.oven.mdp.oven_events import (  # noqa: F401
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


def set_prim_local_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_regex: str,
    scale: tuple[float, float, float],
):
    """Set local scale on matching prims."""
    del env_ids

    if "{ENV_REGEX_NS}" in prim_path_regex:
        prim_path_regex = prim_path_regex.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    stage = get_current_stage()
    for prim_path in sim_utils.find_matching_prim_paths(prim_path_regex, stage):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xformable = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op is None:
            scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

        scale_op.Set(Gf.Vec3d(*scale))
