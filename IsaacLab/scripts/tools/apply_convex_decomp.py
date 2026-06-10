#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply convex collision approximations to mesh prims in one or more USD files.

Examples:
    ./isaaclab.sh --python scripts/tools/apply_convex_decomp.py \
        assets/ArtVIP/Interactive_scene/childrenroom/table_3/model_table_3.usd

    ./isaaclab.sh --python scripts/tools/apply_convex_decomp.py \
        assets/ArtVIP/Interactive_scene/childrenroom/table_3/model_table_3.usd \
        --match /handle/ --match /E_door_
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Iterable

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply convex collision approximations to USD mesh prims.")
    parser.add_argument("usd_paths", nargs="+", help="One or more USD files to patch in place.")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Only patch mesh prims whose prim path contains this substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--approximation",
        choices=("convexDecomposition", "convexHull"),
        default="convexDecomposition",
        help="Collision approximation to apply.",
    )
    parser.add_argument(
        "--max-convex-hulls",
        type=int,
        default=None,
        help="Maximum convex hulls for convex decomposition. If omitted, USD approximation metadata is authored without PhysX tuning attrs.",
    )
    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=None,
        help="Voxel resolution for convex decomposition. If omitted, USD approximation metadata is authored without PhysX tuning attrs.",
    )
    parser.add_argument(
        "--error-percentage",
        type=float,
        default=None,
        help="Approximation error percentage for convex decomposition. If omitted, USD approximation metadata is authored without PhysX tuning attrs.",
    )
    parser.add_argument(
        "--hull-vertex-limit",
        type=int,
        default=None,
        help="Convex hull vertex limit. If omitted, USD approximation metadata is authored without PhysX tuning attrs.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file before patching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching mesh prims without saving changes.",
    )
    return parser


def should_patch(prim_path: str, matches: list[str]) -> bool:
    if not matches:
        return True
    return any(token in prim_path for token in matches)


def iter_mesh_prim_paths(stage: Usd.Stage, matches: list[str]) -> Iterable[str]:
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        prim_path = str(prim.GetPath())
        if should_patch(prim_path, matches):
            yield prim_path


def create_mesh_cfg(args: argparse.Namespace) -> schemas_cfg.MeshCollisionPropertiesCfg:
    if args.approximation == "convexHull":
        return schemas_cfg.ConvexHullPropertiesCfg(
            hull_vertex_limit=args.hull_vertex_limit,
        )
    return schemas_cfg.ConvexDecompositionPropertiesCfg(
        hull_vertex_limit=args.hull_vertex_limit,
        max_convex_hulls=args.max_convex_hulls,
        voxel_resolution=args.voxel_resolution,
        error_percentage=args.error_percentage,
    )


def patch_usd(usd_path: str, args: argparse.Namespace) -> int:
    resolved_path = os.path.abspath(usd_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"USD file not found: {resolved_path}")

    if not args.no_backup:
        backup_path = resolved_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(resolved_path, backup_path)
            print(f"[INFO] Created backup: {backup_path}")

    stage = Usd.Stage.Open(resolved_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {resolved_path}")

    collision_cfg = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_cfg = create_mesh_cfg(args)

    patched_count = 0
    for prim_path in iter_mesh_prim_paths(stage, args.match):
        print(f"[INFO] Patching mesh collision: {prim_path}")
        if not args.dry_run:
            schemas.define_collision_properties(prim_path, collision_cfg, stage=stage)
            schemas.define_mesh_collision_properties(prim_path, mesh_cfg, stage=stage)
        patched_count += 1

    if patched_count == 0:
        print(f"[WARNING] No matching mesh prims found in: {resolved_path}")
        return 0

    if args.dry_run:
        print(f"[INFO] Dry run complete for {resolved_path}: {patched_count} mesh prim(s) matched.")
        return patched_count

    stage.Save()
    print(f"[INFO] Saved {resolved_path}: patched {patched_count} mesh prim(s).")
    return patched_count


def main():
    args = build_argparser().parse_args()

    total_patched = 0
    for usd_path in args.usd_paths:
        total_patched += patch_usd(usd_path, args)

    print(f"[INFO] Finished. Total patched mesh prims: {total_patched}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
