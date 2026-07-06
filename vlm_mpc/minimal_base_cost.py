"""Cost for the minimal VLM-DP base, in the interface used by ``sim_free_mpc``.

Reach, regularization, and downward orientation are reused from ``sim_free_mpc.costs``. This module
adds the terms that base lacks: consistency, a geometry-based obstacle keepout with a table plane, and
a fingertip straddle. Each added term is a standalone function so it can move into
``sim_free_mpc.costs`` unchanged.

Imported from inside a task's ``run()`` (after the Isaac app boots), so heavy imports stay at module top.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from sim_free_mpc.costs import CostWeights, _reach_cost, _regularization, _downward_orientation_cost
from sim_free_mpc.fk import quat_apply_wxyz

# Weights are kept soft so the base stays a steerable policy rather than a rigid solver.
W_ORIENT = 3.0
W_CONSIST = 30.0
W_STRADDLE = 15.0
W_COLL = 20.0
W_FLOOR = 40.0
W_TASK = 25.0       # ReKep relational constraint (J_task) weight
W_PATH = 200.0      # running path-constraint weight
EE_R = 0.035        # gripper collision radius
COLL_CLEAR = 0.02   # keepout margin beyond the object extent
OPEN_HALF = 0.04    # gripper half-span
FINGER_R = 0.012    # fingertip radius


@dataclasses.dataclass(frozen=True)
class CostParams:
    """Tunable cost weights (soft by design) + gripper/collision geometry.

    Defaults match the module constants above; a task supplies overrides from its config.
    """
    orientation: float = W_ORIENT
    consistency: float = W_CONSIST
    straddle: float = W_STRADDLE
    collision: float = W_COLL
    floor: float = W_FLOOR
    task: float = W_TASK
    path: float = W_PATH
    ee_r: float = EE_R
    coll_clear: float = COLL_CLEAR
    finger_r: float = FINGER_R
    open_half: float = OPEN_HALF


def base_weights(orientation=W_ORIENT):
    """CostWeights with the base's orientation weight; the other terms keep their defaults."""
    return CostWeights(orientation=orientation)


def _closing_axis(ee_quat, dev, dt):
    """World-frame finger-closing axis (the gripper's local +y)."""
    axis = torch.zeros((*ee_quat.shape[:-1], 3), device=dev, dtype=dt)
    axis[..., 1] = 1.0
    return quat_apply_wxyz(ee_quat, axis)


def _down_axis(ee_quat, dev, dt):
    """World-frame tool approach axis (the gripper's local +z)."""
    axis = torch.zeros((*ee_quat.shape[:-1], 3), device=dev, dtype=dt)
    axis[..., 2] = 1.0
    return quat_apply_wxyz(ee_quat, axis)


def _consistency_cost(real_actions, context, weight=W_CONSIST):
    """Penalize deviation from the warm-started previous plan to damp step-to-step wander."""
    ref = context.get("plan_ref")
    joints = real_actions[..., :7]
    if ref is None or ref.shape[0] != joints.shape[1]:
        return real_actions.new_zeros(real_actions.shape[0])
    return weight * ((joints - ref.view(1, -1, 7)) ** 2).sum(dim=(1, 2))


def _straddle_cost(ee_pos, ee_quat, context, extents, params):
    """Push the fingertips to bracket the grasp point; active only while approaching it.

    The bracket centers on the stage ``target`` -- the VLM's chosen grasp keypoint -- so the fingers
    straddle exactly where the reach constraint drives the TCP, not an off-center object centroid.
    """
    grasp_obj, payload = context.get("grasp_obj"), context.get("payload")
    objects = context.get("objects", {})
    if payload is not None or grasp_obj not in objects:
        return ee_pos.new_zeros(ee_pos.shape[0])
    dev, dt = ee_pos.device, ee_pos.dtype
    closing = _closing_axis(ee_quat, dev, dt)
    center = torch.as_tensor(context["target"], device=dev, dtype=dt).view(1, 1, 3)
    threshold = extents.get(grasp_obj, (0.03, 0.05, 0.05))[0] + params.finger_r
    left = torch.linalg.vector_norm(ee_pos + params.open_half * closing - center, dim=-1)
    right = torch.linalg.vector_norm(ee_pos - params.open_half * closing - center, dim=-1)
    return params.straddle * (torch.clamp(threshold - left, min=0).pow(2)
                              + torch.clamp(threshold - right, min=0).pow(2)).mean(1)


def _general_collision_cost(ee_pos, ee_quat, context, extents, params):
    """Keep the gripper points clear of every scene object and above the table.

    Only the objects this stage manipulates are excluded -- the grasp target, any carried object, and
    the object being placed onto -- so the pick/place can reach those while full-strength keepout holds
    everywhere else (no distance-based fade, which used to open a hole beside the target).
    """
    dev, dt = ee_pos.device, ee_pos.dtype
    closing = _closing_axis(ee_quat, dev, dt)
    approach = _down_axis(ee_quat, dev, dt)
    gripper_points = torch.stack(
        [ee_pos, ee_pos + params.open_half * closing, ee_pos - params.open_half * closing,
         ee_pos - 0.06 * approach], 0)
    objects = context.get("objects", {})
    excluded = {context.get("grasp_obj"), context.get("payload"), context.get("place_target")}
    positions = [torch.as_tensor(o["pos"], device=dev, dtype=dt) for n, o in objects.items() if n not in excluded]
    radii = [extents.get(n, (0.03, 0.06, 0.05))[1] + params.ee_r + params.coll_clear
             for n in objects if n not in excluded]
    cost = ee_pos.new_zeros(ee_pos.shape[0])
    if positions:
        centers = torch.stack(positions, 0)
        keepout = torch.tensor(radii, device=dev, dtype=dt)
        dist = torch.linalg.vector_norm(gripper_points[..., None, :] - centers.view(1, 1, 1, -1, 3), dim=-1)
        penalty = torch.clamp(keepout.view(1, 1, 1, -1) - dist, min=0).pow(2).sum(-1)
        cost = cost + params.collision * penalty.mean(dim=(0, 2))
    z_table = context.get("z_table")
    if z_table is not None:
        below = torch.clamp(z_table - gripper_points[:3, ..., 2], min=0).pow(2)
        cost = cost + params.floor * below.mean(dim=(0, 2))
    return cost


def _rekep_constraint_cost(ee_pos, ee_quat, context, params):
    """ReKep relational constraint as J_task: the VLM subgoal (+ running path) constraints.

    ``constraint`` / ``path_fns`` are torch callables ``fn(ee_pos[K,H,3], kp[N,K,H,3]) -> [K,H]`` (lower =
    closer to satisfying). Held keypoints ride the candidate gripper (``ee_pos + R(ee_quat)·held_offset``),
    so a place constraint on a carried object actually depends on the candidate motion.
    """
    subgoal = context.get("constraint")
    if subgoal is None:
        return ee_pos.new_zeros(ee_pos.shape[0])
    dev, dt = ee_pos.device, ee_pos.dtype
    keypoints = torch.as_tensor(context["keypoints"], device=dev, dtype=dt)   # [N,3] world
    k, h = ee_pos.shape[0], ee_pos.shape[1]
    kp = keypoints[:, None, None, :].expand(-1, k, h, -1).clone()             # [N,K,H,3]
    held_idx, held_off = context.get("held_idx", ()), context.get("held_offset")
    if held_idx and held_off is not None:
        held_off = torch.as_tensor(held_off, device=dev, dtype=dt)
        for j, i in enumerate(held_idx):
            if ee_quat is not None:                                   # rotate the local offset by each candidate pose
                vec = held_off[j].view(1, 1, 3).expand(k, h, 3)
                kp[i] = ee_pos + quat_apply_wxyz(ee_quat, vec)
            else:
                kp[i] = ee_pos + held_off[j]
    cost = params.task * subgoal(ee_pos, kp).sum(dim=1)                       # [K]
    for path_fn in context.get("path_fns", ()):
        value = path_fn(ee_pos, kp)
        if torch.is_tensor(value) and value.ndim == 2:                        # per-step geometric only
            cost = cost + params.path * torch.clamp(value, min=0).sum(dim=1)
    return cost


class MinimalBaseCost:
    """Minimal base cost: reach, orientation, regularization, consistency, straddle, and collision."""

    def __init__(self, extents=None, params=None, weights=None):
        self.extents = extents or {}   # fallback when the context objects do not carry their extents
        self.params = params or CostParams()
        self.weights = weights or base_weights(self.params.orientation)

    def target(self, context, device, dtype):
        return torch.as_tensor(context["target"], device=device, dtype=dtype)

    def __call__(self, *, real_actions, ee_pos, ee_quat=None, context):
        weights = self.weights
        objects = context.get("objects", {})
        extents = {n: o["extents"] for n, o in objects.items() if "extents" in o} or self.extents
        if context.get("constraint") is not None:   # ReKep relational constraint drives the task
            cost = _rekep_constraint_cost(ee_pos, ee_quat, context, self.params)
        else:                                        # GT / target-point grounding
            cost = _reach_cost(ee_pos, self.target(context, ee_pos.device, ee_pos.dtype), weights)
        cost = cost + _regularization(real_actions, context, weights)
        orientation = _downward_orientation_cost(ee_quat, weights)
        if orientation is not None:
            cost = cost + orientation
        cost = cost + _consistency_cost(real_actions, context, self.params.consistency)
        if ee_quat is not None:
            cost = cost + _straddle_cost(ee_pos, ee_quat, context, extents, self.params)
            cost = cost + _general_collision_cost(ee_pos, ee_quat, context, extents, self.params)
        return cost


def usd_extents(E, scene_objects):
    """Per-object ``(grip, keepout, half_height)`` from the sim's USD bounding boxes, keyed by name.

    Reads geometry straight from the simulator so no per-object radii need hand-specifying. The narrow
    horizontal half-extent is used for grasping and the wide one for collision.
    """
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                       useExtentsHint=True)
    except Exception as exc:
        print(f"[minimal_base] USD extents unavailable ({exc}); using defaults", flush=True)
        return {}
    extents = {}
    for name in scene_objects:
        prim_path = None
        for getter in (lambda: str(E.env.scene[name].root_physx_view.prim_paths[0]),
                       lambda: str(E.env.scene[name].cfg.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")):
            try:
                prim_path = getter()
                break
            except Exception:
                continue
        if prim_path is None:
            continue
        try:
            bounds = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
            half = np.abs(0.5 * (np.array(bounds.GetMax(), dtype=float) - np.array(bounds.GetMin(), dtype=float)))
            if np.all(np.isfinite(half)) and 0 < half.max() < 5.0:
                extents[name] = (float(min(half[0], half[1])), float(max(half[0], half[1])), float(half[2]))
        except Exception:
            continue
    return extents
