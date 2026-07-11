"""Composable cost terms for the VLM-DP base.

Each term is a function term(I: CostInputs) -> Tensor[K] (unweighted, lower = better) registered in
TERMS by name. A cost is a config-chosen {name: weight} combination summed by
base_cost.CompositeCost. Add a term = write a function + @register("name") + list it in the config.

Shapes: ee_pos / ee_quat are [K,H,3] / [K,H,4] (K candidates, H horizon); real_actions
is [K,H,8] (7 joints + gripper); every term returns [K]. Terms self-gate: one that does not
apply to the current stage (for example reach when a relational constraint is set) returns zeros.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from sim_common.geometry import DEFAULT_EXTENT

TERMS: dict[str, Callable[["CostInputs"], torch.Tensor]] = {}


def register(name: str):
    """Register a cost term under name (referenced from the config's cost.terms)."""
    def deco(fn):
        TERMS[name] = fn
        return fn
    return deco


@dataclasses.dataclass
class CostInputs:
    """Everything a term may read: candidate actions/pose, the stage context, and gripper geometry."""
    real_actions: torch.Tensor
    ee_pos: torch.Tensor
    ee_quat: torch.Tensor | None
    context: dict[str, Any]
    extents: dict            # per-object (grip, keepout, half_height); collision/straddle fallback
    geom: Any                # gripper/collision geometry namespace (see base_cost.DEFAULT_GEOM)


def _quat_apply_wxyz(quat, vec):
    """Rotate vec by quat (IsaacLab w,x,y,z), broadcasting over shared leading dims."""
    q = quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8)
    xyz = q[..., 1:]
    t = torch.cross(xyz.expand_as(vec), vec, dim=-1) * 2.0
    return vec + q[..., :1] * t + torch.cross(xyz.expand_as(vec), t, dim=-1)


def _axis(ee_quat, i):
    """World-frame gripper axis i (local +y = closing, +z = tool approach)."""
    a = torch.zeros((*ee_quat.shape[:-1], 3), device=ee_quat.device, dtype=ee_quat.dtype)
    a[..., i] = 1.0
    return _quat_apply_wxyz(ee_quat, a)


def _zeros(I):
    return I.ee_pos.new_zeros(I.ee_pos.shape[0])


def _target(I):
    return torch.as_tensor(I.context["target"], device=I.ee_pos.device, dtype=I.ee_pos.dtype)


def _gripper_points(I):
    """Four gripper points [4,K,H,3]: TCP, both fingertips, and a point back up the tool axis."""
    closing, approach = _axis(I.ee_quat, 1), _axis(I.ee_quat, 2)
    return torch.stack([I.ee_pos, I.ee_pos + I.geom.open_half * closing,
                        I.ee_pos - I.geom.open_half * closing, I.ee_pos - I.geom.tool_back * approach], 0)


def _grasp_frame(I):
    """Grasp-stage geometry (tip, closing, approach, lateral, center, obj_radius), or None off a grasp stage.

    Gated like straddle: needs an orientation, a grasp object in the scene, and no carried payload.
    tip is the TCP shifted along the tool axis by geom.tcp_to_tip; center is the grasp target;
    obj_radius is the object's narrow (grip) half-extent.
    """
    if I.ee_quat is None:
        return None
    ctx = I.context
    grasp_obj, objects = ctx.get("grasp_obj"), ctx.get("objects", {})
    if ctx.get("payload") is not None or grasp_obj not in objects:
        return None
    approach, closing = _axis(I.ee_quat, 2), _axis(I.ee_quat, 1)
    lateral = torch.cross(approach, closing, dim=-1)
    lateral = lateral / torch.clamp(torch.linalg.vector_norm(lateral, dim=-1, keepdim=True), min=1e-8)
    tip = I.ee_pos + I.geom.tcp_to_tip * approach
    center = _target(I).view(1, 1, 3)
    return tip, closing, approach, lateral, center, I.extents.get(grasp_obj, DEFAULT_EXTENT)[0]


def _rekep_keypoints(I):
    """Per-candidate keypoints [N,K,H,3] with held keypoints riding the candidate gripper pose."""
    ee_pos, ee_quat, ctx = I.ee_pos, I.ee_quat, I.context
    dev, dt = ee_pos.device, ee_pos.dtype
    kp = torch.as_tensor(ctx["keypoints"], device=dev, dtype=dt)[:, None, None, :]
    kp = kp.expand(-1, ee_pos.shape[0], ee_pos.shape[1], -1).clone()          # [N,K,H,3]
    held_idx, held_off = ctx.get("held_idx", ()), ctx.get("held_offset")
    if held_idx and held_off is not None:
        held_off = torch.as_tensor(held_off, device=dev, dtype=dt)
        for j, i in enumerate(held_idx):
            if ee_quat is not None:
                vec = held_off[j].view(1, 1, 3).expand(ee_pos.shape[0], ee_pos.shape[1], 3)
                kp[i] = ee_pos + _quat_apply_wxyz(ee_quat, vec)
            else:
                kp[i] = ee_pos + held_off[j]
    return kp


# J_task: reach a target point, or satisfy a ReKep relational constraint. The two are mutually exclusive.

@register("reach")
def reach(I):
    """Mean squared TCP->target distance over the horizon (off when a relational constraint is set)."""
    if I.context.get("constraint") is not None:
        return _zeros(I)
    return ((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1).mean(dim=1)


@register("terminal_reach")
def terminal_reach(I):
    """Terminal squared TCP->target distance (off when a relational constraint is set)."""
    if I.context.get("constraint") is not None:
        return _zeros(I)
    return ((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1)[:, -1]


@register("rekep_subgoal")
def rekep_subgoal(I):
    """ReKep subgoal relational constraint summed over the horizon (off without a constraint)."""
    subgoal = I.context.get("constraint")
    if subgoal is None:
        return _zeros(I)
    return subgoal(I.ee_pos, _rekep_keypoints(I)).sum(dim=1)


@register("rekep_path")
def rekep_path(I):
    """ReKep running path constraints (per-step geometric only) summed over the horizon."""
    if I.context.get("constraint") is None or not I.context.get("path_fns", ()):
        return _zeros(I)
    kp = _rekep_keypoints(I)
    cost = _zeros(I)
    for path_fn in I.context["path_fns"]:
        v = path_fn(I.ee_pos, kp)
        if torch.is_tensor(v) and v.ndim == 2:
            cost = cost + torch.clamp(v, min=0).sum(dim=1)
    return cost


# Regularization and feasibility. These self-gate when their inputs are missing.

@register("smooth")
def smooth(I):
    """Sum of squared consecutive joint changes (motion smoothness)."""
    joints = I.real_actions[..., :7]
    return ((joints[:, 1:] - joints[:, :-1]) ** 2).sum(dim=(-1, -2))


@register("joint_delta")
def joint_delta(I):
    """Sum of squared deviation from the current joints (trust region)."""
    joints = I.real_actions[..., :7]
    current = I.context.get("joint_pos")
    if current is None:
        return _zeros(I)
    current = torch.as_tensor(current, device=joints.device, dtype=joints.dtype)
    if current.ndim > 1 and current.shape[0] == 1:
        current = current[0]
    return ((joints - current[:7].view(1, 1, 7)) ** 2).sum(dim=(-1, -2))


@register("orientation")
def orientation(I):
    """Downward tool-axis alignment (1 - approach . down) meaned over the horizon."""
    if I.ee_quat is None:
        return _zeros(I)
    down = torch.tensor([0.0, 0.0, -1.0], device=I.ee_quat.device, dtype=I.ee_quat.dtype)
    return (1.0 - (_axis(I.ee_quat, 2) * down.view(1, 1, 3)).sum(dim=-1)).mean(dim=1)


@register("consistency")
def consistency(I):
    """Squared deviation from the warm-started previous plan (damps step-to-step wander)."""
    ref = I.context.get("plan_ref")
    joints = I.real_actions[..., :7]
    if ref is None or ref.shape[0] != joints.shape[1]:
        return _zeros(I)
    return ((joints - ref.view(1, -1, 7)) ** 2).sum(dim=(1, 2))


@register("straddle")
def straddle(I):
    """Fingertips bracket the grasp target without penetrating it (XY keepout), grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    _, closing, _, _, center, radius = f
    keepout = radius + I.geom.finger_r
    left = torch.linalg.vector_norm((I.ee_pos + I.geom.open_half * closing - center)[..., :2], dim=-1)
    right = torch.linalg.vector_norm((I.ee_pos - I.geom.open_half * closing - center)[..., :2], dim=-1)
    return (torch.clamp(keepout - left, min=0).pow(2) + torch.clamp(keepout - right, min=0).pow(2)).mean(1)


@register("tip_z")
def tip_z(I):
    """Fingertip at the grasp height (squared z offset), grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, _, _, _, center, _ = f
    return ((tip[..., 2] - center[..., 2]) ** 2).mean(dim=1)


@register("yaw")
def yaw(I):
    """Align the closing axis with a world x/y axis (top-down grasp yaw), grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    closing = f[1]
    return (1.0 - torch.maximum(closing[..., 0].abs(), closing[..., 1].abs())).mean(dim=1)


@register("center_region")
def center_region(I):
    """Keep the grasp point within a region around the gripper center, grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = f
    rel = center - tip
    err = torch.sqrt((rel * closing).sum(-1) ** 2 + (rel * lateral).sum(-1) ** 2
                     + (rel * approach).sum(-1) ** 2 + 1e-12)
    return torch.clamp(err - I.geom.center_scale * radius, min=0.0).pow(2).mean(dim=1)


@register("aperture_region")
def aperture_region(I):
    """Keep the object within the gripper opening along the closing axis, grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, closing, _, _, center, radius = f
    closing_coord = ((center - tip) * closing).sum(-1)
    return torch.clamp(closing_coord.abs() + radius + I.geom.aperture_margin - I.geom.open_half,
                       min=0.0).pow(2).mean(dim=1)


@register("close_gripper")
def close_gripper(I):
    """Gated in-loop gripper close: drive the gripper channel shut when centered and at grasp height."""
    f = _grasp_frame(I)
    if f is None or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = f
    rel = center - tip
    err = torch.sqrt((rel * closing).sum(-1) ** 2 + (rel * lateral).sum(-1) ** 2
                     + (rel * approach).sum(-1) ** 2 + 1e-12)
    center_radius = I.geom.center_scale * radius
    z_scale = I.geom.close_z_scale
    xy_scale = max(center_radius, 1e-3)
    tip_z_sq = (tip[..., 2] - center[..., 2]) ** 2
    center_excess = torch.clamp(err - center_radius, min=0.0)
    z_excess = torch.clamp(torch.sqrt(tip_z_sq + 1e-12) - z_scale, min=0.0)
    gate = torch.exp(-center_excess.pow(2) / max(xy_scale * xy_scale, 1e-8)
                     - z_excess.pow(2) / max(z_scale * z_scale, 1e-8))
    return (I.real_actions[..., 7] - gate).pow(2).mean(dim=1)


@register("carry_hold")
def carry_hold(I):
    """Hold the gripper closed while a payload is carried, releasing at the place target.

    Faithful port of the grasp-flow lift_gripper / place_gripper terms: an object in hand pins the
    gripper channel to fully closed (1), and it opens only once the carried object reaches the place
    target (within release_xy / release_z). Active only on carry/place stages (a payload is set), so
    it is the carry-phase complement of close_gripper (which gates off when a payload is set).
    """
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    gripper = I.real_actions[..., 7]
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    place = objects.get(ctx.get("place_target"))
    if place is None:                              # no place-onto object: pure hold (lift phase)
        return (gripper - 1.0).pow(2).mean(dim=1)
    # Carried object rides the gripper rigidly (offset fixed at grasp); ee_pos[:, 0] is the current TCP.
    payload_pos = torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt)
    carried = I.ee_pos + (payload_pos.view(1, 1, 3) - I.ee_pos[:, :1])
    # Place destination: payload resting on the place object's top surface (extents give the surface).
    place_pos = torch.as_tensor(place["pos"], device=dev, dtype=dt)
    stack = I.extents.get(ctx["place_target"], DEFAULT_EXTENT)[2] + I.extents.get(payload, DEFAULT_EXTENT)[2]
    dest = place_pos + torch.tensor([0.0, 0.0, stack], device=dev, dtype=dt)
    xy_dist = torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
    z_err = (carried[..., 2] - dest[2]).abs()
    release = ((xy_dist < I.geom.release_xy) & (z_err < I.geom.release_z)).to(dt)
    return (gripper - (1.0 - release)).pow(2).mean(dim=1)


@register("collision")
def collision(I):
    """Soft keepout of the gripper points from every non-manipulated scene object."""
    if I.ee_quat is None:
        return _zeros(I)
    ctx = I.context
    objects = ctx.get("objects", {})
    excluded = {ctx.get("grasp_obj"), ctx.get("payload"), ctx.get("place_target")}
    positions = [torch.as_tensor(o["pos"], device=I.ee_pos.device, dtype=I.ee_pos.dtype)
                 for n, o in objects.items() if n not in excluded]
    if not positions:
        return _zeros(I)
    radii = [I.extents.get(n, DEFAULT_EXTENT)[1] + I.geom.ee_r + I.geom.coll_clear
             for n in objects if n not in excluded]
    gp = _gripper_points(I)
    centers = torch.stack(positions, 0)
    keepout = torch.tensor(radii, device=I.ee_pos.device, dtype=I.ee_pos.dtype)
    dist = torch.linalg.vector_norm(gp[..., None, :] - centers.view(1, 1, 1, -1, 3), dim=-1)
    return torch.clamp(keepout.view(1, 1, 1, -1) - dist, min=0).pow(2).sum(-1).mean(dim=(0, 2))


@register("floor")
def floor(I):
    """Keep the gripper points above the table plane (z_table)."""
    if I.ee_quat is None or I.context.get("z_table") is None:
        return _zeros(I)
    below = torch.clamp(I.context["z_table"] - _gripper_points(I)[:3, ..., 2], min=0).pow(2)
    return below.mean(dim=(0, 2))
