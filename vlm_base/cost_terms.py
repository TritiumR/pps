"""Composable cost terms for the VLM-DP base.

Each term is a function ``term(I: CostInputs) -> Tensor[K]`` (unweighted, lower = better) registered in
``TERMS`` by name. A cost is a config-chosen ``{name: weight}`` combination summed by
``base_cost.CompositeCost``. Add a term = write a function + ``@register("name")`` + list it in the config.

Shapes: ``ee_pos`` / ``ee_quat`` are ``[K,H,3]`` / ``[K,H,4]`` (K candidates, H horizon); ``real_actions``
is ``[K,H,8]`` (7 joints + gripper); every term returns ``[K]``. Terms are self-gating -- a term that
does not apply to the current stage (e.g. ``reach`` when a relational constraint is set) returns zeros.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from sim_common.geometry import DEFAULT_EXTENT

TERMS: dict[str, Callable[["CostInputs"], torch.Tensor]] = {}


def register(name: str):
    """Register a cost term under ``name`` (referenced from the config's ``cost.terms``)."""
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
    geom: Any                # namespace: ee_r, coll_clear, finger_r, open_half


def _quat_apply_wxyz(quat, vec):
    """Rotate ``vec`` by ``quat`` (IsaacLab w,x,y,z), broadcasting over shared leading dims."""
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


# --- J_task: reach a target point, OR satisfy a ReKep relational constraint (mutually exclusive) ---

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


# --- regularization / feasibility (always on, but self-gate on missing inputs) ---

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
    """Push the two fingertips to bracket the grasp target (grasp stages only)."""
    if I.ee_quat is None:
        return _zeros(I)
    ctx = I.context
    grasp_obj, objects = ctx.get("grasp_obj"), ctx.get("objects", {})
    if ctx.get("payload") is not None or grasp_obj not in objects:
        return _zeros(I)
    closing, center = _axis(I.ee_quat, 1), _target(I).view(1, 1, 3)
    threshold = I.extents.get(grasp_obj, DEFAULT_EXTENT)[0] + I.geom.finger_r
    left = torch.linalg.vector_norm(I.ee_pos + I.geom.open_half * closing - center, dim=-1)
    right = torch.linalg.vector_norm(I.ee_pos - I.geom.open_half * closing - center, dim=-1)
    return (torch.clamp(threshold - left, min=0).pow(2) + torch.clamp(threshold - right, min=0).pow(2)).mean(1)


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
    """Keep the gripper points above the table plane (``z_table``)."""
    if I.ee_quat is None or I.context.get("z_table") is None:
        return _zeros(I)
    below = torch.clamp(I.context["z_table"] - _gripper_points(I)[:3, ..., 2], min=0).pow(2)
    return below.mean(dim=(0, 2))
