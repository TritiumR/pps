"""Composable cost terms summed config-weighted by CompositeCost.

Each term is term(I: CostInputs) -> Tensor[K], registered in TERMS by name and selected by a config's
cost.terms block. Shapes: ee_pos [K,H,3], real_actions [K,H,8] (7 arm joints plus 1 gripper channel).
Sign convention: a term is a penalty (lower is better, minimized by the sampler) unless its docstring
says reward. Terms self-gate by returning zeros when they do not apply to the current stage, so the
active set is a property of the stage context, not of which terms the config lists (see tests/
test_gating.py). Object extents are (grip, keepout, half_height) half-extents in metres.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from vlm_dp.sim_helpers import DEFAULT_EXTENT

TERMS: dict[str, Callable[["CostInputs"], torch.Tensor]] = {}


def register(name: str):
    """Register a cost term under name, referenced from a config's cost.terms."""
    def deco(fn):
        TERMS[name] = fn
        return fn
    return deco


@dataclasses.dataclass
class CostInputs:
    """Everything a term may read: candidate actions and pose, stage context, extents, gripper geometry."""
    real_actions: torch.Tensor
    ee_pos: torch.Tensor
    ee_quat: torch.Tensor | None
    context: dict[str, Any]
    extents: dict
    geom: Any


def _quat_apply_wxyz(quat, vec):
    """Rotate vec by quat (IsaacLab w,x,y,z), broadcasting over shared leading dims."""
    q = quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8)
    xyz = q[..., 1:]
    t = torch.cross(xyz.expand_as(vec), vec, dim=-1) * 2.0
    return vec + q[..., :1] * t + torch.cross(xyz.expand_as(vec), t, dim=-1)


def _axis(ee_quat, i):
    """World-frame gripper axis i, where local +y is closing and +z is tool approach."""
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
    """Grasp-stage geometry (tip, closing, approach, lateral, center, obj_radius), or None off a grasp.

    obj_radius is the wide horizontal half-extent. The narrow one would understate the close gate.
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
    # Feasibility radius: the local half-width near the VLM keypoint when set (a thin lip fits the
    # gripper, so the pinch terms become satisfiable), else the whole-object extent (compact objects).
    ge = objects.get(grasp_obj, {}).get("grasp_extent")
    radius = ge if ge is not None else I.extents.get(grasp_obj, DEFAULT_EXTENT)[1]
    return tip, closing, approach, lateral, center, radius


def grasp_slack(geom, radius):
    """How far the TCP may sit from the grasp point and still engage the object when the fingers close.

    The single definition of at-the-grasp-pose: the lateral slack between the object and the open
    fingers, open_half - radius - aperture_margin. Derived from gripper geometry and measured object
    size, no free constant. It shrinks as the object grows (a fat object must be centred precisely) and
    grows for a thin one (a lip has most of the aperture to fall into).

    The proportional form it replaces (center_scale * radius) is anti-correlated with the physics: it
    gave a 10 mm lid lip a 4 mm tolerance while the fingers had about 30 mm of room, so the gripper was
    never commanded to close and the press stage waited out the episode. The stage machine calls this
    too, so reached and closing cannot drift apart. Set grasp_dead_zone=proportional to restore the
    legacy form.
    """
    if getattr(geom, "grasp_dead_zone", "aperture") == "proportional":
        return geom.center_scale * radius
    slack = geom.open_half - radius - getattr(geom, "aperture_margin", 0.0)
    return max(slack, getattr(geom, "close_xy_floor", 1e-3))


def _rekep_keypoints(I):
    """Per-candidate keypoints [N,K,H,3] with held keypoints riding the candidate gripper pose."""
    ee_pos, ee_quat, ctx = I.ee_pos, I.ee_quat, I.context
    dev, dt = ee_pos.device, ee_pos.dtype
    kp = torch.as_tensor(ctx["keypoints"], device=dev, dtype=dt)[:, None, None, :]
    kp = kp.expand(-1, ee_pos.shape[0], ee_pos.shape[1], -1).clone()
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


# ============================================================================ goal terms (required)
# Reach a point or satisfy a VLM constraint. Both gate off on the place stage, where place_descent leads.


def _placing(ctx):
    """True on a place stage: a payload is carried and a place target is set (not lift, not grasp)."""
    return ctx.get("payload") is not None and ctx.get("place_target") is not None


@register("reach")
def reach(I):
    """Penalty: mean squared TCP-to-target distance.

    Off when constrained or a payload is held. A held object's height is the payload terms' job, and an
    end-effector height attractor would fight them by the grasp offset.
    """
    if I.context.get("constraint") is not None or I.context.get("payload") is not None:
        return _zeros(I)
    return ((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1).mean(dim=1)


@register("terminal_reach")
def terminal_reach(I):
    """Penalty: terminal squared TCP-to-target distance. Same gating as reach."""
    if I.context.get("constraint") is not None or I.context.get("payload") is not None:
        return _zeros(I)
    return ((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1)[:, -1]


@register("rekep_subgoal")
def rekep_subgoal(I):
    """Constraint approximation: the ReKep subgoal relation over the horizon. Off without a constraint.

    Summed by default so existing configs are unchanged. subgoal_mean averages it instead, so a driven
    cost can balance it against the squared execution terms (collision, grasp) it would otherwise dwarf:
    summed and linear it is about 10x a meaned-squared term and plows straight through obstacles.
    """
    subgoal = I.context.get("constraint")
    if subgoal is None:
        return _zeros(I)
    v = subgoal(I.ee_pos, _rekep_keypoints(I))
    return v.mean(dim=1) if getattr(I.geom, "subgoal_mean", False) else v.sum(dim=1)


@register("rekep_path")
def rekep_path(I):
    """Constraint approximation: ReKep running path constraints (per-step geometric), summed over H."""
    if I.context.get("constraint") is None or not I.context.get("path_fns", ()):
        return _zeros(I)
    kp = _rekep_keypoints(I)
    cost = _zeros(I)
    for path_fn in I.context["path_fns"]:
        v = path_fn(I.ee_pos, kp)
        if torch.is_tensor(v) and v.ndim == 2:
            cost = cost + torch.clamp(v, min=0).sum(dim=1)
    return cost


# ================================================================== regularizers (always applicable)
# Penalties that shape the plan itself. They self-gate only when their inputs are missing.

@register("smooth")
def smooth(I):
    """Penalty: mean squared consecutive joint change. Per-element mean keeps weights shape-independent."""
    joints = I.real_actions[..., :7]
    return ((joints[:, 1:] - joints[:, :-1]) ** 2).mean(dim=(-1, -2))


@register("gripper_smooth")
def gripper_smooth(I):
    """Penalty: mean squared consecutive change of the gripper channel. smooth covers arm joints only."""
    if I.real_actions.shape[-1] <= 7 or I.real_actions.shape[1] < 2:
        return _zeros(I)
    gripper = I.real_actions[..., 7]
    return (gripper[:, 1:] - gripper[:, :-1]).pow(2).mean(dim=1)


@register("joint_delta")
def joint_delta(I):
    """Penalty: trust region toward the current joints. A dwelling plan is what lets the gripper close."""
    joints = I.real_actions[..., :7]
    current = I.context.get("joint_pos")
    if current is None:
        return _zeros(I)
    current = torch.as_tensor(current, device=joints.device, dtype=joints.dtype)
    if current.ndim > 1 and current.shape[0] == 1:
        current = current[0]
    return ((joints - current[:7].view(1, 1, 7)) ** 2).mean(dim=(-1, -2))


@register("orientation")
def orientation(I):
    """Penalty: downward tool-axis misalignment, (1 - approach . down) meaned over the horizon.

    Self-disables when the sub-goal commands a rotation (orient == free): a fixed downward prior is a
    grasp-approach convenience, and left on it fights the VLM's own tilt so a pour could never tip. The
    VLM owns the orientation whenever it constrains one.
    """
    if I.ee_quat is None or I.context.get("orient", "down") != "down":
        return _zeros(I)
    down = torch.tensor([0.0, 0.0, -1.0], device=I.ee_quat.device, dtype=I.ee_quat.dtype)
    return (1.0 - (_axis(I.ee_quat, 2) * down.view(1, 1, 3)).sum(dim=-1)).mean(dim=1)


@register("consistency")
def consistency(I):
    """Penalty: mean squared deviation from the previous plan. Mean-reduced, since summed it would
    collapse the softmax."""
    ref = I.context.get("plan_ref")
    joints = I.real_actions[..., :7]
    if ref is None or ref.shape[0] != joints.shape[1]:
        return _zeros(I)
    return ((joints - ref.view(1, -1, 7)) ** 2).mean(dim=(1, 2))


# ================================================================================= grasp terms (pinch)
# Certify and execute a straddle-pinch. _pinch_gated stands the certification terms down on a press
# contact (a thin or articulated part a pinch cannot engage), where press reaches the point and closes.

def _pinch_gated(I):
    """True on a press contact, where the pinch-certification terms stand down (a lid, not a free body)."""
    return I.context.get("contact") == "press"


@register("straddle")
def straddle(I):
    """Constraint: fingertips bracket the grasp target without penetrating it (xy keepout). Grasp only."""
    if _pinch_gated(I):
        return _zeros(I)
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
    """Penalty: fingertip at the grasp height, squared z offset. Grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, _, _, _, center, _ = f
    return ((tip[..., 2] - center[..., 2]) ** 2).mean(dim=1)


@register("yaw")
def yaw(I):
    """Penalty: align the closing axis with a world x or y axis (top-down grasp yaw). Grasp stages only."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    closing = f[1]
    return (1.0 - torch.maximum(closing[..., 0].abs(), closing[..., 1].abs())).mean(dim=1)


@register("grasp_axis")
def grasp_axis(I):
    """Penalty: close across the object's narrow horizontal axis (the dimension that fits and holds),
    instead of the fixed world-axis yaw.

    Inert (zeros) when the object is round and has no measured axis, so round fruit fall back to the
    yaw term unchanged.
    """
    if _pinch_gated(I):
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    axis = I.context.get("objects", {}).get(I.context.get("grasp_obj"), {}).get("axis")
    if axis is None:
        return _zeros(I)
    closing = f[1]
    n = torch.as_tensor(axis, device=I.ee_pos.device, dtype=I.ee_pos.dtype)
    align = (closing[..., :2] * n[:2].view(1, 1, 2)).sum(-1)   # closing dot narrow-axis, horizontal
    return (1.0 - align.abs()).mean(dim=1)


@register("center_region")
def center_region(I):
    """Constraint: keep the grasp point within a region around the gripper centre. Grasp stages only."""
    if _pinch_gated(I):
        return _zeros(I)
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
    """Constraint: keep the object within the gripper opening along the closing axis. Grasp stages only."""
    if _pinch_gated(I):
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, closing, _, _, center, radius = f
    closing_coord = ((center - tip) * closing).sum(-1)
    return torch.clamp(closing_coord.abs() + radius + I.geom.aperture_margin - I.geom.open_half,
                       min=0.0).pow(2).mean(dim=1)


@register("close_gripper")
def close_gripper(I):
    """Penalty: shut the gripper when the measured TCP is at the grasp pose.

    Gate on context['eef_pos'], never the candidate, which would turn the term into a repulsive barrier.
    Do not raise this weight.
    """
    if I.context.get("gripper_intent") == "open":   # reopen recovery owns the channel
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    _, _, _, _, center, radius = f
    eef = I.context.get("eef_pos")
    # Only term that closes the gripper, so fail loudly rather than silently returning zeros.
    if eef is None:
        raise KeyError("close_gripper needs context['eef_pos'], the measured TCP, to command the grasp.")
    tcp = torch.as_tensor(eef, device=I.ee_pos.device, dtype=I.ee_pos.dtype).reshape(-1)[:3].view(1, 1, 3)
    rel = center - tcp
    err = torch.linalg.vector_norm(rel, dim=-1)
    center_radius = grasp_slack(I.geom, radius)
    z_scale = I.geom.close_z_scale
    # Gate-width floor: a tiny object otherwise makes the gate narrower than TCP wobble and the
    # commanded gripper dithers open and closed.
    xy_scale = max(center_radius, getattr(I.geom, "close_xy_floor", 1e-3))
    gate = torch.exp(-torch.clamp(err - center_radius, min=0.0).pow(2) / max(xy_scale * xy_scale, 1e-8)
                     - torch.clamp(rel[..., 2].abs() - z_scale, min=0.0).pow(2) / max(z_scale * z_scale, 1e-8))
    return (I.real_actions[..., 7] - gate).pow(2).mean(dim=1)


@register("grasp_commit")
def grasp_commit(I):
    """Reward-and-penalty: reward closing while at the grasp pose, penalise closing away from it.

    Optional alternative to close_gripper (use one or the other, not both). close_gripper gates on the
    measured TCP, so its target is one scalar shared by every candidate and horizon step: an action
    chunk cannot express a grasp event (approach, then close) and a steering proxy has no gripper degree
    of freedom to move. Here the proximity factor is evaluated on the candidate, so the term varies
    across samples and along the horizon, and the grasp decision enters the distribution and becomes
    steerable.

    It does not reproduce the repulsive barrier that made the measured gate necessary: in the old
    squared-tracking form a candidate-dependent target scored stay-away-and-open the same as
    arrive-and-close (both zero), so the sampler avoided the grasp. Here staying away and open scores 0
    while arriving and closing scores -1, so arriving is strictly better and the degeneracy is gone.

    Pure candidate evaluation closes prematurely: the sampler can plan a near-and-closed pose before the
    arm is physically there, so the executed gripper shuts on air and thrashes (measured: 85 closed-empty
    vs 23 for close_gripper, N=10). Set commit_measured_gate to modulate the reward by a looser measured
    proximity: no closing reward until the arm is actually in the neighborhood, while the candidate still
    controls the grasp timing within it, so the term stays steerable but stops closing early.
    """
    if I.context.get("gripper_intent") == "open":     # reopen recovery owns the channel
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    tip, _, _, _, center, radius = f
    dead = grasp_slack(I.geom, radius)                # shared at-the-grasp-pose tolerance
    scale = max(dead, getattr(I.geom, "close_xy_floor", 1e-3))
    err = torch.linalg.vector_norm(center - tip, dim=-1)                      # [K,H], per-candidate
    prox = torch.exp(-torch.clamp(err - dead, min=0.0).pow(2) / max(scale * scale, 1e-8))
    grip = I.real_actions[..., 7].clamp(0.0, 1.0)     # bounded, else the reward is unbounded below
    reward = grip * (1.0 - 2.0 * prox)                # near gives -grip (reward), far gives +grip (penalty)
    if getattr(I.geom, "commit_measured_gate", False):
        # Modulate by measured proximity: 1 when the arm is in the neighbourhood, 0 far, so the reward
        # cannot command closing before the arm arrives. The band is a few dead-zones wide, so candidate
        # timing still varies within it (steerable), only premature closing is removed.
        eef = I.context.get("eef_pos")
        if eef is not None:
            tcp = torch.as_tensor(eef, device=I.ee_pos.device, dtype=I.ee_pos.dtype).reshape(-1)[:3]
            m_err = torch.linalg.vector_norm(center.view(3) - tcp)           # scalar, measured
            band = getattr(I.geom, "commit_measured_band", 3.0) * scale
            reward = reward * torch.exp(-torch.clamp(m_err - band, min=0.0).pow(2) / max(band * band, 1e-8))
    return reward.mean(dim=1)


@register("grasp_region")
def grasp_region(I):
    """Penalty: distance to the graspable region (a segment along the object's long axis), not a point.

    Optional alternative to the point attraction of reach and terminal_reach on grasp stages. A single
    grasp point pins the base distribution to one pose, so a steering proxy can only perturb the path to
    a pose already chosen. Scoring distance to a segment leaves the sampler free to pick where along the
    graspable span to grasp, the degree of freedom steering needs to express grasp-the-handle-end.
    Degenerates to the point term for a round object with no measured span, so nothing changes where
    there is genuinely one sensible grasp.
    """
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, _, _, _, center, _ = f
    reg = I.context.get("objects", {}).get(I.context.get("grasp_obj"), {}).get("grasp_region")
    if reg is None:                                    # round or no measured span, point attraction
        return ((tip - center) ** 2).sum(dim=-1).mean(dim=1)
    axis = torch.as_tensor(reg[0], device=tip.device, dtype=tip.dtype).view(1, 1, 3)
    half = float(reg[1])
    d = tip - center
    t = (d * axis).sum(dim=-1, keepdim=True).clamp(-half, half)     # project onto the segment
    closest = center + t * axis
    return ((tip - closest) ** 2).sum(dim=-1).mean(dim=1)


# ==================================================================== carry, lift and place (required)
# Terms active while a payload is held. lift_* drive the lift column, place_* and carry_* drive the
# transit and set-down. carry_hold owns the gripper channel throughout.

@register("release_gripper")
def release_gripper(I):
    """Penalty: open the gripper on stages whose intent is release, such as letting go of a pulled lid."""
    if I.context.get("gripper_intent") != "open" or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    return I.real_actions[..., 7].pow(2).mean(dim=1)


@register("carry_hold")
def carry_hold(I):
    """Penalty: hold the gripper closed while carrying, release once the payload reaches the target."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    gripper = I.real_actions[..., 7]
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    place = objects.get(ctx.get("place_target"))
    if place is None:
        # Lift phase: no place object, so just hold the gripper closed.
        return (gripper - 1.0).pow(2).mean(dim=1)
    if getattr(I.geom, "release_on_subgoal", False) and ctx.get("constraint") is not None:
        # VLM-driven release (opt-in): open once the sub-goal is satisfied (the held object is at the
        # VLM's target), not at a hand-coded geometric seat. Stays a steerable gate on the sub-goal value.
        val = ctx["constraint"](I.ee_pos, _rekep_keypoints(I))
        release = (val < getattr(I.geom, "subgoal_release_eps", 0.05)).to(dt)
        return (gripper - (1.0 - release)).pow(2).mean(dim=1)
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    if getattr(I.geom, "release_commit", False):
        # Commitment-form release (opt-in): a smooth candidate-evaluated commitment instead of the hard
        # AND-threshold below. The hard gate flips the gripper target from closed to open at a boundary.
        # Near it, candidates split and the MBD average sits mid-open, so the gripper dithers and burns
        # rollout (measured: long hesitation at the seat). Here opening is rewarded continuously as the
        # carried payload nears the seat and penalised far from it, so far-and-closed and near-and-open
        # are each strictly best and the transition commits smoothly. Same idea as grasp_commit, opposite
        # polarity.
        dist = torch.linalg.vector_norm(carried - dest.view(1, 1, 3), dim=-1)             # [K,H]
        dead = I.geom.release_xy
        scale = max(dead, getattr(I.geom, "close_xy_floor", 1e-3))
        prox = torch.exp(-torch.clamp(dist - dead, min=0.0).pow(2) / max(scale * scale, 1e-8))
        open_amt = (1.0 - gripper).clamp(0.0, 1.0)     # bounded, else the reward is unbounded below
        return (open_amt * (1.0 - 2.0 * prox)).mean(dim=1)   # near rewards opening, far penalises it
    xy_dist = torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
    # One-sided: at or below the hover releases, so a seated payload does not re-trigger a close.
    z_err = carried[..., 2] - dest[2]
    release = ((xy_dist < I.geom.release_xy) & (z_err < I.geom.release_z)).to(dt)
    return (gripper - (1.0 - release)).pow(2).mean(dim=1)


def _carried_pos(I, payload_pos):
    """Carried payload per candidate: candidate TCP + (measured payload - measured TCP).

    Anchoring on the candidate's own first step instead would make every payload term
    translation-invariant, so a sinking chunk would score like a rising one.
    """
    tcp = torch.as_tensor(I.context["eef_pos"], device=I.ee_pos.device, dtype=I.ee_pos.dtype)
    return I.ee_pos + (payload_pos - tcp.reshape(-1)[:3]).view(1, 1, 3)


def _place_errors(I):
    """Per-step (xy [K,H], z_err [K,H]) of the carried payload to the seat, with the ramped carry
    height. None off a place stage."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    place = objects.get(ctx.get("place_target"))
    if payload is None or payload not in objects or place is None:
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    outer = getattr(I.geom, "place_descend_radius", 0.18)
    inner = getattr(I.geom, "place_seat_radius", 0.10)
    carry_clear = getattr(I.geom, "place_carry_clear", 0.10)
    xy = torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
    # Smooth ramp, not a hard gate: a gate stalls just outside the threshold and never lowers.
    frac = ((xy - inner) / max(outer - inner, 1e-3)).clamp(0.0, 1.0)
    z_target = dest[2] + frac * carry_clear
    return xy, carried[..., 2] - z_target


def _lift_errors(I):
    """Per-step (xy, z) error of the carried payload to the lift column. None off a lift stage."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or _placing(ctx):
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    tgt = _target(I).view(1, 1, 3)
    xy = torch.linalg.vector_norm(carried[..., :2] - tgt[..., :2], dim=-1)
    return xy, carried[..., 2] - tgt[..., 2]


@register("lift_xy")
def lift_xy(I):
    """Penalty: keep the carried payload in the lift column (squared xy error)."""
    e = _lift_errors(I)
    return _zeros(I) if e is None else e[0].pow(2).mean(dim=1)


@register("lift_z")
def lift_z(I):
    """Penalty: drive the carried payload to the lift height (squared z error)."""
    e = _lift_errors(I)
    return _zeros(I) if e is None else e[1].pow(2).mean(dim=1)


@register("lift_terminal")
def lift_terminal(I):
    """Penalty: end-of-chunk pull to the lift point."""
    e = _lift_errors(I)
    return _zeros(I) if e is None else (e[0].pow(2) + e[1].pow(2))[:, -1]


@register("lift_reach")
def lift_reach(I):
    """Penalty: horizon-mean full pull to the lift point."""
    e = _lift_errors(I)
    return _zeros(I) if e is None else (e[0].pow(2) + e[1].pow(2)).mean(dim=1)


def _place_dest(I, objects, payload, ctx):
    """Release point above the seat: top-surface point + payload half-height + clearance."""
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    clearance = getattr(I.geom, "place_release_clearance", 0.0)
    pp = ctx.get("place_point")
    if pp is not None:                       # calibrated top-surface point (root may differ from centre)
        return (torch.as_tensor(pp, device=dev, dtype=dt)
                + torch.tensor([0.0, 0.0, I.extents.get(payload, DEFAULT_EXTENT)[2] + clearance],
                               device=dev, dtype=dt))
    place_pos = torch.as_tensor(objects[ctx["place_target"]]["pos"], device=dev, dtype=dt)
    stack = I.extents.get(ctx["place_target"], DEFAULT_EXTENT)[2] + I.extents.get(payload, DEFAULT_EXTENT)[2]
    return place_pos + torch.tensor([0.0, 0.0, stack + clearance], device=dev, dtype=dt)


def _place_frame(I):
    """Place geometry (xy, z_err, outside, below_carry) of the carried payload, with the binary descend
    gate and the lift-altitude carry height. None off a place stage."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or objects.get(ctx.get("place_target")) is None:
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    carry_z = ctx.get("carry_z")
    carry_z = dest[2] + 0.20 if carry_z is None else torch.as_tensor(float(carry_z), device=dev, dtype=dt)
    xy = torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
    outside = (xy > getattr(I.geom, "place_descend_radius", 0.08)).to(dt)
    z_err = carried[..., 2] - (dest[2] + outside * (carry_z - dest[2]))
    below_carry = torch.clamp(carry_z - carried[..., 2], min=0.0)
    return xy, z_err, outside, below_carry


@register("place_reach")
def place_reach(I):
    """Penalty: horizon-mean full pull to the place point."""
    f = _place_frame(I)
    return _zeros(I) if f is None else (f[0].pow(2) + f[1].pow(2)).mean(dim=1)


@register("place_xy")
def place_xy(I):
    """Penalty: carried payload over the place point (squared xy error)."""
    f = _place_frame(I)
    return _zeros(I) if f is None else f[0].pow(2).mean(dim=1)


@register("place_z")
def place_z(I):
    """Penalty: carried payload at the gated place height (squared z error)."""
    f = _place_frame(I)
    return _zeros(I) if f is None else f[1].pow(2).mean(dim=1)


@register("place_carry_height")
def place_carry_height(I):
    """Penalty: one-sided carry-altitude floor while outside the descend radius."""
    f = _place_frame(I)
    return _zeros(I) if f is None else (f[2] * f[3].pow(2)).mean(dim=1)


@register("carry_liftoff")
def carry_liftoff(I):
    """Penalty: while carrying and still horizontally far from the destination, pull the payload up to a
    lift-off clearance (z_table + margin) so it rises along the grasp approach axis before translating.

    Linear and one-sided so it can stand up to the linear horizon-summed place sub-goal, which a squared
    z term cannot. Off near the destination (the far gate) so the sub-goal still descends and places.
    Derived from grasp geometry and the support plane, not a hand-coded place height.
    """
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    place = objects.get(ctx.get("place_target"))
    z_table = ctx.get("z_table")
    if payload is None or payload not in objects or place is None or z_table is None:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    place_xy = torch.as_tensor(place["pos"], device=dev, dtype=dt)[:2].view(1, 1, 2)
    far = (torch.linalg.vector_norm(carried[..., :2] - place_xy, dim=-1)
           > getattr(I.geom, "place_descend_radius", 0.08)).to(dt)
    liftoff_z = float(z_table) + getattr(I.geom, "carry_liftoff_clear", 0.20)
    lift = torch.clamp(liftoff_z - carried[..., 2], min=0.0)   # linear, rise remaining
    return (far * lift).mean(dim=1)


# ================================================================================ collision (required)
# Soft keepout terms. collision and clear guard the gripper, carry_clear guards the carried payload.

@register("place_descent")
def place_descent(I):
    """Penalty: fly the payload at carry height, then ramp it down onto the place seat.

    Sole arm attractor on the place stage, where the reach terms gate off.
    """
    e = _place_errors(I)
    if e is None:
        return _zeros(I)
    xy, z_err = e
    return (xy.pow(2) + z_err.pow(2)).mean(dim=1)


@register("place_terminal")
def place_terminal(I):
    """Penalty: end-of-chunk pull to the place point."""
    f = _place_frame(I)
    return _zeros(I) if f is None else (f[0].pow(2) + f[1].pow(2))[:, -1]


@register("collision")
def collision(I):
    """Constraint: soft keepout from non-manipulated objects as upright cylinders.

    A sphere would over-approximate flat supports and block top-down grasps.
    """
    if I.ee_quat is None:
        return _zeros(I)
    ctx = I.context
    objects = ctx.get("objects", {})
    excluded = {ctx.get("grasp_obj"), ctx.get("payload"), ctx.get("place_target"),
                ctx.get("destination")}          # the task destination is never an obstacle
    excluded |= set(ctx.get("placed") or ())     # set down on the place target: destination, not obstacle
    names = [n for n in objects if n not in excluded]
    if not names:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    # Assumes an object's root pose is its geometric centre. A base-origin asset would straddle the floor
    # and vanish (contact needs both overlaps), so check a new asset's origin before its extents.
    centers = torch.stack([torch.as_tensor(objects[n]["pos"], device=dev, dtype=dt) for n in names], 0)
    ext = torch.stack([torch.as_tensor(I.extents.get(n, DEFAULT_EXTENT), device=dev, dtype=dt) for n in names], 0)
    margin = I.geom.ee_r + I.geom.coll_clear
    r_xy = (ext[:, 1] + margin).view(1, 1, 1, -1)
    r_z = (ext[:, 2] + margin).view(1, 1, 1, -1)
    d = _gripper_points(I)[..., None, :] - centers.view(1, 1, 1, -1, 3)
    pen_xy = torch.clamp(r_xy - torch.linalg.vector_norm(d[..., :2], dim=-1), min=0)
    pen_z = torch.clamp(r_z - d[..., 2].abs(), min=0)
    return torch.minimum(pen_xy, pen_z).pow(2).sum(-1).mean(dim=(0, 2))


@register("clear")
def clear(I):
    """Constraint: TCP sphere keepout from compact movable objects.

    Objects wider than clear_max_radius are fixtures: a sphere over a board or appliance would wall off
    the workspace.
    """
    ctx = I.context
    objects = ctx.get("objects", {})
    # Placed objects stay obstacles: they repel the next payload's approach.
    excluded = {ctx.get("grasp_obj"), ctx.get("payload"), ctx.get("place_target"), ctx.get("destination")}
    if getattr(I.geom, "clear_exclude_placed", False):   # no-seats ablation restores the old exclusion
        excluded |= set(ctx.get("placed") or ())
    max_r = getattr(I.geom, "clear_max_radius", 0.10)
    names = [n for n in objects if n not in excluded and I.extents.get(n, DEFAULT_EXTENT)[1] <= max_r]
    if not names:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    margin = I.geom.ee_r + I.geom.coll_clear
    cost = _zeros(I)
    for n in names:
        center = torch.as_tensor(objects[n]["pos"], device=dev, dtype=dt)
        keepout = I.extents.get(n, DEFAULT_EXTENT)[1] + margin
        dist = torch.linalg.vector_norm(I.ee_pos - center.view(1, 1, 3), dim=-1)
        cost = cost + torch.clamp(keepout - dist, min=0.0).pow(2).mean(dim=1)
    return cost


@register("carry_clear")
def carry_clear(I):
    """Constraint: keep the carried payload clear of compact obstacles during transit.

    The carry routes over or around obstacles instead of beelining through: emergent obstacle avoidance
    where a lift-over happens only where the geometry needs it, not a hard-coded lift. Sphere keepout,
    so raising z clears. Set carry_clear_include_placed to keep an already-placed object as a soft carry
    obstacle so the next payload seats in the free space beside it (co-placement).
    """
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))  # [K,H,3]
    excluded = {ctx.get("grasp_obj"), payload, ctx.get("place_target"), ctx.get("destination")}
    if not getattr(I.geom, "carry_clear_include_placed", False):
        excluded |= set(ctx.get("placed") or ())   # default: placed objects are destination, not obstacle
    max_r = getattr(I.geom, "clear_max_radius", 0.10)   # fixtures such as board or scale are not obstacles
    names = [n for n in objects if n not in excluded and I.extents.get(n, DEFAULT_EXTENT)[1] <= max_r]
    if not names:
        return _zeros(I)
    pay_r = I.extents.get(payload, DEFAULT_EXTENT)[1]
    margin = getattr(I.geom, "carry_clear_margin", I.geom.coll_clear)
    cost = _zeros(I)
    for n in names:
        center = torch.as_tensor(objects[n]["pos"], device=dev, dtype=dt)
        keepout = pay_r + I.extents.get(n, DEFAULT_EXTENT)[1] + margin
        dist = torch.linalg.vector_norm(carried - center.view(1, 1, 3), dim=-1)
        # Linear penetration (constant gradient), not squared: a squared cm-scale penetration is dwarfed
        # by the sub-goal and the payload plows through. Linear with a wide margin routes it over.
        cost = cost + torch.clamp(keepout - dist, min=0.0).mean(dim=1)
    return cost


@register("carry_altitude")
def carry_altitude(I):
    """Penalty: while carrying and horizontally far from the destination, keep the payload above a
    transit clearance (support plane + margin) so it rises straight up and stays clear of the table and
    other objects.

    Continuous whenever a payload is carried, not a stage. Off near the destination, where place_setdown
    takes over. Linear, to stand up to the balanced sub-goal.
    """
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    place = objects.get(ctx.get("place_target"))
    z_table = ctx.get("z_table")
    if payload is None or payload not in objects or place is None or z_table is None:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    place_xy = torch.as_tensor(place["pos"], device=dev, dtype=dt)[:2].view(1, 1, 2)
    far = (torch.linalg.vector_norm(carried[..., :2] - place_xy, dim=-1)
           > getattr(I.geom, "place_descend_radius", 0.10)).to(dt)
    floor_z = float(z_table) + getattr(I.geom, "carry_altitude_clear", 0.12)
    return (far * torch.clamp(floor_z - carried[..., 2], min=0.0)).mean(dim=1)


@register("place_setdown")
def place_setdown(I):
    """Penalty: once horizontally over the destination, lower the payload onto the surface below it (the
    place object's top) so it is set down, not dropped from the VLM's approach height.

    The support-gated release (carry_hold with release_on_subgoal off) opens once it rests.
    """
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or objects.get(ctx.get("place_target")) is None:
        return _zeros(I)
    if ctx.get("place_mode") == "container":
        # Placing inside: the destination's top surface is its rim, so setting down onto it would seat
        # the object on the rim instead of in the cavity. The VLM's sub-goal already gives the drop point.
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    near = (torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
            <= getattr(I.geom, "place_descend_radius", 0.10)).to(dt)
    return (near * torch.clamp(carried[..., 2] - dest[2], min=0.0)).mean(dim=1)


@register("floor")
def floor(I):
    """Constraint: keep the gripper points above the table plane z_table."""
    if I.ee_quat is None or I.context.get("z_table") is None:
        return _zeros(I)
    below = torch.clamp(I.context["z_table"] - _gripper_points(I)[:3, ..., 2], min=0).pow(2)
    return below.mean(dim=(0, 2))
