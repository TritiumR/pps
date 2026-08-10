"""Composable, stage-gated cost terms for CompositeCost."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from vlm_dp.sim_helpers import DEFAULT_EXTENT

# Release-commitment sigmoid gain: saturates within ~30% of tolerance.
_RELEASE_SHARPNESS = 10.0

TERMS: dict[str, Callable[["CostInputs"], torch.Tensor]] = {}


def register(name: str):
    """Register a cost term under name, referenced from a config's cost.terms."""
    def deco(fn):
        TERMS[name] = fn
        return fn
    return deco


@dataclasses.dataclass
class CostInputs:
    """Everything a term may read: candidate actions and pose, stage context, extents,
    gripper geometry.
    """
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
    """Four gripper points [4,K,H,3]: TCP, both fingertips, and a point back up the tool
    axis.
    """
    closing, approach = _axis(I.ee_quat, 1), _axis(I.ee_quat, 2)
    return torch.stack([I.ee_pos, I.ee_pos + I.geom.open_half * closing,
                        I.ee_pos - I.geom.open_half * closing, I.ee_pos - I.geom.tool_back * approach], 0)


def _grasp_frame(I):
    """Grasp-stage geometry (tip, closing, approach, lateral, center, obj_radius), or None
    off a grasp.
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
    # Feasibility radius: the local half-width near the VLM keypoint when set, else the whole-object.
    ge = objects.get(grasp_obj, {}).get("grasp_extent")
    radius = ge if ge is not None else I.extents.get(grasp_obj, DEFAULT_EXTENT)[1]
    return tip, closing, approach, lateral, center, radius


def grasp_slack(geom, radius):
    """Lateral slack at which closing still engages the object: open_half - radius -
    aperture_margin.
    """
    if getattr(geom, "grasp_dead_zone", "aperture") == "proportional":
        return geom.center_scale * radius
    slack = geom.open_half - radius - getattr(geom, "aperture_margin", 0.0)
    return max(slack, getattr(geom, "close_xy_floor", 1e-3))


def _rekep_keypoints(I):
    """Per-candidate keypoints [N,K,H,3] with held keypoints riding the candidate gripper
    pose.
    """
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


# Goal terms


def _placing(ctx):
    """True on a place stage: a payload is carried and a place target is set (not lift, not
    grasp).
    """
    return ctx.get("payload") is not None and ctx.get("place_target") is not None


def _flat(err, geom):
    """Clamp an error to zero inside a tolerance, so the term is FLAT over the acceptable set.

    MBD weights candidates by softmax(-cost), which is shift-invariant -- subtracting a tolerance
    does nothing. Only clamping makes every configuration inside the tolerance score identically,
    which is what gives the sampler more than one optimum to hold mass on (measured ESS ~1 without
    it). center_region and aperture_region already do this; the goal attractors and tip_z do not,
    and an unclamped term re-selects a single point inside any region the others leave flat.

    Feasibility terms (floor, clear, collision) are deliberately NOT clamped -- a dead zone on
    collision would license penetration.
    """
    tol = float(getattr(geom, "flat_tol", 0.0) or 0.0)
    return err if tol <= 0.0 else torch.clamp(err - tol, min=0.0)


def _progress(values, geom):
    """Reduce a per-row distance series to a cost.

    Default sums the distance itself, which penalises ANY path that is not monotonically
    approaching -- so going up before going across costs more than a straight line even when it
    is the correct motion. That bias is why the config carries explicit permission-slip terms
    (release_rise_first, grasp_standoff, approach_hover) whose only job is to license particular
    detours.

    `potential_shaping` costs the INCREASE in distance instead, max(0, d_t - d_{t-1}). Any
    non-increasing path is then free, so detours need no licensing. This is potential-based
    reward shaping (Ng et al. 1999): adding a potential difference leaves the optimal policy
    unchanged, so it removes the bias without changing what "solved" means.
    """
    if not getattr(geom, "potential_shaping", False):
        return values.mean(dim=1)
    step = values[:, 1:] - values[:, :-1]
    return torch.clamp(step, min=0.0).sum(dim=1)


def _yields_to_constraint(I):
    """True when a live ReKep sub-goal exists and the attractors should stand down for it.

    reach/terminal_reach have always self-disabled under a constraint: the plan's own sub-goal is
    meant to own the objective wherever it speaks, and two attractors on one stage would double-
    charge it. That is right for a controller, and WRONG for an experiment whose whole question is
    "attractor vs ReKep", because under a grounding that supplies a constraint everywhere the
    attractor arm silently has no attractor and the comparison is against nothing.

    `attractors_under_constraint` suspends the hand-off so both can be measured on the same
    grounding. Default False, which is the shipped behaviour exactly.
    """
    return (I.context.get("constraint") is not None
            and not getattr(I.geom, "attractors_under_constraint", False))


@register("reach")
def reach(I):
    """Penalty: mean squared TCP-to-target distance."""
    if I.context.get("payload") is not None or _yields_to_constraint(I):
        return _zeros(I)
    d = _flat(((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1).sqrt(), I.geom) ** 2
    return _progress(d, I.geom)


@register("terminal_reach")
def terminal_reach(I):
    """Penalty: terminal squared TCP-to-target distance."""
    if I.context.get("payload") is not None or _yields_to_constraint(I):
        return _zeros(I)
    return _flat(((I.ee_pos - _target(I).view(1, 1, 3)) ** 2).sum(dim=-1).sqrt(), I.geom)[:, -1] ** 2


@register("rekep_subgoal")
def rekep_subgoal(I):
    """Constraint approximation: the ReKep subgoal relation over the horizon."""
    subgoal = I.context.get("constraint")
    if subgoal is None:
        return _zeros(I)
    v = subgoal(I.ee_pos, _rekep_keypoints(I))
    # ReKep constraints are satisfied at f(x) <= 0, so only POSITIVE values are violations.
    # Summing the raw signed output let a satisfied constraint cancel a violated one, which
    # rekep_path (immediately below) already avoids with the same clamp.
    v = torch.clamp(v, min=0)
    # Clamp FIRST, then the optional shaping: _flat's max(0, err - tol) assumes a non-negative
    # error, and _progress should charge increases in violation, not in signed constraint value.
    v = _flat(v, I.geom)
    if getattr(I.geom, "potential_shaping", False):
        return _progress(v, I.geom)
    return v.mean(dim=1) if getattr(I.geom, "subgoal_mean", False) else v.sum(dim=1)


@register("rekep_keypose")
def rekep_keypose(I):
    """Constraint: the ReKep subgoal evaluated on the LAST chunk row only.

    The keypose-steering term. `rekep_subgoal` sums or means the subgoal over every row, so under
    keypose steering -- where proposals perturb one row of sixteen -- its response to a proposal is
    diluted ~16x (measured: 3.0% relative cost spread across 128 proposals, too flat for the
    Feynman-Kac softmax to discriminate). This scores the keypose row alone, mirroring the terminal
    keypose cost Cory's method weights at 30, and it reads the VLM/GT sub-goal rather than the
    motion scaffold -- so it cannot inherit `carry_hold`'s measured inversion (E4: +22 at insert).

    Requires a stage constraint; inert on stages without one, and on any config that does not
    weight it.
    """
    subgoal = I.context.get("constraint")
    if subgoal is None:
        return _zeros(I)
    v = subgoal(I.ee_pos, _rekep_keypoints(I))
    # ReKep convention: f <= 0 is satisfied. Without the clamp an over-satisfied keypose pays
    # negative cost and dominates the softmax.
    return torch.clamp(v[:, -1], min=0)


@register("rekep_path")
def rekep_path(I):
    """Constraint approximation: ReKep running path constraints (per-step geometric).

    Reduced over H by sum (default, back-compatible) or by mean under the geometry flag
    `rekep_path_mean`. The mean makes it commensurate with the mean-reduced feasibility terms,
    so one weight means the same thing regardless of horizon length; the sum scales with H and
    is kept as the default because the shipped configs weight this term at 200.0 against it.
    """
    if I.context.get("constraint") is None or not I.context.get("path_fns", ()):
        return _zeros(I)
    kp = _rekep_keypoints(I)
    mean_reduce = bool(getattr(I.geom, "rekep_path_mean", False))
    cost = _zeros(I)
    for path_fn in I.context["path_fns"]:
        v = path_fn(I.ee_pos, kp)
        if torch.is_tensor(v) and v.ndim == 2:
            v = torch.clamp(v, min=0)
            cost = cost + (v.mean(dim=1) if mean_reduce else v.sum(dim=1))
    return cost


# Regularizers

@register("smooth")
def smooth(I):
    """Penalty: mean squared consecutive joint change."""
    joints = I.real_actions[..., :7]
    return ((joints[:, 1:] - joints[:, :-1]) ** 2).mean(dim=(-1, -2))


@register("gripper_smooth")
def gripper_smooth(I):
    """Penalty: mean squared gripper change between consecutive steps."""
    if I.real_actions.shape[-1] <= 7 or I.real_actions.shape[1] < 2:
        return _zeros(I)
    gripper = I.real_actions[..., 7]
    return (gripper[:, 1:] - gripper[:, :-1]).pow(2).mean(dim=1)


@register("joint_delta")
def joint_delta(I):
    """Penalty: trust region toward the current joints."""
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
    """Penalty: mean downward tool-axis misalignment over the horizon."""
    if I.ee_quat is None or I.context.get("orient", "down") != "down":
        return _zeros(I)
    down = torch.tensor([0.0, 0.0, -1.0], device=I.ee_quat.device, dtype=I.ee_quat.dtype)
    return (1.0 - (_axis(I.ee_quat, 2) * down.view(1, 1, 3)).sum(dim=-1)).mean(dim=1)


@register("consistency")
def consistency(I):
    """Penalty: mean squared deviation from the previous plan."""
    ref = I.context.get("plan_ref")
    joints = I.real_actions[..., :7]
    if ref is None or ref.shape[0] != joints.shape[1]:
        return _zeros(I)
    return ((joints - ref.view(1, -1, 7)) ** 2).mean(dim=(1, 2))


@register("not_hold")
def not_hold(I):
    """Penalty: similarity to the do-nothing plan."""
    joints = I.real_actions[..., :7]
    hold = I.context.get("joint_pos")
    if hold is None:
        return _zeros(I)
    hold = torch.as_tensor(hold, device=joints.device, dtype=joints.dtype).reshape(-1)[:7]
    sigma = float(getattr(I.geom, "not_hold_sigma", 0.05))  # rad, per-joint scale of "moved"
    d2 = ((joints - hold.view(1, 1, 7)) ** 2).mean(dim=2)  # [K,H] mean squared joint deviation
    return torch.exp(-d2 / max(sigma * sigma, 1e-9)).mean(dim=1)


# Pinch-grasp terms

def _pinch_gated(I):
    """Return whether pinch-specific terms should be disabled for a press contact."""
    return I.context.get("contact") == "press"


@register("straddle")
def straddle(I):
    """Constraint: fingertips bracket the grasp target without penetrating it (xy keepout)."""
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
    """Penalty: fingertip at the grasp height, squared z offset."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, _, _, _, center, _ = f
    err = _flat((tip[..., 2] - center[..., 2]).abs(), I.geom)
    # Linear by default: squared metres vanish near contact.
    return (err.pow(2) if getattr(I.geom, "tip_z_shape", "linear") == "squared" else err).mean(dim=1)


@register("yaw")
def yaw(I):
    """Penalty: align the closing axis with a world x or y axis (top-down grasp yaw)."""
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    closing = f[1]
    return (1.0 - torch.maximum(closing[..., 0].abs(), closing[..., 1].abs())).mean(dim=1)


@register("grasp_axis")
def grasp_axis(I):
    """Penalty: align closing with the object's narrow horizontal axis."""
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
    align = (closing[..., :2] * n[:2].view(1, 1, 2)).sum(-1)  # closing dot narrow-axis, horizontal
    return (1.0 - align.abs()).mean(dim=1)


@register("center_region")
def center_region(I):
    """Constraint: keep the grasp point within a region around the gripper centre."""
    if _pinch_gated(I):
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = f
    rel = center - tip
    err = torch.sqrt((rel * closing).sum(-1) ** 2 + (rel * lateral).sum(-1) ** 2
                     + (rel * approach).sum(-1) ** 2 + 1e-12)
    # grasp_slack, never a re-derived tolerance: this cost decides where the gripper closes and must.
    return torch.clamp(err - grasp_slack(I.geom, radius), min=0.0).pow(2).mean(dim=1)


@register("aperture_region")
def aperture_region(I):
    """Constraint: keep the object within the gripper opening along the closing axis."""
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
    """Penalty: shut the gripper when the measured TCP is at the grasp pose."""
    if I.context.get("gripper_intent") == "open":  # reopen recovery owns the channel
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
    center_radius = grasp_slack(I.geom, radius)
    z_scale = I.geom.close_z_scale
    # Floor: a tiny object otherwise makes the gate narrower than TCP wobble and the command dithers.
    xy_scale = max(center_radius, getattr(I.geom, "close_xy_floor", 1e-3))
    # Lateral error only: sideways misses the object, high is merely early. 
    if getattr(I.geom, "close_gate_split_axes", True):
        err = torch.linalg.vector_norm(rel[..., :2], dim=-1)
    else:
        err = torch.linalg.vector_norm(rel, dim=-1)
    gate = torch.exp(-torch.clamp(err - center_radius, min=0.0).pow(2) / max(xy_scale * xy_scale, 1e-8)
                     - torch.clamp(rel[..., 2].abs() - z_scale, min=0.0).pow(2) / max(z_scale * z_scale, 1e-8))
    pre_z = float(getattr(I.geom, "preshape_z", 0.0))
    if pre_z > 0.0:
        # Pre-shaped descent: begin the close while xy-aligned just above the grasp point.
        xy_tol = float(getattr(I.geom, "preshape_xy_tol", 0.02))
        xy_err = torch.linalg.vector_norm(rel[..., :2], dim=-1)
        above = -rel[..., 2]  # tcp height above the grasp centre
        in_cone = (xy_err <= xy_tol) & (above >= 0.0) & (above <= pre_z)
        gate = torch.maximum(gate, in_cone.to(gate.dtype))
    return (I.real_actions[..., 7] - gate).pow(2).mean(dim=1)


@register("grasp_commit")
def grasp_commit(I):
    """Reward closing at the grasp pose, penalise closing away from it (alternative to
    close_gripper).
    """
    if I.context.get("gripper_intent") == "open":  # reopen recovery owns the channel
        return _zeros(I)
    f = _grasp_frame(I)
    if f is None or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    tip, _, _, _, center, radius = f
    dead = grasp_slack(I.geom, radius)  # shared at-the-grasp-pose tolerance
    scale = max(dead, getattr(I.geom, "close_xy_floor", 1e-3))
    err = torch.linalg.vector_norm(center - tip, dim=-1)  # [K,H], per-candidate
    prox = torch.exp(-torch.clamp(err - dead, min=0.0).pow(2) / max(scale * scale, 1e-8))
    grip = I.real_actions[..., 7].clamp(0.0, 1.0)  # bounded, else the reward is unbounded below
    reward = grip * (1.0 - 2.0 * prox)  # near gives -grip (reward), far gives +grip (penalty)
    if getattr(I.geom, "commit_measured_gate", False):
        # Modulate by measured proximity: 1 when the arm is in the neighbourhood, 0 far.
        eef = I.context.get("eef_pos")
        if eef is not None:
            tcp = torch.as_tensor(eef, device=I.ee_pos.device, dtype=I.ee_pos.dtype).reshape(-1)[:3]
            m_err = torch.linalg.vector_norm(center.view(3) - tcp)
            band = getattr(I.geom, "commit_measured_band", 3.0) * scale
            reward = reward * torch.exp(-torch.clamp(m_err - band, min=0.0).pow(2) / max(band * band, 1e-8))
    return reward.mean(dim=1)


@register("grasp_region")
def grasp_region(I):
    """Penalty: distance to the graspable segment along the object's long axis, not a
    single point.
    """
    f = _grasp_frame(I)
    if f is None:
        return _zeros(I)
    tip, _, _, _, center, _ = f
    reg = I.context.get("objects", {}).get(I.context.get("grasp_obj"), {}).get("grasp_region")
    if reg is None:
        return ((tip - center) ** 2).sum(dim=-1).mean(dim=1)
    axis = torch.as_tensor(reg[0], device=tip.device, dtype=tip.dtype).view(1, 1, 3)
    half = float(reg[1])
    d = tip - center
    t = (d * axis).sum(dim=-1, keepdim=True).clamp(-half, half)  # project onto the segment
    closest = center + t * axis
    return ((tip - closest) ** 2).sum(dim=-1).mean(dim=1)


# Grasp-contact constraints

@register("grasp_approach_corridor")
def grasp_approach_corridor(I):
    """Constraint: touch the grasp object only inside the pinch corridor."""
    frame = _grasp_frame(I)
    if frame is None:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = frame
    ctx = I.context
    obj = ctx["objects"][ctx["grasp_obj"]]
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    obj_pos = torch.as_tensor(obj["pos"], device=dev, dtype=dt).view(1, 1, 3)
    body_r = float(I.extents.get(ctx["grasp_obj"], DEFAULT_EXTENT)[1])
    margin = float(getattr(I.geom, "corridor_margin", 0.01))
    open_half = float(getattr(I.geom, "open_half", 0.04))
    pts = _gripper_points(I)  # [4,K,H,3]
    d_obj_xy = torch.linalg.vector_norm(pts[..., :2] - obj_pos[..., :2], dim=-1)
    d_cor_xy = torch.linalg.vector_norm(pts[..., :2] - center[..., :2], dim=-1)
    inside = torch.clamp(1.0 - d_obj_xy / max(body_r, 1e-3), min=0.0)
    below = torch.clamp((center[..., 2] + 0.005 - pts[..., 2]) / 0.05, min=0.0)
    corridor_r = max(radius, open_half) + margin
    outside = torch.clamp((d_cor_xy - corridor_r) / max(corridor_r, 1e-3), min=0.0)
    return (inside * below * outside).pow(2).sum(dim=0).mean(dim=1)


@register("grasp_descend_rate")
def grasp_descend_rate(I):
    """Constraint: limit grasp approach speed with a braking cone."""
    frame = _grasp_frame(I)
    if frame is None:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = frame
    dist = torch.linalg.vector_norm(tip - center, dim=-1)  # [K,H]
    delta = tip[:, 1:] - tip[:, :-1]  # [K,H-1,3]
    cap = float(getattr(I.geom, "grasp_approach_cap", 0.008))  # m/step at contact
    slope = float(getattr(I.geom, "grasp_brake_slope", 0.25))  # extra cap per m
    if getattr(I.geom, "grasp_descend_closing_only", False):
        # Charge only motion that CLOSES on the grasp point.
        to_center = center.view(1, 1, 3) - tip[:, :-1]
        direction = to_center / torch.clamp(
            torch.linalg.vector_norm(to_center, dim=-1, keepdim=True), min=1e-6)
        speed = (delta * direction).sum(dim=-1)  # [K,H-1], signed
    else:
        speed = torch.linalg.vector_norm(delta, dim=-1)  # [K,H-1]
    v_max = cap + slope * dist[:, :-1]
    return (torch.clamp(speed - v_max, min=0.0) / cap).pow(2).mean(dim=1)


@register("grasp_standoff")
def grasp_standoff(I):
    """Constraint: align laterally before descending from the grasp standoff."""
    frame = _grasp_frame(I)
    if frame is None:
        return _zeros(I)
    tip, closing, approach, lateral, center, radius = frame
    hover = float(getattr(I.geom, "approach_hover", 0.08))
    xy_tol = float(getattr(I.geom, "approach_xy_tol", 0.02))
    xy_err = torch.linalg.vector_norm(tip[..., :2] - center[..., :2], dim=-1)
    below = torch.clamp((center[..., 2] + hover - tip[..., 2]) / hover, min=0.0)
    misaligned = torch.clamp((xy_err - xy_tol) / hover, min=0.0)
    return (below * misaligned).pow(2).mean(dim=1)


@register("release_retreat")
def release_retreat(I):
    """Constraint: gripper points may occupy a set-down object's footprint only above its
    top.
    """
    ctx = I.context
    payload = ctx.get("payload")
    objects = ctx.get("objects", {})
    # Guard every SET-DOWN object for the rest of the episode, not just the payload of this instant.
    guarded = [n for n in (ctx.get("placed") or ()) if n in objects]
    if payload is not None and payload in objects and ctx.get("released", False):
        guarded.append(payload)  # released this step, not yet in `placed`
    if not guarded:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    pts = _gripper_points(I)  # [4,K,H,3]
    cost = _zeros(I)
    for name in dict.fromkeys(guarded):  # de-duplicated, order preserved
        pos = torch.as_tensor(objects[name]["pos"], device=dev, dtype=dt).view(1, 1, 3)
        ext = I.extents.get(name, DEFAULT_EXTENT)
        r = float(ext[1]) + float(getattr(I.geom, "retreat_margin", 0.02))
        top = float(ext[2]) + float(getattr(I.geom, "retreat_clear", 0.04))
        d_xy = torch.linalg.vector_norm(pts[..., :2] - pos[..., :2], dim=-1)
        inside = torch.clamp(1.0 - d_xy / max(r, 1e-3), min=0.0)
        below = torch.clamp((pos[..., 2] + top - pts[..., 2]) / 0.05, min=0.0)
        cost = cost + (inside * below).pow(2).sum(dim=0).mean(dim=1)
    return cost


@register("release_rise_first")
def release_rise_first(I):
    """Constraint: no lateral travel while low over something already set down."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    guarded = [n for n in (ctx.get("placed") or ()) if n in objects]
    if payload is not None and payload in objects and ctx.get("released", False):
        guarded.append(payload)  # released this step, not yet in `placed`
    if not guarded or I.ee_pos.shape[1] < 2:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    pts = _gripper_points(I)  # [4,K,H,3]
    span = float(getattr(I.geom, "rise_first_span", 0.05))  # m of clearance to fade over
    scale = float(getattr(I.geom, "rise_first_step", 0.01))  # m of lateral step per unit
    a, b = pts[..., :-1, :], pts[..., 1:, :]  # segment ends [4,K,H-1,3]
    seg = b[..., :2] - a[..., :2]
    step_xy = torch.linalg.vector_norm(seg, dim=-1)  # exact zeros: a still hand charges nothing
    seg_len2 = seg.pow(2).sum(-1).clamp_min(1e-12)  # clamped only for the projection divide
    cost = _zeros(I)
    for name in dict.fromkeys(guarded):
        pos = torch.as_tensor(objects[name]["pos"], device=dev, dtype=dt).view(1, 1, 1, 3)
        ext = I.extents.get(name, DEFAULT_EXTENT)
        r = float(ext[1]) + float(getattr(I.geom, "retreat_margin", 0.02))
        top = float(ext[2]) + float(getattr(I.geom, "retreat_clear", 0.04))
        # Closest xy approach of each segment to the guard axis (point-to-segment projection).
        t = ((pos[..., :2] - a[..., :2]) * seg).sum(-1).div(seg_len2).clamp(0.0, 1.0)
        near = a[..., :2] + t.unsqueeze(-1) * seg
        d_xy = torch.linalg.vector_norm(near - pos[..., :2], dim=-1)
        z_near = a[..., 2] + t * (b[..., 2] - a[..., 2])
        inside = torch.clamp(1.0 - d_xy / max(r, 1e-3), min=0.0)
        below = torch.clamp((pos[..., 2] + top - z_near) / max(span, 1e-3), min=0.0)
        cost = cost + (inside * below * step_xy / max(scale, 1e-6)).pow(2).sum(dim=0).mean(dim=1)
    return cost


# Articulated pull and press terms


def _artic(I, key):
    """Stage articulation metadata as tensors, or None when this stage has none."""
    spec = I.context.get(key)
    if not spec:
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    out = {}
    for name, value in spec.items():
        t = torch.as_tensor(value, device=dev, dtype=dt)
        out[name] = t.reshape(1, 1, 3) if t.numel() == 3 else float(t)
    for name in ("axis", "bar"):  # direction fields, normalized here so the grounding cannot skew
        if name in out:
            out[name] = out[name] / torch.clamp(
                torch.linalg.vector_norm(out[name], dim=-1, keepdim=True), min=1e-8)
    return out


def _artic_gate(I, point, tol):
    """Return the contact-proximity gate for articulation terms."""
    eef = I.context.get("eef_pos")
    if eef is None:
        return I.ee_pos.new_ones(())
    tcp = torch.as_tensor(eef, device=I.ee_pos.device, dtype=I.ee_pos.dtype).reshape(-1)[:3]
    dist = torch.linalg.vector_norm(point.reshape(3) - tcp)
    band = max(float(getattr(I.geom, "artic_gate_band", 1.0)) * tol, 1e-6)
    return torch.exp(-torch.clamp(dist - tol, min=0.0).pow(2) / (band * band))


@register("hook_pull")
def hook_pull(I):
    """Constraint + goal: ride the handle's travel line and end the chunk advanced along
    it.
    """
    a = _artic(I, "pull")
    if a is None:
        return _zeros(I)
    tol = max(a["tol"], 1e-4)
    rel = I.ee_pos - a["point"]  # [K,H,3]
    off = rel - (rel * a["axis"]).sum(-1, keepdim=True) * a["axis"]  # drop the free slide travel
    along_bar = (off * a["bar"]).sum(-1)  # free inside +-span
    across = torch.linalg.vector_norm(off - along_bar.unsqueeze(-1) * a["bar"], dim=-1)
    line = (across + torch.clamp(along_bar.abs() - a["span"], min=0.0)) / tol
    left = ((a["goal"] - I.ee_pos) * a["axis"]).sum(-1)[:, -1].clamp(min=0.0)
    gate = _artic_gate(I, a["point"], max(a.get("reach", tol), 1e-4))
    return gate * (line.pow(2).mean(dim=1) + (left / max(a["stroke"], 1e-4)).pow(2))


@register("press_axis")
def press_axis(I):
    """Constraint + goal: press an articulated part along the direction its contact point
    travels.
    """
    a = _artic(I, "press")
    if a is None:
        return _zeros(I)
    tol, depth = max(a["tol"], 1e-4), max(a["depth"], 1e-4)
    rel = I.ee_pos - a["point"]
    along = (rel * a["axis"]).sum(-1)  # [K,H]
    lateral = torch.linalg.vector_norm(rel - along.unsqueeze(-1) * a["axis"], dim=-1)
    cone = torch.clamp(lateral - tol, min=0.0) / tol
    short = torch.clamp(depth - along[:, -1], min=0.0) / depth
    gate = _artic_gate(I, a["point"], max(a.get("reach", tol), 1e-4))
    return gate * (cone.pow(2).mean(dim=1) + short.pow(2))


# Carry, lift, and place terms

@register("release_gripper")
def release_gripper(I):
    """Penalty: open the gripper on stages whose intent is release, such as letting go of a
    pulled lid.
    """
    if I.context.get("gripper_intent") != "open" or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    return I.real_actions[..., 7].pow(2).mean(dim=1)


@register("regrasp_penalty")
def regrasp_penalty(I):
    """Penalty: commanding the hand open inside the grace window after a confirmed hold."""
    if I.context.get("gripper_intent") == "open" or I.real_actions.shape[-1] <= 7:
        return _zeros(I)
    grace = float(I.context.get("hold_grace", 0.0))
    if grace <= 0.0:
        return _zeros(I)
    return (grace * (1.0 - I.real_actions[..., 7].clamp(0.0, 1.0))).mean(dim=1)


@register("place_approach_above")
def place_approach_above(I):
    """Constraint: reach the destination over its top face, never through its side."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    place_name = ctx.get("place_target")
    place = objects.get(place_name)
    if payload is None or payload not in objects or place is None:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    seat = ctx.get("place_point")
    if seat is not None:
        s = torch.as_tensor(seat, device=dev, dtype=dt)
        foot_xy, top_z = s[:2], s[2]
    else:
        p = torch.as_tensor(place["pos"], device=dev, dtype=dt)
        foot_xy = p[:2]
        top_z = p[2] + float(I.extents.get(place_name, DEFAULT_EXTENT)[2])
    r_foot = float(I.extents.get(place_name, DEFAULT_EXTENT)[1])
    half = float(I.extents.get(payload, DEFAULT_EXTENT)[2])  # payload centre -> its underside
    clear_z = top_z + half + getattr(I.geom, "place_approach_margin", 0.01)
    xy = torch.linalg.vector_norm(carried[..., :2] - foot_xy.view(1, 1, 2), dim=-1)
    # Ramp, not a step: candidates either side of a binary gate average to a meaningless middle.
    outside = ((xy - r_foot) / max(r_foot * 0.5, 1e-3)).clamp(0.0, 1.0)
    return (outside * (clear_z - carried[..., 2]).clamp(min=0.0)).mean(dim=1)


@register("carry_hold")
def carry_hold(I):
    """Penalty: hold the gripper closed while carrying, release once the payload reaches
    the target.
    """
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
        # VLM-driven release (opt-in): open once the sub-goal is satisfied (the held object is at the.
        val = ctx["constraint"](I.ee_pos, _rekep_keypoints(I))
        release = (val < getattr(I.geom, "subgoal_release_eps", 0.05)).to(dt)
        return (gripper - (1.0 - release)).pow(2).mean(dim=1)
    if ctx.get("place_released", False):
        # Hold open until the stage advances: without the latch the gate below re-closes on the empty.
        return gripper.pow(2).mean(dim=1)
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    if getattr(I.geom, "release_on_stall", False):
        # Opt-in: release on physical evidence.
        tol_xy = max(float(getattr(I.geom, "release_xy", 0.06)), 1e-4)
        m = torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt).reshape(-1)[:3]
        d_xy = torch.linalg.vector_norm(m[:2] - dest[:2]) / tol_xy
        release = torch.sigmoid(_RELEASE_SHARPNESS * (1.0 - d_xy)) \
            * float(ctx.get("seat_contact", 0.0))
        return (gripper - (1.0 - release)).pow(2).mean(dim=1)
    if getattr(I.geom, "release_commit_aniso", False):
        # Opt-in anisotropic release: per-axis tolerance, a sigmoid half-way on the tolerance ellipsoid.
        tol_xy = max(float(getattr(I.geom, "release_xy", 0.06)), 1e-4)
        tol_z = max(float(getattr(I.geom, "release_z", 0.02)), 1e-4)
        m = torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt).reshape(-1)[:3]
        d = torch.sqrt((torch.linalg.vector_norm(m[:2] - dest[:2]) / tol_xy).pow(2)
                       + ((m[2] - dest[2]).clamp(min=0.0) / tol_z).pow(2))  # 1.0 on the ellipsoid
        release = torch.sigmoid(_RELEASE_SHARPNESS * (1.0 - d))
        cap = float(getattr(I.geom, "release_step_motion_cap", 0.0))
        motion = ctx.get("eef_step_motion")
        if cap > 0.0 and motion is not None:
            # Quasi-static release: a moving hand ejects the payload along its motion. 
            import math
            release = release * math.exp(-max(0.0, float(motion) / cap - 1.0) ** 2)
        return (gripper - (1.0 - release)).pow(2).mean(dim=1)
    if getattr(I.geom, "release_commit", False):
        # Opt-in smooth commitment (grasp_commit, opposite polarity): opening is rewarded continuously.
        dist = torch.linalg.vector_norm(carried - dest.view(1, 1, 3), dim=-1)  # [K,H]
        dead = I.geom.release_xy
        scale = max(dead, getattr(I.geom, "close_xy_floor", 1e-3))
        prox = torch.exp(-torch.clamp(dist - dead, min=0.0).pow(2) / max(scale * scale, 1e-8))
        open_amt = (1.0 - gripper).clamp(0.0, 1.0)  # bounded, else the reward is unbounded below
        return (open_amt * (1.0 - 2.0 * prox)).mean(dim=1)  # near rewards opening, far penalises it
    xy_dist = torch.linalg.vector_norm(carried[..., :2] - dest[:2].view(1, 1, 2), dim=-1)
    # One-sided: at or below the hover releases, so a seated payload does not re-trigger a close.
    z_err = carried[..., 2] - dest[2]
    release = ((xy_dist < I.geom.release_xy) & (z_err < I.geom.release_z)).to(dt)
    return (gripper - (1.0 - release)).pow(2).mean(dim=1)


def _carried_pos(I, payload_pos):
    """Carried payload per candidate: candidate TCP + (measured payload - measured TCP)."""
    tcp = torch.as_tensor(I.context["eef_pos"], device=I.ee_pos.device, dtype=I.ee_pos.dtype)
    return I.ee_pos + (payload_pos - tcp.reshape(-1)[:3]).view(1, 1, 3)


def _place_errors(I):
    """Return per-step XY and height errors for the carried payload, or None when inactive."""
    ctx = I.context
    if ctx.get("place_released", False):
        return None
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
    # Stall release: overshoot the seat where the ramp bottoms out; press aid only (see _place_frame).
    over = float(getattr(I.geom, "place_overshoot", 0.08)) \
        if getattr(I.geom, "release_on_stall", False) and ctx.get("overshoot_on", True) else 0.0
    z_target = dest[2] - (1.0 - frac) * over + frac * carry_clear
    return xy, carried[..., 2] - z_target


def _lift_errors(I):
    """Per-step (xy, z) error of the carried payload to the lift column."""
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
    if pp is not None:  # calibrated top-surface point (root may differ from centre)
        return (torch.as_tensor(pp, device=dev, dtype=dt)
                + torch.tensor([0.0, 0.0, I.extents.get(payload, DEFAULT_EXTENT)[2] + clearance],
                               device=dev, dtype=dt))
    place_pos = torch.as_tensor(objects[ctx["place_target"]]["pos"], device=dev, dtype=dt)
    stack = I.extents.get(ctx["place_target"], DEFAULT_EXTENT)[2] + I.extents.get(payload, DEFAULT_EXTENT)[2]
    return place_pos + torch.tensor([0.0, 0.0, stack + clearance], device=dev, dtype=dt)


def _place_frame(I):
    """Return place geometry for the carried payload, or None when inactive."""
    ctx = I.context
    if ctx.get("place_released", False):
        return None
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
    # Stall-release press aid: inside the descend gate aim below the estimated seat.
    over = float(getattr(I.geom, "place_overshoot", 0.08)) \
        if getattr(I.geom, "release_on_stall", False) and ctx.get("overshoot_on", True) else 0.0
    z_err = carried[..., 2] - (dest[2] - over + outside * (carry_z - dest[2] + over))
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
    """Penalty: while carrying and far from the destination, pull the payload up to z_table
    + margin.
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
    lift = torch.clamp(liftoff_z - carried[..., 2], min=0.0)  # linear, rise remaining
    return (far * lift).mean(dim=1)


# Collision terms


def _keepout(pen, geom):
    """Convert penetration depth to a keepout penalty."""
    return pen.pow(2) if getattr(geom, "keepout_shape", "linear") == "squared" else pen

@register("place_descent")
def place_descent(I):
    """Penalty: fly the payload at carry height, then ramp it down onto the place seat."""
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


@register("place_approach_rate")
def place_approach_rate(I):
    """Constraint: quasi-static place approach -- braking cone v_max(d) = cap + slope*d to
    the seat.
    """
    ctx = I.context
    if "overshoot_on" not in ctx:  # only the stall-release bridge emits it: the guarded move
        return _zeros(I)  # guards the press approach and stands down everywhere else
    objects = ctx.get("objects", {})
    payload = ctx.get("payload")
    if payload is None or payload not in objects or objects.get(ctx.get("place_target")) is None:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))
    dest = _place_dest(I, objects, payload, ctx)
    dist = torch.linalg.vector_norm(carried - dest.view(1, 1, 3), dim=-1)  # [K,H]
    speed = torch.linalg.vector_norm(carried[:, 1:] - carried[:, :-1], dim=-1)  # [K,H-1]
    cap = float(getattr(I.geom, "place_approach_cap", 0.012))  # m/step at contact
    slope = float(getattr(I.geom, "place_brake_slope", 0.25))  # extra cap per m
    v_max = cap + slope * dist[:, :-1]
    # Excess in units of the contact cap.
    return (torch.clamp(speed - v_max, min=0.0) / cap).pow(2).mean(dim=1)


@register("place_retreat")
def place_retreat(I):
    """Penalty: after the release-at-seat latch, take the hand vertically out of the
    workpiece.
    """
    ctx = I.context
    if not ctx.get("place_released", False):
        return _zeros(I)
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or objects.get(ctx.get("place_target")) is None:
        return _zeros(I)
    dest = _place_dest(I, objects, payload, ctx)
    hover = dest[2] + float(getattr(I.geom, "place_retreat_height", 0.15))
    short = torch.clamp(hover - I.ee_pos[..., 2], min=0.0) \
        / max(float(getattr(I.geom, "place_retreat_height", 0.15)), 1e-4)
    return short.pow(2).mean(dim=1)


@register("collision")
def collision(I):
    """Constraint: soft keepout from non-manipulated objects as upright cylinders."""
    if I.ee_quat is None:
        return _zeros(I)
    ctx = I.context
    objects = ctx.get("objects", {})
    excluded = {ctx.get("grasp_obj"), ctx.get("payload"), ctx.get("place_target"),
                ctx.get("destination")}  # the task destination is never an obstacle
    # Collision terms
    if not getattr(I.geom, "collision_include_placed", False):
        excluded |= set(ctx.get("placed") or ())
    # Fixtures are not obstacles (same rule as `clear`): the margin is isotropic while a flat.
    max_r = getattr(I.geom, "clear_max_radius", 0.10)
    names = [n for n in objects if n not in excluded
             and I.extents.get(n, DEFAULT_EXTENT)[1] <= max_r]
    if not names:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    # Assumes an object's root pose is its geometric centre. 
    centers = torch.stack([torch.as_tensor(objects[n]["pos"], device=dev, dtype=dt) for n in names], 0)
    ext = torch.stack([torch.as_tensor(I.extents.get(n, DEFAULT_EXTENT), device=dev, dtype=dt) for n in names], 0)
    margin = I.geom.ee_r + I.geom.coll_clear
    r_xy = (ext[:, 1] + margin).view(1, 1, 1, -1)
    r_z = (ext[:, 2] + margin).view(1, 1, 1, -1)
    d = _gripper_points(I)[..., None, :] - centers.view(1, 1, 1, -1, 3)
    pen_xy = torch.clamp(r_xy - torch.linalg.vector_norm(d[..., :2], dim=-1), min=0)
    pen_z = torch.clamp(r_z - d[..., 2].abs(), min=0)
    return _keepout(torch.minimum(pen_xy, pen_z), I.geom).sum(-1).mean(dim=(0, 2))


@register("clear")
def clear(I):
    """Constraint: TCP sphere keepout from compact movable objects."""
    ctx = I.context
    objects = ctx.get("objects", {})
    # Placed objects stay obstacles: they repel the next payload's approach.
    excluded = {ctx.get("grasp_obj"), ctx.get("payload"), ctx.get("place_target"), ctx.get("destination")}
    if getattr(I.geom, "clear_exclude_placed", False):  # no-seats ablation restores the old exclusion
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
        cost = cost + _keepout(torch.clamp(keepout - dist, min=0.0), I.geom).mean(dim=1)
    return cost


@register("carry_clear")
def carry_clear(I):
    """Constraint: keep the carried payload clear of compact obstacles during transit."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects:
        return _zeros(I)
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    carried = _carried_pos(I, torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt))  # [K,H,3]
    excluded = {ctx.get("grasp_obj"), payload, ctx.get("place_target"), ctx.get("destination")}
    if not getattr(I.geom, "carry_clear_include_placed", False):
        excluded |= set(ctx.get("placed") or ())  # default: placed objects are destination, not obstacle
    max_r = getattr(I.geom, "clear_max_radius", 0.10)  # fixtures such as board or scale are not obstacles
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
        # Linear penetration (constant gradient), not squared: a squared cm-scale penetration is dwarfed.
        cost = cost + torch.clamp(keepout - dist, min=0.0).mean(dim=1)
    return cost


@register("carry_altitude")
def carry_altitude(I):
    """Penalty: carried payload stays above the transit clearance (support plane + margin)
    while far from the destination.
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
    """Penalty: once over the destination, lower the payload onto its top surface."""
    ctx = I.context
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if payload is None or payload not in objects or objects.get(ctx.get("place_target")) is None:
        return _zeros(I)
    if ctx.get("place_mode") == "container":
        # Placing inside: the destination's top surface is its rim.
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


# insertion corridor (measured) An insertion is not a set-down: its admissible set narrows.


def _insert(I):
    """Stage insertion metadata as tensors, or None when this stage has none."""
    spec = I.context.get("insert")
    if not spec:
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    out = {}
    for name, value in spec.items():
        t = torch.as_tensor(value, device=dev, dtype=dt)
        out[name] = t.reshape(1, 1, 3) if t.numel() == 3 else float(t)
    out["axis"] = out["axis"] / torch.clamp(
        torch.linalg.vector_norm(out["axis"], dim=-1, keepdim=True), min=1e-8)
    out["r_seat"] = max(out["r_seat"], 1e-4)
    out["r_mouth"] = max(out["r_mouth"], out["r_seat"])
    out["slope"] = (out["r_mouth"] - out["r_seat"]) / max(out["height"], 1e-4)
    return out


def _cone_excess(a, s, u):
    """Metres by which a transverse offset u at axial distance s falls outside the cone."""
    radius = a["r_seat"] + a["slope"] * s.clamp(min=0.0)  # r_seat below the seat, widening above
    return torch.clamp(torch.linalg.vector_norm(u, dim=-1) - radius, min=0.0)


def _insert_frame(I):
    """(a, carried, s, u, excess, gate) of the carried payload, or None off an insertion
    stage.
    """
    ctx = I.context
    a = _insert(I)
    payload, objects = ctx.get("payload"), ctx.get("objects", {})
    if a is None or payload is None or payload not in objects or ctx.get("place_released", False):
        return None
    dev, dt = I.ee_pos.device, I.ee_pos.dtype
    measured = torch.as_tensor(objects[payload]["pos"], device=dev, dtype=dt).reshape(-1)[:3]
    carried = _carried_pos(I, measured)  # [K,H,3]
    rel = carried - a["seat"]
    s = (rel * a["axis"]).sum(-1)
    u = rel - s.unsqueeze(-1) * a["axis"]
    cap = max(a.get("capture", a["r_mouth"]), 1e-4)
    band = max(float(getattr(I.geom, "insert_gate_band", 1.0)) * cap, 1e-6)
    dist = torch.linalg.vector_norm(measured - a["seat"].reshape(3))
    gate = torch.exp(-torch.clamp(dist - cap, min=0.0).pow(2) / (band * band))
    return a, carried, s, u, _cone_excess(a, s, u), gate


@register("insert_funnel")
def insert_funnel(I):
    """Constraint: the carried payload stays inside the receptacle's admissible cone."""
    f = _insert_frame(I)
    if f is None:
        return _zeros(I)
    a, _, _, _, excess, gate = f
    return gate * (excess / a["r_mouth"]).pow(2).mean(dim=1)


@register("descend_gate")
def descend_gate(I):
    """Constraint: descent is free only inside the insertion cone; until then, an altitude
    floor.
    """
    f = _insert_frame(I)
    if f is None or I.ee_pos.shape[1] < 2:
        return _zeros(I)
    a, carried, s, u, excess, gate = f
    step = max(float(getattr(I.geom, "insert_descend_step", 0.02)), 1e-6)  # m of drop per unit
    span = max(float(getattr(I.geom, "insert_floor_span", 0.05)), 1e-6)  # m of shortfall per unit
    ua, ub = u[:, :-1, :], u[:, 1:, :]  # segment ends [K,H-1,3]
    du = ub - ua
    t = (-(ua * du).sum(-1) / du.pow(2).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    s_near = s[:, :-1] + t * (s[:, 1:] - s[:, :-1])
    near = _cone_excess(a, s_near, ua + t.unsqueeze(-1) * du) / a["r_mouth"]
    drop = torch.clamp(carried[:, :-1, 2] - carried[:, 1:, 2], min=0.0)  # exact zeros hold still
    motion = (near.clamp(0.0, 1.0) * drop / step).pow(2).mean(dim=1)
    # Floor: the insertion plane, or the stage's carry altitude when that is higher.
    floor_z = a["seat"].reshape(3)[2]
    carry_z = I.context.get("carry_z")
    if carry_z is not None:
        floor_z = torch.maximum(floor_z, torch.as_tensor(float(carry_z), device=floor_z.device,
                                                         dtype=floor_z.dtype))
    short = torch.clamp(floor_z - carried[..., 2], min=0.0)
    outside = (excess / a["r_mouth"]).clamp(0.0, 1.0)
    return gate * (motion + (outside * short / span).pow(2).mean(dim=1))
