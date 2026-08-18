"""Build simulator-free task contexts for offline priority-cost labeling."""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import torch

from vlm_dp.cost import guard_cost
from vlm_dp.cost.base_cost import CompositeCost
from vlm_dp.sim_helpers import make_torch_constraint


WEIGHT_EXTENTS = {
    "apple":   (0.0374, 0.0413, 0.0377),
    "pear":    (0.0456, 0.0517, 0.0621),
    "mango":   (0.0470, 0.0478, 0.0507),
    "cabbage": (0.0776, 0.0772, 0.0480),
    "board":   (0.1743, 0.1266, 0.0052),
    "scale":   (0.1346, 0.1883, 0.0252),
}
DEFAULT_EXTENT = (0.03, 0.03, 0.03)

WEIGHT_ROLES = {"grasp_objs": ("pear", "apple"), "place_obj": "scale"}


_PLACE_SUCCESS_XY = 0.12
_SCALE_SUCCESS_XY_OFFSET = np.array([0.0, -0.05], dtype=np.float64)
_LIFT_HEIGHT = 0.15
_SUBGOAL_EPS = 0.12
_CARRY_HOVER = 0.10
_CARRY_SLACK = 0.03
_PLACE_CLEARANCE = 0.015
_SCALE_TOP_Z_OFFSET = 0.05238
# Mean visual half-heights across the same 20 current eval perception caches.
_VISUAL_HALF_HEIGHT = {"pear": 0.02385910, "apple": 0.02211143}
# Mean narrow axis across the 20 current eval perception caches; apple is round in 19/20.
_GRASP_AXES = {"pear": (0.6803071, -0.7329272, 0.0), "apple": None}
_HELD_MARGIN = 0.04
_PLACE_APPROACH_XY = 0.18


_UNREPRESENTABLE_TERMS = {
    "release_retreat": "no sensed 'released' offline",
    "place_approach_rate": "needs the stall-release bridge's 'overshoot_on'",
    "consistency": "no previous-chunk plan_ref offline",
}

_UNREPRESENTABLE_GEOM = {
    "release_on_stall": "reads bridge 'seat_contact'",
}


def check_representable(cost_cfg):
    """Reject cost configurations that require unavailable offline state."""
    bad = [f"{n} ({why})" for n, why in _UNREPRESENTABLE_TERMS.items()
           if float(cost_cfg["cost"]["terms"].get(n, 0.0)) != 0.0]
    geom = cost_cfg["cost"].get("geometry", {})
    bad += [f"geometry.{k} ({why})" for k, why in _UNREPRESENTABLE_GEOM.items()
            if float(geom.get(k, 0.0) or 0.0) != 0.0]
    if bad:
        raise ValueError(
            "offline context cannot represent: " + "; ".join(bad)
            + ". Labels would come from a different active cost than evaluation; zero these terms "
              "or generate labels from a runtime trace instead."
        )


def attach_priority_cost(mpc, cost_cfg):
    """Attach the configured priority CompositeCost to an MPC planner."""
    if mpc.config.cost_style != "priority":
        raise ValueError(f"priority context needs cost_style='priority', got {mpc.config.cost_style!r}")
    check_representable(cost_cfg)
    mpc.cost = guard_cost(CompositeCost(cost_cfg["cost"]["terms"], cost_cfg["cost"]["geometry"]))


def _pos(objects, name):
    p = objects.get(name, {}).get("pos")
    return None if p is None else np.asarray(p, dtype=np.float64)


def _on_place(objects, name, place_obj):
    o, p = _pos(objects, name), _pos(objects, place_obj)
    if o is None or p is None:
        return False
    reference = p[:2] + (_SCALE_SUCCESS_XY_OFFSET if place_obj == "scale" else 0.0)
    return float(np.linalg.norm(o[:2] - reference)) < _PLACE_SUCCESS_XY


_HOLD_BAND = (0.05, 0.60)


def infer_stage(objects, eef_pos, gripper_closed, roles=WEIGHT_ROLES, extents=None,
                finger_angle=None):
    """Infer per-frame task roles from object, gripper, and end-effector state."""
    place_obj = roles["place_obj"]
    ext = WEIGHT_EXTENTS if extents is None else extents
    eef = np.asarray(eef_pos, dtype=np.float64)[:3]

    grip_on_object = gripper_closed and (
        finger_angle is None or _HOLD_BAND[0] < float(finger_angle) < _HOLD_BAND[1])
    held_name = None
    if grip_on_object:
        best = None
        for name in roles["grasp_objs"]:
            o = _pos(objects, name)
            if o is None:
                continue
            d = float(np.linalg.norm(o - eef))
            r = float(np.linalg.norm(ext.get(name, DEFAULT_EXTENT))) + _HELD_MARGIN
            if d < r and (best is None or d < best[0]):
                best = (d, name)
        held_name = best[1] if best else None

    placed = frozenset(n for n in roles["grasp_objs"]
                       if _on_place(objects, n, place_obj) and n != held_name)
    if held_name is not None:


        held_pos, dest = _pos(objects, held_name), _pos(objects, place_obj)
        if (held_pos is not None and dest is not None
                and float(np.linalg.norm(held_pos[:2] - dest[:2])) > _PLACE_APPROACH_XY):
            return held_name, held_name, None, placed
        return None, held_name, place_obj, placed
    for name in roles["grasp_objs"]:
        if name in placed:
            continue
        return name, None, None, placed
    return None, roles["grasp_objs"][-1], place_obj, placed


class MonotoneStages:
    """Clamp inferred demo stages to a forward-only sequence."""

    def __init__(self, roles=WEIGHT_ROLES):
        self._stages = []
        for n in roles["grasp_objs"]:
            self._stages.append((n, None))
            self._stages.append((n, n))
            self._stages.append((None, n))
        self._seen = {}
        self.roles = roles

    def _ordinal(self, grasp_obj, payload):
        try:
            return self._stages.index((grasp_obj, payload))
        except ValueError:
            return 0

    def clamp(self, demo_key, step, grasp_obj, payload, placed):
        seen = self._seen.setdefault(demo_key, {})
        ordinal = self._ordinal(grasp_obj, payload)
        floor = max((o for s, o in seen.items() if s < step), default=0)
        ordinal = max(ordinal, floor)
        seen[step] = ordinal
        g, p = self._stages[ordinal]
        place_obj = self.roles["place_obj"]
        done = frozenset(n for n in self.roles["grasp_objs"]
                         if self._stages.index((None, n)) < ordinal) | frozenset(placed)


        target = place_obj if (p is not None and g is None) else None
        return g, p, target, done


def priority_context(base_ctx, gripper_closed, roles=WEIGHT_ROLES, extents=None,
                     finger_angle=None, tracker=None, demo_key=None, step=None):
    """Add priority-cost stage fields to a plain trainer context."""
    ext = dict(WEIGHT_EXTENTS if extents is None else extents)
    ctx = dict(base_ctx)
    objects = {}
    for name, o in base_ctx.get("objects", {}).items():
        objects[name] = {"pos": torch.as_tensor(np.asarray(o["pos"], dtype=np.float32)),
                         "extents": ext.get(name, DEFAULT_EXTENT),
                         "axis": None, "grasp_extent": None, "grasp_region": None}
    ctx["objects"] = objects

    grasp_obj, payload, place_target, placed = infer_stage(
        base_ctx.get("objects", {}), base_ctx["eef_pos"], gripper_closed, roles, ext,
        finger_angle=finger_angle)
    if tracker is not None and demo_key is not None and step is not None:
        grasp_obj, payload, place_target, placed = tracker.clamp(
            demo_key, int(step), grasp_obj, payload, placed)
    ctx.update(grasp_obj=grasp_obj, payload=payload, place_target=place_target,
               destination=roles["place_obj"], placed=placed,
               contact="pinch", orient="down", place_mode="surface",
               gripper_intent=("place" if payload is not None and place_target is not None
                               else "close"))

    place_pos = _pos(base_ctx.get("objects", {}), roles["place_obj"])
    seat = None
    if place_pos is not None:
        half_z = ext.get(roles["place_obj"], DEFAULT_EXTENT)[2]
        seat = np.array([place_pos[0], place_pos[1], place_pos[2] + half_z], dtype=np.float32)
    if payload is not None and place_target is not None and seat is not None:
        ctx["place_point"] = seat
        ctx["target"] = seat
    elif grasp_obj is not None:
        ctx["target"] = np.asarray(_pos(base_ctx.get("objects", {}), grasp_obj), dtype=np.float32)
    elif seat is not None:
        ctx["target"] = seat

    bottoms = [float(np.asarray(o["pos"])[2]) - objects[n]["extents"][2]
               for n, o in base_ctx.get("objects", {}).items()]
    ctx["z_table"] = min(bottoms) if bottoms else None
    ctx.setdefault("plan_ref", None)
    return ctx


def _first_true(mask, start, default):
    idx = np.flatnonzero(np.asarray(mask)[int(start):])
    return int(start + idx[0]) if idx.size else int(default)


def _local_offset(eef_pos, eef_quat, point):
    return _rot_wxyz(eef_quat).T @ (np.asarray(point) - np.asarray(eef_pos))


def weight_episode_signals(demo):
    """Build one fixed eight-stage ReKep plan from a successful weight demo.

    Gripper transitions delimit grasp/release events. Lift->carry uses the active plan's
    one-sided lift residual and subgoal epsilon. Carry->place uses the environment/eval
    placement reference (scale root + [0, -0.05]) and its 12 cm XY threshold. These signals
    only select the stage; they do not change CompositeCost or any simple_auth weight.
    """
    tr = _grip_transitions(demo)
    if len(tr) != 4:
        raise ValueError(f"weight demo has {len(tr)} gripper transitions, expected 4 "
                         "(grasp pear, release pear, grasp apple, release apple)")
    c1, o1, c2, o2 = (int(x) for x in tr)
    eef = np.asarray(demo["obs/eef_pos"], dtype=np.float64)
    eef_q = np.asarray(demo["obs/eef_quat"], dtype=np.float64)
    pear = np.asarray(demo["states/rigid_object/pear/root_pose"][:, :3], dtype=np.float64)
    apple = np.asarray(demo["states/rigid_object/apple/root_pose"][:, :3], dtype=np.float64)
    scale = np.asarray(demo["states/rigid_object/scale/root_pose"][:, :3], dtype=np.float64)

    # Frame zero precedes rigid-body settling in these demos. The last second before the first
    # close is the fixed scene geometry seen by an eval teacher when it builds its plan.
    stable = slice(max(0, c1 - 15), c1)
    initial = {
        "pear": np.median(pear[stable], axis=0),
        "apple": np.median(apple[stable], axis=0),
        "scale": np.median(scale[stable], axis=0),
    }
    lift_rise = _LIFT_HEIGHT - _SUBGOAL_EPS
    pear_lift = _first_true(pear[:, 2] >= initial["pear"][2] + lift_rise, c1, o1)
    apple_lift = _first_true(apple[:, 2] >= initial["apple"][2] + lift_rise, c2, o2)
    scale_ref = scale[:, :2] + _SCALE_SUCCESS_XY_OFFSET
    pear_near = _first_true(np.linalg.norm(pear[:, :2] - scale_ref, axis=1)
                            < _PLACE_SUCCESS_XY, pear_lift, o1)
    apple_near = _first_true(np.linalg.norm(apple[:, :2] - scale_ref, axis=1)
                             < _PLACE_SUCCESS_XY, apple_lift, o2)

    scale0 = initial["scale"].copy()
    place = {
        name: scale0 + np.array([0.0, -0.05,
                                 _SCALE_TOP_Z_OFFSET + _VISUAL_HALF_HEIGHT[name]
                                 + _PLACE_CLEARANCE])
        for name in ("pear", "apple")
    }
    return {
        "bounds": (c1, pear_lift, pear_near, o1, c2, apple_lift, apple_near, o2),
        "initial": initial,
        "place": place,
        "held_offset": {
            "pear": _local_offset(eef[c1], eef_q[c1], pear[c1]),
            "apple": _local_offset(eef[c2], eef_q[c2], apple[c2]),
        },
        "eef": eef.astype(np.float32),
        "z_table": min(float(initial["pear"][2]) - WEIGHT_EXTENTS["pear"][2],
                         float(initial["apple"][2]) - WEIGHT_EXTENTS["apple"][2],
                         float(scale0[2]) - WEIGHT_EXTENTS["scale"][2]),
    }


def _weight_stage(sig, step):
    c1, lift1, near1, o1, c2, lift2, near2, _o2 = sig["bounds"]
    if step < c1:
        return 0
    if step < lift1:
        return 1
    if step < near1:
        return 2
    if step < o1:
        return 3
    if step < c2:
        return 4
    if step < lift2:
        return 5
    if step < near2:
        return 6
    return 7


def _weight_constraints(sig, stage):
    """Return the analytic subgoal/path functions rendered by fake_vlm weight."""
    obj_idx = 0 if stage < 4 else 1
    name = "pear" if obj_idx == 0 else "apple"
    scale_idx = 2
    initial = sig["initial"]
    place_off = np.asarray(sig["place"][name] - initial["scale"], dtype=np.float32)
    hover_off = place_off + np.array([0.0, 0.0, _CARRY_HOVER], dtype=np.float32)
    lift_z = float(initial[name][2] + _LIFT_HEIGHT)
    carry_z = float(initial[name][2] + _LIFT_HEIGHT - _CARRY_SLACK)
    hover_z = float(initial["scale"][2] + hover_off[2])

    if stage in (1, 5):
        sub = lambda _ee, kp, i=obj_idx, z=lift_z: torch.clamp(z - kp[i][..., 2], min=0.0)
        paths = (lambda _ee, _kp: 0.0,)
    elif stage in (2, 6):
        sub = lambda _ee, kp, i=obj_idx, s=scale_idx, off=hover_off: torch.linalg.vector_norm(
            kp[i] - (kp[s] + torch.as_tensor(off, device=kp.device, dtype=kp.dtype)), dim=-1)
        paths = (
            lambda _ee, _kp: 0.0,
            lambda _ee, kp, i=obj_idx, z=carry_z: torch.clamp(z - kp[i][..., 2], min=0.0),
        )
    elif stage in (3, 7):
        sub = lambda _ee, kp, i=obj_idx, s=scale_idx, off=place_off: torch.linalg.vector_norm(
            kp[i] - (kp[s] + torch.as_tensor(off, device=kp.device, dtype=kp.dtype)), dim=-1)

        def gated_descent(_ee, kp, i=obj_idx, s=scale_idx, off=place_off, z=hover_z):
            off_t = torch.as_tensor(off, device=kp.device, dtype=kp.dtype)
            xy = torch.linalg.vector_norm(kp[i][..., :2] - (kp[s][..., :2] + off_t[:2]), dim=-1)
            low = torch.clamp(z - kp[i][..., 2], min=0.0)
            return low * torch.clamp(xy / 0.08, max=1.0)

        paths = (lambda _ee, _kp: 0.0, gated_descent)
    else:
        return None, ()
    return make_torch_constraint([sub]), tuple(make_torch_constraint([p]) for p in paths)


def weight_frame_context(base_ctx, sig, step):
    """Build the complete simple_auth cost context for one demo frame."""
    stage = _weight_stage(sig, int(step))
    raw = base_ctx.get("objects", {})
    objects = {}
    for name, obj in raw.items():
        objects[name] = {
            "pos": torch.as_tensor(np.asarray(obj["pos"], dtype=np.float32)),
            "extents": WEIGHT_EXTENTS.get(name, DEFAULT_EXTENT),
            "axis": _GRASP_AXES.get(name),
            "grasp_extent": None,
            "grasp_region": None,
        }

    name = "pear" if stage < 4 else "apple"
    grasp = stage in (0, 4)
    place = stage in (3, 7)
    payload = None if grasp else name
    obj_idx = 0 if name == "pear" else 1
    keypoints = np.stack([_pos(raw, "pear"), _pos(raw, "apple"), _pos(raw, "scale")]).astype(np.float32)
    constraint, path_fns = _weight_constraints(sig, stage)
    placed = frozenset({"pear"} if stage >= 4 else ())
    if stage == 7 and int(step) >= sig["bounds"][-1] and _on_place(raw, "apple", "scale"):
        placed = frozenset({"pear", "apple"})

    ctx = dict(base_ctx)
    ctx["objects"] = objects
    ctx.update(
        grasp_obj=name if grasp else None,
        payload=payload,
        place_target="scale" if place else None,
        destination="scale",
        placed=placed,
        contact="pinch",
        orient="down",
        place_mode="surface",
        gripper_intent="close" if grasp else ("place" if place else "hold"),
        target=np.asarray(_pos(raw, name), dtype=np.float32),
        z_table=sig["z_table"],
        plan_ref=None,
        stage_label=("grasp", "lift", "carry", "place")[stage % 4],
    )
    if constraint is not None:
        ctx.update(keypoints=keypoints, constraint=constraint, path_fns=path_fns,
                   held_idx=(obj_idx,),
                   held_offset=np.asarray(sig["held_offset"][name], dtype=np.float32)[None])
    start = sig["bounds"][0 if name == "pear" else 4]
    if payload is not None:
        ctx["payload_age_s"] = max(0.0, (int(step) - int(start)) / 15.0)
        if int(step) > 0:
            ctx["eef_hist"] = sig["eef"][max(0, int(step) - 2):int(step)]
    return ctx


TEA_EXTENTS = {
    "teapot": (0.007, 0.068, 0.045),
    "teacup": DEFAULT_EXTENT,
}
TEA_ROLES = {"grasp_objs": ("teapot",), "place_obj": "teacup"}


_TEA_HOLD_MIN = 0.30
_TEA_LIFT = 0.073
_TEA_POUR_XY = 0.16
_TEA_POUR_STANDOFF = 0.114
_TEA_POUR_HEIGHT = 0.072

CAPSULE_EXTENTS = {
    "can": (0.017, 0.020, 0.016),
    "capsule": (0.15, 0.15, 0.15),
    "lid": (0.010, 0.015, 0.010),
}
CAPSULE_ROLES = {"grasp_objs": ("can",), "place_obj": "capsule"}
_CAPSULE_HOLD_BAND = (0.05, 0.60)

_CAPSULE_BAY_LOCAL = np.array([0.0, 0.0, 0.27])
_CAPSULE_BAY_XY, _CAPSULE_BAY_Z = 0.09, (0.255, 0.325)
_CAPSULE_LIFT, _CAPSULE_LIFT_CONFIRM = 0.15, 0.05
_CAPSULE_CARRY = 0.25


def _grip_transitions(demo):
    """Return steps where the commanded gripper state changes."""
    closed = np.asarray(demo["obs/joint_actions"][:, 7]) > 0.5
    return np.flatnonzero(np.diff(closed.astype(int))) + 1


def _rot_wxyz(q):
    """Convert a wxyz quaternion to a rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _objects_ctx(base_ctx, extents, grasp_extents=()):
    """Build CompositeCost object entries from a plain context."""
    objects = {}
    for name, o in base_ctx.get("objects", {}).items():
        objects[name] = {"pos": torch.as_tensor(np.asarray(o["pos"], dtype=np.float32)),
                         "extents": extents.get(name, DEFAULT_EXTENT),
                         "axis": None, "grasp_extent": dict(grasp_extents).get(name),
                         "grasp_region": None}
    return objects


def tea_episode_signals(demo):
    """Extract latched tea-task events and calibrated targets."""
    finger = np.asarray(demo["obs/joint_pos"][:, 7])
    eef = np.asarray(demo["obs/eef_pos"])
    pot = np.asarray(demo["states/rigid_object/teapot/root_pose"][:, :3])
    cup = np.asarray(demo["states/rigid_object/teacup/root_pose"][:, :3])
    T = len(finger)
    tr = _grip_transitions(demo)
    c1 = int(tr[0]) if len(tr) else T
    holding = finger[c1:] > _TEA_HOLD_MIN
    hold = c1 + int(np.argmax(holding)) if holding.any() else T
    d_pc = np.linalg.norm(pot[:, :2] - cup[:, :2], axis=1)
    near = d_pc[min(hold, T - 1):] < _TEA_POUR_XY
    pour = hold + int(np.argmax(near)) if near.any() else T
    rest_z = float(np.median(pot[5:max(c1, 6), 2]))
    cup_z = float(np.median(cup[5:, 2]))
    return {"hold": hold, "pour": pour,

            "handle": eef[min(hold, T - 1)].astype(np.float32),
            "rest_z": rest_z,
            "z_table": min(rest_z - TEA_EXTENTS["teapot"][2],
                           cup_z - TEA_EXTENTS["teacup"][2])}


def _tea_frame_context(base_ctx, sig, step, **_):
    """Build the tea-task context for one frame."""
    ctx = dict(base_ctx)
    raw = base_ctx.get("objects", {})
    ctx["objects"] = _objects_ctx(base_ctx, TEA_EXTENTS,
                                  grasp_extents={"teapot": TEA_EXTENTS["teapot"][0]})
    pot, cup = _pos(raw, "teapot"), _pos(raw, "teacup")
    if step < sig["hold"]:
        stage, grasp_obj, payload, orient = "grasp", "teapot", None, "down"
        target = sig["handle"]
    elif step < sig["pour"]:
        stage, grasp_obj, payload, orient = "lift", "teapot", "teapot", "down"
        target = np.array([pot[0], pot[1], sig["rest_z"] + _TEA_LIFT], dtype=np.float32)
    else:

        stage, grasp_obj, payload, orient = "pour", None, "teapot", "free"
        radial = pot[:2] - cup[:2]
        n = float(np.linalg.norm(radial))
        radial = radial / n if n > 1e-6 else np.array([-1.0, 0.0])
        target = np.array([*(cup[:2] + _TEA_POUR_STANDOFF * radial),
                           cup[2] + _TEA_POUR_HEIGHT], dtype=np.float32)
    ctx.update(grasp_obj=grasp_obj, payload=payload, place_target=None,
               destination=TEA_ROLES["place_obj"], placed=frozenset(),
               contact="pinch", orient=orient, place_mode="surface", gripper_intent="close",
               target=np.asarray(target, dtype=np.float32),
               z_table=sig["z_table"], stage_label=stage)
    ctx.setdefault("plan_ref", None)
    return ctx


def capsule_episode_signals(demo):
    """Extract capsule-task stage events and fixture geometry."""
    tr = _grip_transitions(demo)
    if len(tr) != 4:
        raise ValueError(f"capsule demo has {len(tr)} gripper transitions, expected 4 "
                         "(hook lid, release lid, grasp pod, release pod)")
    c1, o1, c2, o2 = (int(t) for t in tr)
    finger = np.asarray(demo["obs/joint_pos"][:, 7])
    eef = np.asarray(demo["obs/eef_pos"])
    can = np.asarray(demo["states/rigid_object/can/root_pose"][:, :3])
    mroot = np.asarray(demo["states/articulation/capsule/root_pose"][0])
    in_band = (finger[c2:] > _CAPSULE_HOLD_BAND[0]) & (finger[c2:] < _CAPSULE_HOLD_BAND[1])
    hold = c2 + int(np.argmax(in_band)) if in_band.any() else c2
    rest_z = float(np.median(can[5:c2, 2]))
    risen = can[hold:, 2] > rest_z + _CAPSULE_LIFT_CONFIRM
    rise = hold + int(np.argmax(risen)) if risen.any() else hold
    rot = _rot_wxyz(mroot[3:7])
    bay = mroot[:3] + rot @ _CAPSULE_BAY_LOCAL
    seat = (bay - np.array([0.0, 0.0, CAPSULE_EXTENTS["can"][2]])).astype(np.float32)
    return {"c1": c1, "o1": o1, "c2": c2, "o2": o2, "hold": hold, "rise": rise,
            "lip0": eef[c1].astype(np.float32), "lip_open": eef[o1].astype(np.float32),
            "machine_pos": mroot[:3].astype(np.float32), "machine_rot": rot,
            "seat": seat, "rest_z": rest_z,
            "z_table": rest_z - CAPSULE_EXTENTS["can"][2]}


def _capsule_frame_context(base_ctx, sig, step, **_):
    """Build the capsule-task context for one frame."""
    ctx = dict(base_ctx)
    raw = base_ctx.get("objects", {})
    objects = _objects_ctx(base_ctx, CAPSULE_EXTENTS)

    objects["capsule"] = {"pos": torch.as_tensor(sig["machine_pos"]),
                          "extents": CAPSULE_EXTENTS["capsule"],
                          "axis": None, "grasp_extent": None, "grasp_region": None}
    can = _pos(raw, "can")
    place_target = place_point = carry_z = None
    placed = frozenset()
    contact, orient, intent = "pinch", "down", "close"
    if step < sig["c1"]:

        objects["lid"] = {"pos": torch.as_tensor(sig["lip0"]),
                          "extents": CAPSULE_EXTENTS["lid"],
                          "axis": None, "grasp_extent": CAPSULE_EXTENTS["lid"][0],
                          "grasp_region": None}
        stage, grasp_obj, payload = "hook_lid", "lid", None
        target, contact = sig["lip0"], "press"
    elif step < sig["o1"]:


        stage, grasp_obj, payload = "open_lid", None, None
        target, contact, orient = sig["lip_open"], "press", "free"
    elif step < sig["hold"]:
        stage, grasp_obj, payload = "grasp", "can", None
        target = np.asarray(can, dtype=np.float32)
    elif step < sig["rise"]:
        stage, grasp_obj, payload = "lift", "can", "can"
        target = np.array([can[0], can[1], sig["rest_z"] + _CAPSULE_LIFT], dtype=np.float32)
    else:
        stage, grasp_obj, payload = "place", None, "can"
        place_target, place_point, target = "capsule", sig["seat"], sig["seat"]
        carry_z = float(sig["seat"][2]) + _CAPSULE_CARRY
        intent = "place"
        local = sig["machine_rot"].T @ (np.asarray(can) - sig["machine_pos"])
        if (step >= sig["o2"]
                and float(np.linalg.norm(local[:2] - _CAPSULE_BAY_LOCAL[:2])) < _CAPSULE_BAY_XY
                and _CAPSULE_BAY_Z[0] <= local[2] <= _CAPSULE_BAY_Z[1]):
            placed = frozenset({"can"})
    ctx["objects"] = objects
    ctx.update(grasp_obj=grasp_obj, payload=payload, place_target=place_target,
               destination=CAPSULE_ROLES["place_obj"], placed=placed,
               contact=contact, orient=orient, place_mode="container", gripper_intent=intent,
               target=np.asarray(target, dtype=np.float32),
               z_table=sig["z_table"], stage_label=stage)
    if place_point is not None:
        ctx["place_point"] = place_point
    if carry_z is not None:
        ctx["carry_z"] = carry_z
    ctx.setdefault("plan_ref", None)
    return ctx


_CAPSULE_LID_DOF = 1


def capsule_subtask_flags(sig, step):
    """Return latched capsule subtask flags."""
    return {"open_coffee_lid": bool(step >= sig["o1"]), "grasp_pod": bool(step >= sig["hold"])}


def capsule_flow_context(base_ctx, demo, step, sig, *, heuristic=True):
    """Add capsule-flow fixture, joint, and subtask state."""
    ctx = dict(base_ctx)
    objects = dict(base_ctx.get("objects", {}))
    pose = np.asarray(demo["states/articulation/capsule/root_pose"][step], dtype=np.float32)
    objects["capsule"] = {"pos": pose[:3], "quat": pose[3:7]}
    ctx["objects"] = objects
    ctx["capsule_lid_joint_pos"] = float(
        np.asarray(demo["states/articulation/capsule/joint_position"][step])[_CAPSULE_LID_DOF]
    )
    if heuristic:
        ctx["subtasks"] = capsule_subtask_flags(sig, step)
    return ctx


def _weight_frame_context(base_ctx, sig, step, **_):
    """Build the weight-task context for one frame."""
    return weight_frame_context(base_ctx, sig, step)


@dataclasses.dataclass(frozen=True)
class OfflineTask:
    """Describe offline dataset routing and context hooks for a task."""
    task_id: str
    hdf5: str
    frame_context: Callable
    episode_signals: Callable = lambda demo: None
    make_tracker: Callable | None = None
    rollout_dir: str | None = None
    stage_order: tuple[str, ...] = ("grasp", "carry", "place")


TASKS = {
    "weight": OfflineTask(
        task_id="Isaac-Weight-Droid-Visuomotor-v0",
        hdf5="/workspace/pps/data/weight/generated_dataset.hdf5",
        frame_context=_weight_frame_context,
        episode_signals=weight_episode_signals,
        rollout_dir="results/Isaac-Weight-Droid-Visuomotor-v0/q_base_rd_n20"),
    "tea": OfflineTask(
        task_id="Isaac-Tea-Droid-Visuomotor-v0",
        hdf5="/workspace/pps/data/tea/new_generated_dataset.hdf5",
        frame_context=_tea_frame_context,
        episode_signals=tea_episode_signals,
        stage_order=("grasp", "lift", "pour")),
    "capsule": OfflineTask(
        task_id="Isaac-Capsule-Droid-Visuomotor-v0",
        hdf5="/workspace/pps/data/capsule/generated_dataset.hdf5",
        frame_context=_capsule_frame_context,
        episode_signals=capsule_episode_signals,
        stage_order=("hook_lid", "open_lid", "grasp", "lift", "place")),
}
