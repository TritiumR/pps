"""Build simulator-free task contexts for offline priority-cost labeling."""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import torch

from vlm_dp.cost import guard_cost
from vlm_dp.cost.base_cost import CompositeCost


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


_ON_XY, _ON_Z, _HELD_MARGIN = 0.09, 0.04, 0.04


_PLACE_APPROACH_XY = 0.18


_UNREPRESENTABLE_TERMS = {
    "rekep_subgoal": "no VLM constraint offline",
    "rekep_path": "no VLM path functions offline",
    "release_retreat": "no sensed 'released' offline",
    "place_approach_rate": "needs the stall-release bridge's 'overshoot_on'",
    "consistency": "no previous-chunk plan_ref offline",
    "grasp_axis": "no per-object grasp axis offline",
}

_UNREPRESENTABLE_GEOM = {
    "release_on_stall": "reads bridge 'seat_contact'",
    "release_on_subgoal": "reads the VLM constraint",
    "release_step_motion_cap": "reads bridge 'eef_step_motion'",
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
    return (float(np.linalg.norm(o[:2] - p[:2])) < _ON_XY) and (o[2] > p[2] + _ON_Z)


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


def _weight_frame_context(base_ctx, sig, step, *, gripper_closed, finger_angle, tracker, demo_key):
    """Build the weight-task context for one frame."""
    del sig
    return priority_context(base_ctx, gripper_closed, finger_angle=finger_angle,
                            tracker=tracker, demo_key=demo_key, step=step)


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
        make_tracker=MonotoneStages,
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
