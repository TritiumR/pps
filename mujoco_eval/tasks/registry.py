"""Define per-task offline stage signals and cost contexts from demonstration data."""
from __future__ import annotations


import numpy as np
import torch

from .. import paths
paths.ensure_repo_on_path()

from vlm_dp.offline_context import OfflineTask

DATA = paths.DATA


STACK_EXTENTS = {
    "cubeA": (0.02, 0.02, 0.02),
    "cubeB": (0.025, 0.025, 0.025),
    "cubeC": (0.02, 0.02, 0.02),
}
SQUARE_EXTENTS = {
    "nut": (0.015875, 0.0793, 0.01),
    "peg1": (0.016, 0.016, 0.1),
}
PEG_POS = np.array([0.23, 0.10, 0.85], dtype=np.float32)
PEG_TOP = np.array([0.23, 0.10, 0.95], dtype=np.float32)
NUT_HANDLE_LOCAL = np.array([0.054, 0.0, 0.0])
LIFT_EXTENTS = {"cube": (0.021, 0.021, 0.021)}
CAN_EXTENTS = {
    "can": (0.025, 0.025, 0.0407),
    "bin2_q3": (0.0975, 0.1225, 0.04),
}
CAN_SEAT = np.array([0.1975, 0.4025, 0.8604], dtype=np.float32)


CAN_DROP = np.array([0.1975, 0.4025, 1.04], dtype=np.float32)
THREADING_EXTENTS = {
    "needle": (0.02, 0.08, 0.02),
    "tripod": (0.05, 0.05, 0.1),
}
NEEDLE_HANDLE_LOCAL = np.array([0.0, 0.06, 0.0])
NEEDLE_BAR_LOCAL = np.array([0.0, -0.02, 0.0])
RING_LOCAL = np.array([0.0, 0.0, 0.088])
RING_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])
RING_RADIUS = 0.012
NEEDLE_GRASP_EXTENT = 0.016
INSERT_ROOT_OFFSET = 0.03
COFFEE_EXTENTS = {
    "coffee_pod": (0.0243, 0.0243, 0.0231),
    "coffee_machine": (0.0865, 0.1505, 0.1105),
    "coffee_pod_holder": (0.0295, 0.0295, 0.028),
    "coffee_machine_lid": (0.0295, 0.044, 0.0095),
}
POD_GRASP_LOCAL = np.array([0.0, 0.0, 0.0156])
POD_DROP_OFFSET = np.array([0.0, 0.0, 0.0353])
LID_PRESS_LOCAL = np.array([-0.045, 0.031, 0.016])
POD_R_DIFF = 0.0052
LID_CLOSED = 15.0 * np.pi / 180.0
MUG_EXTENTS = {
    "mug": (0.009, 0.0463, 0.031),
    "drawer": (0.12, 0.12, 0.13),
}
HANDLE_EXTENTS = (0.009, 0.059, 0.009)
DRAWER_LINK_LOCAL = np.array([0.0, -0.01, 0.076])
DRAWER_HANDLE_LOCAL = np.array([0.0, -0.16, 0.04])
DRAWER_BOTTOM_TOP = -0.032
DRAWER_INTERIOR_X = 0.064
DRAWER_INTERIOR_Y = (-0.085, 0.057)
MUG_SEAT_LOCAL = np.array([0.0, 0.0, DRAWER_BOTTOM_TOP + 0.031])
DRAWER_OPEN_QPOS = -0.135
TPA_EXTENTS = {
    "base": (0.095, 0.095, 0.019),
    "piece_1": (0.017, 0.051, 0.051),
    "piece_2": (0.02, 0.06, 0.08),
}
P1_GRASP_LOCAL = np.array([0.0, 0.0, 0.027])
P2_GRASP_LOCAL = np.array([0.0, 0.0, 0.058])
SEAT1_OFFSET = np.array([0.0, 0.0, 0.031])
SEAT2_OFFSET = np.array([0.0, 0.0, 0.099])
TPA_LIFT = {"piece_1": 0.10, "piece_2": 0.15}
HC_EXTENTS = {
    "hammer": (0.012, 0.051, 0.0124),
    "handle": (0.009, 0.05, 0.009),
    "drawer": (0.08, 0.105, 0.05),
    "CabinetObject": (0.118, 0.112, 0.065),
}
CAB_ROOT = np.array([0.0, 0.3, 0.905])
HOOK_OFFSET = np.array([0.0, -0.03, -0.03])
CAVITY_LOCAL = np.array([0.0, -0.014, 0.0])
HC_DROP_Z = 1.06
DRAWER_OPEN_Q = -0.10
DRAWER_CLOSED_Q = -0.01
DRAWER_MOVING_Q = -0.005
HC_IN_XY = (0.064, 0.084)
HC_IN_Z = 0.999
KITCHEN_EXTENTS = {
    "pot": (0.007, 0.067, 0.044),
    "bread": (0.015, 0.025, 0.02),
    "stove": (0.095, 0.095, 0.02),
    "button": (0.04, 0.04, 0.05),
    "serving_region": (0.07, 0.10, 0.001),
}
STOVE_POS = np.array([0.03, 0.095, 0.895], dtype=np.float32)
BUTTON_POS = np.array([-0.14, 0.10, 0.895], dtype=np.float32)
SERVING_POS = np.array([0.145, -0.15, 0.878], dtype=np.float32)
POT_HANDLE_LOCAL = np.array([0.0, 0.06, 0.077])
POT_GRASP_EXTENT = 0.007
POT_STOVE_SEAT_Z = 0.920
POT_TABLE_REST_Z = 0.9025
BREAD_DROP_DZ = 0.122
BTN_PRESS_ON = BUTTON_POS + np.array([0.0, 0.027, 0.097], dtype=np.float32)
BTN_PRESS_OFF = BUTTON_POS + np.array([0.002, -0.029, 0.097], dtype=np.float32)
SERVE_RELEASE = SERVING_POS + np.array([-0.145, 0.01, 0.025], dtype=np.float32)
PUSH_EEF_OFFSET = np.array([-0.071, 0.01, 0.026], dtype=np.float32)
SERVE_BOX = (0.05, 0.10, 0.05)
COFFEE_PREP_EXTENTS = {
    "mug": (0.005, 0.051, 0.0402),
    "coffee_pod": (0.0243, 0.0243, 0.0231),
    "coffee_machine": (0.0865, 0.1505, 0.1105),
    "coffee_pod_holder": (0.0295, 0.0295, 0.028),
    "coffee_machine_lid": (0.0295, 0.044, 0.0095),
    "cabinet": (0.118, 0.19, 0.065),
}
MUG_GRASP_LOCAL = np.array([-0.0234, 0.0011, 0.0139])
MUG_SEAT_MACHINE_LOCAL = np.array([-0.013, 0.122, -0.061])
CP_POD_GRASP_LOCAL = np.array([0.0, 0.0, 0.0122])
POD_RELEASE_HOLDER_DZ = 0.034
LID_OPEN_PRESS_LOCAL = np.array([0.0087, 0.0194, 0.0121])
LID_CLOSE_PRESS_LOCAL = np.array([-0.0494, 0.0393, 0.0299])
DRAWER_HOOK_LOCAL = np.array([0.0094, -0.2569, 0.0954])
LID_OPEN_Q = 2.08
CP_DRAWER_OPEN_Q = -0.19


HANDLE_BAR_LOCAL = np.array([1.0, 0.0, 0.0])
HANDLE_BAR_HALF = 0.05
DRAWER_SLIDE_LOCAL = np.array([0.0, 1.0, 0.0])
DRAWER_STROKE = 0.136
HOOK_LINE_TOL = 0.031
MC_HOOK_OPEN = np.array([-0.0099, -0.0288, -0.0198])
MC_HOOK_CLOSE = np.array([-0.0069, -0.0122, -0.0079])
HC_HOOK_OPEN = np.array([-0.0207, -0.0279, -0.0264])
HC_HOOK_CLOSE = np.array([-0.0167, -0.0029, -0.0116])
LID_HINGE_LOCAL = np.array([0.0, -0.044, 0.0])
LID_HINGE_AXIS = np.array([1.0, 0.0, 0.0])
LID_PRESS_DEPTH = 0.058
LID_PRESS_TOL = 0.057
LID_PRESS_REACH = 0.024


def slide_pull(rot, handle_now, handle_goal, hook, opening):
    """Return drawer pull metadata for the hook-pull cost."""
    return {"axis": rot @ ((-1.0 if opening else 1.0) * DRAWER_SLIDE_LOCAL),
            "point": handle_now + rot @ hook, "goal": handle_goal + rot @ hook,
            "bar": rot @ HANDLE_BAR_LOCAL, "span": HANDLE_BAR_HALF,
            "tol": HOOK_LINE_TOL, "stroke": DRAWER_STROKE}


def hinge_press(body_pos, body_rot, press_local, hinge_local, hinge_axis, depth, tol, reach):
    """Return hinge contact metadata for the press-axis cost."""
    point = body_pos + body_rot @ press_local
    vel = np.cross(body_rot @ (-hinge_axis), point - (body_pos + body_rot @ hinge_local))
    norm = float(np.linalg.norm(vel))
    return None if norm < 1e-9 else {"axis": vel / norm, "point": point, "depth": depth,
                                     "tol": tol, "reach": reach}


RING_FIT = 0.003
RING_MOUTH = 0.080
RING_CONE_H = 0.10
RING_CAPTURE = 0.30
PEG_FIT = 0.00675
PEG_MOUTH = 0.085
PEG_CONE_H = 0.015
PEG_CAPTURE = 0.05


def insert_cone(seat, axis, fit, mouth, height, capture):
    """Return insertion-corridor metadata."""
    return {"seat": seat, "axis": axis, "r_seat": fit, "r_mouth": mouth,
            "height": height, "capture": capture}


_CUBE_HOLD_BAND = (0.030, 0.060)
_NUT_HOLD_BAND = (0.024, 0.050)
_CAN_HOLD_BAND = (0.040, 0.065)
_NEEDLE_HOLD_BAND = (0.026, 0.045)
_POD_HOLD_BAND = (0.040, 0.060)
_MUG_HOLD_BAND = (0.005, 0.075)
_HAMMER_HOLD_BAND = (0.016, 0.035)
_POT_HOLD_BAND = (0.012, 0.024)
_BREAD_HOLD_BAND = (0.024, 0.045)
_CP_MUG_HOLD_BAND = (0.004, 0.020)
_CP_POD_HOLD_BAND = (0.040, 0.055)
_AP_STALL = 0.001
_AP_REOPEN = 0.065
_LIFT_CONFIRM = 0.02

_LIFT = {"stack": 0.10, "square": 0.15, "lift": 0.06, "can": 0.18, "threading": 0.15,
         "coffee": 0.20, "mug_cleanup": 0.15, "hammer_cleanup": 0.20,
         "kitchen_pot": 0.08, "kitchen_bread": 0.16,
         "coffee_prep_mug": 0.10, "coffee_prep_pod": 0.21}
_ON_XY, _ON_Z = 0.05, 0.02
_DRAWER_MOVE_EPS = 0.005
_MUG_IN_Z = 0.09
_HINGE_MOVING = 0.05
_SLIDE_MOVING = -0.005
_HINGE_CLOSING_RATE = 0.005
_MUG_ON_XY = 0.05
_MUG_UPRIGHT = 1e-3


STACK_LAYOUT = {"cubeA": (0, 3), "cubeB": (7, 10)}
STACK3_LAYOUT = {"cubeA": (0, 3), "cubeB": (7, 10), "cubeC": (23, 26)}
SQUARE_LAYOUT = {"nut": (0, 3)}
SQUARE_QUAT_XYZW = (3, 7)
LIFT_LAYOUT = {"cube": (0, 3)}
CAN_LAYOUT = {"can": (0, 3)}
THREADING_LAYOUT = {"needle": (0, 3), "tripod": (14, 17)}
THREADING_NEEDLE_QUAT_XYZW = (3, 7)
THREADING_TRIPOD_QUAT_XYZW = (17, 21)
COFFEE_LAYOUT = {"coffee_pod": (0, 3), "coffee_machine": (14, 17),
                 "coffee_pod_holder": (28, 31), "coffee_machine_lid": (42, 45)}
MUG_LAYOUT = {"mug": (0, 3), "drawer": (14, 17)}
TPA_LAYOUT = {"base": (0, 3), "piece_1": (14, 17), "piece_2": (28, 31)}
HC_LAYOUT = {"hammer": (0, 3), "CabinetObject": (14, 17)}
KITCHEN_LAYOUT = {"stove": (0, 3), "bread": (14, 17), "pot": (28, 31),
                  "serving_region": (42, 45)}
COFFEE_PREP_LAYOUT = {"coffee_pod": (0, 3), "coffee_machine": (14, 17),
                      "coffee_pod_holder": (28, 31), "coffee_machine_lid": (42, 45),
                      "cabinet": (56, 59), "mug": (70, 73)}
STATES_TASKS = ("hammer_cleanup",)


def object_positions(obj_row, layout):
    return {n: np.asarray(obj_row[a:b], dtype=np.float32) for n, (a, b) in layout.items()}


def grip_flips(actions):
    """Return steps where the commanded gripper changes state."""
    return np.flatnonzero(np.diff((np.asarray(actions)[:, 6] > 0).astype(int))) + 1


def aperture(gripper_qpos):
    qp = np.asarray(gripper_qpos)
    return qp[:, 0] - qp[:, 1]


def _hold_step(ap, close, open_, band):
    """Return the first stalled aperture frame inside the hold band."""
    for t in range(close, open_):
        if band[0] < ap[t] < band[1] and (t + 1 >= len(ap) or ap[t] - ap[t + 1] < _AP_STALL):
            return t
    return open_


def _rise_step(z, hold, end, rest):
    risen = z[hold:end] > rest + _LIFT_CONFIRM
    return hold + int(np.argmax(risen)) if risen.any() else end


def _rot_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _objects_ctx(raw, extents, grasp_extents=(), axes=()):
    """Build CompositeCost object dictionaries."""
    objects = {}
    for name, pos in raw.items():
        objects[name] = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float32)),
                         "extents": extents[name],
                         "axis": dict(axes).get(name), "grasp_extent": dict(grasp_extents).get(name),
                         "grasp_region": None}
    return objects


def _carry_p90(z, a, b, fallback):
    return float(np.percentile(z[a:b], 90)) if b > a else fallback


def context_extras(task, obj_row, states_row=None):
    """Extract task-specific orientation and articulation state."""
    if task == "square":
        return {"nut_quat_xyzw": obj_row[3:7]}
    if task == "threading":
        return {"needle_quat_xyzw": obj_row[3:7], "tripod_quat_xyzw": obj_row[17:21]}
    if task == "coffee":
        return {"pod_quat_xyzw": obj_row[3:7], "lid_quat_xyzw": obj_row[45:49],
                "hinge_angle": float(obj_row[56])}
    if task == "mug_cleanup":
        return {"drawer_quat_xyzw": obj_row[17:21], "drawer_joint_pos": float(obj_row[28])}
    if task == "three_piece_assembly":
        return {"piece_1_quat_xyzw": obj_row[17:21], "piece_2_quat_xyzw": obj_row[31:35]}
    if task == "hammer_cleanup":
        return {"hammer_quat_xyzw": obj_row[3:7], "drawer_q": float(states_row[1])}
    if task == "kitchen":
        return {"pot_quat_xyzw": obj_row[31:35]}
    if task == "coffee_prep":
        return {"pod_quat_xyzw": obj_row[3:7], "machine_quat_xyzw": obj_row[17:21],
                "lid_quat_xyzw": obj_row[45:49], "cabinet_quat_xyzw": obj_row[59:63],
                "mug_quat_xyzw": obj_row[73:77], "slide_q": float(obj_row[85])}
    return {}


def stack_episode_signals(demo):
    """Extract latched stack-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    A = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    c1 = int(tr[0]) if len(tr) > 0 else T
    o1 = int(tr[1]) if len(tr) > 1 else T
    hold = _hold_step(ap, c1, o1, _CUBE_HOLD_BAND)
    rest = float(np.median(A[5:max(c1, 6), 2]))
    rise = _rise_step(A[:, 2], hold, o1, rest)
    return {"c1": c1, "hold": hold, "rise": rise, "o1": o1, "rest": rest,
            "carry_z": _carry_p90(A[:, 2], rise, o1, rest + _LIFT["stack"]),
            "z_table": rest - STACK_EXTENTS["cubeA"][2]}


def _stack_stage(base_ctx, raw, sig, step, *, grasp, dest, placed, lift_key="stack"):
    """Build one grasp, lift, and place context sequence."""
    pos = raw[grasp]
    seat = raw[dest] + np.array([0.0, 0.0, STACK_EXTENTS[dest][2]], dtype=np.float32)
    if step < sig["hold"]:
        return dict(stage_label="grasp", grasp_obj=grasp, payload=None, place_target=None,
                    target=np.asarray(pos, dtype=np.float32), gripper_intent="close",
                    placed=placed)
    if step < sig["rise"]:
        target = np.array([pos[0], pos[1], sig["rest"] + _LIFT[lift_key]], dtype=np.float32)
        return dict(stage_label="lift", grasp_obj=grasp, payload=grasp, place_target=None,
                    target=target, gripper_intent="close", placed=placed)
    upd = dict(stage_label="place", grasp_obj=None, payload=grasp, place_target=dest,
               target=seat, place_point=seat, carry_z=sig["carry_z"],
               gripper_intent="place", placed=placed)
    if step >= sig["o1"]:
        upd.update(released=True, place_released=True)
    return upd


def _stack_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = base_ctx["objects"]
    ctx["objects"] = _objects_ctx(raw, STACK_EXTENTS)
    ctx.update(_stack_stage(base_ctx, raw, sig, step, grasp="cubeA", dest="cubeB",
                            placed=frozenset()))
    ctx.update(destination="cubeB", contact="pinch", orient="down", place_mode="surface",
               z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def stack_three_episode_signals(demo):
    """Extract events for the two stack-three ladders."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    A, B, C = obj[:, 0:3], obj[:, 7:10], obj[:, 23:26]
    tr = grip_flips(act)
    if len(tr) != 4:
        raise ValueError(f"stack_three demo has {len(tr)} gripper flips, expected 4")
    c1, o1, c2, o2 = (int(t) for t in tr)
    holdA = _hold_step(ap, c1, o1, _CUBE_HOLD_BAND)
    restA = float(np.median(A[5:max(c1, 6), 2]))
    riseA = _rise_step(A[:, 2], holdA, o1, restA)
    reopened = ap[o1:] > _AP_REOPEN
    openA = o1 + int(np.argmax(reopened)) if reopened.any() else o1
    holdC = _hold_step(ap, c2, o2, _CUBE_HOLD_BAND)
    restC = float(np.median(C[5:max(c2, 6), 2]))
    riseC = _rise_step(C[:, 2], holdC, o2, restC)
    a_on_b = (np.linalg.norm(A[openA, :2] - B[openA, :2]) < _ON_XY
              and A[openA, 2] > B[openA, 2] + _ON_Z)
    return {"A": {"c1": c1, "hold": holdA, "rise": riseA, "o1": o1, "rest": restA,
                  "carry_z": _carry_p90(A[:, 2], riseA, o1, restA + _LIFT["stack"]),
                  "z_table": restA - STACK_EXTENTS["cubeA"][2]},
            "C": {"c1": c2, "hold": holdC, "rise": riseC, "o1": o2, "rest": restC,
                  "carry_z": _carry_p90(C[:, 2], riseC, o2, restC + _LIFT["stack"]),
                  "z_table": restA - STACK_EXTENTS["cubeA"][2]},
            "openA": openA, "a_on_b": a_on_b}


def _stack_three_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = base_ctx["objects"]
    ctx["objects"] = _objects_ctx(raw, STACK_EXTENTS)
    if step < sig["openA"]:
        upd = _stack_stage(base_ctx, raw, sig["A"], step, grasp="cubeA", dest="cubeB",
                           placed=frozenset())
        dest = "cubeB"
        z_table = sig["A"]["z_table"]
    else:
        placed = frozenset({"cubeA"}) if sig["a_on_b"] else frozenset()
        upd = _stack_stage(base_ctx, raw, sig["C"], step, grasp="cubeC", dest="cubeA",
                           placed=placed)
        dest = "cubeA"
        z_table = sig["C"]["z_table"]
    ctx.update(upd)
    ctx.update(destination=dest, contact="pinch", orient="down", place_mode="surface",
               z_table=z_table)
    ctx.setdefault("plan_ref", None)
    return ctx


def square_episode_signals(demo):
    """Extract latched square-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    nut = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    c1 = int(tr[0]) if len(tr) > 0 else T
    o1 = int(tr[1]) if len(tr) > 1 else T
    hold = _hold_step(ap, c1, o1, _NUT_HOLD_BAND)
    rest = float(np.median(nut[5:max(c1, 6), 2]))
    rise = _rise_step(nut[:, 2], hold, o1, rest)
    return {"c1": c1, "hold": hold, "rise": rise, "o1": o1, "rest": rest,
            "carry_z": _carry_p90(nut[:, 2], rise, o1, rest + _LIFT["square"]),
            "z_table": rest - SQUARE_EXTENTS["nut"][2]}


def _square_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    raw["peg1"] = PEG_POS
    nut_quat = base_ctx.get("nut_quat_xyzw")
    rot = _rot_xyzw(nut_quat) if nut_quat is not None else np.eye(3)
    handle = (raw["nut"] + rot @ NUT_HANDLE_LOCAL).astype(np.float32)
    axis = rot @ np.array([0.0, 1.0, 0.0])
    axis = np.array([axis[0], axis[1], 0.0], dtype=np.float32)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else None
    ctx["objects"] = _objects_ctx(raw, SQUARE_EXTENTS,
                                  grasp_extents={"nut": SQUARE_EXTENTS["nut"][0]},
                                  axes={"nut": axis})
    if step < sig["hold"]:
        upd = dict(stage_label="grasp", grasp_obj="nut", payload=None, place_target=None,
                   target=handle, gripper_intent="close")
    elif step < sig["rise"]:
        target = np.array([raw["nut"][0], raw["nut"][1], sig["rest"] + _LIFT["square"]],
                          dtype=np.float32)
        upd = dict(stage_label="lift", grasp_obj="nut", payload="nut", place_target=None,
                   target=target, gripper_intent="close")
    else:


        upd = dict(stage_label="place", grasp_obj=None, payload="nut", place_target="peg1",
                   target=PEG_TOP, place_point=PEG_TOP, carry_z=sig["carry_z"],
                   gripper_intent="place",
                   insert=insert_cone(PEG_TOP, np.array([0.0, 0.0, 1.0], dtype=np.float32),
                                      PEG_FIT, PEG_MOUTH, PEG_CONE_H, PEG_CAPTURE))
        if step >= sig["o1"]:
            upd.update(released=True, place_released=True)
    ctx.update(upd)
    ctx.update(destination="peg1", placed=frozenset(), contact="pinch", orient="down",
               place_mode="container", z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def lift_episode_signals(demo):
    """Extract latched lift-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    cube = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    c1 = int(tr[0]) if len(tr) > 0 else T
    hold = _hold_step(ap, c1, T, _CUBE_HOLD_BAND)
    rest = float(np.median(cube[5:max(c1, 6), 2]))
    rise = _rise_step(cube[:, 2], hold, T, rest)
    return {"c1": c1, "hold": hold, "rise": rise, "o1": T, "rest": rest,
            "carry_z": _carry_p90(cube[:, 2], rise, T, rest + _LIFT["lift"]),
            "z_table": rest - LIFT_EXTENTS["cube"][2]}


def _lift_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = base_ctx["objects"]
    ctx["objects"] = _objects_ctx(raw, LIFT_EXTENTS)
    pos = raw["cube"]
    if step < sig["hold"]:
        upd = dict(stage_label="grasp", grasp_obj="cube", payload=None, place_target=None,
                   target=np.asarray(pos, dtype=np.float32), gripper_intent="close")
    else:
        target = np.array([pos[0], pos[1], sig["rest"] + _LIFT["lift"]], dtype=np.float32)
        upd = dict(stage_label="lift", grasp_obj="cube", payload="cube", place_target=None,
                   target=target, gripper_intent="close")
    ctx.update(upd)
    ctx.update(destination=None, placed=frozenset(), contact="pinch", orient="down",
               place_mode="surface", z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def can_episode_signals(demo):
    """Extract latched can-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    can = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    c1 = int(tr[0]) if len(tr) > 0 else T
    o1 = int(tr[1]) if len(tr) > 1 else T
    hold = _hold_step(ap, c1, o1, _CAN_HOLD_BAND)
    rest = float(np.median(can[5:max(c1, 6), 2]))
    rise = _rise_step(can[:, 2], hold, o1, rest)
    return {"c1": c1, "hold": hold, "rise": rise, "o1": o1, "rest": rest,
            "carry_z": _carry_p90(can[:, 2], rise, o1, rest + _LIFT["can"]),
            "z_table": rest - CAN_EXTENTS["can"][2]}


def _can_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    raw["bin2_q3"] = CAN_SEAT
    ctx["objects"] = _objects_ctx(raw, CAN_EXTENTS)
    if step < sig["hold"]:
        upd = dict(stage_label="grasp", grasp_obj="can", payload=None, place_target=None,
                   target=np.asarray(raw["can"], dtype=np.float32), gripper_intent="close")
    elif step < sig["rise"]:
        target = np.array([raw["can"][0], raw["can"][1], sig["rest"] + _LIFT["can"]],
                          dtype=np.float32)
        upd = dict(stage_label="lift", grasp_obj="can", payload="can", place_target=None,
                   target=target, gripper_intent="close")
    else:


        upd = dict(stage_label="place", grasp_obj=None, payload="can", place_target="bin2_q3",
                   target=CAN_DROP, place_point=CAN_DROP, carry_z=sig["carry_z"],
                   gripper_intent="place")
        if step >= sig["o1"]:
            upd.update(released=True, place_released=True)
    ctx.update(upd)
    ctx.update(destination="bin2_q3", placed=frozenset(), contact="pinch", orient="down",
               place_mode="container", z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def _ring_frame(tripod_pos, tripod_quat_xyzw, needle_pos):
    """Return the ring center and approach-signed axis."""
    rot = _rot_xyzw(tripod_quat_xyzw)
    centre = (np.asarray(tripod_pos) + rot @ RING_LOCAL).astype(np.float32)
    axis = (rot @ RING_AXIS_LOCAL).astype(np.float32)
    if float(axis @ (np.asarray(needle_pos, dtype=np.float32) - centre)) < 0:
        axis = -axis
    return centre, axis


def threading_episode_signals(demo):
    """Extract latched threading-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    needle = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    c1 = int(tr[0]) if len(tr) > 0 else T
    hold = _hold_step(ap, c1, T, _NEEDLE_HOLD_BAND)
    rest = float(np.median(needle[5:max(c1, 6), 2]))
    rise = _rise_step(needle[:, 2], hold, T, rest)

    insert = T
    for t in range(rise, T):
        ring_c, _ = _ring_frame(obj[t, 14:17], obj[t, 17:21], needle[t])
        bar = needle[t] + _rot_xyzw(obj[t, 3:7]) @ NEEDLE_BAR_LOCAL
        if np.linalg.norm(bar - ring_c) < RING_RADIUS:
            insert = t
            break
    return {"c1": c1, "hold": hold, "rise": rise, "insert": insert, "o1": T, "rest": rest,
            "carry_z": _carry_p90(needle[:, 2], rise, T, rest + _LIFT["threading"]),
            "z_table": rest - THREADING_EXTENTS["needle"][2]}


def _threading_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    needle_quat = base_ctx.get("needle_quat_xyzw")
    tripod_quat = base_ctx.get("tripod_quat_xyzw")
    rot = _rot_xyzw(needle_quat) if needle_quat is not None else np.eye(3)
    handle = (raw["needle"] + rot @ NEEDLE_HANDLE_LOCAL).astype(np.float32)
    ring_c, ring_ax = _ring_frame(raw["tripod"], tripod_quat, raw["needle"])
    seat = ring_c + INSERT_ROOT_OFFSET * ring_ax

    axis = rot @ np.array([1.0, 0.0, 0.0])
    axis = np.array([axis[0], axis[1], 0.0], dtype=np.float32)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else None
    ctx["objects"] = _objects_ctx(raw, THREADING_EXTENTS,
                                  grasp_extents={"needle": NEEDLE_GRASP_EXTENT},
                                  axes={"needle": axis})
    if step < sig["hold"]:
        upd = dict(stage_label="grasp", grasp_obj="needle", payload=None, place_target=None,
                   target=handle, gripper_intent="close")
    elif step < sig["rise"]:
        target = np.array([raw["needle"][0], raw["needle"][1], sig["rest"] + _LIFT["threading"]],
                          dtype=np.float32)
        upd = dict(stage_label="lift", grasp_obj="needle", payload="needle", place_target=None,
                   target=target, gripper_intent="close")
    else:


        upd = dict(stage_label="place", grasp_obj=None, payload="needle", place_target="tripod",
                   target=seat, place_point=seat, carry_z=sig["carry_z"],
                   gripper_intent="close",
                   insert=insert_cone(seat, ring_ax, RING_FIT, RING_MOUTH, RING_CONE_H,
                                      RING_CAPTURE))
    ctx.update(upd)
    ctx.update(destination="tripod", placed=frozenset(), contact="pinch", orient="down",
               place_mode="container", z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def _coffee_pod_inserted(row):
    """Evaluate the coffee pod insertion predicate."""
    pod, holder, lid = row[0:3], row[28:31], row[42:45]
    if np.linalg.norm(pod[:2] - holder[:2]) > POD_R_DIFF:
        return False
    z_low = holder[2] - COFFEE_EXTENTS["coffee_pod_holder"][2]
    z_high = lid[2] - COFFEE_EXTENTS["coffee_machine_lid"][2]
    half = COFFEE_EXTENTS["coffee_pod"][2]
    return bool(pod[2] - half > z_low and pod[2] + half < z_high)


def coffee_episode_signals(demo):
    """Extract latched coffee-task stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    pod = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    if len(tr) != 2:
        raise ValueError(f"coffee demo has {len(tr)} gripper flips, expected 2")
    c1, o1 = int(tr[0]), int(tr[1])
    hold = _hold_step(ap, c1, o1, _POD_HOLD_BAND)
    rest = float(np.median(pod[5:max(c1, 6), 2]))
    rise = _rise_step(pod[:, 2], hold, o1, rest)
    inserted = next((t for t in range(o1, T) if _coffee_pod_inserted(obj[t])), T)
    hinge = obj[:, 56]
    cl = np.flatnonzero(hinge < LID_CLOSED)
    closed = int(cl[0]) if len(cl) else T
    return {"c1": c1, "hold": hold, "rise": rise, "o1": o1, "inserted": inserted,
            "closed": closed, "rest": rest,
            "carry_z": _carry_p90(pod[:, 2], rise, o1, rest + _LIFT["coffee"]),
            "z_table": rest - COFFEE_EXTENTS["coffee_pod"][2]}


def _coffee_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    pod_quat = base_ctx.get("pod_quat_xyzw")
    lid_quat = base_ctx.get("lid_quat_xyzw")
    pod_rot = _rot_xyzw(pod_quat) if pod_quat is not None else np.eye(3)
    lid_rot = _rot_xyzw(lid_quat) if lid_quat is not None else np.eye(3)
    grasp = (raw["coffee_pod"] + pod_rot @ POD_GRASP_LOCAL).astype(np.float32)
    drop = (raw["coffee_pod_holder"] + POD_DROP_OFFSET).astype(np.float32)
    press = (raw["coffee_machine_lid"] + lid_rot @ LID_PRESS_LOCAL).astype(np.float32)
    ctx["objects"] = _objects_ctx(raw, COFFEE_EXTENTS,
                                  grasp_extents={"coffee_pod": COFFEE_EXTENTS["coffee_pod"][0]})
    if step < sig["hold"]:
        upd = dict(stage_label="grasp", grasp_obj="coffee_pod", payload=None, place_target=None,
                   target=grasp, gripper_intent="close", contact="pinch",
                   destination="coffee_pod_holder", placed=frozenset())
    elif step < sig["rise"]:
        target = np.array([raw["coffee_pod"][0], raw["coffee_pod"][1],
                           sig["rest"] + _LIFT["coffee"]], dtype=np.float32)
        upd = dict(stage_label="lift", grasp_obj="coffee_pod", payload="coffee_pod",
                   place_target=None, target=target, gripper_intent="close", contact="pinch",
                   destination="coffee_pod_holder", placed=frozenset())
    elif step < sig["o1"]:


        upd = dict(stage_label="place", grasp_obj=None, payload="coffee_pod",
                   place_target="coffee_pod_holder", target=drop, place_point=drop,
                   carry_z=sig["carry_z"], gripper_intent="place", contact="pinch",
                   destination="coffee_pod_holder", placed=frozenset())
    else:


        placed = frozenset({"coffee_pod"}) if step >= sig["inserted"] else frozenset()
        upd = dict(stage_label="close_lid", grasp_obj=None, payload=None, place_target=None,
                   target=press, gripper_intent="open", contact="press",
                   destination="coffee_machine_lid", placed=placed,
                   released=True, place_released=True)
        spec = hinge_press(raw["coffee_machine_lid"], lid_rot, LID_PRESS_LOCAL, LID_HINGE_LOCAL,
                           LID_HINGE_AXIS, LID_PRESS_DEPTH, LID_PRESS_TOL, LID_PRESS_REACH)
        if spec is not None:
            ctx["press"] = spec
    ctx.update(upd)
    ctx.update(orient="down", place_mode="container", z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def _drawer_frame(drawer_root, drawer_quat_xyzw, qpos):
    """Return the drawer-link pose for a slide position."""
    rot = _rot_xyzw(drawer_quat_xyzw) if drawer_quat_xyzw is not None else np.eye(3)
    link = np.asarray(drawer_root) + rot @ (DRAWER_LINK_LOCAL + np.array([0.0, qpos, 0.0]))
    return link.astype(np.float32), rot


def _mug_in_drawer(mug_pos, link, rot):
    local = rot.T @ (np.asarray(mug_pos) - link)
    return bool(abs(local[0]) < DRAWER_INTERIOR_X
                and DRAWER_INTERIOR_Y[0] < local[1] < DRAWER_INTERIOR_Y[1]
                and local[2] < _MUG_IN_Z)


def mug_cleanup_episode_signals(demo):
    """Extract latched mug-cleanup stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    mug = obj[:, 0:3]
    djnt = obj[:, 28]
    T = len(act)
    tr = grip_flips(act)
    if len(tr) != 2:
        raise ValueError(f"mug_cleanup demo has {len(tr)} gripper flips, expected 2")
    c1, o1 = int(tr[0]), int(tr[1])
    moving = djnt < -_DRAWER_MOVE_EPS
    open_start = int(np.argmax(moving)) if moving.any() else T
    qmin = float(djnt.min())
    open_end = int(np.argmax(djnt < qmin + _DRAWER_MOVE_EPS))
    after = djnt[o1:] > qmin + _DRAWER_MOVE_EPS
    close_start = o1 + int(np.argmax(after)) if after.any() else T
    hold = _hold_step(ap, c1, o1, _MUG_HOLD_BAND)
    rest = float(np.median(mug[5:max(c1, 6), 2]))
    rise = _rise_step(mug[:, 2], hold, o1, rest)
    return {"open_start": open_start, "open_end": open_end, "c1": c1, "hold": hold,
            "rise": rise, "o1": o1, "close_start": close_start, "qmin": qmin, "rest": rest,
            "carry_z": _carry_p90(mug[:, 2], rise, o1, rest + _LIFT["mug_cleanup"]),
            "z_table": rest - MUG_EXTENTS["mug"][2]}


def _mug_cleanup_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    dq = base_ctx.get("drawer_quat_xyzw")
    qpos = float(base_ctx.get("drawer_joint_pos", 0.0))
    link, rot = _drawer_frame(raw["drawer"], dq, qpos)
    handle = (link + rot @ DRAWER_HANDLE_LOCAL).astype(np.float32)
    seat = (link + rot @ MUG_SEAT_LOCAL).astype(np.float32)
    objects = _objects_ctx(raw, MUG_EXTENTS, grasp_extents={"mug": MUG_EXTENTS["mug"][0]})
    placed = frozenset()
    place_target = place_point = carry_z = None
    contact, orient, intent = "pinch", "down", "close"

    handle_at = lambda q: (_drawer_frame(raw["drawer"], dq, q)[0]
                           + rot @ DRAWER_HANDLE_LOCAL).astype(np.float32)
    pull = None
    if step < sig["open_start"]:

        objects["handle"] = {"pos": torch.as_tensor(handle), "extents": HANDLE_EXTENTS,
                             "axis": None, "grasp_extent": HANDLE_EXTENTS[0],
                             "grasp_region": None}
        stage, grasp_obj, payload = "hook_handle", "handle", None
        target, contact, intent = handle, "press", "open"
        pull = slide_pull(rot, handle, handle_at(-DRAWER_STROKE), MC_HOOK_OPEN, True)
    elif step < sig["open_end"]:

        open_handle, _ = _drawer_frame(raw["drawer"], dq, DRAWER_OPEN_QPOS)
        stage, grasp_obj, payload = "open_drawer", None, None
        target = (open_handle + rot @ DRAWER_HANDLE_LOCAL).astype(np.float32)
        contact, orient, intent = "press", "free", "open"
        pull = slide_pull(rot, handle, handle_at(-DRAWER_STROKE), MC_HOOK_OPEN, True)
    elif step < sig["hold"]:
        stage, grasp_obj, payload = "grasp", "mug", None
        target = np.asarray(raw["mug"], dtype=np.float32)
    elif step < sig["rise"]:
        stage, grasp_obj, payload = "lift", "mug", "mug"
        target = np.array([raw["mug"][0], raw["mug"][1], sig["rest"] + _LIFT["mug_cleanup"]],
                          dtype=np.float32)
    elif step < sig["close_start"]:
        stage, grasp_obj, payload = "place", None, "mug"
        place_target, place_point, target = "drawer", seat, seat
        carry_z = sig["carry_z"]
        intent = "place"
        if step >= sig["o1"]:
            ctx.update(released=True, place_released=True)
            if _mug_in_drawer(raw["mug"], link, rot):
                placed = frozenset({"mug"})
    else:

        closed_handle, _ = _drawer_frame(raw["drawer"], dq, 0.0)
        stage, grasp_obj, payload = "close_drawer", None, None
        target = (closed_handle + rot @ DRAWER_HANDLE_LOCAL).astype(np.float32)
        contact, intent = "press", "open"
        pull = slide_pull(rot, handle, handle_at(0.0), MC_HOOK_CLOSE, False)
        ctx.update(released=True, place_released=True)
        if _mug_in_drawer(raw["mug"], link, rot):
            placed = frozenset({"mug"})
    if pull is not None:
        ctx["pull"] = pull
    ctx["objects"] = objects
    ctx.update(stage_label=stage, grasp_obj=grasp_obj, payload=payload,
               place_target=place_target, target=np.asarray(target, dtype=np.float32),
               destination="drawer", placed=placed, contact=contact, orient=orient,
               place_mode="container", gripper_intent=intent, z_table=sig["z_table"])
    if place_point is not None:
        ctx["place_point"] = place_point
    if carry_z is not None:
        ctx["carry_z"] = carry_z
    ctx.setdefault("plan_ref", None)
    return ctx


def tpa_episode_signals(demo):
    """Extract events for both assembly ladders."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    base, p1, p2 = obj[:, 0:3], obj[:, 14:17], obj[:, 28:31]
    tr = grip_flips(act)
    if len(tr) != 4:
        raise ValueError(f"three_piece_assembly demo has {len(tr)} gripper flips, expected 4")
    c1, o1, c2, o2 = (int(t) for t in tr)
    hold1 = _hold_step(ap, c1, o1, _CUBE_HOLD_BAND)
    rest1 = float(np.median(p1[5:max(c1, 6), 2]))
    rise1 = _rise_step(p1[:, 2], hold1, o1, rest1)
    reopened = ap[o1:] > _AP_REOPEN
    open1 = o1 + int(np.argmax(reopened)) if reopened.any() else o1
    hold2 = _hold_step(ap, c2, o2, _CUBE_HOLD_BAND)
    rest2 = float(np.median(p2[5:max(c2, 6), 2]))
    rise2 = _rise_step(p2[:, 2], hold2, o2, rest2)
    p1_on_base = np.linalg.norm(p1[open1, :2] - base[open1, :2]) < _ON_XY
    return {"p1": {"c1": c1, "hold": hold1, "rise": rise1, "o1": o1, "rest": rest1,
                   "carry_z": _carry_p90(p1[:, 2], rise1, o1, rest1 + TPA_LIFT["piece_1"]),
                   "z_table": rest1 - TPA_EXTENTS["piece_1"][2]},
            "p2": {"c1": c2, "hold": hold2, "rise": rise2, "o1": o2, "rest": rest2,
                   "carry_z": _carry_p90(p2[:, 2], rise2, o2, rest2 + TPA_LIFT["piece_2"]),
                   "z_table": rest1 - TPA_EXTENTS["piece_1"][2]},
            "open1": open1, "p1_on_base": p1_on_base}


def _tpa_ladder(sig, step, *, grasp, pos, seat, target_grasp, lift, placed):
    """Build one assembly grasp, lift, and place sequence."""
    if step < sig["hold"]:
        return dict(stage_label="grasp", grasp_obj=grasp, payload=None, place_target=None,
                    target=target_grasp, gripper_intent="close", placed=placed)
    if step < sig["rise"]:
        target = np.array([pos[0], pos[1], sig["rest"] + lift], dtype=np.float32)
        return dict(stage_label="lift", grasp_obj=grasp, payload=grasp, place_target=None,
                    target=target, gripper_intent="close", placed=placed)
    upd = dict(stage_label="place", grasp_obj=None, payload=grasp,
               target=seat, place_point=seat, carry_z=sig["carry_z"],
               gripper_intent="place", placed=placed)
    if step >= sig["o1"]:
        upd.update(released=True, place_released=True)
    return upd


def _tpa_frame_context(base_ctx, sig, step, **_):
    ctx = dict(base_ctx)
    raw = base_ctx["objects"]
    p1_quat = base_ctx.get("piece_1_quat_xyzw")
    p2_quat = base_ctx.get("piece_2_quat_xyzw")
    r1 = _rot_xyzw(p1_quat) if p1_quat is not None else np.eye(3)
    r2 = _rot_xyzw(p2_quat) if p2_quat is not None else np.eye(3)

    axis = r1 @ np.array([1.0, 0.0, 0.0])
    axis = np.array([axis[0], axis[1], 0.0], dtype=np.float32)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else None
    ctx["objects"] = _objects_ctx(raw, TPA_EXTENTS,
                                  grasp_extents={"piece_1": TPA_EXTENTS["piece_1"][0],
                                                 "piece_2": TPA_EXTENTS["piece_2"][0]},
                                  axes={"piece_1": axis})
    if step < sig["open1"]:

        seat = (raw["base"] + SEAT1_OFFSET).astype(np.float32)
        grasp_t = (raw["piece_1"] + r1 @ P1_GRASP_LOCAL).astype(np.float32)
        upd = _tpa_ladder(sig["p1"], step, grasp="piece_1", pos=raw["piece_1"], seat=seat,
                          target_grasp=grasp_t, lift=TPA_LIFT["piece_1"], placed=frozenset())
        dest = "base"
        z_table = sig["p1"]["z_table"]
    else:


        seat = np.array([raw["piece_1"][0], raw["piece_1"][1],
                         raw["base"][2] + SEAT2_OFFSET[2]], dtype=np.float32)
        grasp_t = (raw["piece_2"] + r2 @ P2_GRASP_LOCAL).astype(np.float32)
        placed = frozenset({"piece_1"}) if sig["p1_on_base"] else frozenset()
        upd = _tpa_ladder(sig["p2"], step, grasp="piece_2", pos=raw["piece_2"], seat=seat,
                          target_grasp=grasp_t, lift=TPA_LIFT["piece_2"], placed=placed)
        dest = "piece_1"
        z_table = sig["p2"]["z_table"]
    if upd["stage_label"] == "place":
        upd["place_target"] = dest
    ctx.update(upd)
    ctx.update(destination=dest, contact="pinch", orient="down", place_mode="container",
               z_table=z_table)
    ctx.setdefault("plan_ref", None)
    return ctx


def _hc_drawer_link(drawer_q):
    """Return the hammer cabinet drawer-link position."""
    return (CAB_ROOT + DRAWER_LINK_LOCAL + np.array([0.0, drawer_q, 0.0])).astype(np.float32)


def _hammer_in_drawer(ham, drawer_q):
    local = np.asarray(ham, dtype=np.float32) - _hc_drawer_link(drawer_q)
    return bool(abs(local[0]) < HC_IN_XY[0] and abs(local[1]) < HC_IN_XY[1]
                and ham[2] < HC_IN_Z)


def hammer_cleanup_episode_signals(demo):
    """Extract latched hammer-cleanup stage events."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    dq = np.asarray(demo["states"])[:, 1]
    ham = obj[:, 0:3]
    T = len(act)
    tr = grip_flips(act)
    if len(tr) != 2:
        raise ValueError(f"hammer_cleanup demo has {len(tr)} gripper flips, expected 2")
    c1, o1 = int(tr[0]), int(tr[1])
    moving = np.flatnonzero(dq < DRAWER_MOVING_Q)
    pull_start = int(moving[0]) if len(moving) else T
    opened = np.flatnonzero(dq < DRAWER_OPEN_Q)
    pull_done = int(opened[0]) if len(opened) else T
    hold = _hold_step(ap, c1, o1, _HAMMER_HOLD_BAND)
    rest = float(np.median(ham[5:max(c1, 6), 2]))
    rise = _rise_step(ham[:, 2], hold, o1, rest)
    closed = np.flatnonzero(dq[o1:] > DRAWER_CLOSED_Q)
    close_done = o1 + int(closed[0]) if len(closed) else T
    return {"pull_start": pull_start, "pull_done": pull_done, "c1": c1, "hold": hold,
            "rise": rise, "o1": o1, "close_done": close_done, "rest": rest,
            "carry_z": _carry_p90(ham[:, 2], rise, o1, rest + _LIFT["hammer_cleanup"]),
            "z_table": rest - HC_EXTENTS["hammer"][2]}


def _hammer_cleanup_frame_context(base_ctx, sig, step, **_):
    """Build the hammer-cleanup cost context for one frame."""
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    dq = float(base_ctx.get("drawer_q", 0.0))
    ham_quat = base_ctx.get("hammer_quat_xyzw")
    dlink = _hc_drawer_link(dq)
    raw["drawer"] = dlink
    handle = (dlink + DRAWER_HANDLE_LOCAL).astype(np.float32)
    cavity = (dlink + CAVITY_LOCAL).astype(np.float32)

    axis = None
    if ham_quat is not None:
        long_ax = _rot_xyzw(ham_quat) @ np.array([0.0, 0.0, 1.0])
        a = np.array([-long_ax[1], long_ax[0], 0.0], dtype=np.float32)
        n = np.linalg.norm(a)
        axis = a / n if n > 1e-6 else None
    ctx["objects"] = _objects_ctx(raw, HC_EXTENTS,
                                  grasp_extents={"hammer": HC_EXTENTS["hammer"][0]},
                                  axes={"hammer": axis})
    ham = raw["hammer"]
    placed = frozenset({"hammer"}) if _hammer_in_drawer(ham, dq) else frozenset()
    contact, orient, intent = "pinch", "down", "close"
    released = step >= sig["o1"]

    handle_at = lambda q: (_hc_drawer_link(q) + DRAWER_HANDLE_LOCAL).astype(np.float32)
    pull = None
    if step < sig["pull_start"]:

        stage, grasp_obj, payload = "hook_handle", "handle", None
        target, contact, intent = handle + HOOK_OFFSET, "press", "open"
        ctx["objects"]["handle"] = _objects_ctx(
            {"handle": handle}, HC_EXTENTS,
            grasp_extents={"handle": HC_EXTENTS["handle"][0]})["handle"]
        pull = slide_pull(np.eye(3), handle, handle_at(-DRAWER_STROKE), HC_HOOK_OPEN, True)
    elif step < sig["pull_done"]:

        stage, grasp_obj, payload = "pull_open", None, None
        target = (CAB_ROOT + DRAWER_LINK_LOCAL + DRAWER_HANDLE_LOCAL + HOOK_OFFSET
                  + np.array([0.0, -0.135, 0.0]))
        contact, orient, intent = "press", "free", "open"
        pull = slide_pull(np.eye(3), handle, handle_at(-DRAWER_STROKE), HC_HOOK_OPEN, True)
    elif step < sig["hold"]:
        stage, grasp_obj, payload = "grasp", "hammer", None
        target = np.asarray(ham, dtype=np.float32)
    elif step < sig["rise"]:
        stage, grasp_obj, payload = "lift", "hammer", "hammer"
        target = np.array([ham[0], ham[1], sig["rest"] + _LIFT["hammer_cleanup"]],
                          dtype=np.float32)
    elif step < sig["o1"]:


        stage, grasp_obj, payload = "place", None, "hammer"
        target = np.array([cavity[0], cavity[1], HC_DROP_Z], dtype=np.float32)
        ctx.update(place_point=target, carry_z=sig["carry_z"])
        intent = "place"
    else:

        stage, grasp_obj, payload = "push_close", None, None
        target = CAB_ROOT + DRAWER_LINK_LOCAL + DRAWER_HANDLE_LOCAL
        contact, intent = "press", "open"
        pull = slide_pull(np.eye(3), handle, handle_at(0.0), HC_HOOK_CLOSE, False)
    if pull is not None:
        ctx["pull"] = pull
    place_target = "drawer" if stage in ("place", "push_close") else None
    ctx.update(stage_label=stage, grasp_obj=grasp_obj, payload=payload,
               place_target=place_target, target=np.asarray(target, dtype=np.float32),
               gripper_intent=intent, contact=contact, orient=orient,
               destination="drawer", placed=placed, place_mode="container",
               z_table=sig["z_table"])
    if released:
        ctx.update(released=True, place_released=True)
    ctx.setdefault("plan_ref", None)
    return ctx


def kitchen_episode_signals(demo):
    """Latched kitchen events: button ON/OFF (states hinge), 3 grasp pairs, holds, rises, pot-on-stove, bread-in-pot, pot-in-serving-box."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    btn = np.asarray(demo["states"])[:, 1]
    bread, pot = obj[:, 14:17], obj[:, 28:31]
    T = len(act)
    tr = grip_flips(act)
    if len(tr) != 6:
        raise ValueError(f"kitchen demo has {len(tr)} gripper flips, expected 6")
    c1, o1, c2, o2, c3, o3 = (int(t) for t in tr)
    on = np.flatnonzero(btn >= 0.0)
    t_on = int(on[0]) if len(on) else T
    off = np.flatnonzero((btn[:-1] >= 0.0) & (btn[1:] < 0.0)) + 1
    t_off = int(off[-1]) if len(off) else T
    h1 = _hold_step(ap, c1, o1, _POT_HOLD_BAND)
    h2 = _hold_step(ap, c2, o2, _BREAD_HOLD_BAND)
    h3 = _hold_step(ap, c3, o3, _POT_HOLD_BAND)
    r1 = _rise_step(pot[:, 2], h1, o1, POT_TABLE_REST_Z)
    r2 = _rise_step(bread[:, 2], h2, o2, float(np.median(bread[20:c1, 2])))
    r3 = _rise_step(pot[:, 2], h3, o3, POT_STOVE_SEAT_Z)
    srel = SERVING_POS - pot
    served = (np.abs(srel[:, 0]) < SERVE_BOX[0]) & (np.abs(srel[:, 1]) < SERVE_BOX[1]) \
        & (np.abs(srel[:, 2]) < SERVE_BOX[2])
    t_serve = int(np.argmax(served)) if served.any() else T
    return {"t_on": t_on, "t_off": t_off, "t_serve": t_serve,
            "c1": c1, "o1": o1, "c2": c2, "o2": o2, "c3": c3, "o3": o3,
            "h1": h1, "h2": h2, "h3": h3, "r1": r1, "r2": r2, "r3": r3,
            "bread_rest": float(np.median(bread[20:c1, 2])),
            "pot_carry1": _carry_p90(pot[:, 2], r1, o1,
                                     POT_TABLE_REST_Z + _LIFT["kitchen_pot"]),
            "bread_carry": _carry_p90(bread[:, 2], r2, o2, 0.9198 + _LIFT["kitchen_bread"]),
            "pot_carry3": _carry_p90(pot[:, 2], r3, o3,
                                     POT_STOVE_SEAT_Z + _LIFT["kitchen_pot"]),
            "z_table": 0.90}


def _kitchen_frame_context(base_ctx, sig, step, **_):
    """12-stage ladder: button_on -> (grasp/lift/place pot on stove) -> (grasp/lift/drop bread in pot) -> (regrasp/lift/place pot short of serving) -> push_pot_serve -> button_off."""
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    pot_quat = base_ctx.get("pot_quat_xyzw")
    rot = _rot_xyzw(pot_quat) if pot_quat is not None else np.eye(3)
    handle = (raw["pot"] + rot @ POT_HANDLE_LOCAL).astype(np.float32)

    axis = rot @ np.array([0.0, 1.0, 0.0])
    axis = np.array([axis[0], axis[1], 0.0], dtype=np.float32)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else None
    ctx["objects"] = _objects_ctx(raw, KITCHEN_EXTENTS,
                                  grasp_extents={"pot": POT_GRASP_EXTENT},
                                  axes={"pot": axis})
    stove_seat = np.array([STOVE_POS[0], STOVE_POS[1], POT_STOVE_SEAT_Z], dtype=np.float32)
    drop_point = (raw["pot"] + np.array([0.0, 0.0, BREAD_DROP_DZ])).astype(np.float32)
    contact, intent, orient = "pinch", "close", "down"
    place_mode = "surface"
    placed = frozenset()
    released = False
    dest = "stove"
    if step < sig["t_on"]:
        upd = dict(stage_label="button_on", grasp_obj="button", payload=None,
                   place_target=None, target=BTN_PRESS_ON)
        contact, intent = "press", "open"
    elif step < sig["h1"]:
        upd = dict(stage_label="grasp_pot", grasp_obj="pot", payload=None,
                   place_target=None, target=handle)
    elif step < sig["r1"]:
        target = np.array([raw["pot"][0], raw["pot"][1],
                           POT_TABLE_REST_Z + _LIFT["kitchen_pot"]], dtype=np.float32)
        upd = dict(stage_label="lift_pot", grasp_obj="pot", payload="pot",
                   place_target=None, target=target)
    elif step < sig["c2"]:
        upd = dict(stage_label="place_pot_stove", grasp_obj=None, payload="pot",
                   place_target="stove", target=stove_seat, place_point=stove_seat,
                   carry_z=sig["pot_carry1"])
        intent = "place"
        released = step >= sig["o1"]
    elif step < sig["h2"]:
        upd = dict(stage_label="grasp_bread", grasp_obj="bread", payload=None,
                   place_target=None, target=np.asarray(raw["bread"], dtype=np.float32))
        placed, dest = frozenset({"pot"}), "pot"
    elif step < sig["r2"]:
        target = np.array([raw["bread"][0], raw["bread"][1],
                           sig["bread_rest"] + _LIFT["kitchen_bread"]], dtype=np.float32)
        upd = dict(stage_label="lift_bread", grasp_obj="bread", payload="bread",
                   place_target=None, target=target)
        placed, dest = frozenset({"pot"}), "pot"
    elif step < sig["c3"]:

        upd = dict(stage_label="place_bread_pot", grasp_obj=None, payload="bread",
                   place_target="pot", target=drop_point, place_point=drop_point,
                   carry_z=sig["bread_carry"])
        intent, place_mode = "place", "container"
        placed, dest = frozenset({"pot"}), "pot"
        released = step >= sig["o2"]
    elif step < sig["h3"]:
        upd = dict(stage_label="regrasp_pot", grasp_obj="pot", payload=None,
                   place_target=None, target=handle)
        placed, dest = frozenset({"bread"}), "serving_region"
    elif step < sig["r3"]:
        target = np.array([raw["pot"][0], raw["pot"][1],
                           POT_STOVE_SEAT_Z + _LIFT["kitchen_pot"]], dtype=np.float32)
        upd = dict(stage_label="lift_pot2", grasp_obj="pot", payload="pot",
                   place_target=None, target=target)
        placed, dest = frozenset({"bread"}), "serving_region"
    elif step < sig["o3"]:

        upd = dict(stage_label="place_pot_serve", grasp_obj=None, payload="pot",
                   place_target="serving_region", target=SERVE_RELEASE,
                   place_point=SERVE_RELEASE, carry_z=sig["pot_carry3"])
        intent = "place"
        placed, dest = frozenset({"bread"}), "serving_region"
    elif step < sig["t_serve"]:
        push_point = (raw["pot"] + PUSH_EEF_OFFSET).astype(np.float32)
        upd = dict(stage_label="push_pot_serve", grasp_obj=None, payload=None,
                   place_target="serving_region", target=push_point)
        contact, intent = "press", "open"
        placed, dest = frozenset({"bread"}), "serving_region"
        released = True
    else:
        upd = dict(stage_label="button_off", grasp_obj="button", payload=None,
                   place_target=None, target=BTN_PRESS_OFF)
        contact, intent = "press", "open"
        placed, dest = frozenset({"bread", "pot"}), "serving_region"
        released = True
    ctx.update(upd)
    if released:
        ctx.update(released=True, place_released=True)
    ctx.update(destination=dest, placed=placed, contact=contact,
               orient=orient, place_mode=place_mode, gripper_intent=intent,
               z_table=sig["z_table"])
    ctx.setdefault("plan_ref", None)
    return ctx


def _first_step(pred, lo, hi, fallback):
    """First step in [lo, hi) where pred holds, else fallback."""
    return next((t for t in range(lo, hi) if pred(t)), fallback)


def coffee_prep_episode_signals(demo):
    """Nine-stage ladder events: two pinch ladders (mug, pod) + three press stages."""
    act = np.asarray(demo["actions"])
    ap = aperture(demo["obs/robot0_gripper_qpos"])
    obj = np.asarray(demo["obs/object"])
    mug, pod = obj[:, 70:73], obj[:, 0:3]
    machine, holder = obj[:, 14:17], obj[:, 28:31]
    hinge, slide = obj[:, 84], obj[:, 85]
    T = len(act)
    tr = grip_flips(act)
    if len(tr) != 4:
        raise ValueError(f"coffee_prep demo has {len(tr)} gripper flips, expected 4")
    c1, o1, c2, o2 = (int(t) for t in tr)

    hold1 = _hold_step(ap, c1, o1, _CP_MUG_HOLD_BAND)
    rest1 = float(np.median(mug[5:max(c1, 6), 2]))
    rise1 = _rise_step(mug[:, 2], hold1, o1, rest1)

    lid_start = _first_step(lambda t: hinge[t] > _HINGE_MOVING, o1, T, T)
    lid_open = _first_step(lambda t: hinge[t] > LID_OPEN_Q, lid_start, T, T)
    drw_start = _first_step(lambda t: slide[t] < _SLIDE_MOVING, lid_open, T, T)
    drw_open = _first_step(lambda t: slide[t] < CP_DRAWER_OPEN_Q, drw_start, T, T)

    hold2 = _hold_step(ap, c2, o2, _CP_POD_HOLD_BAND)
    rest2 = float(np.median(pod[max(c2 - 5, 0):max(c2, 1), 2]))
    rise2 = _rise_step(pod[:, 2], hold2, o2, rest2)
    close_start = _first_step(lambda t: hinge[t] - hinge[t + 1] > _HINGE_CLOSING_RATE,
                              o2, T - 1, o2)

    mR = _rot_xyzw(obj[0, 17:21])
    seat = machine[min(lid_start, T - 1)] + mR @ MUG_SEAT_MACHINE_LOCAL
    mugR = _rot_xyzw(obj[min(lid_start, T - 1), 73:77])
    mug_on = (np.linalg.norm(mug[min(lid_start, T - 1), :2] - seat[:2]) < _MUG_ON_XY
              and 1.0 - mugR[2, 2] < _MUG_UPRIGHT)
    d_pod = pod[T - 1] - holder[T - 1]
    pod_in = (np.linalg.norm(d_pod[:2]) < POD_R_DIFF and 0.0 < d_pod[2] < 0.02)
    return {"c1": c1, "o1": o1, "c2": c2, "o2": o2,
            "hold1": hold1, "rise1": rise1, "rest1": rest1,
            "carry1": _carry_p90(mug[:, 2], rise1, o1, rest1 + _LIFT["coffee_prep_mug"]),
            "lid_start": lid_start, "lid_open": lid_open,
            "drw_start": drw_start, "drw_open": drw_open,
            "hold2": hold2, "rise2": rise2, "rest2": rest2,
            "carry2": _carry_p90(pod[:, 2], rise2, o2, rest2 + _LIFT["coffee_prep_pod"]),
            "close_start": close_start, "mug_on": mug_on, "pod_in": pod_in,
            "z_table": rest1 - 0.0386}


def _coffee_prep_frame_context(base_ctx, sig, step, **_):
    """grasp -> lift -> place (mug) -> open_lid -> open_drawer -> grasp -> lift -> place (pod) -> close_lid, all events measured."""
    ctx = dict(base_ctx)
    raw = dict(base_ctx["objects"])
    quat_rot = lambda key: (_rot_xyzw(base_ctx[key])
                            if base_ctx.get(key) is not None else np.eye(3))
    mR, lidR = quat_rot("machine_quat_xyzw"), quat_rot("lid_quat_xyzw")
    mugR, podR = quat_rot("mug_quat_xyzw"), quat_rot("pod_quat_xyzw")
    cabR = quat_rot("cabinet_quat_xyzw")
    slide = float(base_ctx.get("slide_q", 0.0))
    mug_grasp = (raw["mug"] + mugR @ MUG_GRASP_LOCAL).astype(np.float32)
    pod_grasp = (raw["coffee_pod"] + podR @ CP_POD_GRASP_LOCAL).astype(np.float32)
    mug_seat = (raw["coffee_machine"] + mR @ MUG_SEAT_MACHINE_LOCAL).astype(np.float32)
    pod_seat = (raw["coffee_pod_holder"]
                + np.array([0.0, 0.0, POD_RELEASE_HOLDER_DZ])).astype(np.float32)
    lid_open_press = (raw["coffee_machine_lid"] + lidR @ LID_OPEN_PRESS_LOCAL).astype(
        np.float32)
    lid_close_press = (raw["coffee_machine_lid"] + lidR @ LID_CLOSE_PRESS_LOCAL).astype(
        np.float32)

    hook = (raw["cabinet"] + cabR @ (DRAWER_HOOK_LOCAL
                                     + np.array([0.0, slide, 0.0]))).astype(np.float32)
    ctx["objects"] = _objects_ctx(
        raw, COFFEE_PREP_EXTENTS,
        grasp_extents={"mug": COFFEE_PREP_EXTENTS["mug"][0],
                       "coffee_pod": COFFEE_PREP_EXTENTS["coffee_pod"][0]})
    placed = frozenset()
    if sig["mug_on"] and step >= sig["lid_start"]:
        placed = placed | {"mug"}
    if sig["pod_in"] and step >= sig["close_start"]:
        placed = placed | {"coffee_pod"}
    contact, orient, intent = "pinch", "down", "close"
    dest, place_mode = "coffee_machine", "surface"
    place_target = place_point = carry_z = None
    released = False
    if step < sig["hold1"]:
        stage, grasp_obj, payload, target = "grasp", "mug", None, mug_grasp
    elif step < sig["rise1"]:
        stage, grasp_obj, payload = "lift", "mug", "mug"
        target = np.array([raw["mug"][0], raw["mug"][1],
                           sig["rest1"] + _LIFT["coffee_prep_mug"]], dtype=np.float32)
    elif step < sig["lid_start"]:
        stage, grasp_obj, payload = "place", None, "mug"
        place_target, target, place_point = "coffee_machine", mug_seat, mug_seat
        carry_z, intent = sig["carry1"], "place"
        released = step >= sig["o1"]
    elif step < sig["lid_open"]:

        stage, grasp_obj, payload = "open_lid", None, None
        target, contact, intent = lid_open_press, "press", "open"
        dest = "coffee_machine"
    elif step < sig["drw_open"]:
        stage, grasp_obj, payload = "open_drawer", None, None
        target, contact, intent = hook, "press", "open"
        dest = "cabinet"
    elif step < sig["hold2"]:
        stage, grasp_obj, payload, target = "grasp", "coffee_pod", None, pod_grasp
        dest = "coffee_pod_holder"
    elif step < sig["rise2"]:
        stage, grasp_obj, payload = "lift", "coffee_pod", "coffee_pod"
        target = np.array([raw["coffee_pod"][0], raw["coffee_pod"][1],
                           sig["rest2"] + _LIFT["coffee_prep_pod"]], dtype=np.float32)
        dest = "coffee_pod_holder"
    elif step < sig["close_start"]:


        stage, grasp_obj, payload = "place", None, "coffee_pod"
        place_target, target, place_point = "coffee_pod_holder", pod_seat, pod_seat
        carry_z, intent = sig["carry2"], "place"
        dest, place_mode = "coffee_pod_holder", "container"
        released = step >= sig["o2"]
    else:
        stage, grasp_obj, payload = "close_lid", None, None
        target, contact, intent = lid_close_press, "press", "open"
        dest = "coffee_machine"
    ctx.update(stage_label=stage, grasp_obj=grasp_obj, payload=payload,
               place_target=place_target, target=np.asarray(target, dtype=np.float32),
               gripper_intent=intent, destination=dest, placed=placed, contact=contact,
               orient=orient, place_mode=place_mode, z_table=sig["z_table"])
    if place_point is not None:
        ctx["place_point"] = place_point
    if carry_z is not None:
        ctx["carry_z"] = carry_z
    if released:
        ctx.update(released=True, place_released=True)
    ctx.setdefault("plan_ref", None)
    return ctx


MG_TASKS = {
    "stack": OfflineTask(
        task_id="Stack_D0",
        hdf5=str(DATA / "stack_d0/demo.hdf5"),
        frame_context=_stack_frame_context,
        episode_signals=stack_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "stack_three": OfflineTask(
        task_id="StackThree_D0",
        hdf5=str(DATA / "stack_three_d0/demo.hdf5"),
        frame_context=_stack_three_frame_context,
        episode_signals=stack_three_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "square": OfflineTask(
        task_id="Square_D0",
        hdf5=str(DATA / "square_d0/demo.hdf5"),
        frame_context=_square_frame_context,
        episode_signals=square_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "lift": OfflineTask(
        task_id="Lift",
        hdf5=str(DATA / "lift/demo.hdf5"),
        frame_context=_lift_frame_context,
        episode_signals=lift_episode_signals,
        stage_order=("grasp", "lift")),
    "can": OfflineTask(
        task_id="PickPlaceCan",
        hdf5=str(DATA / "can/demo.hdf5"),
        frame_context=_can_frame_context,
        episode_signals=can_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "threading": OfflineTask(
        task_id="Threading_D0",
        hdf5=str(DATA / "threading_d0/demo.hdf5"),
        frame_context=_threading_frame_context,
        episode_signals=threading_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "coffee": OfflineTask(
        task_id="Coffee_D0",
        hdf5=str(DATA / "coffee_d0/demo.hdf5"),
        frame_context=_coffee_frame_context,
        episode_signals=coffee_episode_signals,
        stage_order=("grasp", "lift", "place", "close_lid")),
    "mug_cleanup": OfflineTask(
        task_id="MugCleanup_D0",
        hdf5=str(DATA / "mug_cleanup_d0/demo.hdf5"),
        frame_context=_mug_cleanup_frame_context,
        episode_signals=mug_cleanup_episode_signals,
        stage_order=("hook_handle", "open_drawer", "grasp", "lift", "place",
                     "close_drawer")),
    "three_piece_assembly": OfflineTask(
        task_id="ThreePieceAssembly_D0",
        hdf5=str(DATA / "three_piece_assembly_d0/demo.hdf5"),
        frame_context=_tpa_frame_context,
        episode_signals=tpa_episode_signals,
        stage_order=("grasp", "lift", "place")),
    "hammer_cleanup": OfflineTask(
        task_id="HammerCleanup_D0",
        hdf5=str(DATA / "hammer_cleanup_d0/demo.hdf5"),
        frame_context=_hammer_cleanup_frame_context,
        episode_signals=hammer_cleanup_episode_signals,
        stage_order=("hook_handle", "pull_open", "grasp", "lift", "place", "push_close")),
    "kitchen": OfflineTask(
        task_id="Kitchen_D0",
        hdf5=str(DATA / "kitchen/demo.hdf5"),
        frame_context=_kitchen_frame_context,
        episode_signals=kitchen_episode_signals,
        stage_order=("button_on", "grasp_pot", "lift_pot", "place_pot_stove",
                     "grasp_bread", "lift_bread", "place_bread_pot", "regrasp_pot",
                     "lift_pot2", "place_pot_serve", "push_pot_serve", "button_off")),
    "coffee_prep": OfflineTask(
        task_id="CoffeePreparation_D0",
        hdf5=str(DATA / "coffee_preparation_d0/demo.hdf5"),
        frame_context=_coffee_prep_frame_context,
        episode_signals=coffee_prep_episode_signals,
        stage_order=("grasp", "lift", "place", "open_lid", "open_drawer", "close_lid")),
}

OBJ_LAYOUT = {"stack": STACK_LAYOUT, "stack_three": STACK3_LAYOUT, "square": SQUARE_LAYOUT,
              "lift": LIFT_LAYOUT, "can": CAN_LAYOUT, "threading": THREADING_LAYOUT,
              "coffee": COFFEE_LAYOUT, "mug_cleanup": MUG_LAYOUT,
              "three_piece_assembly": TPA_LAYOUT, "hammer_cleanup": HC_LAYOUT,
              "kitchen": KITCHEN_LAYOUT, "coffee_prep": COFFEE_PREP_LAYOUT}


def _timeline(task, n_demos=3):
    """Print per-demo stage timelines with object heights, for eyeball validation."""
    import h5py
    spec = MG_TASKS[task]
    layout = OBJ_LAYOUT[task]
    with h5py.File(spec.hdf5, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:n_demos]
        for name in names:
            demo = f[f"data/{name}"]
            sig = spec.episode_signals(demo)
            obj = np.asarray(demo["obs/object"])
            states = np.asarray(demo["states"]) if task in STATES_TASKS else None
            ap = aperture(demo["obs/robot0_gripper_qpos"])
            T = obj.shape[0]
            print(f"\n--- {task} {name} (T={T}) events "
                  + {"stack_three": lambda s: (f"A: c{ s['A']['c1']} h{s['A']['hold']} "
                                               f"r{s['A']['rise']} o{s['A']['o1']} "
                                               f"open{s['openA']} (a_on_b={s['a_on_b']})  "
                                               f"C: c{s['C']['c1']} h{s['C']['hold']} "
                                               f"r{s['C']['rise']} o{s['C']['o1']}"),
                     "three_piece_assembly": lambda s: (
                         f"p1: c{s['p1']['c1']} h{s['p1']['hold']} r{s['p1']['rise']} "
                         f"o{s['p1']['o1']} open{s['open1']} (p1_on_base={s['p1_on_base']})  "
                         f"p2: c{s['p2']['c1']} h{s['p2']['hold']} r{s['p2']['rise']} "
                         f"o{s['p2']['o1']}"),
                     "kitchen": lambda s: (
                         f"on={s['t_on']} c1={s['c1']} o1={s['o1']} c2={s['c2']} "
                         f"o2={s['o2']} c3={s['c3']} o3={s['o3']} serve={s['t_serve']} "
                         f"off={s['t_off']}"),
                     "coffee_prep": lambda s: (
                         f"c1={s['c1']} o1={s['o1']} lid={s['lid_start']}..{s['lid_open']} "
                         f"drw={s['drw_start']}..{s['drw_open']} c2={s['c2']} o2={s['o2']} "
                         f"close={s['close_start']}")}
                  .get(task, lambda s: f"c1={s['c1']} hold={s['hold']} rise={s['rise']} "
                                       f"o1={s['o1']} carry_z={s['carry_z']:.3f}")(sig))
            step_grid = sorted(set(list(range(0, T, max(T // 14, 1))) + [T - 1]))
            print(f" {'t':>4} {'stage':<7} {'grasp':<6} {'payload':<7} {'place':<6} "
                  f"{'ap':>6} " + " ".join(f"{n}_z" for n in layout))
            for t in step_grid:
                raw = object_positions(obj[t], layout)
                base = {"objects": raw, "eef_pos": np.zeros(3)}
                base.update(context_extras(task, obj[t],
                                           states[t] if states is not None else None))
                ctx = spec.frame_context(base, sig, t)
                zs = " ".join(f"{raw[n][2]:.3f}" for n in layout)
                flags = ("R" if ctx.get("released") else "") + \
                        ("P" if ctx.get("placed") else "")
                print(f" {t:>4} {ctx['stage_label']:<7} {str(ctx['grasp_obj']):<6} "
                      f"{str(ctx['payload']):<7} {str(ctx['place_target']):<6} "
                      f"{ap[t]:>6.4f} {zs}  {flags}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="print stage timelines for validation")
    p.add_argument("--task", default="all")
    p.add_argument("--n_demos", type=int, default=3)
    a = p.parse_args()
    for t in (MG_TASKS if a.task == "all" else [a.task]):
        _timeline(t, a.n_demos)
