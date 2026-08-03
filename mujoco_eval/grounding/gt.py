"""Build ground-truth task groundings from live MuJoCo simulator state."""
from __future__ import annotations


import numpy as np

from .. import paths
paths.ensure_repo_on_path()

from vlm_dp.grounding import Grounding, SceneObject, Stage


EXTENTS = {
    "cubeA": (0.02, 0.02, 0.02),
    "cubeB": (0.025, 0.025, 0.025),
    "cubeC": (0.02, 0.02, 0.02),
    "nut": (0.015875, 0.0793, 0.01),
    "peg1": (0.016, 0.016, 0.1),
    "cube": (0.021, 0.021, 0.021),
    "can": (0.025, 0.025, 0.0407),
    "bin2_q3": (0.0975, 0.1225, 0.04),
    "needle": (0.02, 0.08, 0.02),
    "tripod": (0.05, 0.05, 0.1),
    "coffee_pod": (0.0243, 0.0243, 0.0231),
    "coffee_machine": (0.0865, 0.1505, 0.1105),
    "coffee_pod_holder": (0.0295, 0.0295, 0.028),
    "coffee_machine_lid": (0.0295, 0.044, 0.0095),
    "base": (0.095, 0.095, 0.019),
    "piece_1": (0.017, 0.051, 0.051),
    "piece_2": (0.02, 0.06, 0.08),
    "pot": (0.007, 0.067, 0.044),
    "bread": (0.015, 0.025, 0.02),
    "stove": (0.095, 0.095, 0.02),
    "button": (0.04, 0.04, 0.05),
    "serving_region": (0.07, 0.10, 0.001),
}

MC_EXTENTS = {"mug": (0.009, 0.0463, 0.031), "drawer": (0.12, 0.12, 0.13),
              "handle": (0.009, 0.059, 0.009)}
HC_EXTENTS = {"hammer": (0.012, 0.051, 0.0124), "handle": (0.009, 0.05, 0.009),
              "drawer": (0.08, 0.105, 0.05),
              "CabinetObject": (0.118, 0.112, 0.065)}
CP_EXTENTS = {"mug": (0.005, 0.051, 0.0402), "cabinet": (0.118, 0.19, 0.065),
              **{k: EXTENTS[k] for k in ("coffee_pod", "coffee_machine",
                                         "coffee_pod_holder", "coffee_machine_lid")}}
NUT_GRASP_EXTENT = 0.015875
NUT_HANDLE_LOCAL = np.array([0.054, 0.0, 0.0])
PEG_POS = np.array([0.23, 0.10, 0.85])
PEG_TOP = np.array([0.23, 0.10, 0.95])
CAN_SEAT = np.array([0.1975, 0.4025, 0.8604])
CAN_DROP = np.array([0.1975, 0.4025, 1.04])
NEEDLE_HANDLE_LOCAL = np.array([0.0, 0.06, 0.0])
NEEDLE_BAR_LOCAL = np.array([0.0, -0.02, 0.0])
RING_LOCAL = np.array([0.0, 0.0, 0.088])
RING_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])
RING_RADIUS = 0.012
NEEDLE_GRASP_EXTENT = 0.016
INSERT_ROOT_OFFSET = 0.03


LID_JOINT = "coffee_machine_lid_main_joint0"
MC_DRAWER_JOINT = "DrawerObject_goal_slidey"
CABINET_JOINT = "CabinetObject_goal_slidey"
BTN_JOINT = "Button1_hinge"


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


def _slide_pull(rot, handle_now, handle_goal, hook, opening):
    """Return drawer pull metadata for the hook-pull cost."""
    return {"axis": rot @ ((-1.0 if opening else 1.0) * DRAWER_SLIDE_LOCAL),
            "point": handle_now + rot @ hook, "goal": handle_goal + rot @ hook,
            "bar": rot @ HANDLE_BAR_LOCAL, "span": HANDLE_BAR_HALF,
            "tol": HOOK_LINE_TOL, "stroke": DRAWER_STROKE}


def _hinge_press(body_pos, body_rot, press_local, hinge_local, hinge_axis, depth, tol, reach):
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


def _insert_cone(seat, axis, fit, mouth, height, capture):
    """Return insertion-corridor metadata."""
    return {"seat": seat, "axis": axis, "r_seat": fit, "r_mouth": mouth,
            "height": height, "capture": capture}

POD_GRASP_LOCAL = np.array([0.0, 0.0, 0.0156])
POD_DROP_OFFSET = np.array([0.0, 0.0, 0.0353])
LID_PRESS_LOCAL = np.array([-0.045, 0.031, 0.016])
POD_R_DIFF = 0.0052
LID_CLOSED = 15.0 * np.pi / 180.0

DRAWER_LINK_LOCAL = np.array([0.0, -0.01, 0.076])
DRAWER_HANDLE_LOCAL = np.array([0.0, -0.16, 0.04])
DRAWER_BOTTOM_TOP = -0.032
DRAWER_INTERIOR_X = 0.064
DRAWER_INTERIOR_Y = (-0.085, 0.057)
MUG_SEAT_LOCAL = np.array([0.0, 0.0, DRAWER_BOTTOM_TOP + 0.031])
_MUG_IN_Z = 0.09
_DRAWER_OPEN_LATCH = -0.12
_DRAWER_CLOSED_LATCH = -0.005
_MC_RETREAT = np.array([0.0, -0.05, 0.10])
_MC_RETREAT_EPS = 0.06
_MUG_CARRY_OVER_SEAT = 0.135

P1_GRASP_LOCAL = np.array([0.0, 0.0, 0.027])
P2_GRASP_LOCAL = np.array([0.0, 0.0, 0.058])
SEAT1_OFFSET = np.array([0.0, 0.0, 0.031])
SEAT2_OFFSET = np.array([0.0, 0.0, 0.099])
Z_CORRECT_OFFSET = 0.08
P1_GRASP_EXTENT = 0.017
P2_GRASP_EXTENT = 0.02
_ASSEMBLED_XY = 0.02
_ASSEMBLED_Z = 0.02
_SEAT1_DONE_Z = 0.01
TPA_LIFT = {"piece_1": 0.10, "piece_2": 0.15}

CAB_ROOT = np.array([0.0, 0.3, 0.905])
HOOK_OFFSET = np.array([0.0, -0.03, -0.03])
CAVITY_LOCAL = np.array([0.0, -0.014, 0.0])
HC_DROP_Z = 1.06
DRAWER_OPEN_Q = -0.10
DRAWER_CLOSED_Q = -0.01
HC_IN_XY = (0.064, 0.084)
HC_IN_Z = 0.999
_PULL_STANDOFF = 0.135
_HC_RETREAT = np.array([0.0, -0.06, 0.10])
_HC_RETREAT_EPS = 0.04

STOVE_POS = np.array([0.03, 0.095, 0.895])
BUTTON_POS = np.array([-0.14, 0.10, 0.895])
SERVING_POS = np.array([0.145, -0.15, 0.878])
POT_HANDLE_LOCAL = np.array([0.0, 0.06, 0.077])
POT_GRASP_EXTENT = 0.007
POT_RIM_Z = 0.075
POT_STOVE_SEAT_Z = 0.920
POT_TABLE_REST_Z = 0.9025
BREAD_DROP_DZ = 0.122
BTN_PRESS_ON = BUTTON_POS + np.array([0.0, 0.027, 0.097])
BTN_PRESS_OFF = BUTTON_POS + np.array([0.002, -0.029, 0.097])
SERVE_RELEASE = SERVING_POS + np.array([-0.145, 0.01, 0.025])
PUSH_EEF_OFFSET = np.array([-0.071, 0.01, 0.026])
SERVE_BOX = (0.05, 0.10, 0.05)

MUG_GRASP_LOCAL = np.array([-0.0234, 0.0011, 0.0139])
MUG_RELEASE_MACHINE_LOCAL = np.array([-0.0105, 0.1459, -0.0567])
MUG_SEAT_MACHINE_LOCAL = np.array([-0.013, 0.122, -0.061])
CP_POD_GRASP_LOCAL = np.array([0.0, 0.0, 0.0122])
POD_RELEASE_HOLDER_DZ = 0.034
LID_OPEN_PRESS_LOCAL = np.array([0.0087, 0.0194, 0.0121])
LID_CLOSE_PRESS_LOCAL = np.array([-0.0494, 0.0393, 0.0299])
DRAWER_HOOK_LOCAL = np.array([0.0094, -0.2569, 0.0954])
LID_OPEN_Q = 2.08
CP_DRAWER_OPEN_Q = -0.19
_MUG_ON_XY = 0.05
_MUG_UPRIGHT = 1e-3

_LIFT = {"stack": 0.10, "stack_three": 0.10, "square": 0.15,
         "lift": 0.06, "can": 0.18, "threading": 0.15,
         "coffee": 0.20, "mug_cleanup": 0.15, "hammer_cleanup": 0.20,
         "kitchen_pot": 0.08, "kitchen_bread": 0.16,
         "coffee_prep_mug": 0.10, "coffee_prep_pod": 0.21}
_LIFT_CONFIRM = 0.05
_PLACE_CLEARANCE = 0.10
_ON_XY, _ON_Z = 0.05, 0.02
_SQ_XY, _SQ_Z_TOP = 0.03, 0.93
_CAN_XY = (0.0975, 0.1225)
_CAN_Z_BAND = (0.8, 0.9)
_INSERT_STANDOFF = 0.05

TASKS = {
    "stack": {"grasp_objs": ["cubeA"], "place_obj": "cubeB",
              "movable": ["cubeA", "cubeB"]},
    "stack_three": {"grasp_objs": ["cubeA", "cubeC"], "place_obj": "cubeB",
                    "movable": ["cubeA", "cubeB", "cubeC"]},
    "square": {"grasp_objs": ["nut"], "place_obj": "peg1",
               "movable": ["nut"]},
    "lift": {"grasp_objs": ["cube"], "place_obj": None,
             "movable": ["cube"]},
    "can": {"grasp_objs": ["can"], "place_obj": "bin2_q3",
            "movable": ["can"]},
    "threading": {"grasp_objs": ["needle"], "place_obj": "tripod",
                  "movable": ["needle", "tripod"]},
    "coffee": {"grasp_objs": ["coffee_pod"], "place_obj": "coffee_pod_holder",
               "movable": ["coffee_pod"]},
    "mug_cleanup": {"grasp_objs": ["mug"], "place_obj": "drawer",
                    "movable": ["mug"]},
    "three_piece_assembly": {"grasp_objs": ["piece_1", "piece_2"], "place_obj": "base",
                             "movable": ["base", "piece_1", "piece_2"]},
    "hammer_cleanup": {"grasp_objs": ["hammer"], "place_obj": "drawer",
                       "movable": ["hammer", "drawer"]},
    "kitchen": {"grasp_objs": ["pot", "bread"], "place_obj": "serving_region",
                "movable": ["pot", "bread"]},
    "coffee_prep": {"grasp_objs": ["mug", "coffee_pod"], "place_obj": "coffee_machine",
                    "movable": ["mug", "coffee_pod", "coffee_machine", "cabinet",
                                "coffee_pod_holder", "coffee_machine_lid"]},
}


class MGGroundingSource:
    """Build task stages and targets from exact MuJoCo poses."""

    def __init__(self, task):
        if task not in TASKS:
            raise ValueError(f"unknown mg task {task!r} (have {sorted(TASKS)})")
        self.task = task
        spec = TASKS[task]
        self.movable = list(spec["movable"])
        self.roles = {"grasp_obj": spec["grasp_objs"][0], "grasp_objs": spec["grasp_objs"],
                      "place_obj": spec["place_obj"]}

    def ground(self, env, world) -> Grounding:
        if self.task == "square":
            return self._ground_square(world)
        if self.task == "lift":
            return self._ground_lift(world)
        if self.task == "can":
            return self._ground_can(world)
        if self.task == "threading":
            return self._ground_threading(world)
        if self.task == "coffee":
            return self._ground_coffee(world)
        if self.task == "mug_cleanup":
            return self._ground_mug_cleanup(env, world)
        if self.task == "three_piece_assembly":
            return self._ground_tpa(world)
        if self.task == "hammer_cleanup":
            return self._ground_hammer_cleanup(env, world)
        if self.task == "kitchen":
            return self._ground_kitchen(world)
        if self.task == "coffee_prep":
            return self._ground_coffee_prep(world)
        return self._ground_stack(world)


    def _ground_stack(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        objects = [SceneObject(name=n, pos=(lambda n=n: pos(n)), extents=EXTENTS[n])
                   for n in self.movable]
        ladders = [("cubeA", "cubeB")]
        if self.task == "stack_three":
            ladders.append(("cubeC", "cubeA"))
        stages = []
        for name, dest in ladders:
            stages += self._ladder(pos, name, dest)
        kp_names = [ladders[0][0], ladders[0][1]] + [n for n, _ in ladders[1:]]
        keypoints = lambda: np.stack([pos(n) for n in kp_names]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset(self.movable), keypoints=keypoints)

    def _ladder(self, pos, name, dest):
        """Build a grasp, lift, and place ladder for one object."""
        z0 = float(pos(name)[2])
        lift = _LIFT[self.task]
        seat = lambda: pos(dest) + np.array([0.0, 0.0, EXTENTS[dest][2]])
        hover = lambda: seat() + np.array([0.0, 0.0, _PLACE_CLEARANCE])
        lift_t = lambda: np.array([*pos(name)[:2], z0 + lift])

        def stacked():
            d = pos(name) - pos(dest)
            return bool(np.linalg.norm(d[:2]) < _ON_XY and d[2] > _ON_Z)

        return [
            Stage(name=f"grasp {name}", gripper="close", grasp_obj=name,
                  target=(lambda n=name: pos(n))),
            Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                  target=lift_t,
                  done=(lambda n=name, z=z0: float(pos(n)[2]) > z + _LIFT_CONFIRM)),
            Stage(name=f"place {name} on {dest}", gripper="place", payload=name,
                  place_target=dest, target=hover, place_point=seat,
                  carry_z=(lambda z=z0 + lift: z), done=stacked),
        ]


    def _ground_square(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        handle = lambda: pos("nut") + world.object_pose("nut")[1] @ NUT_HANDLE_LOCAL

        ax = (world.object_pose("nut")[1] @ np.array([0.0, 1.0, 0.0]))[:2]
        n = float(np.linalg.norm(ax))
        axis = tuple(np.array([*(ax / n), 0.0])) if n > 1e-6 else None
        objects = [
            SceneObject(name="nut", pos=(lambda: pos("nut")), extents=EXTENTS["nut"],
                        axis=axis, grasp_extent=NUT_GRASP_EXTENT),
            SceneObject(name="peg1", pos=(lambda: PEG_POS.copy()), extents=EXTENTS["peg1"]),
        ]
        z0 = float(pos("nut")[2])
        lift_t = lambda: np.array([*pos("nut")[:2], z0 + _LIFT["square"]])

        def threaded():
            d = pos("nut") - PEG_TOP
            return bool(np.linalg.norm(d[:2]) < _SQ_XY and pos("nut")[2] < _SQ_Z_TOP)

        stages = [
            Stage(name="grasp nut handle", gripper="close", grasp_obj="nut", target=handle),
            Stage(name="lift nut", gripper="hold", grasp_obj="nut", payload="nut",
                  target=lift_t, done=(lambda: float(pos("nut")[2]) > z0 + _LIFT_CONFIRM)),
            Stage(name="place nut on peg1", gripper="place", payload="nut", place_target="peg1",
                  target=(lambda: PEG_TOP + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=(lambda: PEG_TOP.copy()), carry_z=(lambda: z0 + _LIFT["square"]),
                  done=threaded, place_mode="container",
                  insert=(lambda: _insert_cone(PEG_TOP.copy(), np.array([0.0, 0.0, 1.0]), PEG_FIT,
                                               PEG_MOUTH, PEG_CONE_H, PEG_CAPTURE))),
        ]
        keypoints = lambda: np.stack([handle(), PEG_TOP]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"nut"}), keypoints=keypoints)


    def _ground_lift(self, world):


        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        objects = [SceneObject(name="cube", pos=(lambda: pos("cube")), extents=EXTENTS["cube"])]
        z0 = float(pos("cube")[2])
        lift_t = lambda: np.array([*pos("cube")[:2], z0 + _LIFT["lift"]])
        stages = [
            Stage(name="grasp cube", gripper="close", grasp_obj="cube",
                  target=(lambda: pos("cube"))),
            Stage(name="lift cube", gripper="hold", grasp_obj="cube", payload="cube",
                  target=lift_t,
                  done=(lambda: float(pos("cube")[2]) > z0 + _LIFT_CONFIRM)),
        ]
        keypoints = lambda: np.stack([pos("cube")]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"cube"}), keypoints=keypoints)


    def _ground_can(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        objects = [
            SceneObject(name="can", pos=(lambda: pos("can")), extents=EXTENTS["can"]),
            SceneObject(name="bin2_q3", pos=(lambda: CAN_SEAT.copy()),
                        extents=EXTENTS["bin2_q3"]),
        ]
        z0 = float(pos("can")[2])
        lift_t = lambda: np.array([*pos("can")[:2], z0 + _LIFT["can"]])

        def in_bin():
            d = pos("can") - CAN_SEAT
            return bool(abs(d[0]) < _CAN_XY[0] and abs(d[1]) < _CAN_XY[1]
                        and _CAN_Z_BAND[0] < pos("can")[2] < _CAN_Z_BAND[1])

        stages = [
            Stage(name="grasp can", gripper="close", grasp_obj="can",
                  target=(lambda: pos("can"))),
            Stage(name="lift can", gripper="hold", grasp_obj="can", payload="can",
                  target=lift_t, done=(lambda: float(pos("can")[2]) > z0 + _LIFT_CONFIRM)),

            Stage(name="place can in bin2", gripper="place", payload="can",
                  place_target="bin2_q3",
                  target=(lambda: CAN_DROP + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=(lambda: CAN_DROP.copy()), carry_z=(lambda: z0 + _LIFT["can"]),
                  done=in_bin, place_mode="container"),
        ]
        keypoints = lambda: np.stack([pos("can"), CAN_DROP]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"can"}), keypoints=keypoints)


    def _ground_threading(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        rot = lambda n: world.object_pose(n)[1]
        handle = lambda: pos("needle") + rot("needle") @ NEEDLE_HANDLE_LOCAL
        bar = lambda: pos("needle") + rot("needle") @ NEEDLE_BAR_LOCAL

        def ring_frame():
            centre = pos("tripod") + rot("tripod") @ RING_LOCAL
            axis = rot("tripod") @ RING_AXIS_LOCAL
            if float(axis @ (pos("needle") - centre)) < 0:
                axis = -axis
            return centre, axis


        ax = (rot("needle") @ np.array([1.0, 0.0, 0.0]))[:2]
        n = float(np.linalg.norm(ax))
        axis = tuple(np.array([*(ax / n), 0.0])) if n > 1e-6 else None
        objects = [
            SceneObject(name="needle", pos=(lambda: pos("needle")), extents=EXTENTS["needle"],
                        axis=axis, grasp_extent=NEEDLE_GRASP_EXTENT),
            SceneObject(name="tripod", pos=(lambda: pos("tripod")), extents=EXTENTS["tripod"]),
        ]
        z0 = float(pos("needle")[2])
        lift_t = lambda: np.array([*pos("needle")[:2], z0 + _LIFT["threading"]])
        seat = lambda: (lambda cf: cf[0] + INSERT_ROOT_OFFSET * cf[1])(ring_frame())
        standoff = lambda: (lambda cf: cf[0] + _INSERT_STANDOFF * cf[1])(ring_frame())

        cone = lambda: (lambda cf: _insert_cone(cf[0] + INSERT_ROOT_OFFSET * cf[1], cf[1], RING_FIT,
                                                RING_MOUTH, RING_CONE_H, RING_CAPTURE))(ring_frame())

        def threaded():
            centre, _ = ring_frame()
            return bool(np.linalg.norm(bar() - centre) < RING_RADIUS)

        stages = [
            Stage(name="grasp needle handle", gripper="close", grasp_obj="needle",
                  target=handle),
            Stage(name="lift needle", gripper="hold", grasp_obj="needle", payload="needle",
                  target=lift_t, done=(lambda: float(pos("needle")[2]) > z0 + _LIFT_CONFIRM)),

            Stage(name="thread needle through ring", gripper="hold", payload="needle",
                  place_target="tripod", target=standoff, place_point=seat,
                  carry_z=(lambda: z0 + _LIFT["threading"]), done=threaded,
                  place_mode="container", insert=cone),
        ]
        keypoints = lambda: np.stack([handle(), ring_frame()[0]]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"needle"}), keypoints=keypoints)


    def _ground_coffee(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        rot = lambda n: world.object_pose(n)[1]
        grasp = lambda: pos("coffee_pod") + rot("coffee_pod") @ POD_GRASP_LOCAL
        drop = lambda: pos("coffee_pod_holder") + POD_DROP_OFFSET
        press = lambda: (pos("coffee_machine_lid")
                         + rot("coffee_machine_lid") @ LID_PRESS_LOCAL)

        def pod_inserted():
            pod, holder = pos("coffee_pod"), pos("coffee_pod_holder")
            lid = pos("coffee_machine_lid")
            if np.linalg.norm(pod[:2] - holder[:2]) > POD_R_DIFF:
                return False
            z_low = holder[2] - EXTENTS["coffee_pod_holder"][2]
            z_high = lid[2] - EXTENTS["coffee_machine_lid"][2]
            half = EXTENTS["coffee_pod"][2]
            return bool(pod[2] - half > z_low and pod[2] + half < z_high)

        def lid_closed():
            return bool(world.joint_angle("coffee_machine", LID_JOINT) < LID_CLOSED)


        objects = [
            SceneObject(name="coffee_pod", pos=(lambda: pos("coffee_pod")),
                        extents=EXTENTS["coffee_pod"],
                        grasp_extent=EXTENTS["coffee_pod"][0]),
            SceneObject(name="coffee_machine", pos=(lambda: pos("coffee_machine")),
                        extents=EXTENTS["coffee_machine"]),
            SceneObject(name="coffee_pod_holder", pos=(lambda: pos("coffee_pod_holder")),
                        extents=EXTENTS["coffee_pod_holder"]),
            SceneObject(name="coffee_machine_lid", pos=(lambda: pos("coffee_machine_lid")),
                        extents=EXTENTS["coffee_machine_lid"]),
        ]
        z0 = float(pos("coffee_pod")[2])
        lift_t = lambda: np.array([*pos("coffee_pod")[:2], z0 + _LIFT["coffee"]])

        stages = [
            Stage(name="grasp coffee_pod", gripper="close", grasp_obj="coffee_pod",
                  target=grasp),
            Stage(name="lift coffee_pod", gripper="hold", grasp_obj="coffee_pod",
                  payload="coffee_pod", target=lift_t,
                  done=(lambda: float(pos("coffee_pod")[2]) > z0 + _LIFT_CONFIRM)),

            Stage(name="place coffee_pod in coffee_pod_holder", gripper="place",
                  payload="coffee_pod", place_target="coffee_pod_holder",
                  target=(lambda: drop() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=drop, carry_z=(lambda: z0 + _LIFT["coffee"]),
                  done=pod_inserted, place_mode="container"),


            Stage(name="close coffee_machine_lid", gripper="open", contact="press",
                  target=press, done=lid_closed, advance_on_done=True,
                  press=(lambda: _hinge_press(pos("coffee_machine_lid"),
                                              rot("coffee_machine_lid"), LID_PRESS_LOCAL,
                                              LID_HINGE_LOCAL, LID_HINGE_AXIS, LID_PRESS_DEPTH,
                                              LID_PRESS_TOL, LID_PRESS_REACH))),
        ]
        keypoints = lambda: np.stack([grasp(), drop(), press()]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"coffee_pod", "coffee_machine_lid"}),
                         keypoints=keypoints)


    def _ground_mug_cleanup(self, env, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        root0 = pos("drawer")
        rot = world.object_pose("drawer")[1]
        qpos = lambda: float(world.joint_angle("drawer", MC_DRAWER_JOINT))
        link = lambda: pos("drawer_link")
        handle = lambda: link() + rot @ DRAWER_HANDLE_LOCAL
        handle_at = lambda q: root0 + rot @ (DRAWER_LINK_LOCAL + np.array([0.0, q, 0.0])
                                             + DRAWER_HANDLE_LOCAL)
        seat = lambda: link() + rot @ MUG_SEAT_LOCAL
        retreat_point = lambda: handle() + rot @ _MC_RETREAT

        def drawer_open():
            return qpos() <= _DRAWER_OPEN_LATCH

        def drawer_closed():
            return qpos() >= _DRAWER_CLOSED_LATCH

        def retreated():
            return bool(np.linalg.norm(np.asarray(env.tcp()) - retreat_point())
                        < _MC_RETREAT_EPS)

        def mug_in_drawer():
            local = rot.T @ (pos("mug") - link())
            return bool(abs(local[0]) < DRAWER_INTERIOR_X
                        and DRAWER_INTERIOR_Y[0] < local[1] < DRAWER_INTERIOR_Y[1]
                        and local[2] < _MUG_IN_Z)

        objects = [
            SceneObject(name="mug", pos=(lambda: pos("mug")), extents=MC_EXTENTS["mug"],
                        grasp_extent=MC_EXTENTS["mug"][0]),
            SceneObject(name="handle", pos=handle, extents=MC_EXTENTS["handle"],
                        grasp_extent=MC_EXTENTS["handle"][0]),
            SceneObject(name="drawer", pos=(lambda: root0.copy()),
                        extents=MC_EXTENTS["drawer"]),
        ]
        z0 = float(pos("mug")[2])
        lift_t = lambda: np.array([*pos("mug")[:2], z0 + _LIFT["mug_cleanup"]])

        stages = [


            Stage(name="open drawer", gripper="open", grasp_obj="handle", target=handle,
                  contact="press", done=drawer_open, advance_on_done=True,
                  pull=(lambda: _slide_pull(rot, handle(), handle_at(-DRAWER_STROKE),
                                            MC_HOOK_OPEN, True))),
            Stage(name="release handle", gripper="open", target=retreat_point,
                  done=retreated),
            Stage(name="grasp mug", gripper="close", grasp_obj="mug",
                  target=(lambda: pos("mug"))),
            Stage(name="lift mug", gripper="hold", grasp_obj="mug", payload="mug",
                  target=lift_t, done=(lambda: float(pos("mug")[2]) > z0 + _LIFT_CONFIRM)),
            Stage(name="place mug in drawer", gripper="place", payload="mug",
                  place_target="drawer",
                  target=(lambda: seat() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=seat,
                  carry_z=(lambda: float(seat()[2]) + _MUG_CARRY_OVER_SEAT),
                  done=mug_in_drawer, place_mode="container"),
            Stage(name="close drawer", gripper="open", grasp_obj="handle",
                  target=(lambda: handle_at(0.0)), contact="press",
                  done=drawer_closed, advance_on_done=True,
                  pull=(lambda: _slide_pull(rot, handle(), handle_at(0.0),
                                            MC_HOOK_CLOSE, False))),
        ]
        keypoints = lambda: np.stack([handle(), pos("mug"), seat()]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"mug", "handle", "drawer"}),
                         keypoints=keypoints)


    def _ground_tpa(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        rot = lambda n: world.object_pose(n)[1]
        grasp1 = lambda: pos("piece_1") + rot("piece_1") @ P1_GRASP_LOCAL
        grasp2 = lambda: pos("piece_2") + rot("piece_2") @ P2_GRASP_LOCAL
        seat1 = lambda: pos("base") + SEAT1_OFFSET

        seat2 = lambda: np.array([*pos("piece_1")[:2], pos("base")[2] + SEAT2_OFFSET[2]])


        ax = (rot("piece_1") @ np.array([1.0, 0.0, 0.0]))[:2]
        n = float(np.linalg.norm(ax))
        axis = tuple(np.array([*(ax / n), 0.0])) if n > 1e-6 else None
        objects = [
            SceneObject(name="base", pos=(lambda: pos("base")), extents=EXTENTS["base"]),
            SceneObject(name="piece_1", pos=(lambda: pos("piece_1")),
                        extents=EXTENTS["piece_1"], axis=axis,
                        grasp_extent=P1_GRASP_EXTENT),
            SceneObject(name="piece_2", pos=(lambda: pos("piece_2")),
                        extents=EXTENTS["piece_2"], grasp_extent=P2_GRASP_EXTENT),
        ]
        z1 = float(pos("piece_1")[2])
        z2 = float(pos("piece_2")[2])
        lift1_t = lambda: np.array([*pos("piece_1")[:2], z1 + TPA_LIFT["piece_1"]])
        lift2_t = lambda: np.array([*pos("piece_2")[:2], z2 + TPA_LIFT["piece_2"]])

        def assembled_1():


            d = pos("piece_1") - pos("base")
            return bool(np.linalg.norm(d[:2]) < _ASSEMBLED_XY
                        and pos("piece_1")[2] < pos("base")[2] + SEAT1_OFFSET[2]
                        + _SEAT1_DONE_Z)

        def assembled_2():

            d_xy = np.linalg.norm(pos("piece_2")[:2] - pos("piece_1")[:2])
            d_z = abs(pos("piece_2")[2] - (pos("base")[2] + Z_CORRECT_OFFSET))
            return bool(d_xy < _ASSEMBLED_XY and d_z < _ASSEMBLED_Z)

        stages = [
            Stage(name="grasp piece_1 wall", gripper="close", grasp_obj="piece_1",
                  target=grasp1),
            Stage(name="lift piece_1", gripper="hold", grasp_obj="piece_1",
                  payload="piece_1", target=lift1_t,
                  done=(lambda: float(pos("piece_1")[2]) > z1 + _LIFT_CONFIRM)),
            Stage(name="insert piece_1 into base hole", gripper="place", payload="piece_1",
                  place_target="base",
                  target=(lambda: seat1() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=seat1, carry_z=(lambda: z1 + TPA_LIFT["piece_1"]),
                  done=assembled_1, place_mode="container"),
            Stage(name="grasp piece_2 knob", gripper="close", grasp_obj="piece_2",
                  target=grasp2),
            Stage(name="lift piece_2", gripper="hold", grasp_obj="piece_2",
                  payload="piece_2", target=lift2_t,
                  done=(lambda: float(pos("piece_2")[2]) > z2 + _LIFT_CONFIRM)),
            Stage(name="cap piece_2 over the wall", gripper="place", payload="piece_2",
                  place_target="piece_1",
                  target=(lambda: seat2() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=seat2, carry_z=(lambda: z2 + TPA_LIFT["piece_2"]),
                  done=assembled_2, place_mode="container"),
        ]
        keypoints = lambda: np.stack([grasp1(), seat1(), grasp2()]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"piece_1", "piece_2"}), keypoints=keypoints)


    def _ground_hammer_cleanup(self, env, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        drawer_q = lambda: float(world.joint_angle("drawer", CABINET_JOINT))
        handle = lambda: pos("drawer") + DRAWER_HANDLE_LOCAL
        cavity = lambda: pos("drawer") + CAVITY_LOCAL
        drop = lambda: np.array([*cavity()[:2], HC_DROP_Z])
        pull_to = lambda: (CAB_ROOT + DRAWER_LINK_LOCAL + DRAWER_HANDLE_LOCAL + HOOK_OFFSET
                           + np.array([0.0, -_PULL_STANDOFF, 0.0]))
        push_to = lambda: CAB_ROOT + DRAWER_LINK_LOCAL + DRAWER_HANDLE_LOCAL

        handle_at = lambda q: (CAB_ROOT + DRAWER_LINK_LOCAL + np.array([0.0, q, 0.0])
                               + DRAWER_HANDLE_LOCAL)

        def retreated():
            return bool(np.linalg.norm(np.asarray(env.tcp()) - (handle() + _HC_RETREAT))
                        < _HC_RETREAT_EPS)

        def hammer_in_drawer():
            local = pos("hammer") - pos("drawer")
            return bool(abs(local[0]) < HC_IN_XY[0] and abs(local[1]) < HC_IN_XY[1]
                        and pos("hammer")[2] < HC_IN_Z)

        def success_twin():
            z = float(pos("hammer")[2])
            return bool(0.94 < z < 1.0 and pos("hammer")[1] > 0.22
                        and drawer_q() > DRAWER_CLOSED_Q)


        long_ax = world.object_pose("hammer")[1] @ np.array([0.0, 0.0, 1.0])
        a = np.array([-long_ax[1], long_ax[0], 0.0])
        n = float(np.linalg.norm(a))
        axis = tuple(a / n) if n > 1e-6 else None
        objects = [
            SceneObject(name="hammer", pos=(lambda: pos("hammer")),
                        extents=HC_EXTENTS["hammer"], axis=axis,
                        grasp_extent=HC_EXTENTS["hammer"][0]),
            SceneObject(name="handle", pos=handle, extents=HC_EXTENTS["handle"],
                        grasp_extent=HC_EXTENTS["handle"][0]),
            SceneObject(name="drawer", pos=(lambda: pos("drawer")),
                        extents=HC_EXTENTS["drawer"]),
            SceneObject(name="CabinetObject", pos=(lambda: CAB_ROOT.copy()),
                        extents=HC_EXTENTS["CabinetObject"]),
        ]
        z0 = float(pos("hammer")[2])
        lift_t = lambda: np.array([*pos("hammer")[:2], z0 + _LIFT["hammer_cleanup"]])

        stages = [


            Stage(name="open drawer", gripper="open", grasp_obj="handle", target=pull_to,
                  contact="press", done=(lambda: drawer_q() < DRAWER_OPEN_Q),
                  advance_on_done=True,
                  pull=(lambda: _slide_pull(np.eye(3), handle(), handle_at(-DRAWER_STROKE),
                                            HC_HOOK_OPEN, True))),
            Stage(name="release handle", gripper="open",
                  target=(lambda: handle() + _HC_RETREAT), done=retreated),
            Stage(name="grasp hammer", gripper="close", grasp_obj="hammer",
                  target=(lambda: pos("hammer"))),
            Stage(name="lift hammer", gripper="hold", grasp_obj="hammer", payload="hammer",
                  target=lift_t,
                  done=(lambda: float(pos("hammer")[2]) > z0 + _LIFT_CONFIRM)),
            Stage(name="place hammer in drawer", gripper="place", payload="hammer",
                  place_target="drawer", target=drop, place_point=drop,
                  carry_z=(lambda: z0 + _LIFT["hammer_cleanup"]), done=hammer_in_drawer,
                  place_mode="container"),
            Stage(name="close drawer", gripper="open", contact="press",
                  place_target="drawer", target=push_to, done=success_twin,
                  advance_on_done=True, place_mode="container",
                  pull=(lambda: _slide_pull(np.eye(3), handle(), handle_at(0.0),
                                            HC_HOOK_CLOSE, False))),
        ]
        keypoints = lambda: np.stack([handle(), pos("hammer"), cavity()]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"hammer", "drawer"}), keypoints=keypoints)


    def _ground_kitchen(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        rot = lambda n: world.object_pose(n)[1]
        handle = lambda: pos("pot") + rot("pot") @ POT_HANDLE_LOCAL

        btn_q = lambda: float(world.joint_angle("button", BTN_JOINT))


        ax = (rot("pot") @ np.array([0.0, 1.0, 0.0]))[:2]
        n = float(np.linalg.norm(ax))
        axis = tuple(np.array([*(ax / n), 0.0])) if n > 1e-6 else None
        objects = [
            SceneObject(name="pot", pos=(lambda: pos("pot")), extents=EXTENTS["pot"],
                        axis=axis, grasp_extent=POT_GRASP_EXTENT),
            SceneObject(name="bread", pos=(lambda: pos("bread")), extents=EXTENTS["bread"]),
            SceneObject(name="stove", pos=(lambda: STOVE_POS.copy()),
                        extents=EXTENTS["stove"]),
            SceneObject(name="button", pos=(lambda: BUTTON_POS.copy()),
                        extents=EXTENTS["button"]),
            SceneObject(name="serving_region", pos=(lambda: SERVING_POS.copy()),
                        extents=EXTENTS["serving_region"]),
        ]
        stove_seat = np.array([STOVE_POS[0], STOVE_POS[1], POT_STOVE_SEAT_Z])
        drop = lambda: pos("pot") + np.array([0.0, 0.0, BREAD_DROP_DZ])
        push = lambda: pos("pot") + PUSH_EEF_OFFSET

        def pot_on_stove():
            d = pos("pot") - STOVE_POS
            return bool(np.linalg.norm(d[:2]) < 0.05
                        and abs(pos("pot")[2] - POT_STOVE_SEAT_Z) < 0.01)

        def bread_in_pot():
            d = pos("bread") - pos("pot")
            return bool(np.all(np.abs(d[:2]) < 0.05) and 0.0 < d[2] < POT_RIM_Z + 0.015)

        def pot_served():
            d = SERVING_POS - pos("pot")
            return bool(np.all(np.abs(d) < np.array(SERVE_BOX)))

        stages = [


            Stage(name="press button on", gripper="open",
                  target=(lambda: BTN_PRESS_ON.copy()), contact="press",
                  done=(lambda: btn_q() >= 0.0), advance_on_done=True),
            Stage(name="grasp pot handle", gripper="close", grasp_obj="pot", target=handle),
            Stage(name="lift pot", gripper="hold", grasp_obj="pot", payload="pot",
                  target=(lambda: np.array([*pos("pot")[:2],
                                            POT_TABLE_REST_Z + _LIFT["kitchen_pot"]])),
                  done=(lambda: float(pos("pot")[2]) > POT_TABLE_REST_Z + _LIFT_CONFIRM)),
            Stage(name="place pot on stove", gripper="place", payload="pot",
                  place_target="stove", target=(lambda: stove_seat.copy()),
                  place_point=(lambda: stove_seat.copy()),
                  carry_z=(lambda: POT_TABLE_REST_Z + _LIFT["kitchen_pot"]),
                  done=pot_on_stove),
            Stage(name="grasp bread", gripper="close", grasp_obj="bread",
                  target=(lambda: pos("bread"))),
            Stage(name="lift bread", gripper="hold", grasp_obj="bread", payload="bread",
                  target=(lambda: np.array([*pos("bread")[:2],
                                            0.92 + _LIFT["kitchen_bread"]])),
                  done=(lambda: float(pos("bread")[2]) > 0.92 + _LIFT_CONFIRM)),


            Stage(name="drop bread in pot", gripper="place", payload="bread",
                  place_target="pot", target=drop, place_point=drop,
                  carry_z=(lambda: 0.92 + _LIFT["kitchen_bread"]), done=bread_in_pot,
                  place_mode="container"),
            Stage(name="regrasp pot handle", gripper="close", grasp_obj="pot",
                  target=handle),
            Stage(name="lift pot off stove", gripper="hold", grasp_obj="pot", payload="pot",
                  target=(lambda: np.array([*pos("pot")[:2],
                                            POT_STOVE_SEAT_Z + _LIFT["kitchen_pot"]])),
                  done=(lambda: float(pos("pot")[2])
                        > POT_STOVE_SEAT_Z + _LIFT_CONFIRM)),

            Stage(name="place pot short of serving", gripper="place", payload="pot",
                  place_target="serving_region", target=(lambda: SERVE_RELEASE.copy()),
                  place_point=(lambda: SERVE_RELEASE.copy()),
                  carry_z=(lambda: POT_STOVE_SEAT_Z + _LIFT["kitchen_pot"]),
                  done=(lambda: bool(np.linalg.norm(
                      pos("pot")[:2] - SERVE_RELEASE[:2]) < 0.03))),
            Stage(name="push pot into serving region", gripper="open", target=push,
                  contact="press", place_target="serving_region",
                  done=pot_served, advance_on_done=True),
            Stage(name="press button off", gripper="open",
                  target=(lambda: BTN_PRESS_OFF.copy()), contact="press",
                  done=(lambda: btn_q() < 0.0), advance_on_done=True),
        ]
        keypoints = lambda: np.stack([
            handle(), pos("bread"), stove_seat, SERVING_POS, BTN_PRESS_ON,
        ]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"pot", "bread"}), keypoints=keypoints)


    def _ground_coffee_prep(self, world):
        pos = lambda n: np.asarray(world.object_pose(n)[0], dtype=np.float64)
        rot = lambda n: world.object_pose(n)[1]
        hinge = lambda: float(world.joint_angle("coffee_machine", LID_JOINT))
        slide = lambda: float(world.joint_angle("cabinet", CABINET_JOINT))
        mug_grasp = lambda: pos("mug") + rot("mug") @ MUG_GRASP_LOCAL
        pod_grasp = lambda: pos("coffee_pod") + rot("coffee_pod") @ CP_POD_GRASP_LOCAL
        seat = lambda: (pos("coffee_machine")
                        + rot("coffee_machine") @ MUG_SEAT_MACHINE_LOCAL)
        release = lambda: (pos("coffee_machine")
                           + rot("coffee_machine") @ MUG_RELEASE_MACHINE_LOCAL)
        lid_open_press = lambda: (pos("coffee_machine_lid")
                                  + rot("coffee_machine_lid") @ LID_OPEN_PRESS_LOCAL)
        lid_close_press = lambda: (pos("coffee_machine_lid")
                                   + rot("coffee_machine_lid") @ LID_CLOSE_PRESS_LOCAL)
        hook = lambda: pos("cabinet") + rot("cabinet") @ (
            DRAWER_HOOK_LOCAL + np.array([0.0, slide(), 0.0]))
        pod_entry = lambda: (pos("coffee_pod_holder")
                             + np.array([0.0, 0.0, POD_RELEASE_HOLDER_DZ]))

        objects = [
            SceneObject(name="mug", pos=(lambda: pos("mug")), extents=CP_EXTENTS["mug"],
                        grasp_extent=CP_EXTENTS["mug"][0]),
            SceneObject(name="coffee_pod", pos=(lambda: pos("coffee_pod")),
                        extents=CP_EXTENTS["coffee_pod"],
                        grasp_extent=CP_EXTENTS["coffee_pod"][0]),
            SceneObject(name="coffee_machine", pos=(lambda: pos("coffee_machine")),
                        extents=CP_EXTENTS["coffee_machine"]),
            SceneObject(name="coffee_pod_holder", pos=(lambda: pos("coffee_pod_holder")),
                        extents=CP_EXTENTS["coffee_pod_holder"]),
            SceneObject(name="coffee_machine_lid", pos=(lambda: pos("coffee_machine_lid")),
                        extents=CP_EXTENTS["coffee_machine_lid"]),
            SceneObject(name="cabinet", pos=(lambda: pos("cabinet")),
                        extents=CP_EXTENTS["cabinet"]),
        ]
        z_mug = float(pos("mug")[2])

        def mug_placed():
            d = pos("mug") - seat()
            upright = 1.0 - rot("mug")[2, 2] < _MUG_UPRIGHT
            return bool(np.linalg.norm(d[:2]) < _MUG_ON_XY and upright and d[2] < 0.02)

        def pod_seated():
            d = pos("coffee_pod") - pos("coffee_pod_holder")
            return bool(np.linalg.norm(d[:2]) < POD_R_DIFF and 0.0 < d[2] < 0.02)

        stages = [
            Stage(name="grasp mug rim", gripper="close", grasp_obj="mug", target=mug_grasp),
            Stage(name="lift mug", gripper="hold", grasp_obj="mug", payload="mug",
                  target=(lambda: np.array([*pos("mug")[:2],
                                            z_mug + _LIFT["coffee_prep_mug"]])),
                  done=(lambda: float(pos("mug")[2]) > z_mug + _LIFT_CONFIRM)),
            Stage(name="place mug on machine base", gripper="place", payload="mug",
                  place_target="coffee_machine",
                  target=(lambda: seat() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=release,
                  carry_z=(lambda: z_mug + _LIFT["coffee_prep_mug"]), done=mug_placed),


            Stage(name="open machine lid", gripper="open", target=lid_open_press,
                  contact="press", advance_on_done=True,
                  done=(lambda: hinge() > LID_OPEN_Q)),
            Stage(name="open drawer", gripper="open", target=hook, contact="press",
                  advance_on_done=True, done=(lambda: slide() < CP_DRAWER_OPEN_Q)),
            Stage(name="grasp coffee pod", gripper="close", grasp_obj="coffee_pod",
                  target=pod_grasp),
            Stage(name="lift pod clear of machine", gripper="hold", grasp_obj="coffee_pod",
                  payload="coffee_pod",
                  target=(lambda: np.array([*pos("coffee_pod")[:2],
                                            float(pos("coffee_pod_holder")[2])
                                            + _LIFT["coffee_prep_pod"]])),
                  done=(lambda: float(pos("coffee_pod")[2])
                        > float(pos("coffee_pod_holder")[2]) + 0.05)),
            Stage(name="insert pod into holder", gripper="place", payload="coffee_pod",
                  place_target="coffee_pod_holder",
                  target=(lambda: pod_entry() + np.array([0.0, 0.0, _PLACE_CLEARANCE])),
                  place_point=pod_entry,
                  carry_z=(lambda: float(pos("coffee_pod_holder")[2])
                           + _LIFT["coffee_prep_pod"]),
                  done=pod_seated, place_mode="container"),
            Stage(name="close machine lid", gripper="open", target=lid_close_press,
                  contact="press", advance_on_done=True,
                  done=(lambda: hinge() < LID_CLOSED)),
        ]
        keypoints = lambda: np.stack([
            mug_grasp(), seat(), lid_open_press(), hook(), pod_grasp(), pod_entry(),
        ]).astype(np.float32)
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({"mug", "coffee_pod", "coffee_machine_lid",
                                                "cabinet"}),
                         keypoints=keypoints)
