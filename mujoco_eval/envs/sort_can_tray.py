"""Continuous-goal can placement into an undivided tray, on the stock robosuite bins arena.

The sibling of :mod:`sort_can`. One can starts in bin1 and has to be placed at a COMMANDED
CONTINUOUS POSITION inside bin2, which here is stripped of its cross divider and used as a
single walled tray. The goal is a 3-vector rather than one of two discrete bins, so the
dataset can ask whether a policy interpolates over goal space rather than merely selecting
between two modes.

Everything except the divider is left alone: same arena, same camera, same Panda mount, same
can, same placement sampler, so ``fk_fit_can.json`` and the sort_can conversion path apply
unchanged. No coloured pads are added -- this experiment has no colours, and bin2's own
dark-wood floor is the neutral surface.

Success is four conjuncts: released, settled, the can centre within :data:`GOAL_TOL` of the
commanded goal, and the can inside the tray.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.arenas import BinsArena
from robosuite.models.objects import CanObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from .sort_can import (BIN_FLOOR_TOP, BIN_RIM, CAN_HALF_HEIGHT, CAN_RADIUS, FLOOR_TOL,
                       SEAT_TOL, SETTLE_DISP, SETTLE_STEPS)

# bins_arena.xml bin2 carries two divider geoms -- one collision (group 0) and one visual
# (group 1) per axis -- and they are the ONLY geoms in that body centred at this local
# position, which is how they are identified without relying on unnamed-geom ordering.
DIVIDER_LOCAL_POS = (0.0, 0.0, 0.05)
DIVIDER_N_GEOMS = 4

# Interior half-extents of bin2, measured to the INNER faces of its walls, from the arena XML:
# the floor geom is 0.2 x 0.25 (half), the x walls sit at +-0.2 with half-thickness 0.01 and the
# y walls at +-0.25 with half-thickness 0.01, so the usable floor is 0.38 x 0.48 m. This is
# deliberately NOT ``bin_size / 2`` (= 0.195 x 0.245): ``bin_size`` is the arena's nominal
# ``table_full_size``, which the placement sampler uses, and it overshoots the real walls by
# 5 mm on each axis. ``verify_geometry`` re-derives these from the loaded model.
TRAY_HALF_INTERIOR = (0.19, 0.24)

# Erosions that define the goal regions, all as insets from TRAY_HALF_INTERIOR.
#   valid    = interior - (can radius + wall margin): a can seated anywhere here is fully
#              clear of the walls, so no goal can require wall contact.
#   envelope = valid - TRAIN_INSET: where TRAINING goals are drawn.
#   ring     = valid \ envelope: the extrapolation ring, physically safe by construction.
# The regions are additionally clipped to a reachability box (see REACH_HALF), measured with
# agent_tests/_gct_reach.py -- the far corners of the tray are outside the Panda's usable
# workspace on this mount and no erosion of the tray alone would exclude them.
WALL_MARGIN = 0.010
TRAIN_INSET = 0.020
GOAL_TOL = 0.035

# Reach clip, in bin2-local metres: (x_min, x_max, y_min, y_max) of the largest axis-aligned
# box inside both the eroded tray and the empirically reachable set. None means "no clip", which
# is what the reach probe itself runs under.
#
# From agent_tests/_gct_reach.py (108 points, free-arm, each from the same home configuration):
# the placement residual is < 1 mm everywhere inside a base radius of 0.821 m and blows up past
# 0.827 m -- a clean radial workspace boundary, not an orientation artifact (agent_tests/
# _gct_yaw_probe.py: yawing the wrist +-60 deg changes the far-corner residual by at most a few
# mm). The clip is the largest-area axis-aligned box inside radius 0.795 m, i.e. ~3 cm of margin
# on the measured boundary, rounded to the nearest 5 mm. Its far corner (local +0.045, +0.075)
# sits at 0.789 m, and the probe was free-armed, so the margin also absorbs the extra load of
# carrying the can.
REACH_BOX = (-0.155, 0.045, -0.205, 0.075)

# ---------------------------------------------------------------- pre-registered goal protocol
#
# Frozen before collection. Fixed by the PI after agent_tests/_gct_region_study.py showed the
# reach-limited envelope (0.160 x 0.240 m) cannot hold 5 goals at 12 cm separation -- acceptance
# 0.0% with the holdout and 3.2% without. The ruling keeps the 12 cm separation and the ~200-demo
# budget by trading goals-per-scene for scenes: 3 goals x 67 training scenes.
GOALS_PER_SCENE = 3
MIN_GOAL_SEP = 0.120
N_TRAIN_SCENES = 67
N_VAL_SCENES = 5

# Spatial holdout: a GRID_N x GRID_N grid over the training envelope; the two named cells and a
# HOLDOUT_BORDER ring around each are excluded from every TRAINING goal, and the cell interiors
# form an evaluation set.
#
# The cells are diagonally opposite CORNERS. With a 0.160 x 0.240 m envelope the 2.5 cm border is
# half a cell wide, so any same-row or same-column pair wipes out a full band and leaves training
# goals in two disjoint ~5.5 cm strips -- fatal for a task whose point is continuous goal
# coverage. Only the diagonal pairs leave every x with >= 13.5 cm of y and every y with >= 8.2 cm
# of x. The consequence, which every analysis must state: the held-out cells sit OUTSIDE the
# convex hull of the training goals, so this set measures SPATIAL EXTRAPOLATION, never
# interpolation. The interpolation claim lives on `sample_interpolation_goals` instead.
GRID_N = 3
HOLDOUT_CELLS = ((0, 0), (2, 2))
HOLDOUT_BORDER = 0.025

# Interpolation evaluation goals must be strictly inside the training hull and this far from
# every collected training goal, so they are genuinely unseen rather than near-duplicates.
#
# 8 mm is set against the measured density, not chosen for comfort: 201 goals over 0.0218 m^2
# tile the region at a mean spacing of ~10 mm (median nearest-neighbour 4.2 mm), so 8 mm is
# about twice the typical inter-goal gap -- clearly a new goal -- while remaining satisfiable.
# A larger buffer is not available: the largest empty circle in the training region has radius
# 27 mm, so NO point is more than 27 mm from some training goal, and a buffer near the 35 mm
# success tolerance admits almost nothing (at 25 mm it admitted exactly one goal).
#
# The consequence for evaluation, which must be stated wherever this set is used: because an
# unseen goal always has a training goal within ~27 mm and the success radius is 35 mm, a policy
# that merely snapped to the nearest goal it had seen could still score binary success here.
# The interpolation set therefore has to be scored on CONTINUOUS placement error, where such a
# policy shows an error floor at the nearest-neighbour distance and a true interpolator does not.
INTERP_HULL_MARGIN = 0.010
INTERP_MIN_DIST = 0.008

PROMPT = "put the can in the tray"

# ---------------------------------------------------------------- semantic target marker
#
# A flat disc lying in the tray at the episode's semantic target. It exists so a PLAN can refer to
# a physical referent that something in the scene could actually perceive, rather than to a number
# handed in from outside: the constraint says "the can goes on the marker", and swapping the fake
# VLM for a real one later changes only how the marker is found, not what the plan means.
#
# It is deliberately inert: collision groups are off, so the can, the tray and the gripper pass
# through it and the physics of the task is bit-for-bit what it was without the marker. The body
# centre sits exactly ON the tray floor top, so a can seated on the marker has its centre at
# marker + CAN_HALF_HEIGHT with no residual offset to correct for.
MARKER_BODY = "traymarker"
MARKER_GEOM = "traymarker_disc"
MARKER_RADIUS = 0.020
MARKER_HALF_THICKNESS = 0.002
MARKER_RGBA = (0.10, 0.80, 0.45, 1.0)


def _local_box(half_x, half_y, clip=REACH_BOX):
    """Axis-aligned region in bin2-local xy as (x_min, x_max, y_min, y_max)."""
    box = (-half_x, half_x, -half_y, half_y)
    if clip is None:
        return box
    return (max(box[0], clip[0]), min(box[1], clip[1]),
            max(box[2], clip[2]), min(box[3], clip[3]))


def valid_box(clip=REACH_BOX):
    """Local xy box in which a commanded goal is physically placeable."""
    return _local_box(TRAY_HALF_INTERIOR[0] - CAN_RADIUS - WALL_MARGIN,
                      TRAY_HALF_INTERIOR[1] - CAN_RADIUS - WALL_MARGIN, clip)


def train_box(clip=REACH_BOX):
    """Local xy box from which TRAINING goals are drawn: valid_box eroded by TRAIN_INSET."""
    x0, x1, y0, y1 = valid_box(clip)
    return (x0 + TRAIN_INSET, x1 - TRAIN_INSET, y0 + TRAIN_INSET, y1 - TRAIN_INSET)


def in_box(xy, box):
    return bool(box[0] <= xy[0] <= box[1] and box[2] <= xy[1] <= box[3])


def grid_cells(clip=REACH_BOX):
    """The GRID_N x GRID_N holdout grid over the training envelope, keyed by (ix, iy)."""
    x0, x1, y0, y1 = train_box(clip)
    xs = np.linspace(x0, x1, GRID_N + 1)
    ys = np.linspace(y0, y1, GRID_N + 1)
    return {(i, j): (xs[i], xs[i + 1], ys[j], ys[j + 1])
            for i in range(GRID_N) for j in range(GRID_N)}


def holdout_boxes(clip=REACH_BOX):
    """Interiors of the held-out cells; the SPATIAL EXTRAPOLATION evaluation set."""
    cells = grid_cells(clip)
    return [cells[tuple(c)] for c in HOLDOUT_CELLS]


def blocked_boxes(clip=REACH_BOX):
    """Held-out cells dilated by HOLDOUT_BORDER; no training goal may fall in these."""
    return [(x0 - HOLDOUT_BORDER, x1 + HOLDOUT_BORDER,
             y0 - HOLDOUT_BORDER, y1 + HOLDOUT_BORDER)
            for x0, x1, y0, y1 in holdout_boxes(clip)]


def in_train_region(xy, clip=REACH_BOX):
    """A legal TRAINING goal: inside the envelope and clear of both blocked cells."""
    return bool(in_box(xy, train_box(clip))
                and not any(in_box(xy, b) for b in blocked_boxes(clip)))


def in_holdout_cell(xy, clip=REACH_BOX):
    """Inside a held-out cell interior -- the spatial EXTRAPOLATION evaluation region."""
    return bool(any(in_box(xy, b) for b in holdout_boxes(clip)))


def in_ring(xy, clip=REACH_BOX):
    """Inside the extrapolation ring: placeable, but outside the training envelope."""
    return bool(in_box(xy, valid_box(clip)) and not in_box(xy, train_box(clip)))


def _propose_goal_set(rng, n, sep, box, clip, slot_tries=600):
    """One uniformly drawn set of n mutually separated goals, or None if the draw jams."""
    chosen = []
    for _ in range(n):
        for _ in range(slot_tries):
            p = np.array([rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3])])
            if not in_train_region(p, clip):
                continue
            if any(np.linalg.norm(p - q) < sep for q in chosen):
                continue
            chosen.append(p)
            break
        else:
            return None
    return np.stack(chosen)


def sample_train_goals(rng, n=GOALS_PER_SCENE, sep=MIN_GOAL_SEP, coverage=None,
                       n_sets=64, set_budget=600, clip=REACH_BOX):
    """Draw n well-separated training goals for one scene, biased toward thin coverage.

    The coverage bias is applied to whole SETS, not goal by goal. Choosing each goal greedily
    from the least-populated bin deadlocks: the first goals get pulled to whichever corner is
    thinnest, and no third point 12 cm from both survives -- and because the greedy choice is
    near-deterministic, restarting reproduces the same jam. Proposing complete separated sets
    uniformly and then keeping the one that lands in the thinnest bins keeps the within-scene
    geometry unbiased while still steering the POOLED set toward even coverage, which is where
    coverage has to come from: at 12 cm separation a single scene cannot cover anything.

    Args:
        coverage (None or CoverageGrid): updated in place with the chosen set.

    Returns:
        (n, 2) array of local xy, in a random order that carries no coverage-bias signal.
    """
    box = train_box(clip)
    sets = []
    for _ in range(set_budget):
        if len(sets) >= n_sets:
            break
        s = _propose_goal_set(rng, n, sep, box, clip)
        if s is not None:
            sets.append(s)
    if not sets:
        raise RuntimeError(f"could not place {n} goals at {sep} m separation in {box}")
    if coverage is None:
        goals = sets[rng.integers(len(sets))]
    else:
        scores = np.array([sum(coverage.count(p) for p in s) for s in sets], dtype=np.float64)
        tied = np.flatnonzero(scores == scores.min())
        goals = sets[tied[rng.integers(len(tied))]]
        for p in goals:
            coverage.add(p)
    # The slot order is the proposal order, so shuffle before goal_index is assigned.
    return goals[rng.permutation(n)]


class CoverageGrid:
    """Occupancy histogram over the training region, in local tray coordinates."""

    def __init__(self, pitch=0.02, clip=REACH_BOX):
        self.box = train_box(clip)
        self.pitch = float(pitch)
        self.shape = (max(1, int(np.ceil((self.box[1] - self.box[0]) / self.pitch))),
                      max(1, int(np.ceil((self.box[3] - self.box[2]) / self.pitch))))
        self.counts = np.zeros(self.shape, dtype=np.int64)

    def _bin(self, xy):
        i = int(np.clip((xy[0] - self.box[0]) / self.pitch, 0, self.shape[0] - 1))
        j = int(np.clip((xy[1] - self.box[2]) / self.pitch, 0, self.shape[1] - 1))
        return i, j

    def count(self, xy):
        i, j = self._bin(xy)
        return int(self.counts[i, j])

    def add(self, xy):
        i, j = self._bin(xy)
        self.counts[i, j] += 1


def sample_interpolation_goals(train_goals, rng, n=24, clip=REACH_BOX,
                               hull_margin=INTERP_HULL_MARGIN, min_dist=INTERP_MIN_DIST):
    """Unseen goals strictly INSIDE the convex hull of the collected training goals.

    This is the interpolation evaluation set. The held-out corner cells cannot serve that role
    -- being envelope corners they fall outside the training hull by construction, which is why
    they are labelled extrapolation everywhere. Points here are inside the hull shrunk by
    `hull_margin` and at least `min_dist` from every training goal, so they are genuinely new
    without being near-duplicates of something demonstrated.
    """
    from scipy.spatial import ConvexHull, Delaunay

    pts = np.asarray(train_goals, dtype=np.float64)[:, :2]
    hull = ConvexHull(pts)
    centroid = pts[hull.vertices].mean(axis=0)
    shrunk = centroid + (pts[hull.vertices] - centroid) * (
        1.0 - hull_margin / np.linalg.norm(pts[hull.vertices] - centroid, axis=1, keepdims=True))
    tri = Delaunay(shrunk)
    box = train_box(clip)
    out = []
    for _ in range(200000):
        if len(out) >= n:
            break
        p = np.array([rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3])])
        if tri.find_simplex(p) < 0 or not in_train_region(p, clip):
            continue
        if np.linalg.norm(pts - p, axis=1).min() < min_dist:
            continue
        if out and min(np.linalg.norm(np.asarray(out) - p, axis=1)) < min_dist:
            continue
        out.append(p)
    return np.asarray(out)


class SortCanTray(PickPlace):
    """Place one can at a commanded continuous position inside the undivided bin2 tray.

    Args:
        goal (None or 3-tuple): pin the commanded goal in WORLD coordinates. None leaves the
            goal at the tray centre until :meth:`set_goal` is called, which is what the
            collector and the evaluator both do.
        reach_box (None or 4-tuple): override :data:`REACH_BOX` for the region helpers. The
            reach probe passes None so it can measure the unclipped tray.
        marker (bool): add the inert semantic-target marker disc to the scene. Off by default,
            so every experiment written before it sees the identical scene it always saw.
    """

    def __init__(self, goal=None, reach_box=REACH_BOX, z_rotation=(0.0, np.pi / 2.0),
                 marker=False, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs
        self._reach_box = None if reach_box is None else tuple(float(v) for v in reach_box)
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64)
        self._marker = bool(marker)
        self._can_track = []
        super().__init__(single_object_mode=2, object_type="can", z_rotation=z_rotation,
                         **kwargs)

    # ---------------------------------------------------------------- scene

    def _construct_visual_objects(self):
        """No robosuite visual objects; the tray is the arena's own bin2."""
        self.visual_objects = []

    def _construct_objects(self):
        self.objects = [CanObject(name="Can")]
        # PickPlace.__init__ set object_id from object_to_id ("can" -> 3), but this task
        # builds a single object, so the active index is 0.
        self.object_id = 0

    @staticmethod
    def _strip_dividers(bin2_body):
        """Delete bin2's cross divider, turning four quadrants into one walled tray.

        Both the collision and the visual copy of each divider have to go: the converter
        renders with the collision group disabled, so a surviving visual divider would show
        up in every training image while the physics said the tray was open.
        """
        removed = 0
        for geom in list(bin2_body.findall("geom")):
            pos = geom.get("pos")
            if pos is None:
                continue
            if np.allclose(np.fromstring(pos, sep=" "), DIVIDER_LOCAL_POS):
                bin2_body.remove(geom)
                removed += 1
        if removed != DIVIDER_N_GEOMS:
            raise RuntimeError(f"expected {DIVIDER_N_GEOMS} bin2 divider geoms at "
                               f"{DIVIDER_LOCAL_POS}, removed {removed}; the arena XML changed")
        return removed

    def _load_model(self):
        """Build the bins arena with bin2's divider removed, then place the can.

        A near-copy of PickPlace._load_model: the divider has to be dropped before
        ManipulationTask merges the arena, so the tray is baked into the per-demo XML and a
        restored scene cannot silently get its divider back.
        """
        # Skip PickPlace._load_model and go straight to its own parent, which sets up the
        # robot; the arena and objects are rebuilt here.
        super(PickPlace, self)._load_model()

        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["bins"])

        mujoco_arena = BinsArena(bin1_pos=self.bin1_pos, table_full_size=self.table_full_size,
                                 table_friction=self.table_friction)
        mujoco_arena.set_origin([0, 0, 0])
        self.bin_size = mujoco_arena.table_full_size
        self._strip_dividers(mujoco_arena.bin2_body)
        if self._marker:
            self._add_marker(mujoco_arena)

        self._construct_visual_objects()
        self._construct_objects()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )
        self._get_placement_initializer()

    @staticmethod
    def _add_marker(arena):
        """Weld the inert marker disc into the arena, parked until `set_marker` moves it.

        A body with no joint is welded to the world, so its pose lives in ``model.body_pos`` and
        survives ``sim.reset``/``set_state_from_flattened`` -- a snapshot restore therefore keeps
        the marker where the scene put it without the marker ever entering the state vector.
        """
        body = ET.SubElement(arena.worldbody, "body", name=MARKER_BODY, pos="0 0 -1")
        ET.SubElement(body, "geom", name=MARKER_GEOM, type="cylinder",
                      size=f"{MARKER_RADIUS} {MARKER_HALF_THICKNESS}",
                      rgba=" ".join(str(v) for v in MARKER_RGBA),
                      contype="0", conaffinity="0", group="1")
        return body

    @property
    def has_marker(self):
        return self._marker

    def _marker_body_id(self):
        if not self._marker:
            raise RuntimeError("this env was built without the marker; pass marker=True")
        return self.sim.model.body_name2id(MARKER_BODY)

    def set_marker(self, world_xyz):
        """Place the marker disc, its centre ON the tray floor top under the given point."""
        pos = np.asarray(world_xyz, dtype=np.float64).reshape(3).copy()
        pos[2] = self.bin2_pos[2] + BIN_FLOOR_TOP
        self.sim.model.body_pos[self._marker_body_id()] = pos
        self.sim.forward()
        return pos

    def marker_pos(self):
        """Live world position of the marker disc centre."""
        return np.array(self.sim.data.body_xpos[self._marker_body_id()], dtype=np.float64)

    def _get_placement_initializer(self):
        """Sample the can anywhere inside bin1, with the D0 yaw range."""
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=[-bin_x_half, bin_x_half],
            y_range=[-bin_y_half, bin_y_half],
            rotation=self.z_rotation,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.bin1_pos,
            z_offset=self.z_offset,
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._can_track = []

    def _post_action(self, action):
        self._can_track.append(self.can_pos())
        if len(self._can_track) > SETTLE_STEPS:
            self._can_track.pop(0)
        return super()._post_action(action)

    def reset_settle_history(self):
        """Clear the settle window; a snapshot restore is not a reset."""
        self._can_track = []

    def verify_geometry(self):
        """Re-derive the tray interior from the LOADED model and check the constants.

        Guards against a robosuite arena change silently moving the walls out from under
        TRAY_HALF_INTERIOR, which every region definition is anchored to.
        """
        model, half = self.sim.model, np.zeros(2)
        bin2 = model.body_name2id("bin2")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bin2 or model.geom_group[gid] != 0:
                continue
            pos, size = model.geom_pos[gid], model.geom_size[gid]
            if abs(pos[2]) < 1e-9:
                continue  # the floor
            for axis in (0, 1):
                if size[axis] < 0.05:  # a wall's thin axis
                    half[axis] = max(half[axis], abs(pos[axis]) - size[axis])
        found = tuple(np.round(half, 6))
        if not np.allclose(found, TRAY_HALF_INTERIOR, atol=1e-6):
            raise RuntimeError(f"tray interior {found} != TRAY_HALF_INTERIOR "
                               f"{TRAY_HALF_INTERIOR}")
        n_div = sum(1 for gid in range(model.ngeom)
                    if model.geom_bodyid[gid] == bin2
                    and np.allclose(model.geom_pos[gid], DIVIDER_LOCAL_POS))
        if n_div:
            raise RuntimeError(f"{n_div} divider geoms survived into the loaded model")
        return {"tray_half_interior": list(found), "divider_geoms": 0}

    # ---------------------------------------------------------------- goal

    def tray_centre(self):
        """World xyz of the tray centre at SEAT height (a can resting on the tray floor)."""
        return np.array([self.bin2_pos[0], self.bin2_pos[1],
                         self.bin2_pos[2] + BIN_FLOOR_TOP + CAN_HALF_HEIGHT], dtype=np.float64)

    @property
    def seat_z(self):
        """The constant z every commanded goal carries."""
        return float(self.bin2_pos[2] + BIN_FLOOR_TOP + CAN_HALF_HEIGHT)

    def to_world(self, local_xy):
        """Local tray xy -> world goal xyz at seat height."""
        return np.array([self.bin2_pos[0] + local_xy[0], self.bin2_pos[1] + local_xy[1],
                         self.seat_z], dtype=np.float64)

    def to_local(self, world_xyz):
        """World goal xyz -> local tray xy."""
        return np.asarray(world_xyz, dtype=np.float64)[:2] - np.asarray(
            self.bin2_pos, dtype=np.float64)[:2]

    def valid_box(self):
        return valid_box(self._reach_box)

    def train_box(self):
        return train_box(self._reach_box)

    def set_goal(self, goal):
        """Pin the commanded goal, in world coordinates."""
        g = np.asarray(goal, dtype=np.float64).reshape(3)
        if not in_box(self.to_local(g), self.valid_box()):
            raise ValueError(f"goal {g.round(4).tolist()} (local "
                             f"{self.to_local(g).round(4).tolist()}) is outside the valid box "
                             f"{np.round(self.valid_box(), 4).tolist()}")
        self._goal = g

    @property
    def goal(self):
        """The commanded goal; the tray centre until one is set."""
        return self.tray_centre() if self._goal is None else np.array(self._goal)

    def g_task(self):
        """Canonical goal, in the same (position, tool-down quaternion) form as sort_can."""
        return self.goal, np.asarray(self.tool_down_quat(), dtype=np.float64)

    def goal_prompt(self):
        return PROMPT

    def tool_down_quat(self):
        """Current end-effector orientation (wxyz); the home pose points the tool down."""
        return np.array(self.sim.data.body_xquat[
            self.sim.model.body_name2id(self.robots[0].robot_model.eef_name)],
            dtype=np.float64)

    def layout(self):
        """Everything needed to reconstruct the episode's scene semantics."""
        return {
            "tray_centre": self.tray_centre().tolist(),
            "tray_half_interior": list(TRAY_HALF_INTERIOR),
            "valid_box_local": list(self.valid_box()),
            "train_box_local": list(self.train_box()),
            "reach_box_local": None if self._reach_box is None else list(self._reach_box),
            "holdout_cells": [list(c) for c in HOLDOUT_CELLS],
            "holdout_boxes_local": [list(b) for b in holdout_boxes(self._reach_box)],
            "holdout_border_m": HOLDOUT_BORDER,
            "min_goal_sep_m": MIN_GOAL_SEP,
            "seat_z": self.seat_z,
            "goal_tol_m": GOAL_TOL,
            "goal_world": self.goal.tolist(),
            "goal_local": self.to_local(self.goal).tolist(),
            "bin1_pos": np.asarray(self.bin1_pos).tolist(),
            "bin2_pos": np.asarray(self.bin2_pos).tolist(),
            "bin_size": np.asarray(self.bin_size).tolist(),
        }

    # ---------------------------------------------------------------- state

    def can_pos(self):
        return np.array(self.sim.data.body_xpos[self.obj_body_id["Can"]], dtype=np.float64)

    def can_speed(self):
        qvel = self.sim.data.get_joint_qvel(self.objects[0].joints[0])
        return float(np.linalg.norm(np.asarray(qvel)[:3]))

    def can_grasped(self):
        return bool(self._check_grasp(gripper=self.robots[0].gripper,
                                     object_geoms=self.objects[0].contact_geoms))

    def in_tray(self):
        """The can is physically inside the tray: within the walls and down at seat height.

        Same test sort_can applies per quadrant, with the tray's own half extents, so a can
        held or dropped above the open tray is not counted as inside it.
        """
        pos = self.can_pos()
        local = pos[:2] - np.asarray(self.bin2_pos, dtype=np.float64)[:2]
        if abs(local[0]) >= TRAY_HALF_INTERIOR[0] - CAN_RADIUS:
            return False
        if abs(local[1]) >= TRAY_HALF_INTERIOR[1] - CAN_RADIUS:
            return False
        floor_top = self.bin2_pos[2] + BIN_FLOOR_TOP
        seat_max = min(self.bin2_pos[2] + BIN_RIM, floor_top + CAN_HALF_HEIGHT + SEAT_TOL)
        return bool(floor_top - FLOOR_TOL <= pos[2] <= seat_max)

    def goal_error(self):
        """Distance from the can centre to the commanded goal, in metres."""
        return float(np.linalg.norm(self.can_pos() - self.goal))

    def settled(self):
        """The can centre has not moved over the last SETTLE_STEPS control steps."""
        if len(self._can_track) < SETTLE_STEPS:
            return False
        track = np.stack(self._can_track)
        return bool(np.abs(track - track[-1]).max() < SETTLE_DISP)

    def settle_displacement(self):
        if len(self._can_track) < 2:
            return float("inf")
        track = np.stack(self._can_track)
        return float(np.abs(track - track[-1]).max())

    def place_info(self):
        """Per-step placement telemetry, so a near miss is recorded rather than just a miss."""
        return {
            "in_tray": self.in_tray(),
            "goal_error_m": self.goal_error(),
            "goal_world": self.goal.tolist(),
            "released": not self.can_grasped(),
            "settled": self.settled(),
            "can_speed": self.can_speed(),
            "settle_disp": self.settle_displacement(),
        }

    def _check_success(self):
        """Released, at rest, within GOAL_TOL of the commanded goal, and inside the tray."""
        return bool(not self.can_grasped()
                    and self.settled()
                    and self.goal_error() <= GOAL_TOL
                    and self.in_tray())
