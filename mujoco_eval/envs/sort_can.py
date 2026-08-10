"""Two-bin can sorting on the stock robosuite bins arena.

One can starts in bin1. Two of bin2's four walled quadrants act as a red bin and a
blue bin; the requested destination is drawn per episode and the colour-to-quadrant
assignment is drawn independently, so the destination is not predictable from geometry.

The quadrants come from ``bins_arena.xml``: bin2 carries a cross divider, so each
quadrant already has four walls. Only two visual-only overlay geoms are added, which
is why this task needs no MJCF authoring and reuses ``fk_fit_can.json`` (the Panda
mount offset for the bins arena is unchanged).
"""

from __future__ import annotations

import collections

import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.arenas import BinsArena
from robosuite.models.objects import CanObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

COLOURS = ("red", "blue")

# Overlay pads. Visual only (group 1, no contype/conaffinity), so they neither collide
# nor appear in the collision-geom render pass.
PAD_RGBA = {"red": (0.85, 0.10, 0.10, 1.0), "blue": (0.10, 0.20, 0.85, 1.0)}
PAD_INSET = 0.012
PAD_THICKNESS = 0.002

# The diagonal quadrant pair, so the two destinations differ in both x and y (0.195 m and
# 0.245 m apart) rather than along one axis. Measured: q3 -- the quadrant the existing `can`
# task names as its cost target -- sits 0.864 m from the Panda base and the arm stalls 4-6 cm
# short of it at every carry height, so the usable diagonal is q1/q2, not q0/q3.
QUADRANTS = (1, 2)

# bins_arena.xml bin2: floor geom half-thickness 0.02 (top at bin2_pos.z + 0.02) and
# wall geoms centred at z=0.05 with half-height 0.05 (rim at bin2_pos.z + 0.10).
BIN_FLOOR_TOP = 0.02
BIN_RIM = 0.10

# CanObject half extents, as already fitted for the `can` task (tasks/registry.py).
CAN_RADIUS = 0.025
CAN_HALF_HEIGHT = 0.0407

# Success tolerances. The bin interior is BIN_RIM - BIN_FLOOR_TOP = 0.08 m tall while the
# can is 2 * CAN_HALF_HEIGHT = 0.0814 m, so an upright can resting on the bin floor pokes
# ~1 mm past the rim and a literal "can fully below the rim" test is unsatisfiable. The
# height test is therefore on the can CENTRE against its seated height, which is both
# orientation-agnostic (a toppled can sits lower still) and strictly tighter than the rim:
# seat bound 0.8707 < rim 0.90, so nothing hovering at or above the rim can pass.
FLOOR_TOL = 0.005
SEAT_TOL = 0.010
# "Settled" is measured as positional stillness, not speed. A can resting on the bin floor
# still reports 7-15 mm/s of contact-solver chatter while its centre holds to within 1e-5 m,
# so any velocity threshold tight enough to be meaningful sits below that noise floor.
SETTLE_DISP = 0.002
SETTLE_STEPS = 10

# Set to (pos, quat) to re-aim agentview; None keeps the arena's own camera.
AGENTVIEW_POSE = None

PROMPT = "put the can in the {colour} bin"


class SortCanTwoBin(PickPlace):
    """Sort one can into the requested coloured bin.

    Args:
        target_colour (None or str): pin the destination for every episode. None draws
            it per reset, which is what evaluation uses; the paired collector pins it
            through :meth:`set_target_colour` instead.
        quadrants (2-tuple of int): bin2 quadrant ids used as the two bins.
    """

    def __init__(self, target_colour=None, quadrants=QUADRANTS, z_rotation=(0.0, np.pi / 2.0),
                 **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs
        if target_colour is not None and target_colour not in COLOURS:
            raise ValueError(f"target_colour must be one of {COLOURS}, got {target_colour!r}")
        self._pinned_colour = target_colour
        self._quadrants = tuple(int(q) for q in quadrants)
        if len(self._quadrants) != 2 or len(set(self._quadrants)) != 2:
            raise ValueError(f"need two distinct quadrants, got {quadrants!r}")
        # Which quadrant carries which colour. Drawn in _load_model so it is baked into
        # the XML: a state-only reset_to cannot restore model fields, but the per-demo
        # model_file can, and that is the path the 224 converter re-renders through.
        self._pad_colour = {}
        self._target_colour = target_colour or COLOURS[0]
        self._can_track = collections.deque(maxlen=SETTLE_STEPS)
        super().__init__(single_object_mode=2, object_type="can", z_rotation=z_rotation,
                         **kwargs)
        if not self.hard_reset:
            raise ValueError("SortCanTwoBin randomizes the colour assignment inside the model, "
                             "so it requires hard_reset=True")

    # ---------------------------------------------------------------- scene

    def _construct_visual_objects(self):
        """No robosuite visual objects; the coloured pads are arena geoms instead."""
        self.visual_objects = []

    def _construct_objects(self):
        self.objects = [CanObject(name="Can")]
        # PickPlace.__init__ set object_id from object_to_id ("can" -> 3), but this task
        # builds a single object, so the active index is 0.
        self.object_id = 0

    def quadrant_frame(self, qid):
        """Return (centre_xyz, half_xy) of a bin2 quadrant, in world coordinates.

        Mirrors the arithmetic PickPlace uses for its own target bin placements.
        """
        x_low, y_low = float(self.bin2_pos[0]), float(self.bin2_pos[1])
        if qid in (0, 2):
            x_low -= self.bin_size[0] / 2.0
        if qid < 2:
            y_low -= self.bin_size[1] / 2.0
        half = (self.bin_size[0] / 4.0, self.bin_size[1] / 4.0)
        centre = np.array([x_low + half[0], y_low + half[1], float(self.bin2_pos[2])])
        return centre, half

    def _pad_geoms(self, colour_of_quadrant):
        """Build the two visual-only overlay geoms, in bin2-local coordinates."""
        geoms = []
        for qid, colour in colour_of_quadrant.items():
            centre, half = self.quadrant_frame(qid)
            local = centre - np.asarray(self.bin2_pos, dtype=np.float64)
            geoms.append(new_geom(
                name=f"sortcan_pad_q{qid}",
                type="box",
                size=array_to_string((half[0] - PAD_INSET, half[1] - PAD_INSET, PAD_THICKNESS)),
                pos=array_to_string((local[0], local[1], BIN_FLOOR_TOP + PAD_THICKNESS)),
                group=1,
                rgba=array_to_string(PAD_RGBA[colour]),
                contype="0",
                conaffinity="0",
            ))
        return geoms

    def _load_model(self):
        """Build the bins arena, paint two quadrants, and place the can.

        A near-copy of PickPlace._load_model: the pads have to be inserted into the arena
        before ManipulationTask merges it, so the colour assignment lands in the XML.
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
        if AGENTVIEW_POSE is not None:
            mujoco_arena.set_camera("agentview", pos=AGENTVIEW_POSE[0], quat=AGENTVIEW_POSE[1])

        # Fresh colour assignment per hard reset, then baked into the arena XML.
        order = np.random.permutation(len(COLOURS))
        self._pad_colour = {q: COLOURS[order[i]] for i, q in enumerate(self._quadrants)}
        for geom in self._pad_geoms(self._pad_colour):
            mujoco_arena.bin2_body.append(geom)

        self._construct_visual_objects()
        self._construct_objects()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )
        self._get_placement_initializer()

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

    def _sync_pad_colour(self):
        """Recover the colour assignment from the LOADED model, not from the last draw.

        _load_model draws the assignment and bakes it into the XML, but a scene is re-entered
        through reset_from_xml_string (that is how the 224 converter, the replay gate and any
        stored episode restore a demo), and robomimic's reset_to calls a plain reset FIRST --
        which re-draws. The Python bookkeeping would then name the wrong quadrant for a colour
        while the render showed the right one, so success got judged against the other bin.
        Reading the pad rgba back out of the model makes the XML the single source of truth.
        """
        for qid in self._quadrants:
            rgba = np.asarray(
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(f"sortcan_pad_q{qid}")],
                dtype=np.float64)
            self._pad_colour[qid] = min(
                COLOURS, key=lambda c: float(np.linalg.norm(rgba - np.asarray(PAD_RGBA[c]))))

    def _reset_internal(self):
        super()._reset_internal()
        self._sync_pad_colour()
        if self._pinned_colour is None:
            # Independent of the colour-to-quadrant draw, so colour and geometry are
            # uncorrelated across episodes.
            self._target_colour = COLOURS[int(np.random.randint(len(COLOURS)))]
        else:
            self._target_colour = self._pinned_colour
        self._can_track.clear()

    def _post_action(self, action):
        self._can_track.append(self.can_pos())
        return super()._post_action(action)

    def reset_settle_history(self):
        """Clear the settle window; a snapshot restore is not a reset."""
        self._can_track.clear()

    # ---------------------------------------------------------------- goal

    def set_target_colour(self, colour):
        """Pin the destination, for the counterfactually paired collection protocol."""
        if colour not in COLOURS:
            raise ValueError(f"colour must be one of {COLOURS}, got {colour!r}")
        self._pinned_colour = colour
        self._target_colour = colour

    @property
    def target_colour(self):
        return self._target_colour

    @property
    def pad_colour(self):
        """Quadrant id -> colour, as baked into the current model."""
        return dict(self._pad_colour)

    def quadrant_of(self, colour):
        return next(q for q, c in self._pad_colour.items() if c == colour)

    @property
    def target_quadrant(self):
        return self.quadrant_of(self._target_colour)

    @property
    def distractor_quadrant(self):
        return next(q for q in self._quadrants if q != self.target_quadrant)

    def goal_prompt(self, colour=None):
        return PROMPT.format(colour=colour or self._target_colour)

    def g_task(self, colour=None):
        """Canonical geometric destination for a colour: the quadrant's seat pose.

        This is the goal a keypoint/ReKep front end would emit -- quadrant centre, can
        resting height, tool pointing down -- and is independent of what the expert did.
        """
        qid = self.quadrant_of(colour) if colour is not None else self.target_quadrant
        centre, _ = self.quadrant_frame(qid)
        seat = np.array([centre[0], centre[1],
                         centre[2] + BIN_FLOOR_TOP + CAN_HALF_HEIGHT], dtype=np.float64)
        return seat, np.asarray(self.tool_down_quat(), dtype=np.float64)

    def tool_down_quat(self):
        """Current end-effector orientation (wxyz); the home pose points the tool down."""
        return np.array(self.sim.data.body_xquat[
            self.sim.model.body_name2id(self.robots[0].robot_model.eef_name)],
            dtype=np.float64)

    def layout(self):
        """Everything needed to reconstruct the episode's scene semantics."""
        centres = {q: self.quadrant_frame(q)[0].tolist() for q in self._quadrants}
        return {
            "quadrants": list(self._quadrants),
            "quadrant_centres": centres,
            "pad_colour": {str(q): c for q, c in self._pad_colour.items()},
            "target_colour": self._target_colour,
            "target_quadrant": int(self.target_quadrant),
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

    def contained_quadrant(self):
        """Quadrant the can is physically inside, or None.

        Containment is inside the walls (xy shrunk by the can radius) and down at seat
        height between the bin floor and the rim, so a can held or dropped above an open
        bin is not counted as inside it.
        """
        pos = self.can_pos()
        for qid in self._quadrants:
            centre, half = self.quadrant_frame(qid)
            if abs(pos[0] - centre[0]) >= half[0] - CAN_RADIUS:
                continue
            if abs(pos[1] - centre[1]) >= half[1] - CAN_RADIUS:
                continue
            floor_top = centre[2] + BIN_FLOOR_TOP
            rim = centre[2] + BIN_RIM
            seat_max = min(rim, floor_top + CAN_HALF_HEIGHT + SEAT_TOL)
            if not floor_top - FLOOR_TOL <= pos[2] <= seat_max:
                continue
            return qid
        return None

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

    def sort_info(self):
        """Per-step sorting telemetry, so a mis-sort is recorded rather than just a miss."""
        inside = self.contained_quadrant()
        grasped = self.can_grasped()
        return {
            "contained_quadrant": inside,
            "target_quadrant": int(self.target_quadrant),
            "target_colour": self._target_colour,
            "mis_sort": bool(inside is not None and inside != self.target_quadrant),
            "released": not grasped,
            "settled": self.settled(),
            "can_speed": self.can_speed(),
            "settle_disp": self.settle_displacement(),
        }

    def _check_success(self):
        """The can is in the REQUESTED bin, released, and at rest."""
        return bool(self.contained_quadrant() == self.target_quadrant
                    and not self.can_grasped()
                    and self.settled())
