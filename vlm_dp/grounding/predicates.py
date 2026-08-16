"""Load and evaluate VLM-authored stage COMPLETION PREDICATES.

A ReKep sub-goal constraint is a scalar: the runtime decides a stage is finished when that
scalar dips below ``subgoal_eps``. A completion predicate answers a different question --
"has the event this stage names actually happened" -- as a composition of named booleans over
a short history window. This module is the loader for that second kind of block.

Format
------
Completion predicates live in the same ``gt_vlm_output/<task>/raw.txt`` the constraints do, as
column-0 functions named ``stage<N>_completion``. :func:`write_files` splits them exactly the
way ``fake_vlm._render`` splits constraints (a function runs from ``def `` to its first
``    return `` line) and writes one ``stage<N>_completion.txt`` per stage next to the
``stage<N>_subgoal_constraints.txt`` files. Stages with no predicate get an empty file, so a
loader can read every stage unconditionally.

Contract
--------
Each predicate is called as::

    stage<N>_completion(history, end_effector, keypoints, robot_state, components) -> bool

``history`` -- recent frames, OLDEST FIRST, sampled ``history["dt"]`` seconds apart
    ``history["kp"]``               list[T] of (K, 3) arrays: per-frame keypoint BELIEFS
    ``history["eef"]``              list[T] of (3,) arrays: end-effector position
    ``history["gripper_aperture"]`` list[T] of float: finger joint angle. 0.0 is fully open,
                                    ~0.785 (pi/4) is a free close on air, anything stalled in
                                    between means the fingers are held apart by an object.
    ``history["dt"]``               float, seconds between frames (1/15 for the 15 Hz loop)
``end_effector`` -- (3,) current end-effector position (equals ``history["eef"][-1]``)
``keypoints``    -- (K, 3) current keypoint beliefs (equals ``history["kp"][-1]``)
``robot_state``  -- ``{"gripper_aperture": float, "gripper_closed_cmd": bool}``
``components``   -- a caller-supplied dict the predicate FILLS IN. Keys named ``margin_*`` hold
                    the raw float behind a boolean so a log shows how close to its edge each
                    component ran; every other key is one named boolean of the composition.

Predicates see beliefs and proprioception only. No privileged simulator state is ever passed.

Short windows are the point: a predicate is expected to fail gracefully (return False) when the
history is shorter than it needs, so the first frames after a reset never fire a stage.

Evaluation happens in a sandbox whose globals are ``np`` plus the seven-name primitive API of
:class:`PredicateRuntime` (see its docstring). A plan that uses the primitives states WHAT event
it means; the sensing, the window, and the thresholds live here, in one place, reviewed once.
"""
from __future__ import annotations

import os

import numpy as np

_PREDICATE_SUFFIX = "_completion"


def split_functions(text):
    """Split rendered plan text into ``name -> source`` blocks.

    Mirrors ``fake_vlm._render``: a function runs from a column-0 ``def `` to its first line
    starting with ``    return ``. Keeping the two splitters identical means a predicate block
    is parsed by exactly the rule a constraint block is.
    """
    lines, functions, start, name = text.split("\n"), {}, None, None
    for i, line in enumerate(lines):
        if line.startswith("def "):
            start, name = i, line.split("(")[0].split("def ")[1]
        elif line.startswith("    return ") and name is not None:
            functions[name] = "\n".join(lines[start:i + 1])
            start, name = None, None
    return functions


def write_files(text, out_dir, num_stages):
    """Write ``stage<N>_completion.txt`` for every stage from already-rendered plan text.

    Returns the number of stages that actually carry a predicate. Stages without one get an
    empty file so :func:`load` can read all of them unconditionally.
    """
    functions, found = split_functions(text), 0
    for idx in range(1, int(num_stages) + 1):
        body = functions.get(f"stage{idx}{_PREDICATE_SUFFIX}", "")
        found += bool(body)
        with open(os.path.join(out_dir, f"stage{idx}_completion.txt"), "w", encoding="utf-8") as f:
            f.write(body + "\n" if body else "")
    return found


def _compile(source, idx):
    """Exec one predicate block in a sandbox whose only global is numpy."""
    sandbox = {"np": np}
    exec(compile(source, f"<stage{idx}_completion>", "exec"), sandbox)  # noqa: S102
    fn = sandbox.get(f"stage{idx}{_PREDICATE_SUFFIX}")
    if fn is None:
        raise SystemExit(f"[predicates] stage{idx}_completion.txt defines no "
                         f"stage{idx}{_PREDICATE_SUFFIX}()")
    return fn


class PredicateContractError(Exception):
    """A predicate asked a primitive for something the primitive is not allowed to answer.

    Distinct from a bug in the predicate's arithmetic: this is the plan using a tolerance on a
    keypoint the tolerance is not defined for (see :meth:`PredicateRuntime.clearance_margin`).
    It is raised, never papered over with a plausible-looking number.
    """


class PredicateRuntime:
    """The trusted primitives a completion predicate is written in.

    A plan states WHAT event a stage names; this states HOW the event is sensed. Everything
    about sensing -- which signals, over how long a window, against which thresholds -- lives
    in the constants block below and nowhere else, so the whole set is reviewed in one place
    instead of being re-derived (and drifting) in every stage of every task's raw.txt.

    Seven names are injected into a predicate's sandbox:

    EVENTS -- each returns a bool AND fills ``components`` with its own named boolean and the
    ``margin_*`` floats behind it, so a log shows how close to its edge the event ran:

    ``grasped(kp, name, mode)``
        The gripper owns the body that carries keypoint ``kp``. ``mode="acquire"`` is the
        pick-up test: the fingers are stalled on something for the whole window, the object
        POSITIVELY co-moved with a hand that actually travelled, and the object is within
        reach. ``mode="maintain"`` is the still-holding test: fingers stalled, and the object
        did not slip relative to a hand that moved -- a still hand proves nothing about slip,
        so a still hand passes. Acquisition must show evidence; maintenance must show no
        counter-evidence. Collapsing the two would make a pick-up fire on a resting object.
    ``released(kp, name)``
        The gripper let go: either the hand departed (it travelled while the object did not
        follow) or the fingers opened for a quarter of the window.
    ``stationary(kp, name)``
        The body carrying ``kp`` is not moving -- a statement about speed, which no distance
        to a place point can see.
    ``sustained(cost_fn, name)``
        The plan's own ReKep-convention cost (``<= 0`` satisfied, called per frame as
        ``cost_fn(end_effector, keypoints)``) held over the last ``SUSTAIN`` belief frames,
        worst-of-window, with ``BELIEF_TOL`` of slack for belief noise. This is how a plan
        states a geometric condition without also stating a window or a tolerance.

    TOLERANCES -- each returns a length in metres, resolved from the SAME grounding geometry
    the cost terms use, so a plan never writes a magic number:

    ``grasp_tolerance(kp)``    half-width of the grasp feature that owns ``kp``.
    ``target_region_radius(kp)``  radius of the placement region around an anchor keypoint.
    ``clearance_margin(kp)``   vertical band separating "clear above" from "descended onto",
        defined ONLY for a keypoint on a body the plan carries.

    Numerical note (deliberate, see the aperture band): the band literals are NOT derived from
    the aperture sensor in this revision even though the sensor's own stall threshold is
    ``q_free - stall_margin``. Deriving them is a threshold change, not an API change, and is
    kept out of this one so the refactor can be checked for equivalence. The sensor config is
    carried here and printed once precisely so that follow-up has the number.
    """

    # ---- sensing constants. The single place any of these may be written. ----
    WINDOW = 8                  # frames of history an event test spans (8 @ 15 Hz ~ 0.53 s)
    SUSTAIN = 5                 # frames a sustained() geometric condition must hold for
    BELIEF_TOL = 0.005          # m, slack sustained() allows for belief noise
    APERTURE_LO = 0.08          # rad, below this the fingers closed on air
    APERTURE_HI = 0.60          # rad, above this they are stalled on nothing (free close ~0.785)
    SLIP_RATIO_MAX = 0.5        # object-vs-hand displacement ratio that still counts as held
    CARRY_HAND_DISP_M = 0.02    # m, hand travel below which co-motion cannot be judged
    DEPART_HAND_DISP_M = 0.03   # m, hand travel a departure test needs
    DEPART_RATIO_MIN = 0.7      # ratio above which the object stayed behind
    REACH_M = 0.10              # m, object-to-TCP distance that counts as "in the hand"
    OPEN_Q = 0.06               # rad, aperture at or below which the fingers read as open
    OPEN_FRAC_MIN = 0.25        # fraction of the window that must read open for a release
    OPEN_MIN_FRAMES = 4         # frames before the open-fraction test means anything
    QUIESCENT_MPS = 0.015       # m/s, speed below which a placed object counts as at rest
    REGION_R_M = 0.08           # m, placement region radius
    CLEARANCE_M = 0.05          # m, clear-above / descended-onto separation band
    MIN_GRASP_EXT_M = 0.005     # m, floor on a resolved grasp half-width

    def __init__(self, owner_of=None, grasp_ext_of=None, extents=None, carried=(),
                 declared=(), open_half=0.04, geom=None, sensor=None):
        self._owner_of = owner_of or (lambda i: None)
        self._grasp_ext_of = dict(grasp_ext_of or {})
        self._extents = dict(extents or {})
        self._carried = frozenset(carried or ())
        self._declared = frozenset(declared or ())
        self.open_half = float(open_half)
        self.geom = dict(geom or {})
        self.sensor = dict(sensor or {})
        # Per-evaluation frame state, installed by begin().
        self._hist = None
        self._components = None
        self._eef = None
        self._kp = None
        self._robot = None

    # ------------------------------------------------------------------ binding

    def api(self):
        """Return the ``name -> callable`` mapping injected into a predicate sandbox."""
        return {"grasped": self.grasped, "released": self.released,
                "stationary": self.stationary, "sustained": self.sustained,
                "grasp_tolerance": self.grasp_tolerance,
                "target_region_radius": self.target_region_radius,
                "clearance_margin": self.clearance_margin}

    def begin(self, history, end_effector, keypoints, robot_state, components):
        """Install the frame the primitives read. Called once per predicate evaluation."""
        self._hist = history
        self._eef = end_effector
        self._kp = keypoints
        self._robot = robot_state
        self._components = components

    def describe(self):
        """One line naming the thresholds, for the episode log."""
        q_free = float(self.sensor.get("q_free", float("nan")))
        stall = float(self.sensor.get("stall_margin", float("nan")))
        return (f"window={self.WINDOW} sustain={self.SUSTAIN} belief_tol={self.BELIEF_TOL} "
                f"aperture=({self.APERTURE_LO}, {self.APERTURE_HI}) "
                f"region_r={self.REGION_R_M} clearance={self.CLEARANCE_M} "
                f"carried={sorted(self._carried)} "
                f"[sensor stall threshold q_free-stall_margin={q_free - stall:.4f}, "
                f"NOT used for the band in this revision]")

    # ------------------------------------------------------------------ helpers

    def _series(self, kp_idx):
        """Return ``(aperture, eef, obj, w, dt)`` for one keypoint over the event window."""
        hist = self._hist
        if hist is None:
            raise PredicateContractError("predicate primitives called with no history installed")
        ap = np.asarray(hist["gripper_aperture"], dtype=np.float64)
        eef = np.asarray(hist["eef"], dtype=np.float64)
        obj = np.asarray([f[int(kp_idx)] for f in hist["kp"]], dtype=np.float64)
        w = int(min(len(ap), len(eef), len(obj), self.WINDOW))
        return ap, eef, obj, w, max(float(hist.get("dt", 1.0 / 15.0)), 1e-6)

    def _put(self, key, value):
        if self._components is not None:
            self._components[key] = value

    @staticmethod
    def _disp(arr, w):
        return float(np.linalg.norm(arr[-1] - arr[-w])) if w > 1 else 0.0

    @staticmethod
    def _slip(obj, eef, w, absent):
        """Displacement of the object's offset from the hand across the window."""
        if w <= 1:
            return float(absent)
        return float(np.linalg.norm((obj[-1] - eef[-1]) - (obj[-w] - eef[-w])))

    # ------------------------------------------------------------------- events

    def grasped(self, kp, name="grasped", mode="acquire"):
        """Whether the gripper owns the body carrying ``kp``. See the class docstring."""
        if mode not in ("acquire", "maintain"):
            raise PredicateContractError(
                f"grasped(mode={mode!r}): mode must be 'acquire' (pick-up: positive co-motion "
                f"required) or 'maintain' (still holding: absence of slip)")
        ap, eef, obj, w, _ = self._series(kp)
        closed = bool(w >= self.WINDOW
                      and np.all(ap[-w:] > self.APERTURE_LO) and np.all(ap[-w:] < self.APERTURE_HI))
        hand = self._disp(eef, w)
        if mode == "acquire":
            # Pick-up needs EVIDENCE the object came along: the hand must have moved, and the
            # object's offset from it must have held. A hand that never moved proves nothing,
            # so the ratio is driven far out of band rather than treated as passing.
            slip = self._slip(obj, eef, w, absent=9.9)
            ratio = slip / hand if hand > 1e-9 else 9.9
            rides = bool(w >= self.WINDOW and hand > self.CARRY_HAND_DISP_M
                         and ratio < self.SLIP_RATIO_MAX)
            reach = float(np.linalg.norm(obj[-1] - eef[-1]))
            at_hand = bool(reach < self.REACH_M)
            verdict = bool(closed and rides and at_hand)
            self._put("at_hand", at_hand)
            self._put("margin_reach_m", reach)
        else:
            # Maintenance needs only the ABSENCE of counter-evidence: a still hand cannot slip.
            slip = self._slip(obj, eef, w, absent=9.9)
            ratio = slip / hand if hand > self.CARRY_HAND_DISP_M else 0.0
            rides = bool(ratio < self.SLIP_RATIO_MAX)
            verdict = bool(closed and rides)
        self._put("closed_on_object", closed)
        self._put("rides_with_hand", rides)
        self._put("margin_aperture", float(ap[-1]) if len(ap) else 0.0)
        self._put("margin_slip_ratio", float(ratio))
        self._put("margin_hand_disp_m", hand)
        self._put(name, verdict)
        return verdict

    def released(self, kp, name="released"):
        """Whether the gripper let go of the body carrying ``kp`` (departure or opening)."""
        ap, eef, obj, w, _ = self._series(kp)
        hand = self._disp(eef, w)
        slip = self._slip(obj, eef, w, absent=0.0)
        ratio = slip / hand if hand > self.CARRY_HAND_DISP_M else 0.0
        departed = bool(w >= self.WINDOW and hand > self.DEPART_HAND_DISP_M
                        and ratio > self.DEPART_RATIO_MIN)
        open_frac = float(np.mean(ap[-w:] <= self.OPEN_Q)) if w >= self.OPEN_MIN_FRAMES else 0.0
        verdict = bool(departed or open_frac >= self.OPEN_FRAC_MIN)
        self._put("margin_slip_ratio", float(ratio))
        self._put("margin_hand_disp_m", hand)
        self._put("margin_open_frac", open_frac)
        self._put(name, verdict)
        return verdict

    def stationary(self, kp, name="stationary"):
        """Whether the body carrying ``kp`` has stopped moving."""
        _ap, _eef, obj, w, dt = self._series(kp)
        speed = float(np.linalg.norm(obj[-1] - obj[-w]) / ((w - 1) * dt)) if w > 1 else 9.9
        verdict = bool(w >= self.WINDOW and speed < self.QUIESCENT_MPS)
        self._put("margin_speed_mps", speed)
        self._put(name, verdict)
        return verdict

    def sustained(self, cost_fn, name="sustained"):
        """Whether a ReKep-convention cost held (``<= 0``) over the last ``SUSTAIN`` frames."""
        hist = self._hist
        if hist is None:
            raise PredicateContractError("predicate primitives called with no history installed")
        frames, eefs = hist["kp"], hist["eef"]
        k = int(min(len(frames), len(eefs), self.SUSTAIN))
        worst = float("inf")
        if k >= 1:
            worst = max(float(np.max(np.asarray(
                cost_fn(np.asarray(eefs[-i], dtype=np.float64),
                        np.asarray(frames[-i], dtype=np.float64)), dtype=np.float64)))
                for i in range(1, k + 1))
        verdict = bool(k >= self.SUSTAIN and worst <= self.BELIEF_TOL)
        self._put(f"margin_{name}", worst)
        self._put(name, verdict)
        return verdict

    # --------------------------------------------------------------- tolerances

    def _owner(self, kp, who):
        owner = self._owner_of(int(kp))
        if not owner:
            raise PredicateContractError(
                f"{who}(kp={kp}): no object owns that keypoint, so no geometry of its can be "
                f"resolved. Tolerances are grounded in a body, never guessed.")
        return owner

    def grasp_tolerance(self, kp):
        """Half-width of the grasp FEATURE that owns ``kp``.

        A declared local extent (a lid rim, a handle) is the geometry the gripper closes on and
        always wins; the owner's whole-body extent is a last resort for a body with no declared
        or measured grasp feature, never a substitute for one that has it.
        """
        owner = self._owner(kp, "grasp_tolerance")
        ext = self._grasp_ext_of.get(owner)
        if ext is None:
            ext = self._extents.get(owner, (self.open_half, self.open_half, self.open_half))[0]
        return max(float(ext), self.MIN_GRASP_EXT_M)

    def target_region_radius(self, kp):
        """Radius of the placement region around the anchor keypoint ``kp``."""
        if self._owner_of(int(kp)) is None and int(kp) < 0:
            raise PredicateContractError(f"target_region_radius(kp={kp}): not a keypoint index")
        return float(self.REGION_R_M)

    def clearance_margin(self, kp):
        """Vertical band separating "clear above a surface" from "descended onto it".

        DEFINED ONLY for a keypoint on a body the plan CARRIES, because the band is a property
        of the carried body's own geometry. Asking for it on a fixture-owned or declared
        feature -- a machine's lid rim, a cup's mouth -- is a category error: those clearances
        belong to the fixture, are authored by the plan, and must not be laundered through a
        primitive that would return a carried body's number for them.
        """
        owner = self._owner(kp, "clearance_margin")
        # TWO ways to be out of scope, and the declared one is not implied by the ownership one.
        # A DECLARED feature can sit on a body that also appears in grasp_keypoints -- the capsule
        # machine owns both its lid rim and the grasp the plan performs on it -- so an
        # owner-in-carried test alone would hand a lid recess the pod's number. A declared
        # feature's geometry is authored precisely because the owner's extents do not describe
        # it, which is exactly the condition under which this may not answer.
        if int(kp) in self._declared:
            raise PredicateContractError(
                f"clearance_margin(kp={kp}): keypoint {kp} is a DECLARED feature on {owner!r}. Its "
                f"geometry was declared because {owner!r}'s own extents do not describe it, so no "
                f"clearance can be derived from them. State this clearance in the plan.")
        if owner not in self._carried:
            raise PredicateContractError(
                f"clearance_margin(kp={kp}): keypoint {kp} belongs to {owner!r}, which this plan "
                f"never carries (carried: {sorted(self._carried) or 'none'}). A clearance for a "
                f"fixture feature is the plan's to state, not this primitive's to invent.")
        return float(self.CLEARANCE_M)


class CompletionPredicates:
    """The per-stage predicates of one plan, compiled and callable."""

    def __init__(self, fns):
        self._fns = dict(fns)
        self.runtime = None

    def __len__(self):
        return len(self._fns)

    def __contains__(self, stage_idx):
        return int(stage_idx) in self._fns

    @property
    def stages(self):
        """Zero-based indices of the stages that carry a predicate."""
        return sorted(self._fns)

    @classmethod
    def from_dir(cls, vlm_dir, num_stages):
        """Load ``stage<N>_completion.txt`` for stages 1..num_stages (missing/empty = no predicate)."""
        fns = {}
        for idx in range(1, int(num_stages) + 1):
            path = os.path.join(vlm_dir, f"stage{idx}_completion.txt")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                source = f.read().strip()
            if source:
                fns[idx - 1] = _compile(source, idx)     # keyed zero-based, like stage_idx
        return cls(fns)

    @classmethod
    def from_text(cls, text, num_stages):
        """Compile predicates straight out of already-rendered plan text (no files involved)."""
        functions, fns = split_functions(text), {}
        for idx in range(1, int(num_stages) + 1):
            source = functions.get(f"stage{idx}{_PREDICATE_SUFFIX}")
            if source:
                fns[idx - 1] = _compile(source, idx)
        return cls(fns)

    def bind(self, runtime):
        """Install the primitive runtime every predicate of this plan is written against.

        Must run BEFORE the advanceability preflight, so the preflight exercises the same bound
        primitives the rollout will. Injects into each compiled function's OWN globals dict --
        which is the sandbox it was exec'd in -- so the primitives are as reachable to a
        predicate as ``np`` is, and no predicate can reach anything else.
        """
        self.runtime = runtime
        api = runtime.api()
        for fn in self._fns.values():
            fn.__globals__.update(api)
        return self

    def evaluate(self, stage_idx, history, end_effector=None, keypoints=None, robot_state=None):
        """Evaluate one stage's predicate.

        Returns ``(fired, components)``. A stage with no predicate, or one that raises, returns
        ``(False, {...})`` with the reason under ``"error"`` -- a shadow signal must never be
        able to take down a rollout. A :class:`PredicateContractError` is additionally printed,
        because it means the PLAN is wrong rather than the state being unsatisfied, and it must
        not be lost in a components dict nobody reads.
        """
        idx = int(stage_idx)
        components = {}
        fn = self._fns.get(idx)
        if fn is None:
            components["error"] = "no predicate for stage"
            return False, components
        if end_effector is None:
            end_effector = np.asarray(history["eef"][-1], dtype=np.float64)
        if keypoints is None:
            keypoints = np.asarray(history["kp"][-1], dtype=np.float64)
        if robot_state is None:
            robot_state = {"gripper_aperture": float(history["gripper_aperture"][-1]),
                           "gripper_closed_cmd": False}
        if self.runtime is not None:
            fn.__globals__.update(self.runtime.api())
            self.runtime.begin(history, end_effector, keypoints, robot_state, components)
        try:
            fired = bool(fn(history, end_effector, keypoints, robot_state, components))
        except PredicateContractError as exc:         # the plan is wrong; say so out loud
            components["error"] = f"PredicateContractError: {exc}"
            print(f"[predicates] stage{idx + 1}_completion violates the primitive contract: {exc}",
                  flush=True)
            return False, components
        except Exception as exc:                      # diagnostics must never break a rollout
            components["error"] = f"{type(exc).__name__}: {exc}"
            return False, components
        return fired, components


def load(vlm_dir, num_stages):
    """Convenience wrapper around :meth:`CompletionPredicates.from_dir`."""
    return CompletionPredicates.from_dir(vlm_dir, num_stages)


class HistoryBuffer:
    """Fixed-length recent-frame buffer in the shape predicates expect.

    Push once per applied control step; read :meth:`view` at decision time. ``maxlen`` frames
    at 15 Hz is one second of history, which is longer than any window a weight predicate asks
    for (8 frames) and short enough that the buffer costs nothing.
    """

    def __init__(self, maxlen=15, dt=1.0 / 15.0):
        self.maxlen = int(maxlen)
        self.dt = float(dt)
        self.kp = []
        self.eef = []
        self.gripper_aperture = []

    def reset(self):
        """Drop all frames (call on episode reset -- stale frames would fire a stage)."""
        self.kp, self.eef, self.gripper_aperture = [], [], []

    def push(self, keypoints, eef, aperture):
        """Append one frame and evict the oldest."""
        self.kp.append(np.asarray(keypoints, dtype=np.float64))
        self.eef.append(np.asarray(eef, dtype=np.float64).reshape(3))
        self.gripper_aperture.append(float(aperture))
        for buf in (self.kp, self.eef, self.gripper_aperture):
            del buf[:-self.maxlen]

    def __len__(self):
        return len(self.eef)

    def view(self):
        """Return the ``history`` dict of the contract."""
        return {"kp": self.kp, "eef": self.eef,
                "gripper_aperture": self.gripper_aperture, "dt": self.dt}
