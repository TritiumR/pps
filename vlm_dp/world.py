"""Object-state seam: the only place the controller reads object poses and grasp flags.

GTWorld reads the simulator, the privileged upper bound. SensedWorld reads sensors only: perception
fixes poses, FK carries the held object while the finger joint reports contact, and lost contact marks
the belief stale.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.sim_helpers import quat_wxyz_to_R

# Placement tolerances are our own sensing slack, deliberately not the env's success thresholds: a
# sensed flag calibrated to the evaluator's predicate is tuned to the answer key and will not transfer.
_PLACE_MARGIN = 0.02      # m, slack added to the destination's measured footprint (grounding error)
_PLACE_FOOT_FALLBACK = 0.10   # m, destination half-footprint used only when its extent is unavailable
_REST_TOL = 0.10          # m, vertical slack so an object still in the air does not read as placed


class GTWorld:
    """Object state straight from the simulator: the privileged upper bound (raw IsaacLab env)."""

    def __init__(self, env):
        self.env = env
        self.names = list(getattr(env.scene, "rigid_objects", {}) or {})

    def object_pose(self, name):
        st = self.env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy()
        return st[:3].astype(np.float64), quat_wxyz_to_R(st[3:7])

    def body_pose(self, asset, body):
        """World pose of one articulation link such as a lid. Returns (pos[3], R[3,3])."""
        data = self.env.scene[asset].data
        i = list(data.body_names).index(body)
        pos = data.body_pos_w[0, i].detach().cpu().numpy().astype(np.float64)
        return pos, quat_wxyz_to_R(data.body_quat_w[0, i].detach().cpu().numpy())

    def joint_angle(self, asset, joint):
        """Current angle of one articulation joint (rad)."""
        data = self.env.scene[asset].data
        return float(data.joint_pos[0, list(data.joint_names).index(joint)].detach())

    def flags(self):
        try:
            group = self.env.observation_manager.compute_group("subtask_terms")
            return {key: bool(val.detach().flatten()[0].item()) for key, val in group.items()}
        except Exception:
            return {}

    def observe(self, env, commanded_close, candidates=None):
        pass

    def sync_fk(self, env):
        pass

    def refresh(self, env):
        pass

    def stale(self):
        return set()


class SensedWorld:
    """Object state from perception and proprioception only."""

    _SLIP_MARGIN = 0.18   # rad the aperture may close past the acquired grip before it counts as a slip
    _MAX_JUMP = 0.15      # m a per-step vision correction may move a belief, above which it is a latch

    def __init__(self, perception, sensor, place_obj=None, tcp_offset=(0.0, 0.0, 0.0),
                 track="fk", reperceive_every=8, tracker=None):
        self.perception = perception          # observe(env) returns {name: pos}, the only camera consumer
        self.sensor = sensor                  # ApertureGraspSensor, reports what is between the fingers
        self.place_obj = place_obj
        self.tcp_offset = tcp_offset
        # Between-look tracking: fk is dead-reckoning only (default). reperceive re-segments every
        # reperceive_every steps. visual corrects per step from a point tracker, occlusion-aware.
        self.track = track
        self.reperceive_every = reperceive_every
        self.visual = tracker                 # a VisualTracker (CoTracker) for track=visual, else None
        self._step = 0
        self.names = []
        self._pos = {}                        # name -> estimated world position
        self._rot = {}                        # name -> estimated world rotation (identity until carried)
        self._held = None                     # object the gripper reports holding, if any
        self._grip0 = None                    # hand pose when the current hold began
        self._pose0 = None                    # held object's pose when the current hold began
        self._pending = None                  # hand pose at contact onset (certification lags it)
        self._grip_aperture = None            # aperture the current object was gripped at (slip datum)
        self._stale = set()                   # objects whose belief is known wrong, look again

    def seed(self, name, pos):
        """Register a calibrated point (such as an articulation's lip) as a trackable object.

        Its initial estimate comes from workcell calibration, not detection. It is carried by FK while
        held and corrected by the visual tracker like any other object.
        """
        self._pos[name] = np.asarray(pos, dtype=np.float64)
        self._rot[name] = np.eye(3)
        self.names = list(self._pos)

    def refresh(self, env):
        """Look at the scene, and re-fix every object that is not currently in the hand."""
        seen = self.perception.observe(env)
        displaced = False
        for name, pos in seen.items():
            if name == self._held:            # the hand is in front of it, kinematics knows better
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if name in self._stale and name in self._pos \
                    and float(np.linalg.norm(pos - self._pos[name])) > 0.03:
                displaced = True              # the drop moved it, its tracker query is now invalid
            self._pos[name] = pos
            # A single view does not recover orientation, and a dropped object's carried rotation is
            # a leftover.
            self._rot[name] = np.eye(3)
            self._stale.discard(name)
        self.names = list(self._pos)
        if displaced and self.visual is not None:
            self._reprime_tracker(env)

    def _reprime_tracker(self, env):
        """A dropped object re-perceived elsewhere invalidates its CoTracker query: the point
        latched onto the occluder/background and would confidently re-assert the old location,
        clobbering re-perception. Re-prime on the current frame with the current estimates."""
        from vlm_dp.visual_tracker import VisualTracker
        self.visual = VisualTracker(env.cam, self.names,
                                    {n: self._pos[n] for n in self.names},
                                    device=self.visual.device)
        print("[world:visual] tracker re-primed after displaced re-perception", flush=True)

    def observe(self, env, commanded_close, candidates=None):
        """One control step: update contact and carry the held object with the hand.

        candidates are the objects currently reached for, needed to name what the fingers hold.
        """
        self.sensor.observe(env, commanded_close)

        # Contact-onset anchor: certification lags first touch by the settle window, and the hand keeps
        # moving in between. Anchoring the ride at certification would bake that drift into the carried
        # offset, so the object reads metres below the hand and the lift never confirms.
        if (not commanded_close) or self.sensor.closed_on_air():
            self._pending = None              # open hand, or closed on air: no live contact
        elif not self.sensor.is_open() and self._pending is None:
            self._pending = self._hand(env)

        # Proximity names the object at acquisition, proprioception sustains the hold, keyed on the
        # aperture staying near the grip it was acquired at (measured live at onset, no per-object table).
        # An open hand releases. Closing past the grip by more than _SLIP_MARGIN means the object is
        # squeezing out (a slip that never reaches closed-on-air), so drop the hold and let re-perception
        # and re-grasp fire. This band tolerates the lift-jerk jitter that holding()'s settle window did
        # not, yet catches the in-band slip FK is blind to.
        if (self._held is not None and self._grip_aperture is not None
                and not self.sensor.is_open()
                and self.sensor.aperture() < self._grip_aperture + self._SLIP_MARGIN):
            held = self._held
        else:
            near = ({n: p for n, p in self._pos.items() if n in candidates} if candidates
                    else self._pos)
            held = self.sensor.held_object(near, env.tcp())

        if held != self._held:
            if self._held is not None:        # contact lost, the belief about that object is now stale
                self._stale.add(self._held)
            # Grip signature: the aperture at the moment this hold was acquired (measured live, no table).
            self._grip_aperture = self.sensor.aperture() if held is not None else None
            self._held = held
            self._grip0 = (self._pending or self._hand(env)) if held else None
            self._pose0 = ((self._pos[held].copy(), self._rot[held].copy())
                           if held is not None and held in self._pos else None)

        self._fk_held(env)                    # dead-reckoning: the held object rides the hand
        if self.track != "fk":                # ... and vision periodically corrects it (and catches slip)
            self._vision_step(env)

    def sync_fk(self, env):
        """Re-carry the held object to the current joint state.

        observe() runs at the end of a replan, after the chunk executed, so without this the estimate
        the next replan's advance and cost read would be a whole chunk stale, the arm having already
        moved past it.
        """
        self._fk_held(env)

    def _fk_held(self, env):
        """Carry the held object rigidly with the hand: apply the hand's motion since the grasp began."""
        if self._held is not None and self._pose0 is not None:
            tcp, rot = self._hand(env)
            tcp0, rot0 = self._grip0
            d_rot = rot @ rot0.T
            pos0, obj_rot0 = self._pose0
            self._pos[self._held] = tcp + d_rot @ (pos0 - tcp0)
            self._rot[self._held] = d_rot @ obj_rot0

    def _vision_step(self, env):
        """A fresh visual estimate, at the cadence the tracking mode allows, layered over dead-reckoning."""
        self._step += 1
        if self.track == "reperceive":
            if self._step % self.reperceive_every == 0:
                self._vision_correct(env, self.perception.observe(env))
        elif self.track == "visual" and self.visual is not None:
            self._vision_correct(env, self.visual.step(env))

    def _vision_correct(self, env, seen):
        """Snap non-held beliefs to a visual estimate.

        The held object is owned by FK: contact licenses the kinematic ride, and this is exactly the
        phase where vision is worst. The hand occludes the object, so the tracker latches onto a
        background point and would drag the estimate off the hand, breaking the carry and faking a lost
        lift.
        """
        if not seen:
            return
        moved = []                            # names where vision disagrees with the current belief
        for name, pos in seen.items():
            if name in self._stale:           # dropped, the track is broken, only re-perception re-fixes it
                continue
            if name == self._held:            # held, FK is authoritative and vision is occluded here
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if name in self._pos:
                d = float(np.linalg.norm(pos - self._pos[name]))
                if d > self._MAX_JUMP:        # implausible one-step jump, the tracker point latched onto
                    continue                  # the gripper or background. Keep the belief, re-perception re-fixes.
                if d > 0.015:                 # a real move (a carried object, or a slip), not centroid jitter
                    moved.append((name, round(d, 3)))
            self._pos[name] = pos
            self._rot[name] = np.eye(3)       # a single view does not recover orientation
            self._stale.discard(name)
        self.names = list(self._pos)
        if moved:                             # the whole point of vision over dead-reckoning: it sees the move
            print(f"[world:{self.track}] corrected {moved}", flush=True)

    def object_pose(self, name):
        if name not in self._pos:
            raise KeyError(f"[world] no estimate for {name!r}: perception never found it. The controller "
                           f"may not fall back to simulator state.")
        return self._pos[name], self._rot[name]

    def flags(self):
        """The same booleans the driver and the cost already consume, from sensors instead of the sim."""
        out = {}
        for name in self._pos:
            out[f"grasp_{name}"] = (name == self._held)
            if self.place_obj and name != self.place_obj:
                out[f"{name}_on_{self.place_obj}"] = self._placed(name)
        return out

    def stale(self):
        """Objects whose belief is known to be wrong: contact was lost while one was being carried."""
        return set(self._stale)

    def _placed(self, name):
        """name is resting on the place object: released, and sitting over its surface.

        Judged against the destination's measured footprint, not the env's success thresholds. This flag
        is the sensed twin of <obj>_on_<place>, and calibrating it to the evaluator's predicate would
        tune the policy to the answer key.
        """
        if name == self._held or self.place_obj not in self._pos:
            return False
        d = self._pos[name] - self._pos[self.place_obj]
        ext = self.perception.object_extents(self.place_obj) if self.perception is not None else None
        foot = float(ext[1]) if ext is not None else _PLACE_FOOT_FALLBACK
        return bool(np.linalg.norm(d[:2]) < foot + _PLACE_MARGIN and abs(d[2]) < _REST_TOL)

    def _hand(self, env):
        """World pose of the gripper from the joint encoders: ``(tcp[3], R[3,3])``."""
        pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), self.tcp_offset)
        return (pos[0].detach().cpu().numpy().astype(np.float64),
                rot[0].detach().cpu().numpy().astype(np.float64))
