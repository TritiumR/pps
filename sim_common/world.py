"""What the controller believes about object state, and the only place it may read it from.

Exactly two dynamic quantities drive the controller: where each object is, and whether one is in the
gripper. Everything else it uses -- object sizes, workspace bounds, camera calibration -- is static, known
once, and legitimately available to a real robot. So those two, and only those two, sit behind this seam.

Putting them here makes the privileged and the sensed arms differ in one place instead of a dozen:

    GTWorld       reads the simulator: object poses from the physics state, grasp flags from its contact
                  detector. The upper bound, and the control the sensed arm is measured against.
    SensedWorld   reads sensors only: object poses from segmentation, then propagated by forward
                  kinematics for as long as the gripper reports contact; grasp from the finger joint.

``SensedWorld`` rests on one claim, and that claim is what licenses forward kinematics to stand in for a
visual tracker:

    An object moves only when something touches it, and the gripper is the only thing that does. So while
    contact holds, the object rides the hand rigidly and the joint encoders give its motion exactly -- and
    that is also the phase where vision is worst, because the hand is in front of the object. While contact
    does not hold, the object is where perception last saw it. When contact is lost, that belief is stale
    and perception has to look again.

The contact sensor is what licenses the propagation, and it is a genuinely independent measurement: the
finger joint reports what is *between the fingers*, which no amount of arm kinematics can tell you. Without
that gate the two signals would collapse into one -- an object whose position is derived from the arm
always "rises" when the arm rises, so an empty gripper would report a perfect lift, forever.
"""
from __future__ import annotations

import numpy as np

from sim_common.geometry import quat_wxyz_to_R

_PLACE_XY = 0.12    # m: horizontal tolerance for "placed on", matching the task's own success predicate
_PLACE_Z = 0.10     # m: vertical tolerance, so an object still in the air does not read as placed


class GTWorld:
    """Object state read straight from the simulator: the privileged upper bound.

    Faithful to what the controller did before this seam existed, so this arm is unchanged by the refactor
    and remains the reference the sensed arm is scored against. Takes the raw IsaacLab env.
    """

    def __init__(self, env):
        self.env = env
        self.names = list(getattr(env.scene, "rigid_objects", {}) or {})

    def object_pose(self, name):
        st = self.env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy()
        return st[:3].astype(np.float64), quat_wxyz_to_R(st[3:7])

    def flags(self):
        try:
            group = self.env.observation_manager.compute_group("subtask_terms")
            return {key: bool(val.detach().flatten()[0].item()) for key, val in group.items()}
        except Exception:
            return {}

    def observe(self, env, commanded_close, candidates=None):
        pass

    def refresh(self, env):
        pass

    def stale(self):
        return set()


class SensedWorld:
    """Object state from perception and proprioception only: the real stack.

    Perception fixes each object's pose whenever the scene is looked at. From then on that pose is held --
    a still object stays put -- except while the gripper reports contact, when the object is carried rigidly
    with the hand. Losing contact invalidates the belief and marks the object for a fresh look.
    """

    def __init__(self, perception, sensor, place_obj=None, tcp_offset=(0.0, 0.0, 0.0),
                 track="fk", reperceive_every=8, tracker=None):
        self.perception = perception          # .observe(env) -> {name: pos}; the only consumer of the camera
        self.sensor = sensor                  # ApertureGraspSensor: is something between the fingers?
        self.place_obj = place_obj
        self.tcp_offset = tcp_offset
        # How an object's position is kept between the looks that fix it. All three ride the same seam --
        # a periodic correction on top of the kinematic dead-reckoning below -- so downstream is identical.
        #   fk         : dead-reckoning only. The held object rides the hand; nothing corrects it until
        #                contact is lost (today's behaviour, and the default).
        #   reperceive : + a full re-segmentation every ``reperceive_every`` steps snaps every object,
        #                held included, back onto what the camera sees. Genuinely vision-based, but the
        #                grasp is exactly when the hand occludes the object, so it is the weaker option.
        #   visual     : + a point tracker (``tracker``) corrects every object each step it reports one,
        #                which is occlusion-aware in a way re-segmentation is not.
        self.track = track
        self.reperceive_every = reperceive_every
        self.visual = tracker                 # a VisualTracker (CoTracker) for track="visual", else None
        self._step = 0
        self.names = []
        self._pos = {}                        # name -> estimated world position
        self._rot = {}                        # name -> estimated world rotation (identity until carried)
        self._held = None                     # object the gripper reports holding, if any
        self._grip0 = None                    # hand pose when the current hold began
        self._pose0 = None                    # held object's pose when the current hold began
        self._stale = set()                   # objects whose belief is known to be wrong -> look again

    def refresh(self, env):
        """Look at the scene, and re-fix every object that is not currently in the hand."""
        seen = self.perception.observe(env)
        for name, pos in seen.items():
            if name == self._held:            # the hand is in front of it; kinematics knows better
                continue
            self._pos[name] = np.asarray(pos, dtype=np.float64)
            # Back to the frame the keypoint offsets were measured in. A single view does not recover
            # orientation, and for an object that was just dropped the rotation it happened to be carried
            # at is not an estimate of anything -- it is a leftover.
            self._rot[name] = np.eye(3)
            self._stale.discard(name)
        self.names = list(self._pos)

    def observe(self, env, commanded_close, candidates=None):
        """One control step: update contact, and carry the held object with the hand.

        ``candidates`` are the objects the controller is currently trying to grasp or carry. The fingers
        report that *something* is between them, not what, so naming it needs both what was reached for and
        where the hand is. Without that, the nearest estimate wins -- and the board the fruit is sitting on
        has a centroid close enough to steal the answer.
        """
        self.sensor.observe(env, commanded_close)
        near = ({n: p for n, p in self._pos.items() if n in candidates} if candidates
                else self._pos)
        held = self.sensor.held_object(near, env.tcp())

        if held != self._held:
            if self._held is not None:        # contact lost: what we believe about that object is now stale
                self._stale.add(self._held)
            self._held = held
            self._grip0 = self._hand(env) if held else None
            self._pose0 = ((self._pos[held].copy(), self._rot[held].copy())
                           if held is not None and held in self._pos else None)

        self._fk_held(env)                    # dead-reckoning: the held object rides the hand
        if self.track != "fk":                # ... and vision periodically corrects it (and catches slip)
            self._vision_step(env)

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
        """Snap object beliefs to a visual estimate, and re-anchor the held object's ride there.

        Vision sees what kinematics cannot: the object leaving the hand. Correcting onto it is what turns a
        slip -- invisible to FK, which rides the object with the arm no matter what -- into an observable jump
        away from the gripper. The held object is re-anchored at the corrected pose so its dead-reckoning
        continues from where vision last saw it, rather than snapping back on the next step.
        """
        if not seen:
            return
        moved = []                            # where vision disagrees with the current belief: the correction
        for name, pos in seen.items():
            pos = np.asarray(pos, dtype=np.float64)
            if name in self._pos:
                d = float(np.linalg.norm(pos - self._pos[name]))
                if d > 0.015:                 # a real move (a carried object, or a slip), not centroid jitter
                    moved.append((name, round(d, 3)))
            self._pos[name] = pos
            if name != self._held:
                self._rot[name] = np.eye(3)   # a single view does not recover orientation
            self._stale.discard(name)
        self.names = list(self._pos)
        if self._held is not None and self._held in seen:
            self._grip0 = self._hand(env)
            self._pose0 = (self._pos[self._held].copy(), self._rot[self._held].copy())
        if moved:                             # the whole point of vision over dead-reckoning: it sees the move
            print(f"[world:{self.track}] corrected {moved}"
                  f"{f' (held {self._held})' if self._held else ''}", flush=True)

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
        """``name`` is resting on the place object: let go of, and sitting over its surface."""
        if name == self._held or self.place_obj not in self._pos:
            return False
        d = self._pos[name] - self._pos[self.place_obj]
        return bool(np.linalg.norm(d[:2]) < _PLACE_XY and abs(d[2]) < _PLACE_Z)

    def _hand(self, env):
        """World pose of the gripper from the joint encoders: ``(tcp[3], R[3,3])``."""
        pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), self.tcp_offset)
        return (pos[0].detach().cpu().numpy().astype(np.float64),
                rot[0].detach().cpu().numpy().astype(np.float64))
