"""Provide simulator-backed and sensor-backed object state for vlm_dp."""
from __future__ import annotations

import numpy as np

from vlm_dp.hold import HoldLatch
from vlm_dp.sim_helpers import quat_wxyz_to_R


_PLACE_MARGIN = 0.02
_PLACE_FOOT_FALLBACK = 0.10
_REST_TOL = 0.10


class GTWorld:
    """Read exact object state from the simulator."""

    def __init__(self, env, sensor=None):
        self.env = env
        self.names = list(getattr(env.scene, "rigid_objects", {}) or {})


        self._latch = HoldLatch(sensor) if sensor is not None else None

    def object_pose(self, name):
        st = self.env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy()
        return st[:3].astype(np.float64), quat_wxyz_to_R(st[3:7])

    @property
    def _pos(self):
        """Believed object positions, for the eval loop's belief logger.

        Only SensedWorld defined this, and the logger guards on
        getattr(world, "_pos", None) -- so every --vlm_state gt run silently logged no
        object_beliefs at all, leaving GT target error unmeasurable and GT unusable as a
        reference arm. GTWorld reads exact simulator state, so the belief IS the pose.
        """
        return {n: self.object_pose(n)[0] for n in self.names}

    def body_pose(self, asset, body):
        """Return the world pose of an articulation body."""
        data = self.env.scene[asset].data
        i = list(data.body_names).index(body)
        pos = data.body_pos_w[0, i].detach().cpu().numpy().astype(np.float64)
        return pos, quat_wxyz_to_R(data.body_quat_w[0, i].detach().cpu().numpy())

    def joint_angle(self, asset, joint):
        """Return an articulation joint angle in radians."""
        data = self.env.scene[asset].data
        return float(data.joint_pos[0, list(data.joint_names).index(joint)].detach())

    def flags(self):
        try:
            group = self.env.observation_manager.compute_group("subtask_terms")
            return {key: bool(val.detach().flatten()[0].item()) for key, val in group.items()}
        except Exception:
            return {}

    def held(self):
        """Return the latched held object, if available."""
        return self._latch.held() if self._latch is not None else None

    def observe(self, env, commanded_close, candidates=None, points=None):
        """Update the GT hold latch for one control step."""
        if self._latch is None:
            return
        self._latch.sensor.observe(env, commanded_close)
        positions = {n: self.object_pose(n)[0] for n in self.names}
        positions.update(points or {})   # grasp points, not centroids: see SensedWorld.observe
        self._latch.update(positions, env.tcp(), candidates)

    def sync_fk(self, env):
        pass

    def refresh(self, env):
        pass

    def stale(self):
        return set()

    def mark_stale(self, name):
        """Mark a non-held object estimate as stale."""
        return False


class SensedWorld:
    """Track object state from perception and proprioception."""

    _SLIP_MARGIN = 0.18


    _MAX_JUMP_PER_STEP = 0.0375

    def __init__(self, perception, sensor, place_obj=None, tcp_offset=(0.0, 0.0, 0.0),
                 track="fk", reperceive_every=32, tracker=None):
        self.perception = perception
        self.sensor = sensor
        self.place_obj = place_obj
        self.tcp_offset = tcp_offset


        self.track = track
        self.reperceive_every = reperceive_every
        self.visual = tracker
        self._step = 0


        self._last_correct = {}
        self.names = []
        self._pos = {}
        self._rot = {}
        self._held = None
        self._grip0 = None
        self._pose0 = None
        self._pending = None
        self._grip_aperture = None
        self._stale = set()

    def seed(self, name, pos):
        """Register a calibrated point as a tracked object."""
        self._pos[name] = np.asarray(pos, dtype=np.float64)
        self._rot[name] = np.eye(3)
        self.names = list(self._pos)

    def refresh(self, env):
        """Refresh visible non-held object estimates from perception."""


        if getattr(self, "relax_identity_when_stale", False):
            self.perception.distrust = set(self._stale)
        seen = self.perception.observe(env)
        displaced = False
        for name, pos in seen.items():
            if name == self._held:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if name in self._stale and name in self._pos \
                    and float(np.linalg.norm(pos - self._pos[name])) > 0.03:
                displaced = True
            self._pos[name] = pos


            self._rot[name] = np.eye(3)
            self._stale.discard(name)
        self.names = list(self._pos)


        if self.visual is not None and seen:
            rebase = getattr(self.visual, "rebase", None)
            if callable(rebase):
                rebase({n: self._pos[n] for n in seen if n in self._pos})
        if displaced and self.visual is not None:
            self._reprime_tracker(env)

    def _reprime_tracker(self, env):
        """Reinitialize visual tracking after a displaced re-perception."""
        from vlm_dp.visual_tracker import VisualTracker
        self.visual = VisualTracker(env.cam, self.names,
                                    {n: self._pos[n] for n in self.names},
                                    device=self.visual.device)
        print("[world:visual] tracker re-primed after displaced re-perception", flush=True)

    def observe(self, env, commanded_close, candidates=None, points=None):
        """Update the hold latch for one control step.

        ``points`` overrides, for the hold test only, where an object is considered to BE. The
        hold test asks "is the thing the fingers stalled on within reach of the TCP", and the
        answer has to be measured at the point the stage actually grasps: a declared lid rim is
        0.25-0.30m from its machine's centroid, so resolving the hold against centroids made a
        physically perfect grasp (TCP 1.2mm from the rim, fingers stalled on it) unrecognisable and
        the stage unadvanceable by construction. Object TRACKING keeps using the centroid belief --
        that is what _fk_held carries and what _placed measures -- so only this dictionary changes.
        """
        self.sensor.observe(env, commanded_close)


        if self.sensor.is_open() or self.sensor.closed_on_air():
            self._pending = None
        elif self._pending is None:
            self._pending = self._hand(env)


        if (self._held is not None and self._grip_aperture is not None
                and not self.sensor.is_open()
                and self.sensor.aperture() < self._grip_aperture + self._SLIP_MARGIN):
            held = self._held
        else:
            at = dict(self._pos)
            at.update(points or {})
            near = ({n: p for n, p in at.items() if n in candidates} if candidates else at)
            held = self.sensor.held_object(near, env.tcp())

        if held != self._held:
            if self._held is not None:
                self._stale.add(self._held)

            self._grip_aperture = self.sensor.aperture() if held is not None else None
            self._held = held
            self._grip0 = (self._pending or self._hand(env)) if held else None
            self._pose0 = ((self._pos[held].copy(), self._rot[held].copy())
                           if held is not None and held in self._pos else None)

        self._fk_held(env)
        if self.track != "fk":
            self._vision_step(env)

    def sync_fk(self, env):
        """Update the held-object estimate from the current hand pose."""
        self._fk_held(env)

    def _fk_held(self, env):
        """Carry the held object rigidly with the hand."""
        if self._held is not None and self._pose0 is not None:
            tcp, rot = self._hand(env)
            tcp0, rot0 = self._grip0
            d_rot = rot @ rot0.T
            pos0, obj_rot0 = self._pose0
            self._pos[self._held] = tcp + d_rot @ (pos0 - tcp0)
            self._rot[self._held] = d_rot @ obj_rot0

    def _vision_step(self, env):
        """Apply visual corrections at the configured cadence."""
        self._step += 1
        if self.track == "reperceive":
            if self._step % self.reperceive_every == 0:
                self._vision_correct(env, self.perception.observe(env))
        elif self.track == "visual" and self.visual is not None:
            self._vision_correct(env, self.visual.step(env))

    def set_jump_rate(self, rate: float) -> None:
        """Set the maximum accepted visual correction per control step."""
        self._MAX_JUMP_PER_STEP = float(rate)

    def _max_jump(self, name) -> float:
        """Return the current correction limit for an object."""
        elapsed = max(1, int(self._step) - int(self._last_correct.get(name, 0)))
        return self._MAX_JUMP_PER_STEP * elapsed

    def _vision_correct(self, env, seen):
        """Apply plausible visual corrections to non-held objects."""
        if not seen:
            return
        moved = []
        for name, pos in seen.items():
            if name in self._stale:
                continue
            if name == self._held:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if name in self._pos:
                d = float(np.linalg.norm(pos - self._pos[name]))
                if d > self._max_jump(name):
                    continue


                if d > 0.015:
                    moved.append((name, round(d, 3)))
            self._last_correct[name] = int(self._step)
            self._pos[name] = pos
            self._rot[name] = np.eye(3)
            self._stale.discard(name)
        self.names = list(self._pos)
        if moved:
            print(f"[world:{self.track}] corrected {moved}", flush=True)

    def held(self):
        """Return the latched held object, if available."""
        return self._held

    def object_pose(self, name):
        if name not in self._pos:
            raise KeyError(f"[world] no estimate for {name!r}: perception never found it. The controller "
                           f"may not fall back to simulator state.")
        return self._pos[name], self._rot[name]

    def flags(self):
        """Return sensed grasp and placement flags."""
        out = {}
        for name in self._pos:
            out[f"grasp_{name}"] = (name == self._held)
            if self.place_obj and name != self.place_obj:
                out[f"{name}_on_{self.place_obj}"] = self._placed(name)
        return out

    def stale(self):
        """Return objects whose estimates are known to be stale."""
        return set(self._stale)

    def mark_stale(self, name):
        """Mark a non-held object estimate as stale."""
        if name is None or name == self._held or name in self._stale:
            return False
        self._stale.add(name)
        return True

    def _placed(self, name):
        """Return whether an object is resting on the destination."""
        if name == self._held or self.place_obj not in self._pos:
            return False
        d = self._pos[name] - self._pos[self.place_obj]
        ext = self.perception.object_extents(self.place_obj) if self.perception is not None else None
        foot = float(ext[1]) if ext is not None else _PLACE_FOOT_FALLBACK
        return bool(np.linalg.norm(d[:2]) < foot + _PLACE_MARGIN and abs(d[2]) < _REST_TOL)

    def _hand(self, env):
        """Return the gripper world pose from joint encoders."""
        pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), self.tcp_offset)
        return (pos[0].detach().cpu().numpy().astype(np.float64),
                rot[0].detach().cpu().numpy().astype(np.float64))
