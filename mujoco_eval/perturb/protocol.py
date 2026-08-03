"""Apply one rollout disturbance and record recovery metrics."""
from __future__ import annotations

import numpy as np

from ..env.mujoco_env import _BODY_ALIASES

KINDS = ("none", "nudge", "displace", "drop")


STALE_APERTURE = 0.068
_DROP_MAX_STEPS = 40
_DROP_MIN_STEPS = 3
_FREE_JOINT = 0
_HOLD_SLACK = 0.03


_GRIP_MIN, _GRIP_MAX = 0.008, 0.066


def parse_at(spec):
    """Parse a perturbation trigger specification."""
    s = str(spec).strip().lower()
    for pre in ("step:", "replan:"):
        if s.startswith(pre):
            return pre[:-1], int(s[len(pre):]), 0
    if s.startswith("hold"):
        tail = s[len("hold"):]
        return "hold", None, int(tail) if tail else 0
    if s.startswith("stage:"):
        key, _, off = s[len("stage:"):].partition("+")
        if not key:
            raise SystemExit(f"--perturb_at {spec!r}: stage: needs an index or a name substring")
        return "stage", (int(key) if key.isdigit() else key), (int(off) if off else 0)
    raise SystemExit(f"--perturb_at {spec!r}: expected step:N, replan:N, hold+K, "
                     f"stage:S or stage:S+K")


def _body_of(env, name):
    """Resolve a grounding name to a robosuite body name."""
    names = list(env.sim.model.body_names)
    alias = _BODY_ALIASES.get(name, ())
    alias = (alias,) if isinstance(alias, str) else alias
    return next((c for c in (*alias, f"{name}_main", name) if c in names), None)


def _free_joint(env, name):
    """Return the free-joint addresses for a named body."""
    body = _body_of(env, name)
    if body is None:
        return None
    model = env.sim.model
    bid = model.body_name2id(body)
    adr, num = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
    for i in range(num):
        if int(model.jnt_type[adr + i]) == _FREE_JOINT:
            return int(model.jnt_qposadr[adr + i]), int(model.jnt_dofadr[adr + i])
    return None


def teleport(env, name, delta):
    """Offset a free body pose and clear its velocities."""
    joint = _free_joint(env, name)
    if joint is None:
        return None
    qadr, dadr = joint
    data = env.sim.data
    before = np.array(data.qpos[qadr:qadr + 3], dtype=np.float64)
    data.qpos[qadr:qadr + 3] = before + np.asarray(delta, dtype=np.float64)
    data.qvel[dadr:dadr + 6] = 0.0
    env.sim.forward()
    return before, np.array(data.qpos[qadr:qadr + 3], dtype=np.float64)


def _grounded(bridge, name):
    return next((o for o in bridge.grounding.objects if o.name == name), None)


def release_gate_open(bridge, stage, payload):
    """Return whether the current place-stage release gate is open."""
    if stage.place_point is None:
        return False
    obj = _grounded(bridge, payload)
    if obj is None:
        return False
    geom = bridge.geom
    seat = np.asarray(stage.place_point(), dtype=np.float64)
    pos = np.asarray(obj.pos(), dtype=np.float64)
    half = float(obj.extents[2]) if obj.extents is not None else 0.0
    dest_z = seat[2] + half + float(geom.get("place_release_clearance", 0.0))
    xy = float(np.linalg.norm(pos[:2] - seat[:2]))
    return xy < float(geom.get("release_xy", 0.06)) \
        and float(pos[2] - dest_z) < float(geom.get("release_z", 0.02))


def phys_held(env, bridge, payload):
    """Evaluate whether a payload is physically held."""
    obj = _grounded(bridge, payload)
    half = float(max(obj.extents)) if obj is not None and obj.extents is not None else 0.02
    dist = float(np.linalg.norm(np.asarray(env.object_pose(payload)[0], dtype=np.float64)
                                - np.asarray(env.tcp(), dtype=np.float64)))
    grip = float(env.gripper_q())
    return bool(dist <= _HOLD_SLACK + half and _GRIP_MIN < grip < _GRIP_MAX), dist


def expects_hold(bridge, payload):
    """Return whether the current stage expects the payload to remain held."""
    if payload is None:
        return False
    st = bridge.stage()
    if st.gripper == "close":
        return st.grasp_obj == payload
    if st.gripper == "hold":
        return st.payload == payload
    if st.gripper == "place":
        return st.payload == payload and not release_gate_open(bridge, st, payload)
    return False


class Perturbation:
    """Apply one disturbance and monitor recovery for the episode."""

    def __init__(self, kind, at, mag, obj, seed, movable):
        if kind not in KINDS or kind == "none":
            raise SystemExit(f"--perturb {kind!r}: expected one of {KINDS[1:]}")
        self.kind = kind
        self.at_spec = str(at)
        self.at_kind, self.at_key, self.at_off = parse_at(at)
        self.mag = float(mag)
        self.obj_override = None if obj in (None, "", "auto") else str(obj)
        self.movable = list(movable)
        self.rng = np.random.RandomState((int(seed) * 1000003 + KINDS.index(kind)) % (2 ** 31 - 1))
        self.event = None
        self._replan = -1
        self._hold_replan = None
        self._stage_entry = {}
        self._drop_left = 0
        self._drop_run = 0

        self._t0_step = None
        self._t0_stage = None
        self._t0_target = None
        self._payload = None
        self._lost_now = False
        self._prev_stage = None
        self.post_replans = 0
        self.post_steps = 0
        self.loss_events = 0
        self.release_events = 0
        self.backtracks_after = 0
        self.advances_after = 0
        self.stale_latch_after = 0
        self.stage_max_after = 0
        self.detect_signal = None
        self.detect_replans = None
        self.detect_steps = None
        self.latch_detect_signal = None
        self.latch_detect_replans = None
        self.latch_detect_steps = None
        self.reacquire_replans = None
        self.reacquire_steps = None
        self.phys_held_at_event = None
        self.phys_loss_replans = None
        self.phys_loss_steps = None


    def on_replan(self, step, env, bridge):
        """Advance the perturbation monitor at each replan."""
        self._replan += 1
        self._stage_entry.setdefault(bridge.stage_idx, self._replan)
        if self._hold_replan is None and bridge.world.held() is not None:
            self._hold_replan = self._replan
        if self.event is not None:
            self._observe(step, env, bridge)
            return None
        return self._fire(step, env, bridge) if self._due(bridge) else None

    def on_step(self, step, env, bridge):
        """Handle step triggers and physical hold sampling."""
        if self.event is None:
            if self.at_kind == "step" and step >= self.at_key:
                return self._fire(step, env, bridge)
            return None
        self._sample_phys(step, env, bridge)
        return None

    def _sample_phys(self, step, env, bridge):
        """Update physical loss and reacquisition metrics."""
        if self._payload is None:
            return
        rel_step = step - self._t0_step
        self.post_steps = max(self.post_steps, rel_step)
        in_hand, _ = phys_held(env, bridge, self._payload)
        if not in_hand and self.phys_loss_steps is None:
            self.phys_loss_steps = rel_step
            self.phys_loss_replans = self._replan - self.event["replan"]
        elif in_hand and self.phys_loss_steps is not None and self.reacquire_steps is None:
            self.reacquire_steps = rel_step
            self.reacquire_replans = self._replan - self.event["replan"]

    def filter_action(self, action, env, bridge):
        """Force the gripper open during a drop disturbance."""
        if self._drop_left <= 0:
            return action
        self._drop_left -= 1
        self._drop_run += 1
        out = np.array(action, dtype=np.float64, copy=True)
        out[7] = 0.0
        if self._drop_run >= _DROP_MIN_STEPS and self._payload is not None \
                and not phys_held(env, bridge, self._payload)[0]:
            self._drop_left = 0
        return out

    def record(self):
        """Return the perturbation event record."""
        if self.event is not None:
            return self.event
        return {"type": self.kind, "at": self.at_spec, "mag": self.mag, "fired": False,
                "reason": "trigger never became due"}

    def summary(self, success):
        """Return episode recovery metrics."""
        fired = self.event is not None
        applied = bool(fired and self.event.get("applied"))
        lost = self.phys_loss_replans is not None
        disturbed = applied and (lost if self._payload is not None else True)
        detected = self.detect_replans is not None
        latch_detected = self.latch_detect_replans is not None
        reacquired = self.reacquire_replans is not None
        lag = None
        if lost and self.latch_detect_replans is not None:
            lag = self.latch_detect_replans - self.phys_loss_replans
        return {
            "fired": fired, "applied": applied, "disturbed": disturbed,
            "detected": detected, "detect_signal": self.detect_signal,
            "detect_replans": self.detect_replans, "detect_steps": self.detect_steps,
            "latch_detected": latch_detected, "latch_detect_signal": self.latch_detect_signal,
            "latch_detect_replans": self.latch_detect_replans,
            "latch_detect_steps": self.latch_detect_steps,
            "detect_lag_replans": lag,
            "blind": bool(disturbed and not latch_detected),
            "reacquired": reacquired, "reacquire_replans": self.reacquire_replans,
            "reacquire_steps": self.reacquire_steps, "recover_steps": self.reacquire_steps,
            "success": bool(success),
            "recovered": bool(disturbed and reacquired and success),
            "lucky": bool(success and disturbed and not latch_detected),
            "payload_at_event": self._payload,
            "phys_held_at_event": self.phys_held_at_event,
            "phys_loss_replans": self.phys_loss_replans,
            "phys_loss_steps": self.phys_loss_steps,
            "payload_never_lost": (None if self._payload is None else not lost),
            "loss_events": self.loss_events, "release_events": self.release_events,
            "backtracks_after": self.backtracks_after, "advances_after": self.advances_after,
            "stale_latch_after": self.stale_latch_after,
            "stale_latch_frac": round(self.stale_latch_after / max(self.post_replans, 1), 4),
            "post_replans": self.post_replans, "post_steps": self.post_steps,
            "stage_at_event": self._t0_stage, "stage_max_after": self.stage_max_after,
        }


    def _due(self, bridge):
        if self.at_kind == "replan":
            return self._replan >= self.at_key
        if self.at_kind == "hold":
            return self._hold_replan is not None and self._replan >= self._hold_replan + self.at_off
        if self.at_kind == "stage":
            entry = self._entry_replan(bridge)
            return entry is not None and self._replan >= entry + self.at_off
        return False

    def _entry_replan(self, bridge):
        """Return the first replan that entered the configured stage."""
        if isinstance(self.at_key, int):
            return self._stage_entry.get(self.at_key)
        for idx, entry in sorted(self._stage_entry.items()):
            if self.at_key in bridge.grounding.stages[idx].name.lower():
                return entry
        return None

    def _resolve(self, bridge, held):
        """Resolve the object targeted by the disturbance."""
        if self.obj_override:
            return self.obj_override
        st = bridge.stage()
        if self.kind in ("nudge", "drop"):
            return held or st.payload
        for cand in (st.place_target, st.grasp_obj, st.payload):
            if cand and cand in self.movable and cand != held:
                return cand
        return None


    def _fire(self, step, env, bridge):
        st = bridge.stage()
        held = bridge.world.held()
        target = self._resolve(bridge, held)
        try:
            t0_target = np.asarray(st.target(), dtype=np.float64)
        except Exception:
            t0_target = None
        ev = {"type": self.kind, "at": self.at_spec, "mag": self.mag, "fired": True,
              "step": step, "replan": self._replan, "obj": target,
              "stage_idx": bridge.stage_idx, "stage": st.name, "held": held,
              "gripper_read": round(float(env.gripper_q()), 4),
              "tcp": np.round(env.tcp(), 4).tolist(),
              "expects_hold": bool(expects_hold(bridge, held))}
        if self.kind == "drop":
            if held is None:
                ev.update(applied=False, reason="no payload held")
            else:
                self._drop_left, self._drop_run = _DROP_MAX_STEPS, 0
                pose = np.round(env.object_pose(held)[0], 4).tolist()
                ev.update(applied=True, pose_before=pose, pose_after=pose)
        elif target is None:
            ev.update(applied=False, reason="no perturbable object for this stage")
        else:
            azimuth = float(self.rng.uniform(0.0, 2.0 * np.pi))
            delta = self.mag * np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            moved = teleport(env, target, delta)
            if moved is None:
                ev.update(applied=False, reason=f"{target} has no free joint (fixture)")
            else:
                ev.update(applied=True, azimuth=round(azimuth, 4),
                          delta=np.round(delta, 4).tolist(),
                          pose_before=np.round(moved[0], 4).tolist(),
                          pose_after=np.round(moved[1], 4).tolist())
        self.event = ev
        self._arm(step, env, bridge, held, t0_target)
        return ev

    def _arm(self, step, env, bridge, held, t0_target):
        """Initialize the post-event recovery monitor."""
        self._t0_step = step
        self._t0_stage = bridge.stage_idx
        self._prev_stage = bridge.stage_idx
        self.stage_max_after = bridge.stage_idx
        self._t0_target = t0_target
        self._payload = held if self.kind in ("nudge", "drop") else None
        if self._payload is not None:
            in_hand, dist = phys_held(env, bridge, self._payload)
            self.phys_held_at_event = bool(in_hand)
            self.event["phys_dist"] = round(dist, 4)


    def _observe(self, step, env, bridge):
        rel = self._replan - self.event["replan"]
        self.post_replans += 1
        self.post_steps = max(self.post_steps, step - self._t0_step)
        held = bridge.world.held()
        stage = bridge.stage_idx
        if stage < self._prev_stage:
            self.backtracks_after += 1
        elif stage > self._prev_stage:
            self.advances_after += 1
        self._prev_stage = stage
        self.stage_max_after = max(self.stage_max_after, stage)
        if held is not None and float(env.gripper_q()) >= STALE_APERTURE:
            self.stale_latch_after += 1
        if self._payload is not None:
            lost = held != self._payload
            if lost and not self._lost_now:
                if expects_hold(bridge, self._payload):
                    self.loss_events += 1
                else:
                    self.release_events += 1
            self._lost_now = lost
        signal = self._detect(bridge, held)
        if signal is not None:
            if self.detect_replans is None:
                self.detect_signal, self.detect_replans = signal, rel
                self.detect_steps = self.post_steps
            if signal != "backtrack" and self.latch_detect_replans is None:
                self.latch_detect_signal, self.latch_detect_replans = signal, rel
                self.latch_detect_steps = self.post_steps
        if self._payload is None and self.reacquire_replans is None \
                and self._stage_restored(bridge):
            self.reacquire_replans, self.reacquire_steps = rel, self.post_steps

    def _detect(self, bridge, held):
        """Return the first post-event signal observed by the controller."""
        if self._payload is not None:
            if held != self._payload and expects_hold(bridge, self._payload):
                return "held_lost"
            if bridge.sensor is not None and bridge.sensor.closed_on_air() \
                    and expects_hold(bridge, self._payload):
                return "closed_empty"
            if self.backtracks_after > 0 and self.phys_loss_replans is not None:
                return "backtrack"
            return None
        if self._t0_target is not None:
            try:
                now = np.asarray(bridge.stage().target(), dtype=np.float64)
            except Exception:
                return None
            if float(np.linalg.norm(now - self._t0_target)) >= 0.5 * self.mag:
                return "target_moved"
        return None

    def _stage_restored(self, bridge):
        """Return whether a non-held disturbance has been recovered."""
        if bridge.stage_idx > self._t0_stage:
            return True
        return bool(bridge.stage_idx == self._t0_stage and bridge.stage().done())
