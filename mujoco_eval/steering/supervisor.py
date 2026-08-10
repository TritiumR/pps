"""Supervisory composition over the expert proxy, with an optional base fallback.

Entirely opt-in: nothing in this module runs unless `--supervisor on` is passed, and the
`--supv_base_fallback` arm is a second, independent switch. No sampler, no cost and no
action is modified anywhere in here -- the supervisor only decides, at chunk boundaries,

  * which controller drives the next chunk (the expert, or the MBD base), and
  * whether the expert should be asked for a FRESH sample of its own distribution.

The retry mechanism is deliberately the weakest intervention that is still an intervention:
the expert's sampling context is reset and its RNG substream advanced, then it re-plans from
the current observation. Same policy, same physical state, no extra task information, no
modification of the chunk it returns.

Milestone predicates are read from GT simulator state and the aperture hold latch. The base's
aggregate cost is never consulted: it was measured as an inverted judge of working behaviour.
"""

from __future__ import annotations

import numpy as np


PHASES = ("grasp", "carry", "align")

# Bridge ladder stage that owns each phase's milestone. Used to pin the base to the ONE stage
# whose milestone broke during a recovery, rather than letting it run the whole task.
STAGE_OF_PHASE = {"grasp": 0, "carry": 1, "align": 2}

# Square task geometry, verified against the live MuJoCo model (see the report):
#   peg1 body xpos      = (0.230, 0.100, 0.850)     robosuite on_peg() reference
#   peg top             = (0.230, 0.100, 0.950)     grounding/gt.PEG_TOP
#   Square_D0 nut init  x in (-0.115, -0.110), y in (0.110, 0.225)   placement sampler
PEG_XY = (0.230, 0.100)
PEG_TOP_Z = 0.950
NUT_INIT_X = (-0.115, -0.110)
NUT_INIT_Y = (0.110, 0.225)
TABLE_TOP_Z = 0.820

# Improvement smaller than this does not reset the stall window: it is sensor/contact wobble,
# not progress. 2 mm is well under the 30 mm success tolerance and over the settled noise floor.
PROGRESS_EPS = 0.002
# "Stationary" for the handback test, per boundary.
STATIONARY_EPS = 0.003
# Substream stride for a retry. Large and prime so a retry's seed can never collide with the
# seed any later replan of this episode would have drawn on its own.
RETRY_STRIDE = 7_000_003


def workspace_box(pad):
    """Axis-aligned task workspace: the D0 nut init hull UNION the peg, padded.

    The init hull alone is NOT the workspace -- the nut has to travel to the peg to succeed, so
    a box drawn round the init region would flag every correct carry as out-of-bounds. The
    union is the region the task is defined over; leaving it means the nut was bulldozed or
    swept off, which is exactly the unrecoverable-by-retry state escalation is for.
    """
    xs = (NUT_INIT_X[0], NUT_INIT_X[1], PEG_XY[0])
    ys = (NUT_INIT_Y[0], NUT_INIT_Y[1], PEG_XY[1])
    return {"x": (min(xs) - pad, max(xs) + pad),
            "y": (min(ys) - pad, max(ys) + pad),
            "z_min": TABLE_TOP_Z - pad}


class Milestones:
    """Per-boundary milestone predicates for square, from GT state and the hold latch."""

    def __init__(self, env, bridge, *, lift_m, over_peg_m, ws_pad):
        self.env = env
        self.bridge = bridge
        self.lift_m = float(lift_m)
        self.over_peg_m = float(over_peg_m)
        self.box = workspace_box(float(ws_pad))
        self.nut_z0 = float(self._nut()[2])

    def _nut(self):
        return np.asarray(self.env.object_pose("nut")[0], dtype=np.float64)

    def _handle(self):
        """The grasp keypoint the GT grounding publishes (nut handle, world frame)."""
        kp = self.bridge.grounding.keypoints
        if kp is None:
            return self._nut()
        return np.asarray(kp(), dtype=np.float64)[0]

    def read(self):
        """Snapshot every predicate the supervisor is allowed to look at."""
        nut = self._nut()
        tcp = np.asarray(self.env.tcp(), dtype=np.float64)
        held = self.bridge.world.held() == "nut"
        peg_xy = float(np.linalg.norm(nut[:2] - np.asarray(PEG_XY)))
        box = self.box
        in_box = bool(box["x"][0] <= nut[0] <= box["x"][1]
                      and box["y"][0] <= nut[1] <= box["y"][1]
                      and nut[2] >= box["z_min"])
        return {
            "nut": nut, "tcp": tcp, "handle": self._handle(),
            "held": held,
            "lifted": bool(nut[2] > self.nut_z0 + self.lift_m),
            "over_peg": bool(peg_xy < self.over_peg_m),
            "peg_xy_m": peg_xy,
            "tcp_nut_m": float(np.linalg.norm(tcp - self._handle())),
            "in_box": in_box,
            "success": bool(self.env.success()),
        }

    def residual(self, phase, m):
        """Distance-to-milestone for `phase`, in metres. Lower is progress.

        Piecewise by phase, and offset so that acquiring the hold is always an improvement over
        any approach distance -- otherwise the window would reset on approach wobble alone.
        """
        if phase == "grasp":
            if not m["held"]:
                return 1.0 + m["tcp_nut_m"]
            return max(0.0, (self.nut_z0 + self.lift_m) - float(m["nut"][2]))
        if phase == "carry":
            return m["peg_xy_m"]
        return m["peg_xy_m"] + max(0.0, float(m["nut"][2]) - PEG_TOP_Z)


class Supervisor:
    """Phase machine over chunk boundaries: advance, regress, retry, escalate, hand back."""

    def __init__(self, *, milestones, base_fallback, stall_w, dwell, max_retry_phase,
                 max_retry_episode, base_max_chunks, max_recoveries):
        self.m = milestones
        self.base_fallback = bool(base_fallback)
        self.stall_w = int(stall_w)
        self.dwell = int(dwell)
        self.max_retry_phase = int(max_retry_phase)
        self.max_retry_episode = int(max_retry_episode)
        self.base_max_chunks = int(base_max_chunks)
        self.max_recoveries = int(max_recoveries)

        self.drive = "expert"
        self.phase = "grasp"
        self.chunk = 0
        self.events = []
        self.pin_stage = None

        self._best = None
        self._since_improve = 0
        self._last_event_chunk = -10_000
        self._retries_phase = 0
        self._retries_episode = 0
        self._retry_fail_streak = 0
        self._retry_pending = False
        self._base_chunks = 0
        self._recoveries = 0
        self._prev_nut = None
        self._recovery_open = None
        self.retry_salt = 0
        self._probe_due = False
        self._probe_seed = None

        self.tally = {"advance": 0, "regress": 0, "retry": 0, "retry_blocked": 0,
                      "escalate": 0, "handback": 0, "recovery_timeout": 0,
                      "stall_exhausted": 0}

    # -- bookkeeping -------------------------------------------------------------------

    def _log(self, kind, step, m, **extra):
        rec = {"kind": "supv", "event": kind, "step": step, "chunk": self.chunk,
               "phase": self.phase, "drive": self.drive,
               "held": bool(m["held"]), "lifted": bool(m["lifted"]),
               "over_peg": bool(m["over_peg"]), "in_box": bool(m["in_box"]),
               "nut": np.round(m["nut"], 4).tolist(),
               "tcp": np.round(m["tcp"], 4).tolist(),
               "peg_xy_m": round(m["peg_xy_m"], 4),
               "tcp_nut_m": round(m["tcp_nut_m"], 4),
               "since_improve": self._since_improve,
               "retries_phase": self._retries_phase,
               "retries_episode": self._retries_episode,
               "recoveries": self._recoveries}
        rec.update(extra)
        self.tally[kind] = self.tally.get(kind, 0) + 1
        self.events.append(rec)
        return rec

    def _enter_phase(self, phase, step, m, why):
        kind = "advance" if PHASES.index(phase) > PHASES.index(self.phase) else "regress"
        prev, self.phase = self.phase, phase
        self._best = None
        self._since_improve = 0
        self._retries_phase = 0
        self._retry_fail_streak = 0
        rec = self._log(kind, step, m, frm=prev, to=phase, why=why)
        self._last_event_chunk = self.chunk
        return rec

    def _dwell_ok(self):
        return self.chunk - self._last_event_chunk >= self.dwell

    # -- the boundary hook -------------------------------------------------------------

    def boundary(self, step, steer, bridge):
        """Run one supervision step. Returns the records produced at this boundary."""
        m = self.m.read()
        out = []
        self.chunk += 1

        if m["success"]:
            self._prev_nut = m["nut"]
            return out

        # Phase estimation first: retries and escalations are defined relative to the phase
        # whose milestone is actually outstanding.
        out += self._transition(step, m)

        if self.drive == "base":
            out += self._supervise_base(step, m, steer, bridge)
        else:
            out += self._supervise_expert(step, m, steer, bridge)

        self._prev_nut = m["nut"]
        return out

    def _transition(self, step, m):
        """Advance on the phase's predicate, regress when a prerequisite goes false."""
        out = []
        if self.phase == "grasp" and m["held"] and m["lifted"]:
            out.append(self._enter_phase("carry", step, m, "grasped+lifted"))
        elif self.phase == "carry" and not m["held"]:
            out.append(self._enter_phase("grasp", step, m, "grasp lost"))
        elif self.phase == "carry" and m["over_peg"]:
            out.append(self._enter_phase("align", step, m, "over peg"))
        elif self.phase == "align" and not m["over_peg"]:
            out.append(self._enter_phase("carry", step, m, "left peg xy"))
        return out

    def _progress(self, m):
        """Update the stall window; True when the phase residual improved."""
        r = self.m.residual(self.phase, m)
        if self._best is None or r < self._best - PROGRESS_EPS:
            self._best = r if self._best is None else min(self._best, r)
            self._since_improve = 0
            return True
        self._best = min(self._best, r)
        self._since_improve += 1
        return False

    def _supervise_expert(self, step, m, steer, bridge):
        out = []
        improved = self._progress(m)
        if improved and self._retry_pending:
            # The last retry produced measurable progress: it worked.
            self._retry_pending = False
            self._retry_fail_streak = 0
        if self._since_improve < self.stall_w or not self._dwell_ok():
            return out

        if self._retry_pending:                 # the previous retry stalled again
            self._retry_pending = False
            self._retry_fail_streak += 1

        out_of_box = not m["in_box"]
        can_escalate = (self.base_fallback and self._recoveries < self.max_recoveries)
        want_escalate = out_of_box or self._retry_fail_streak >= 2

        if can_escalate and want_escalate:
            out.append(self._escalate(step, m, bridge,
                                      "nut out of workspace box" if out_of_box
                                      else "two consecutive retry failures"))
        elif (self._retries_phase < self.max_retry_phase
              and self._retries_episode < self.max_retry_episode):
            out.append(self._retry(step, m, steer))
        elif can_escalate:
            out.append(self._escalate(step, m, bridge, "retry budget exhausted"))
        else:
            out.append(self._log("stall_exhausted", step, m,
                                 reason="retry budget exhausted, no base authority available"))
            self._last_event_chunk = self.chunk
        # Restart the window either way, so one stall produces one event.
        self._since_improve = 0
        self._best = self.m.residual(self.phase, m)
        return out

    def _retry(self, step, m, steer):
        """STRICT fresh sample: reset the sampling context, advance the RNG substream."""
        # The seed the imminent replan would have used WITHOUT this retry, kept so the probe can
        # measure how far the fresh sample actually moved. Diagnostic only.
        self._probe_seed = steer.base_seed * 100003 + steer.replan_idx + self.retry_salt
        self._probe_due = True
        self.retry_salt += RETRY_STRIDE
        steer.retry_salt = self.retry_salt
        steer._x0_real = None                    # drop the cached chain; nothing else is state
        self._retries_phase += 1
        self._retries_episode += 1
        self._retry_pending = True
        self._last_event_chunk = self.chunk
        return self._log("retry", step, m, retry_salt=self.retry_salt,
                         residual=round(self.m.residual(self.phase, m), 4))

    def _escalate(self, step, m, bridge, why):
        """Hand authority to the base, pinned to the ladder stage that owns this milestone."""
        self.drive = "base"
        self._base_chunks = 0
        self._recoveries += 1
        self.pin_stage = STAGE_OF_PHASE[self.phase]
        pin_bridge_stage(bridge, self.pin_stage)
        self._last_event_chunk = self.chunk
        self._recovery_open = {"phase": self.phase, "step": step,
                               "nut_before": np.round(m["nut"], 4).tolist(),
                               "held_before": bool(m["held"]),
                               "residual_before": round(self.m.residual(self.phase, m), 4)}
        return self._log("escalate", step, m, why=why, pin_stage=self.pin_stage,
                         recovery_index=self._recoveries)

    def _supervise_base(self, step, m, steer, bridge):
        """Watch the recovery and hand back the moment the state is recoverable."""
        out = []
        self._base_chunks += 1
        moved = (float(np.abs(m["nut"] - self._prev_nut).max())
                 if self._prev_nut is not None else 1.0)
        stationary = moved < STATIONARY_EPS

        if m["held"] and m["lifted"]:
            out.append(self._handback(step, m, steer, bridge, "carry", "base acquired grasp+lift"))
        elif stationary and m["in_box"] and not m["held"] and self._dwell_ok():
            out.append(self._handback(step, m, steer, bridge, "grasp",
                                      "nut stationary, in box, gripper clear"))
        elif self._base_chunks >= self.base_max_chunks:
            rec = self._handback(step, m, steer, bridge, "grasp", "recovery timeout")
            rec["timeout"] = True
            self.tally["recovery_timeout"] += 1
            out.append(rec)
        return out

    def _handback(self, step, m, steer, bridge, phase, why):
        """Return authority to the expert, at `phase`, on a fresh expert sample."""
        self.drive = "expert"
        self.pin_stage = None
        self.retry_salt += RETRY_STRIDE
        steer.retry_salt = self.retry_salt
        steer._x0_real = None
        prev = self.phase
        self.phase = phase
        self._best = None
        self._since_improve = 0
        self._retries_phase = 0
        self._retry_fail_streak = 0
        self._retry_pending = False
        self._last_event_chunk = self.chunk
        rec = self._log("handback", step, m, why=why, frm=prev, to=phase,
                        base_chunks=self._base_chunks,
                        recovery_index=self._recoveries)
        if self._recovery_open is not None:
            rec["recovery"] = dict(self._recovery_open,
                                   nut_after=np.round(m["nut"], 4).tolist(),
                                   held_after=bool(m["held"]),
                                   residual_after=round(self.m.residual(phase, m), 4),
                                   base_chunks=self._base_chunks)
            self._recovery_open = None
        return rec

    def retry_probe(self, steer, env, num_iterations, plan):
        """How far the fresh sample moved from the one the retry replaced. Diagnostic only.

        Re-runs the proxy's chain at the seed the replan WOULD have used, at the same
        observation. The service is stateless in its seed, so this consumes no RNG the rollout
        depends on and cannot change the executed plan.
        """
        if not self._probe_due:
            return None
        self._probe_due = False
        try:
            table = env.rgb("agentview", hw=224)
            wrist = env.rgb("robot0_eye_in_hand", hw=224)
            q0 = env.q0().numpy().astype(np.float32)
            grip = float(np.clip(env.gripper_q() / 0.080, 0.0, 1.0))
            old, _ = steer.client.chain(seed=int(self._probe_seed),
                                        num_iterations=int(num_iterations),
                                        joint_pos=q0, gripper_pos=grip, table=table, wrist=wrist)
            n = int(np.asarray(plan).shape[0])
            ref = np.asarray(old[-1][:n], dtype=np.float64)[:, :7]
            cur = np.asarray(plan, dtype=np.float64)[:n, :7]
            return {"supv_retry_dq_max": round(float(np.abs(ref - cur).max()), 5),
                    "supv_retry_dq_mean": round(float(np.abs(ref - cur).mean()), 5)}
        except Exception:
            return None                        # a diagnostic must never cost the rollout

    def summary(self):
        return {"phase_final": self.phase, "drive_final": self.drive,
                "chunks": self.chunk, "tally": dict(self.tally),
                "retries_episode": self._retries_episode,
                "recoveries": self._recoveries,
                "base_fallback": self.base_fallback,
                "events": self.events}


def pin_bridge_stage(bridge, stage_idx):
    """Force the ladder onto one stage, so a recovery pursues the broken milestone only."""
    from vlm_dp.stage import _capture_held
    if bridge.stage_idx == stage_idx:
        return
    bridge.stage_idx = int(stage_idx)
    bridge.held_offset = _capture_held(bridge.env, bridge.grounding, bridge.stage().held_idx)
    bridge._enter_stage()


def build(args, env, bridge):
    """Construct the supervisor for one episode, or None when the flag is off."""
    if getattr(args, "supervisor", "off") != "on":
        return None
    m = Milestones(env, bridge, lift_m=args.supv_lift_m, over_peg_m=args.supv_over_peg_m,
                   ws_pad=args.supv_workspace_pad)
    return Supervisor(milestones=m,
                      base_fallback=getattr(args, "supv_base_fallback", "off") == "on",
                      stall_w=args.supv_stall_w, dwell=args.supv_dwell,
                      max_retry_phase=args.supv_max_retry_phase,
                      max_retry_episode=args.supv_max_retry_episode,
                      base_max_chunks=args.supv_base_max_chunks,
                      max_recoveries=args.supv_max_recoveries)
