"""Bridge VLM grounding and CompositeCost into the sim_free_mpc planner."""
from __future__ import annotations

import json
import types

import numpy as np
import torch

from sim_common.envs.droid import DroidEnv
from vlm_dp.sim_helpers import ROBOTIQ_GRASP_OFFSET
from vlm_dp.grasp_sensor import ApertureGraspSensor
from vlm_dp.grounding import get_source
from vlm_dp.grounding.predicates import HistoryBuffer
from vlm_dp.world import GTWorld, SensedWorld
from vlm_dp.cost.base_cost import CompositeCost
from vlm_dp.cost.terms import grasp_slack
from vlm_dp.hold import HoldLatch, payload_held
from vlm_dp.stage import _capture_held, _should_advance
from vlm_dp.cost import guard_cost
from vlm_dp.context import build_context
from vlm_dp.grasp_recovery import debounce_gripper, descent_stalled, probe_pattern

# Planner styles that pass tcp_pos instead of ee_pos.
_TCP_STYLES = ("ref_style", "explore", "grasp_flow", "capsule_flow")


class _LatchWorld:
    """A world stand-in whose only job is to answer ``held()`` from a real HoldLatch.

    Used by the advanceability preflight so that the hold branch of ``_stage_reached`` resolves
    through the SAME latch, sensor and grasp points a rollout uses, over the synthesized
    satisfying state. The predecessor of this class was a stub that answered "yes, the nearest
    object is held" unconditionally, which certified as advanceable exactly the stage that could
    never advance: the capsule lid grasp, whose real hold test resolved against a centroid belief
    0.25-0.30m from the rim the fingers were on and therefore never fired.
    """

    def __init__(self, latch):
        self._latch = latch

    def held(self):
        return self._latch.held()


def _opt_float(value):
    """Convert a configured value to float while preserving None."""
    return None if value is None else float(value)


class VlmDpBridge:
    """Connect grounding-derived state and costs to the MPC planner."""

    def __init__(self, ground, roles, cost_cfg, *, task_key=None, device="cuda:0",
                 commit_hold=5, lift_tol=0.01, grasp_confirm=10, rise_confirm=0.01,
                 grasp_eps=0.02, state="gt", track="fk", segment="groundedsam",
                 vocab=None, fixtures=(), reperceive_every=32):
        """Configure grounding, sensing, stage advancement, and recovery."""
        self.ground_name = ground
        self.state = state
        self.track = track
        self.segment = segment
        self.vocab = dict(vocab) if vocab else {}
        self.fixtures = tuple(fixtures)
        self.reperceive_every = int(reperceive_every)
        self.roles = dict(roles)
        self.terms = cost_cfg["cost"]["terms"]
        self.geom = cost_cfg["cost"]["geometry"]
        self._geom_ns = types.SimpleNamespace(**self.geom)  # attribute access for grasp_slack
        self.task_key = task_key
        self.device = device
        self.commit_hold = int(commit_hold)
        self.lift_tol = float(lift_tol)
        self.grasp_confirm = int(grasp_confirm)
        self.rise_confirm = float(rise_confirm)
        self.grasp_eps = float(grasp_eps)
        # Standard advance settings and optional ablations.
        adv = cost_cfg.get("advance", {})
        self.flag_fallback = bool(adv.get("flag_fallback", True))
        self.backtrack_enabled = bool(adv.get("backtrack", True))
        self.advance_mode = adv.get("mode", "sensed")               # sensed | env_flags
        # Search recovery probes nearby poses after a closed-empty grasp.
        self._grasp_recovery = adv.get("grasp_recovery", "reopen")
        self._probe_pts = probe_pattern(float(adv.get("grasp_search_radius", 0.045)))
        # Bounds on the reopen recovery, so it cannot hold the gripper open for an entire episode
        # when the fingers physically cannot reach the open threshold (see _reopen_escape).
        self._reopen_max = int(adv.get("reopen_max_replans", 12))
        self._reopen_stall = int(adv.get("reopen_stall_replans", 4))
        self._reopen_stall_eps = float(adv.get("reopen_stall_eps", 0.01))
        self._reopen_travel = float(adv.get("reopen_min_travel", 0.02))
        self._reopen_cooldown = int(adv.get("reopen_cooldown_replans", 8))
        self.stall_margin = float(adv.get("stall_margin", 0.15))
        self._thin_feature_contact_ratio = float(adv.get("thin_feature_contact_ratio", 0.5))
        self._thin_feature_width_band = float(
            adv.get("thin_feature_width_band", 0.08))
        # Once a thin grasp is certified, use a looser width band plus a short debounce while
        # lifting. This rejects a true free close without treating load-induced motion as air.
        self._thin_feature_loss_ratio = float(adv.get("thin_feature_loss_ratio", 0.8))
        self._thin_feature_loss_grace = int(adv.get("thin_feature_loss_grace_replans", 2))
        # Optional hold hysteresis, grace, and backtrack limits.
        self.hold_enter = _opt_float(adv.get("hold_enter"))
        self.hold_exit = _opt_float(adv.get("hold_exit"))
        self._hold_grace_replans = int(adv.get("hold_grace_replans", 0))
        self._backtrack_budget = int(adv.get("backtrack_budget", 0))  # 0 disables the limit
        self._backtrack_commit = int(adv.get("backtrack_commit_replans", 0))
        self._release_debug = bool(adv.get("release_debug", False))
        # Failure thresholds use demonstration statistics; no stats disables the gate.
        steer_cfg = cost_cfg.get("steering", {}) or {}
        self._gate_dwell_k = float(steer_cfg.get("dwell_k", 2.0))
        self._gate_stats = None
        _stats_path = steer_cfg.get("gate_stats")
        if _stats_path:
            try:
                with open(_stats_path, encoding="utf-8") as fh:
                    self._gate_stats = json.load(fh)
            except OSError:
                print(f"[vlm_dp] gate_stats unreadable ({_stats_path}); failure gate stays closed",
                      flush=True)
        # Allow a confirmed hold to bypass noisy target proximity.
        self._grasp_advance_on_hold = bool(adv.get("grasp_advance_on_hold", False))
        # Opt-in: PLAN-AUTHORITATIVE stage transitions. For every stage the plan itself authored
        # completion evidence for (a sub-goal constraint or a stage<N>_completion predicate), that
        # evidence is the ONLY thing allowed to advance the stage. No hardcoded geometric test
        # second-guesses the plan's stage semantics, and in particular the text-order target()
        # heuristic (rekep._vlm_stages derives a move stage's target from the FIRST keypoint
        # mentioned in the constraint source) never feeds an advance decision: under the one-sided
        # lift rewrite that made the hold-branch z-test compare a payload's belief CENTRE against a
        # SURFACE keypoint of the same object, a permanent stall (8/10 seeds sat in the lift stage
        # 380+ replans with the plan's own sub-goal reading 0.000 satisfied; motion_pred_go).
        # Absent -> strict no-op: every branch keeps its shipped rule.
        self._plan_authoritative = bool(adv.get("plan_authoritative", False))
        # Opt-in: certify PLACE stage completion with the plan's completion predicate (hand let
        # go AND object in the place region AND quiescent) instead of the scalar sub-goal test.
        # Absent -> strict no-op: _stage_reached keeps the seated/settled/released rule, and the
        # predicates stay the shadow signal they were. plan_authoritative folds this in: a place
        # stage's plan-authored evidence IS its completion predicate.
        self._pred_place_transitions = bool(adv.get("predicate_place_transitions", False)) \
            or self._plan_authoritative
        # Advanceability preflight (rekep.advance_preflight, once per episode). On by default;
        # the escape hatch exists so a diagnostic run can be made in spite of a refusal.
        self._advance_preflight = bool(adv.get("advance_preflight", True))
        # Re-perceive after repeated closed-empty grasps.
        self._regrasp_perceive = bool(adv.get("regrasp_perceive", False))
        # Allow stale beliefs to bypass appearance matching.
        self._relax_identity_when_stale = bool(adv.get("relax_identity_when_stale", False))
        self._ground_err_debug = bool(adv.get("ground_error_debug", False))
        self._ground_err_every = int(adv.get("ground_error_every", 5))
        # Suppress brief open commands while holding; zero disables the filter.
        self._grip_debounce = int(adv.get("gripper_open_debounce", 0))
        self._hold_gripper_authoritative = bool(
            adv.get("hold_gripper_authoritative", False))
        # Contact detection from stalled descent.
        self._stall_replans = int(adv.get("stall_replans", 3))
        self._stall_eps = float(adv.get("stall_eps", 0.003))
        # Placement settling may be shortened for release-on-stall.
        self._place_settle = int(adv.get("place_settle", self._PLACE_SETTLE))
        self.settle_eps = float(adv.get("settle_eps", 0.01))
        # "replan" restores one sensor sample per chunk.
        self.sensor_cadence = str(adv.get("sensor_cadence", "step"))
        # "latched" preserves hysteresis; "instant" re-evaluates each call.
        self.hold_authority = str(adv.get("hold_authority", "latched"))
        # Restore the pre-2026-07-30 sensor and world behavior.
        self.legacy_sensor = bool(adv.get("legacy_sensor", False))
        if self.legacy_sensor:
            self.hold_authority = "instant"
        # Visual-tracker jump limit in metres per control step.
        self.jump_rate = adv.get("jump_rate", 0.15 if adv.get("legacy_sensor") else None)
        self.jump_rate = float(self.jump_rate) if self.jump_rate is not None else None
        self.close_steps = int(adv.get("close_steps", 12))
        self.settle_steps = int(adv.get("settle_steps", 12))
        # A thin contact can enter the generic air band before the aperture has settled.
        self._thin_feature_settle_max_steps = int(adv.get(
            "thin_feature_settle_max_steps", 2 * (max(self.close_steps, self.settle_steps) + 1)
        ))
        self.seat_shift = bool(cost_cfg.get("grounding", {}).get("seat_shift", True))
        self.local_grasp = bool(cost_cfg.get("grounding", {}).get("local_grasp", False))
        self.local_grasp_radius = float(cost_cfg.get("grounding", {}).get("local_grasp_radius", 0.05))
        self.rotate_grasp_offset = bool(cost_cfg.get("grounding", {}).get("rotate_grasp_offset", False))
        self.lift_latch_xy = bool(cost_cfg.get("grounding", {}).get("lift_latch_xy", False))
        self.seat_from_plane = bool(cost_cfg.get("grounding", {}).get("seat_from_plane", False))
        self.support_extents = bool(cost_cfg.get("grounding", {}).get("support_extents", False))
        self.kp_source = cost_cfg.get("grounding", {}).get("kp_source", "perception")
        self.subgoal_eps = float(cost_cfg.get("grounding", {}).get("subgoal_eps", 0.06))
        # "feasibility" uses geometry; "plan" uses stage structure.
        self.contact_criterion = cost_cfg.get("grounding", {}).get("contact_criterion", "feasibility")
        if self.state == "real" and (self.flag_fallback or self.advance_mode == "env_flags"):
            print("[vlm_dp] WARNING: --vlm_state real with env-flag advance (flag_fallback or env_flags) "
                  "lets privileged flags co-confirm stages. Use the flag-free advance config for an "
                  "honest sensed-state run.", flush=True)
        # Episode state.
        self.env = None
        self.world = None
        self.grounding = None
        self.stage_idx = 0
        self.held_offset = None
        self.plan_ref = None
        self.sensor = None
        self.stage_replans = 0
        self.grasp_z0 = {}
        self._place_seen = None
        self._place_since = None
        self._reopen = False
        self._reopen_ap = []
        self._reopen_cooldown_left = 0
        self._hold_world = None
        self._grasp_probe = np.zeros(3)
        self._grasp_probe_idx = 0
        self._closed_empty = 0
        self._open_run = 0
        self._close_val = 1.0
        self._z_hist = []                    # Never stage-reset; contact detection spans stage changes.
        # Measured TCP one control step back, for carry_accel's chunk-boundary rows. Sampled
        # per applied step only when a config actually asks for the term, so every other
        # config pays nothing and sees no new context key.
        self._track_eef_hist = "carry_accel" in self.terms
        self._eef_last = None
        # Payload age for carry_accel's settle window (same gating as eef_hist: configs that
        # do not ask for the term see no new context key). Counts env steps since the CURRENT
        # payload was acquired, i.e. since the stage that first names it took over.
        self._env_steps = 0
        self._payload_name = None
        self._payload_acq = None
        self._contact_prev = False
        self._last_cmd_close = False
        # SHADOW completion predicates. One second of belief / TCP / aperture history, evaluated
        # for the CURRENT stage on every replan and logged beside the scalar sub-goal decision.
        # Nothing in advance() reads pred_shadow: this is observation only, and a run with it on
        # must reproduce exactly the episode a run without it would.
        self._pred_hist = HistoryBuffer(maxlen=15, dt=1.0 / 15.0)
        self.pred_shadow = None
        self._pred_transition = None      # last predicate-certified transition decision, logged
        self._reset_churn()

    def _reset_churn(self):
        """Reset episode-level churn counters without refunding backtracks."""
        self._hold_grace = 0
        self._backtracks_run = 0
        self._commit_left = 0
        self._stage_high = 0

    def attach_cost(self, mpc):
        """Replace the planner cost with the guarded composite cost."""
        style = mpc.config.cost_style
        if style != "priority" or style in _TCP_STYLES:
            raise ValueError(
                f"--vlm_cost requires --mpc_cost priority, got cost_style={style!r}. CompositeCost "
                f"has no tcp_pos= parameter, and sim_free_mpc/planner.py calls {_TCP_STYLES} "
                f"with tcp_pos=.")
        mpc.cost = guard_cost(CompositeCost(self.terms, self.geom))

    _SETTLE_SECONDS = 1.0   # Allow objects to settle after reset.

    def _settle(self, raw_env):
        """Advance physics and rendering until the reset scene is stable."""
        for _ in range(int(round(self._SETTLE_SECONDS / raw_env.physics_dt))):
            raw_env.sim.step(render=False)
            raw_env.scene.update(dt=raw_env.physics_dt)
        for _ in range(3):                  # Render extra frames to refresh camera annotators.
            raw_env.sim.render()
            raw_env.scene.update(dt=raw_env.physics_dt)

    def reset(self, raw_env):
        """Rebind the environment and rebuild grounding after reset."""
        self._settle(raw_env)
        try:
            # Re-seat the pot lid after reset because reset-time writes are discarded.
            from pot_scene_fix import seat_pot_lid
            jp = raw_env.scene["robot"].data.joint_pos[0, :7]
            hold = torch.cat([jp, jp.new_zeros(1)])
            if seat_pot_lid(raw_env, hold):
                print("[vlm_dp] pot lid re-seated (pot_scene_fix)", flush=True)
                self._settle(raw_env)
        except ImportError:
            pass
        self._eef_last = None                # no cross-episode TCP history
        self._env_steps = 0                  # no cross-episode payload age
        self._payload_name = None
        self._payload_acq = None
        self.env = DroidEnv.attach(raw_env, device=self.device)
        self.sensor = ApertureGraspSensor(stall_margin=self.stall_margin, settle_eps=self.settle_eps,
                                          close_steps=self.close_steps, settle_steps=self.settle_steps,
                                          legacy=self.legacy_sensor,
                                          stall_margin_enter=self.hold_enter,
                                          stall_margin_exit=self.hold_exit)
        # Real state shares perception between world tracking and grounding.
        percep = self._build_perception() if self.state == "real" else None
        src = get_source(self.ground_name, task_key=self.task_key, perception=percep,
                         seat_shift=self.seat_shift, local_grasp=self.local_grasp,
                         local_grasp_radius=self.local_grasp_radius, kp_source=self.kp_source,
                         contact_criterion=self.contact_criterion, subgoal_eps=self.subgoal_eps,
                         rotate_grasp_offset=self.rotate_grasp_offset,
                         lift_latch_xy=self.lift_latch_xy, seat_from_plane=self.seat_from_plane,
                         open_half=float(self.geom.get("open_half", 0.04)), geom=self.geom,
                         # The live aperture sensor's own thresholds, so the completion
                         # predicates' contact band is auditable against the band the hold
                         # latch actually uses instead of being compared from memory.
                         sensor_cfg={"q_free": float(self.sensor.q_free),
                                     "stall_margin": float(self.sensor.stall_margin),
                                     "q_touch": float(self.sensor.q_touch)},
                         **self.roles)
        self.world = self._build_world(raw_env, percep, src)
        self.grounding = src.ground(self.env, self.world)
        # Advance checks use grounding estimates rather than simulator poses.
        self._obj_pos = {o.name: o.pos for o in self.grounding.objects}
        self._extents = {o.name: o.extents for o in self.grounding.objects}
        # Advanceability preflight: every stage, in a state where it is genuinely satisfied, must
        # be able to advance. Runs once per episode, against the real _stage_reached.
        run_preflight = getattr(self.grounding, "advance_preflight", None)
        if self._advance_preflight and callable(run_preflight):
            run_preflight(self.preflight_probe)
        self.stage_idx = 0
        self.held_offset = _capture_held(self.env, self.grounding,
                                         self.grounding.stages[0].held_idx)
        self.plan_ref = None
        self.grasp_z0 = {}
        self._open_run = 0
        self._close_val = 1.0
        self._z_hist = []
        self._contact_prev = False
        self._last_cmd_close = False
        self._pred_hist.reset()          # stale frames across a reset would fire a stage
        self.pred_shadow = None
        self._pred_transition = None
        self._reset_churn()
        self._enter_stage()

    def _build_perception(self):
        from vlm_dp.perception import Perception           # Deferred to avoid loading vision weights.
        if not self.vocab:
            raise SystemExit("[vlm_dp] --vlm_state real needs the task's 'objects' vocabulary "
                             "in task_prompts.json")
        return Perception(self.vocab, fixtures=self.fixtures, segment=self.segment,
                          support_extents=self.support_extents,
                          support=self.roles.get("support"),
                          support_names=self.roles.get("grasp_objs") or ())

    def _build_world(self, raw_env, percep, src):
        """Build simulator-backed or perception-backed object state."""
        if self.state != "real":
            return GTWorld(raw_env, sensor=self.sensor)
        world = SensedWorld(percep, self.sensor, place_obj=self.roles.get("place_obj"),
                            tcp_offset=ROBOTIQ_GRASP_OFFSET, track=self.track,
                            reperceive_every=self.reperceive_every)
        world.relax_identity_when_stale = self._relax_identity_when_stale
        if self.jump_rate is not None:
            world.set_jump_rate(self.jump_rate)
        percep.calibrate(self.env)
        world.refresh(self.env)
        missing = [n for n in self.vocab if n not in world.names]
        required = {self.roles.get("grasp_obj"), self.roles.get("place_obj"),
                    self.roles.get("support"), *(self.roles.get("grasp_objs") or ())} - {None}
        fatal = [n for n in missing if n in required]
        if fatal:
            raise SystemExit(f"[vlm_dp] perception did not find {fatal}; refusing to run half-blind")
        if missing:
            # Missing obstacles reduce clearance coverage but do not invalidate the run.
            print(f"[vlm_dp] WARNING: obstacles not found this frame: {missing}; "
                  f"continuing without them", flush=True)
        # Seed calibrated fixtures before priming the visual tracker.
        cal = getattr(src, "calibration_points", None)
        for name, pos in (cal(self.env) if cal else {}).items():
            world.seed(name, pos)
        if self.track == "visual":
            from vlm_dp.visual_tracker import VisualTracker
            # Do not track fixed calibration fixtures.
            trackable = [n for n in world.names if n not in self.fixtures]
            init = {n: world.object_pose(n)[0] for n in trackable}
            world.visual = VisualTracker(self.env.cam, trackable, init)
        print(f"[vlm_dp] sensed world: {world.names} (track={self.track})", flush=True)
        return world

    def stage(self):
        return self.grounding.stages[self.stage_idx]

    def _enter_stage(self):
        """Reset stage-local state and capture the initial grasp height."""
        self.stage_replans = 0
        # The previous stage's (or a backtracked-away) plan must not survive into the new one:
        # the `consistency` term penalises deviation from plan_ref, so a stale reference makes
        # recovery pay for not repeating the plan that just failed.
        self.plan_ref = None
        self._place_seen = None
        self._place_since = None
        self._reopen = False
        self._reopen_ap = []
        self._reopen_cooldown_left = 0
        self._contact_seen = False
        self._released_latch = False
        self._grasp_probe = np.zeros(3)
        self._grasp_probe_idx = 0
        self._closed_empty = 0
        self._thin_loss_replans = 0
        # Reset per-stage failure evidence; backtracks re-add their event after entry.
        self.gate_events = {"closed_empty": 0, "backtracks": 0}
        self._stage_env_steps = 0
        st = self.stage()
        if st.on_enter is not None:
            st.on_enter()
        if st.gripper == "close" and st.grasp_obj is not None:
            self.grasp_z0[st.grasp_obj] = float(self._pos(st.grasp_obj)[2])

    def context(self, raw_env, obs, executed_steps=0):
        """Build the current world-frame cost context."""
        placed = frozenset(s.payload for s in self.grounding.stages[:self.stage_idx]
                           if s.gripper == "place" and s.payload is not None)
        # Mark a released payload as placed before the stage advances.
        st_now = self.stage()
        if (st_now.gripper == "place" and st_now.payload is not None
                and self.sensor is not None and self.sensor.released()):
            placed = placed | {st_now.payload}
        ref = None
        if self.plan_ref is not None:
            k = min(max(int(executed_steps), 0), self.plan_ref.shape[0])
            ref = torch.cat([self.plan_ref[k:], self.plan_ref[-1:].expand(k, 7)], dim=0) if k else self.plan_ref
        ctx = build_context(raw_env, obs, self.grounding, self.stage(),
                            plan_ref=ref, held_offset=self.held_offset,
                            placed=placed)
        ctx["destination"] = self.roles.get("place_obj")
        # Reopen after a closed-empty grasp before retrying.
        st = self.stage()
        # Press stages may close without certifying a pinch hold.
        if st.gripper == "close" and self.sensor is not None and getattr(st, "contact", "pinch") != "press":
            if self._reopen_cooldown_left > 0:
                # Escaped a reopen recently: let the gripper close again before re-arming, or the
                # recovery re-latches on the same evidence and the escape buys nothing.
                self._reopen_cooldown_left -= 1
                self._reopen = False
            elif self._thin_feature_held(st.grasp_obj):
                self._reopen = False
            elif self.sensor.closed_on_air() and self._thin_feature_settling(st.grasp_obj):
                self._reopen = False
                print(f"[vlm_dp] thin-feature contact settling: keep close "
                      f"age={self.sensor.close_age()}/{self._thin_feature_settle_max_steps}",
                      flush=True)
            elif self.sensor.closed_on_air():
                if not self._reopen:
                    self._reopen_ap = []
                    self._closed_empty += 1
                    self.gate_events["closed_empty"] += 1
                    print("[vlm_dp] closed-empty at grasp target: commanding reopen", flush=True)
                    if self._grasp_recovery == "search":
                        self._grasp_probe_idx += 1
                        self._grasp_probe = self._probe_pts[self._grasp_probe_idx % len(self._probe_pts)]
                        print(f"[vlm_dp] grasp search -> probe {self._grasp_probe_idx} "
                              f"{np.round(self._grasp_probe, 3)}", flush=True)
                    # Repeated closed-empty grasps mark the target belief stale.
                    if (self._regrasp_perceive and st.grasp_obj is not None
                            and self._closed_empty >= self._REGRASP_PERCEIVE_AFTER):
                        self._closed_empty = 0
                        if self.world.mark_stale(st.grasp_obj):
                            print(f"[vlm_dp] {st.grasp_obj}: {self._REGRASP_PERCEIVE_AFTER} closed-empty "
                                  f"closes -> belief marked stale, re-perceiving", flush=True)
                self._reopen = True
            elif self.sensor.is_open():
                self._reopen = False
            if self._reopen:
                escape = self._reopen_escape()
                if escape is not None:
                    self._reopen = False
                    self._reopen_cooldown_left = self._reopen_cooldown
                    print(f"[vlm_dp] reopen abandoned ({escape}); commanding close again for "
                          f"{self._reopen_cooldown} replans", flush=True)
        else:
            self._reopen = False
        if self._reopen:
            ctx["gripper_intent"] = "open"
        # Apply the same probe offset to the cost target and reach check.
        if self._grasp_recovery == "search" and st.gripper == "close" \
                and getattr(st, "contact", "pinch") != "press":
            ctx["target"] = np.asarray(ctx["target"], dtype=np.float32) + self._grasp_probe.astype(np.float32)
        # Detect seat contact from a stalled, commanded-close descent.
        if self.geom.get("release_on_stall", False):
            self._z_hist.append(float(self.env.tcp()[2]))
        if self.geom.get("release_on_stall", False) and st.gripper == "place" \
                and st.payload is not None and st.place_point is not None:
            # Seat contact requires a close command because sensed hold weakens on contact.
            contact = False
            if self._last_cmd_close:
                seat = np.asarray(st.place_point(), dtype=np.float64)
                d_xy = float(np.linalg.norm(self._pos(st.payload)[:2] - seat[:2]))
                xy_near = d_xy <= float(self.geom.get("release_xy", 0.06)) \
                    * float(self.geom.get("release_commit_band", 3.0))
                contact = xy_near and descent_stalled(self._z_hist, self._stall_replans, self._stall_eps)
            ctx["seat_contact"] = 1.0 if contact else 0.0
            # Disable overshoot after opening so the gripper retreats.
            ctx["overshoot_on"] = bool(self._last_cmd_close)
            # Latch release after contact so retreat terms remain active.
            if contact:
                self._contact_seen = True
            if self._contact_seen and self.sensor is not None and self.sensor.released():
                if not self._released_latch:
                    print(f"[vlm_dp] place_released latched replan={self.stage_replans}", flush=True)
                self._released_latch = True
            ctx["place_released"] = self._released_latch
            if contact != self._contact_prev:
                print(f"[vlm_dp] seat_contact={'ON' if contact else 'off'} replan={self.stage_replans} "
                      f"z={self._z_hist[-1]:.4f} cmd_close={self._last_cmd_close}", flush=True)
                self._contact_prev = contact
        self._stage_env_steps += max(int(executed_steps), 0)
        # Payload age, in seconds since acquisition, for carry_accel's settle window. The
        # dose-response that set carry_accel_max was measured on the first ~0.4-0.5s after
        # lift (an unsettled pinch); past the settle window the cap is a transport speed
        # limit rather than a slip guard, so the term reads this and stands down.
        # A payload change (None->name or name->other) restarts the clock; releasing clears it.
        if self._track_eef_hist:
            self._env_steps += max(int(executed_steps), 0)
            pay = ctx.get("payload")
            if pay != self._payload_name:
                self._payload_name = pay
                self._payload_acq = self._env_steps if pay is not None else None
            if pay is not None and self._payload_acq is not None:
                ctx["payload_age_s"] = (self._env_steps - self._payload_acq) / 15.0
        # Record TCP motion and release state for retreat terms.
        eef_now = np.asarray(ctx["eef_pos"], dtype=np.float64)[:3]
        if getattr(self, "_eef_prev", None) is not None and int(executed_steps) > 0:
            ctx["eef_step_motion"] = float(np.linalg.norm(eef_now - self._eef_prev)) / int(executed_steps)
        self._eef_prev = eef_now
        # Chunk-boundary history for carry_accel: the TCP one control step ago and now, so
        # the acceleration of the first planned row is defined against the executed past.
        # Omitted before the first executed step of an episode -> the term keeps its old shape.
        if self._track_eef_hist and self._eef_last is not None:
            ctx["eef_hist"] = np.stack([self._eef_last,
                                        eef_now.astype(np.float32)]).astype(np.float32)
        ctx["released"] = bool(self.sensor.released()) if self.sensor is not None else False
        ctx["hold_grace"] = self._hold_grace_value(st)
        auth = self._steer_authority(st)
        if auth != getattr(self, "_auth_prev", 0.0):
            print(f"[vlm_dp] steer window {'OPEN' if auth > 0 else 'closed'} stage='{st.name}' "
                  f"events={self.gate_events} dwell={self._stage_env_steps} "
                  f"budget={self._gate_dwell_budget(st)}", flush=True)
        self._auth_prev = auth
        ctx["steer_authority"] = auth
        ctx["steer_events"] = dict(self.gate_events)
        ctx["stage_env_steps"] = int(self._stage_env_steps)
        # Shadow only, and deliberately NOT a ctx key: the planner must not be able to see it.
        self._eval_predicate_shadow(ctx)
        return ctx

    def _reopen_escape(self):
        """Return why the reopen recovery must be abandoned this replan, or None to continue.

        The recovery latches on ``closed_on_air`` and, as shipped, cleared on ``is_open()`` alone
        (aperture <= q_touch). Fingers that close on the EDGE of something wedge part-way -- 0.23
        to 0.38 rad, neither open nor free-closed -- and neither exit condition can ever be met, so
        the open command is held for the rest of the episode and the stage cannot be retried. Both
        escapes below are bounded and preserve the normal path: an ordinary reopen travels from the
        free-close angle to open in a few replans, so it never sits still long enough to stall out
        and never reaches the timeout.
        """
        ap = float(self.sensor.aperture())
        self._reopen_ap.append(ap)
        n = len(self._reopen_ap)
        if n >= self._reopen_max:
            return (f"timeout: {n} replans commanding open, aperture still {ap:.3f} rad "
                    f"(open needs <= {self.sensor.q_touch:.3f})")
        k = self._reopen_stall
        if n >= k and not self.sensor.closed_on_air():
            # WEDGED means part-way: neither open nor free-closed. While the fingers are still in
            # the free-close range this is just an ordinary reopen in progress -- which is allowed
            # to be slow, and is bounded by the timeout above -- so the stall test stands down and
            # normal closed-on-air recovery is preserved.
            window = self._reopen_ap[-k:]
            moved = self._reopen_ap[0] - ap
            if max(window) - min(window) <= self._reopen_stall_eps and moved >= self._reopen_travel:
                return (f"fingers wedged at {ap:.3f} rad: {k} replans within "
                        f"{self._reopen_stall_eps:.3f} rad after opening {moved:.3f} rad")
        return None

    def _hold_grace_value(self, stage):
        """Return whether the post-grasp hold-grace window is active."""
        if self._hold_grace_replans <= 0:
            return 0.0
        if stage.gripper == "place" or self._reopen or self.sensor is None:
            self._hold_grace = 0
            return 0.0
        obj = stage.payload if stage.payload is not None else stage.grasp_obj
        if obj is not None and self._payload_held(obj):
            self._hold_grace = self._hold_grace_replans
        elif self._hold_grace > 0:
            self._hold_grace -= 1
        return 1.0 if self._hold_grace > 0 else 0.0

    def _gate_dwell_budget(self, stage):
        """Return the demo-normalized stage dwell budget in environment steps."""
        if not self._gate_stats:
            return None
        key = "grasp_phase_steps" if stage.gripper == "close" else "hold_phase_steps"
        stats = self._gate_stats.get(key) or {}
        p95 = stats.get("p95")
        return None if not p95 else self._gate_dwell_k * float(p95)

    def _steer_authority(self, stage):
        """Return expert authority when the stage policy and failure gate allow it."""
        policy = getattr(stage, "steer_policy", None)
        if policy in (None, "never"):
            return 0.0
        if policy == "always":
            return 1.0
        if self.gate_events["backtracks"] > 0 or self.gate_events["closed_empty"] >= 2:
            return 1.0
        budget = self._gate_dwell_budget(stage)
        if budget is not None and self._stage_env_steps > budget:
            return 1.0
        return 0.0

    def filter_plan(self, actions, execute_steps):
        """Suppress brief open commands while a payload is held."""
        st = self.stage()
        if getattr(st, "gripper_authoritative", False) and st.gripper == "close":
            suppressed = []
            for i in range(min(int(execute_steps), len(actions))):
                raw = float(actions[i][7])
                if raw < 1.0:
                    actions[i][7] = 1.0
                    suppressed.append((i, raw))
            self._open_run = 0
            self._close_val = 1.0
            return actions, suppressed
        if self._grip_debounce <= 0:
            return actions, []
        name = st.payload or st.grasp_obj
        held = bool(name) and self._payload_held(name)
        # Lift/carry has no legitimate release intent.  Do not let the soft
        # horizon-mean carry cost turn a certified grasp into an empty close.
        # Place stages are excluded, so their authored release is untouched.
        if held and st.gripper == "hold" and self._hold_gripper_authoritative:
            suppressed = []
            for i in range(min(int(execute_steps), len(actions))):
                raw = float(actions[i][7])
                if raw <= 0.5:
                    actions[i][7] = 1.0
                    suppressed.append((i, raw))
            self._open_run = 0
            self._close_val = 1.0
            if suppressed:
                raws = " ".join(f"{i}:{r:.3f}" for i, r in suppressed)
                print(f"[vlm_dp] gripper_latch stage={self.stage_idx} held={name} "
                      f"authoritative-hold suppressed {raws}", flush=True)
            return actions, suppressed
        suppressed, self._open_run, self._close_val = debounce_gripper(
            actions, execute_steps, self._grip_debounce, held, self._open_run, self._close_val)
        if suppressed:
            raws = " ".join(f"{i}:{r:.3f}" for i, r in suppressed)
            print(f"[vlm_dp] gripper_latch stage={self.stage_idx} held={name} "
                  f"suppressed {raws}", flush=True)
        return actions, suppressed

    def observe_plan(self, actions):
        """Record the decoded chunk and any replan-cadence sensor sample."""
        self.plan_ref = torch.as_tensor(actions, dtype=torch.float32, device=self.device)[..., :7]
        if self.sensor_cadence != "replan":
            return
        commanded_close = bool(float(torch.as_tensor(actions).reshape(actions.shape[0], -1)[0, 7]) > 0.5)
        self._last_cmd_close = commanded_close
        self._observe_sensors(commanded_close)

    def observe_step(self, action_step):
        """Record sensor evidence for one applied control step."""
        self._sample_eef()
        self._sample_predicate_history()
        if self.sensor_cadence == "replan":
            return                                   # observe_plan already sampled this chunk
        commanded_close = bool(float(torch.as_tensor(action_step).reshape(-1)[7]) > 0.5)
        self._last_cmd_close = commanded_close
        self._observe_sensors(commanded_close)

    def _sample_eef(self):
        """Record the measured TCP before this control step (carry_accel boundary rows only).

        Called once per APPLIED step, so consecutive samples are exactly one control step
        apart -- the spacing carry_accel's second difference assumes. The frame is the one
        build_context reports as ctx["eef_pos"], which is the frame the planner optimizes in.
        """
        if not self._track_eef_hist or self.env is None:
            return
        try:
            frame = self.env.env.scene["ee_frame"]
            frame.update(0.0, force_recompute=True)
            self._eef_last = np.asarray(
                frame.data.target_pos_w[0, 0].detach().cpu(), dtype=np.float32)
        except Exception:                            # never let bookkeeping break a rollout
            self._eef_last = None

    def _sample_predicate_history(self):
        """Push one (keypoints, TCP, aperture) frame for the shadow completion predicates.

        Once per APPLIED control step, so the buffer is uniformly spaced at the 15 Hz control
        rate the predicates assume. Skipped entirely when the plan authored no predicates, and
        it never raises -- a shadow signal must not be able to end a rollout.
        """
        if self.grounding is None or getattr(self.grounding, "completion", None) is None:
            return
        try:
            kps = self.grounding.keypoints()
            self._pred_hist.push(kps, self.env.tcp(), self.env.gripper_q())
        except Exception:                            # observation only; never break a rollout
            pass

    def _eval_predicate_shadow(self, ctx):
        """Evaluate the current stage's completion predicate beside the scalar sub-goal test.

        Writes self.pred_shadow, which only the debug log reads. The stage machine is untouched:
        advance() does not consult it, so an arm with predicates loaded produces byte-identical
        actions to one without.
        """
        comp = getattr(self.grounding, "completion", None)
        if comp is None or len(self._pred_hist) == 0:
            self.pred_shadow = None
            return
        st = self.stage()
        history = self._pred_hist.view()
        fired, components = comp.evaluate(self.stage_idx, history)
        # The scalar decision this would replace, computed exactly as rekep's subgoal_done does.
        old = None
        if st.constraint is not None and self.grounding.keypoints is not None:
            try:
                dev = self.device
                ee = torch.as_tensor(self.env.tcp(), device=dev, dtype=torch.float32).reshape(1, 1, 3)
                kp = torch.as_tensor(self.grounding.keypoints(), device=dev,
                                     dtype=torch.float32)[:, None, None, :]
                old = float(torch.as_tensor(st.constraint(ee, kp)).reshape(-1)[0])
            except Exception:
                old = None
        self.pred_shadow = {
            "stage_idx": int(self.stage_idx),
            "stage": st.name,
            "predicate": bool(fired),
            "subgoal_value": old,
            "subgoal_fired": None if old is None else bool(old < self.subgoal_eps),
            "frames": int(len(self._pred_hist)),
            "components": {
                k: (bool(v) if isinstance(v, bool)
                    else float(v) if isinstance(v, (int, float, np.number))
                    else str(v))
                for k, v in components.items()
            },
        }
        # advance() ran just before this context(), so its transition decision (when predicate
        # transitions are enabled) belongs to this same replan. Carrying it here puts the
        # certified decision and the scalar decision it replaced in one log record.
        if self._pred_transition is not None:
            self.pred_shadow["transition"] = dict(self._pred_transition)

    def _observe_sensors(self, commanded_close: bool) -> None:
        """Record one shared world and aperture-sensor observation."""
        # Route both state modes through the world to preserve latched holds.
        st = self.stage()
        press = getattr(st, "contact", "pinch") == "press"
        # A press touches an articulated part but never carries it. Passing an explicit empty
        # candidate set keeps the hold latch from claiming the lid and freezing its visual track.
        cand = set() if press else {n for n in (st.grasp_obj, st.payload) if n}
        observe = getattr(self.world, "observe", None)
        if callable(observe):
            observe(self.env, commanded_close,
                    candidates=(cand if press else (cand or None)),
                    points=(None if press else self._hold_points(cand)))
        else:
            self.sensor.observe(self.env, commanded_close)

    def _hold_points(self, names):
        """Return where the hold test should look for each object: its GROUNDED grasp point.

        The grounding's own accessor (``self._obj_pos``, i.e. the tracked keypoint the stage
        grasps, with its registered offset) rather than the world's centroid belief. For an object
        whose grasp point IS its centre the two are the same reading; for a stage that grasps a
        declared feature -- a lid rim on a coffee machine -- they are 0.25-0.30m apart, and the
        centroid reading makes the correct grasp invisible to ApertureGraspSensor.held_object,
        whose proximity gate is 0.10m. The stage's grasp point is what the cost drove the TCP to,
        so it is the point the hold has to be judged against.
        """
        out = {}
        for name in names or ():
            fn = self._obj_pos.get(name)
            if fn is None:
                continue
            try:
                out[name] = np.asarray(fn(), dtype=np.float64)[:3]
            except Exception:
                continue                      # an un-grounded name simply keeps the world's belief
        return out or None

    def _pos(self, name):
        """Return the estimated object position, with a world-state fallback."""
        fn = self._obj_pos.get(name)
        return np.asarray(fn() if fn is not None else self.world.object_pose(name)[0], dtype=np.float64)

    def viz(self):
        """Return drawable state for the current stage."""
        st = self.stage()
        out = {"stage": f"{self.stage_idx}: {st.name}",
               "holding": bool(self.sensor is not None and self.sensor.holding())}
        if self.grounding.keypoints is not None:
            out["keypoints"] = np.asarray(self.grounding.keypoints(), dtype=np.float64)
        try:
            out["target"] = np.asarray(st.target(), dtype=np.float64)
        except Exception:
            pass
        if st.gripper == "place" and st.place_point is not None:
            out["seat"] = np.asarray(st.place_point(), dtype=np.float64)
            if st.payload is not None:
                out["payload"] = self._pos(st.payload)
        elif st.grasp_obj is not None:
            out["payload"] = self._pos(st.grasp_obj)
        return out

    def _log_ground_error(self):
        """Log grounding drift against simulator truth for diagnostics."""
        if self.stage_replans % max(self._ground_err_every, 1):
            return
        for name in sorted(getattr(self.grounding, "manipulated", ()) or ()):
            fn = self._obj_pos.get(name)
            if fn is None:
                continue
            try:
                est = np.asarray(fn(), dtype=np.float64)[:3]
                gt = np.asarray(self.env.object_pose(name)[0], dtype=np.float64)[:3]
                d = est - gt
                # Log both trajectories to distinguish tracker drift from object motion.
                print(f"[vlm_dp] ground_err replan={self.stage_replans} stage={self.stage_idx} "
                      f"{name} |d|={np.linalg.norm(d) * 1000:.1f}mm "
                      f"est=({est[0]:.3f},{est[1]:.3f},{est[2]:.3f}) "
                      f"gt=({gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f})", flush=True)
            except Exception:
                pass

    def _log_grasp_state(self, stage):
        """Log the grasp evidence a close stage advances on, once per replan.

        One line carrying all four quantities that decide a grasp: where the TCP is relative to the
        point the stage actually grasps, the feasibility radius every grasp term is sized by (and
        the dead zone it implies), the gripper channel's state including the reopen recovery, and
        what the hold latch resolves to.
        """
        if stage.gripper != "close" or stage.grasp_obj is None or self.sensor is None:
            return
        name = stage.grasp_obj
        obj = next((o for o in self.grounding.objects if o.name == name), None)
        ge = getattr(obj, "grasp_extent", None) if obj is not None else None
        ext = self._extents.get(name)
        radius = float(ge) if ge is not None else (float(ext[1]) if ext is not None else float("nan"))
        try:
            tgt = np.asarray(stage.target(), dtype=np.float64)[:3] + self._grasp_probe
            d = float(np.linalg.norm(np.asarray(self.env.tcp(), dtype=np.float64)[:3] - tgt))
        except Exception:
            d = float("nan")
        latched = getattr(self.world, "held", None)
        print(f"[vlm_dp] grasp_dbg stage={self.stage_idx} replan={self.stage_replans} obj={name} "
              f"r={radius:.4f}{'' if ge is not None else '(whole-object)'} "
              f"dead={self._grasp_slack(name):.4f} d_tcp={d:.4f} ap={self.sensor.aperture():.3f} "
              f"open={int(self.sensor.is_open())} air={int(self.sensor.closed_on_air())} "
              f"holding={int(self.sensor.holding())} reopen={int(self._reopen)} "
              f"cool={self._reopen_cooldown_left} "
              f"latch={latched() if callable(latched) else None} "
              f"held={int(self._payload_held(name))}", flush=True)

    def _log_release_gate(self, stage):
        """Log placement release errors and simulator-only diagnostics."""
        if stage.gripper != "place" or stage.payload is None or stage.place_point is None:
            return
        if not (self.sensor is not None and self.sensor.holding()):
            return
        try:
            seat = np.asarray(stage.place_point(), dtype=np.float64)
            ext = self._extents.get(stage.payload)
            dest_z = seat[2] + (float(ext[2]) if ext is not None else 0.0) \
                + float(self.geom.get("place_release_clearance", 0.0))
            pos = self._pos(stage.payload)
            xy = float(np.linalg.norm(pos[:2] - seat[:2]))
            dz = float(pos[2] - dest_z)
            tol_xy = float(self.geom.get("release_xy", 0.06))
            tol_z = float(self.geom.get("release_z", 0.02))
            blocked = ("xy" if xy >= tol_xy else "") + ("z" if dz >= tol_z else "")
            gate = "OPEN" if not blocked else "blocked-" + blocked
            # Simulator truth distinguishes a bad estimate from a physical hover.
            try:
                pgt = np.asarray(self.env.object_pose(stage.payload)[0], dtype=np.float64)
                tgt_name = stage.place_target
                dgt = np.asarray(self.env.object_pose(tgt_name)[0], dtype=np.float64)
                d_ext = self._extents.get(tgt_name)
                p_half = float(ext[2]) if ext is not None else 0.0
                d_top = dgt[2] + (float(d_ext[2]) if d_ext is not None else 0.0)
                print(f"[vlm_dp] release_gt payload_z={pgt[2]:.4f} dest_top={d_top:.4f} "
                      f"gap={(pgt[2] - p_half) - d_top:+.4f}m  seat_est={seat[2]:.4f} "
                      f"seat_err={seat[2] - d_top:+.4f}m", flush=True)
            except Exception:
                pass
            print(f"[vlm_dp] release_gate replan={self.stage_replans} "
                  f"xy={xy:.4f}(tol {tol_xy:.3f}, {xy / max(tol_xy, 1e-6):.2f}x) "
                  f"dz={dz:+.4f}(tol {tol_z:.3f}, {dz / max(tol_z, 1e-6):+.2f}x) "
                  f"gate={gate}", flush=True)
        except Exception:
            pass

    def _payload_held(self, payload):
        """Return whether the configured hold authority reports the payload held.

        ``_hold_world`` is the world holding the latch. It is ``self.world`` for a rollout; the
        advanceability preflight substitutes one carrying a REAL HoldLatch over a REAL
        ApertureGraspSensor, so that the preflight exercises this same rule rather than a stub.
        """
        # A visually measured thin feature has an expected stall aperture above the generic
        # air threshold.  During acquisition, require that width-specific certificate instead
        # of accepting any wider obstruction (for Pot this was usually the lid edge).
        half_w = self._grip_half_width(payload)
        thin = False
        if self.sensor is not None and half_w is not None:
            predicted = self.sensor.q_free - self._AP_SLOPE * (2.0 * half_w)
            thin = predicted > self.sensor.q_free - self.sensor.stall_margin_enter
        if thin and self.stage().gripper == "close":
            return self._thin_feature_held(payload)

        world = getattr(self, "_hold_world", None) or self.world
        held = payload_held(payload, self.hold_authority, world, self.sensor,
                            self.env.tcp(), self._pos(payload))
        if held:
            return True
        return self._thin_feature_held(payload)

    def _grip_half_width(self, payload):
        """Return the half-width used to predict the payload stall angle."""
        obj = next((o for o in self.grounding.objects if o.name == payload), None)
        ge = getattr(obj, "grasp_extent", None) if obj is not None else None
        if ge is not None:
            return float(ge)
        ext = self._extents.get(payload)
        return float(ext[0]) if ext is not None else None

    def _thin_feature_contact(self, payload, ratio=None):
        """Return whether aperture, visual width and TCP proximity agree on thin contact."""
        if self.sensor is None or payload is None:
            return False
        half_w = self._grip_half_width(payload)
        if half_w is None:
            return False
        predicted = self.sensor.q_free - self._AP_SLOPE * (2.0 * half_w)
        air_threshold = self.sensor.q_free - self.sensor.stall_margin_enter
        if predicted <= air_threshold:
            return False
        # Reject contacts substantially wider than the visually measured feature.  The old
        # one-sided test rejected only a free close, so a 10mm handle could certify while the
        # fingers were actually stalled wide on the surrounding lid.
        min_aperture = predicted - float(
            getattr(self, "_thin_feature_width_band", 0.08))
        if self.sensor.aperture() <= min_aperture:
            return False
        ratio = float(getattr(self, "_thin_feature_contact_ratio", 0.5)
                      if ratio is None else ratio)
        max_aperture = predicted + ratio * (self.sensor.q_free - predicted)
        if self.sensor.aperture() >= max_aperture:
            return False
        try:
            distance = float(np.linalg.norm(np.asarray(self.env.tcp()) - self._pos(payload)))
        except Exception:
            return False
        return distance <= self.sensor.proximity

    def _thin_feature_held(self, payload):
        """Certify a settled contact on a feature thinner than the generic stall band."""
        return self._thin_feature_contact(payload) and self.sensor.closed()

    def _thin_feature_settling(self, payload):
        """Keep closing while a plausible thin contact gathers settling evidence.

        The generic air threshold is crossed before ``closed()`` can collect its full window.
        Deferring recovery prevents it from erasing that evidence. The bounded close age keeps
        recovery available for a contact that never settles.
        """
        if not getattr(self, "_last_cmd_close", False):
            return False
        if not self._thin_feature_contact(payload) or self.sensor.closed():
            return False
        default_max = 2 * (max(self.sensor.close_steps, self.sensor.settle_steps) + 1)
        max_steps = int(getattr(self, "_thin_feature_settle_max_steps", default_max))
        return self.sensor.close_age() < max_steps

    _REGRASP_PERCEIVE_AFTER = 2   # Closed-empty grasps before re-perception.

    _PLACE_GRACE = 6     # Seat-check flicker tolerance after release.
    _PLACE_SETTLE = 5    # Stable-seat confirmation window.
    _REPERCEIVE_EVERY = 6  # Re-segmentation interval for stale objects.

    # Calibrated mapping from grip width to finger aperture.
    _AP_SLOPE = 9.3
    _AP_BAND = 0.16

    def _log_grip(self, why, payload):
        """Log measured and width-predicted aperture after a lost hold."""
        if not self._ground_err_debug or payload is None:
            return
        try:
            ap = float(self.sensor.aperture())
            half_w = self._grip_half_width(payload)
            if half_w is None:
                print(f"[vlm_dp] grip_lost {why} stage={self.stage_idx} {payload} "
                      f"aperture={ap:.3f} (no measured extent)", flush=True)
                return
            pred = self.sensor.q_free - self._AP_SLOPE * (2.0 * half_w)
            print(f"[vlm_dp] grip_lost {why} stage={self.stage_idx} {payload} "
                  f"aperture={ap:.3f} predicted={pred:.3f} delta={ap - pred:+.3f} "
                  f"band=+/-{self._AP_BAND:.2f} half_w={half_w:.3f}", flush=True)
        except Exception:
            pass

    def _grip_ok(self, payload):
        """Return whether the aperture matches the expected grip width."""
        half_w = self._grip_half_width(payload)
        if half_w is None:
            return True
        predicted = self.sensor.q_free - self._AP_SLOPE * (2.0 * half_w)
        return abs(self.sensor.aperture() - predicted) <= self._AP_BAND

    def advance(self, flags):
        """Backtrack on invariant failure or advance when the stage is reached."""
        stage = self.stage()
        self.stage_replans += 1
        self._pred_transition = None      # one fresh transition record per replan, never stale
        if self._commit_left > 0:
            self._commit_left -= 1
        # Sync FK before reading held-object state.
        self.world.sync_fk(self.env)
        # Re-perceive stale objects at a throttled rate.
        if self.world.stale() and self.stage_replans % self._REPERCEIVE_EVERY == 0:
            self.world.refresh(self.env)
        if self._release_debug:
            self._log_release_gate(stage)
        if self._ground_err_debug:
            self._log_ground_error()
            self._log_grasp_state(stage)
        violated = self._invariant_violated(stage, flags) if self.backtrack_enabled else None
        if violated and self._backtrack_blocked():
            violated = None  # Commit to the current grip.
        if violated:
            if self.stage_idx > 0:
                self.stage_idx -= 1
                self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
                self._enter_stage()
                self.gate_events["backtracks"] += 1  # Preserve failure evidence.
                self._backtracks_run += 1
                print(f"[vlm_dp] BACKTRACK -> {self.stage_idx}: {self.stage().name} ({violated})",
                      flush=True)
            return
        if self._stage_reached(stage, flags) and self.stage_idx + 1 < len(self.grounding.stages):
            self.stage_idx += 1
            self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
            self._enter_stage()
            if self.stage_idx > self._stage_high:
                # Only new maximum stage depth refunds the backtrack budget.
                self._stage_high = self.stage_idx
                self._backtracks_run = 0
            print(f"[vlm_dp] stage -> {self.stage_idx}: {self.stage().name}", flush=True)

    def _backtrack_blocked(self):
        """Return whether the backtrack budget suppresses regression."""
        if self._backtrack_budget <= 0:
            return False
        if self._commit_left > 0:
            return True
        if self._backtracks_run < self._backtrack_budget:
            return False
        self._commit_left = self._backtrack_commit
        self._backtracks_run = 0
        print(f"[vlm_dp] backtrack budget spent ({self._backtrack_budget}); committing to the "
              f"current grip for {self._commit_left} replans", flush=True)
        return True

    def _invariant_violated(self, stage, flags):
        """Return the violated sensed invariant, or None."""
        if stage.payload is None:
            return None
        if getattr(stage, "contact", "pinch") == "press":
            # Press contacts do not certify pinch holds.
            return None
        placed_now = stage.gripper == "place" and (
            bool(stage.done())
            or (self.flag_fallback and stage.done_flag is not None
                and bool(flags.get(stage.done_flag, False))))
        # Allow a brief seat-check flicker after deliberate release.
        if placed_now:
            self._place_seen = self.stage_replans
            if self._place_since is None:  # Start settling once.
                self._place_since = self.stage_replans
            return None
        self._place_since = None  # Restart settling after leaving the seat.
        if self._place_seen is not None and self.stage_replans - self._place_seen < self._PLACE_GRACE:
            return None  # Ignore brief seat-check flicker.
        if self.advance_mode == "env_flags":
            # Privileged-signal ablation.
            return None if flags.get(f"grasp_{stage.payload}", False) else "flag dropped"
        if self.sensor.closed_on_air() and not self._thin_feature_held(stage.payload):
            plausible = self._thin_feature_contact(
                stage.payload, ratio=getattr(self, "_thin_feature_loss_ratio", 0.8))
            if plausible:
                self._thin_loss_replans = 0
                return None
            self._thin_loss_replans = getattr(self, "_thin_loss_replans", 0) + 1
            grace = int(getattr(self, "_thin_feature_loss_grace", 2))
            if self._thin_loss_replans <= grace:
                print(f"[vlm_dp] thin-feature hold fluctuation: defer empty-hand backtrack "
                      f"{self._thin_loss_replans}/{grace}", flush=True)
                return None
            self._log_grip("empty hand", stage.payload)
            return "empty hand"
        self._thin_loss_replans = 0
        if self.sensor.holding() and not self._grip_ok(stage.payload):
            self._log_grip("poor grip", stage.payload)
            return "poor grip"
        # Subgoal mode does not impose an additional lift-height invariant.
        z0 = self.grasp_z0.get(stage.payload)
        if (self.advance_mode != "subgoal" and z0 is not None
                and self.stage_replans >= self.grasp_confirm
                and float(self._pos(stage.payload)[2]) < z0 + self.rise_confirm):
            return "never rose"
        return None

    def _grasp_slack(self, name):
        """Return the cost-aligned grasp tolerance for an object."""
        if name is None:
            return self.grasp_eps
        obj = next((o for o in self.grounding.objects if o.name == name), None)
        radius = getattr(obj, "grasp_extent", None) if obj is not None else None
        if radius is None:
            ext = self._extents.get(name)
            radius = ext[1] if ext is not None else None
        return self.grasp_eps if radius is None else grasp_slack(self._geom_ns, float(radius))

    def _place_certified(self, stage, flags):
        """Return the completion predicate's verdict for a PLACE stage, or None for "no opinion".

        None means "fall through to the shipped rule": the opt-in key is absent, the plan
        authored no predicate for this stage, or there is no history yet (start of an episode).
        The scalar sub-goal test answers "is the held object's keypoint near the place point",
        which a still-gripped object satisfies; the predicate answers "did the placement EVENT
        happen" -- hand let go AND object inside the region AND quiescent -- so a stage cannot
        be signed off while the payload is still owned by the gripper.
        """
        if not self._pred_place_transitions:
            return None
        comp = getattr(self.grounding, "completion", None)
        if comp is None or self.stage_idx not in comp or len(self._pred_hist) == 0:
            return None
        fired, components = comp.evaluate(self.stage_idx, self._pred_hist.view())
        # The decision this replaces, for the evidence trail. Only the sensed rule is
        # reconstructed; the other advance modes are not what this key is for.
        old = None
        if self.advance_mode not in ("subgoal", "env_flags"):
            try:
                seated = (bool(stage.done())
                          or (self.flag_fallback and stage.done_flag is not None
                              and bool(flags.get(stage.done_flag, False))))
                settled = (self._place_since is not None
                           and self.stage_replans - self._place_since >= self._place_settle)
                old = bool(seated and settled and self.sensor.released())
            except Exception:                        # logging must never break a rollout
                old = None
        self._pred_transition = {
            "stage_idx": int(self.stage_idx), "stage": stage.name,
            "replan": int(self.stage_replans), "predicate": bool(fired), "old_rule": old,
            "components": {k: (bool(v) if isinstance(v, bool) else float(v))
                           for k, v in components.items()},
        }
        if fired or (old is not None and old != fired):
            print(f"[vlm_dp] place_certify stage={self.stage_idx} predicate={fired} "
                  f"old_rule={old} replan={self.stage_replans} "
                  f"{ {k: v for k, v in components.items() if not k.startswith('margin_')} }",
                  flush=True)
        return bool(fired)

    def _preflight_history(self, stage, synth):
        """Build the predicate history the synthesized satisfying state implies.

        Two recipes, both in the vocabulary predicates.py documents. PLACE: the hand is open and
        neither it nor the object is moving (released + quiescent). Everything else: the fingers
        are stalled on the payload, the hand travelled horizontally across the window, and the
        payload's keypoints kept their offset from it (closed_on_object + rides_with_hand), with
        the object's height constant so a carry-height component is unaffected by the motion.
        """
        kps = np.asarray(synth["kp"], dtype=np.float64)
        eef = np.asarray(synth["eef"], dtype=np.float64).reshape(3)
        moved = list(synth.get("moved") or ())
        hist = HistoryBuffer(maxlen=self._pred_hist.maxlen, dt=self._pred_hist.dt)
        n = max(int(hist.maxlen), 2)
        for t in range(n):
            if stage.gripper == "place":
                hist.push(kps, eef, 0.0)
                continue
            back = np.array([0.06 * (n - 1 - t) / (n - 1), 0.0, 0.0])
            frame = kps.copy()
            if moved:
                frame[moved] -= back
            hist.push(frame, eef - back, 0.30)
        return hist

    def preflight_probe(self, stage_idx, stage, synth):
        """Run the ACTIVE advance test for one stage against an installed satisfying state.

        rekep.advance_preflight has already installed the synthesized keypoints and TCP in the
        tracker and the env, so every accessor _stage_reached reads -- stage.done(), stage.target(),
        self._pos, the keypoint beliefs -- sees that state. Only the bridge-side evidence (the
        predicate history and the aperture sensor) is supplied here. This calls the REAL
        _stage_reached, so the preflight tests the rule that will run, not a copy of it.
        """
        saved = (self.stage_idx, self._pred_hist, self.sensor, self._hold_world,
                 self._place_seen, self._place_since, self._pred_transition, self._grasp_probe)
        try:
            self.stage_idx = int(stage_idx)
            self._pred_hist = self._preflight_history(stage, synth)
            holding = stage.gripper != "place"
            self.sensor = self._preflight_sensor(holding, stage)
            # The configured hold authority is left ALONE; only the latch's inputs are synthesized.
            self._hold_world = self._preflight_hold_world(stage)
            self._place_seen = self.stage_replans
            self._place_since = self.stage_replans - self._place_settle
            self._grasp_probe = np.zeros(3)
            return bool(self._stage_reached(stage, {})), self.advance_test_name(stage)
        finally:
            (self.stage_idx, self._pred_hist, self.sensor, self._hold_world,
             self._place_seen, self._place_since, self._pred_transition, self._grasp_probe) = saved

    def _preflight_sensor(self, holding, stage=None):
        """Return a REAL ApertureGraspSensor driven into the state the satisfying state implies.

        Fingers stalled mid-band on a payload while carrying, an open hand once the object has
        been placed. Same class, same thresholds and same window lengths the rollout configures,
        driven through its ordinary observe() loop -- so what the preflight certifies is the real
        certificate, not a promise that one would have been issued.
        """
        sensor = ApertureGraspSensor(stall_margin=self.stall_margin, settle_eps=self.settle_eps,
                                     close_steps=self.close_steps, settle_steps=self.settle_steps,
                                     legacy=self.legacy_sensor,
                                     stall_margin_enter=self.hold_enter,
                                     stall_margin_exit=self.hold_exit)
        stalled = (sensor.q_touch + (sensor.q_free - sensor.stall_margin_enter)) / 2.0
        if holding and stage is not None:
            payload = stage.grasp_obj or stage.payload
            half_w = self._grip_half_width(payload) if payload is not None else None
            if half_w is not None:
                predicted = sensor.q_free - self._AP_SLOPE * (2.0 * half_w)
                if predicted > sensor.q_free - sensor.stall_margin_enter:
                    # A thin feature's valid contact lies inside the generic air band.
                    # Synthesize the feature-predicted aperture, not the generic midpoint.
                    stalled = predicted
        q = stalled if holding else 0.0
        env = types.SimpleNamespace(gripper_q=(lambda q=q: q))
        for _ in range(max(sensor.close_steps, sensor.settle_steps) + 2):
            sensor.observe(env, holding)
        return sensor

    def _preflight_hold_world(self, stage):
        """Return a world answering held() from a real latch over the synthesized state."""
        latch = HoldLatch(self.sensor)
        names = {n for n in (stage.grasp_obj, stage.payload) if n}
        latch.update(self._hold_points(names) or {}, self.env.tcp(), names or None)
        return _LatchWorld(latch)

    def _plan_owns(self, stage):
        """Return whether the plan authored its own completion evidence for this stage.

        Evidence means a sub-goal constraint (the scalar the plan wrote, read through
        ``stage.done()``) or a ``stage<N>_completion`` predicate. Under
        ``advance.plan_authoritative`` such a stage advances on that evidence and on nothing else.
        A stage the plan said nothing about is not covered and keeps its shipped rule.
        """
        if not self._plan_authoritative:
            return False
        comp = getattr(self.grounding, "completion", None)
        return stage.constraint is not None or (comp is not None and self.stage_idx in comp)

    def advance_test_name(self, stage):
        """Name the branch of _stage_reached this stage will actually take (diagnostics only).

        Mirrors the branch order below; the preflight quotes it in its refusal so the failing
        test is named rather than guessed. It never decides anything.
        """
        plan_auth = self._plan_owns(stage)
        if stage.gripper == "close" and stage.grasp_obj is not None:
            if getattr(stage, "advance_on_done", False):
                return "stage.done()"
            if self.advance_mode == "env_flags":
                return "env flag grasp_<obj>"
            if getattr(stage, "contact", "pinch") == "press":
                return "target proximity + closed()"
            return "hold certificate" if self._grasp_advance_on_hold else "target proximity + hold"
        if stage.gripper == "hold" and stage.payload is not None:
            if getattr(stage, "advance_on_done", False) or plan_auth \
                    or self.advance_mode == "subgoal":
                return "stage.done() (plan sub-goal)" if plan_auth else "stage.done()"
            return "payload z >= target().z - lift_tol"
        if stage.gripper == "place":
            if self._pred_place_transitions:
                return "completion predicate"
            if self.advance_mode == "subgoal":
                return "released() + seat seen"
            return "seated + settled + released()"
        return "stage.done() (plan sub-goal)" if plan_auth else "done_flag / stage.done()"

    def _stage_reached(self, stage, flags):
        """Return whether the current stage target is satisfied."""
        plan_auth = self._plan_owns(stage)
        if stage.gripper == "close" and stage.grasp_obj is not None:
            # Deliberately UNCHANGED by plan_authoritative. The grasp rule is an aperture
            # certificate -- the fingers are stalled on something of the expected width, in the
            # place the object is believed to be -- which is a PHYSICAL measurement, not a
            # geometric re-reading of the plan's prose. The plan cannot observe a grip, so there
            # is nothing here for it to be authoritative about.
            if getattr(stage, "advance_on_done", False):  # Task-state completion.
                return bool(stage.done())
            if self.advance_mode == "env_flags":
                return bool(flags.get(f"grasp_{stage.grasp_obj}", False))
            # Measure proximity against the probe-shifted target.
            tgt = np.asarray(stage.target()) + self._grasp_probe
            near = float(np.linalg.norm(np.asarray(self.env.tcp()) - tgt)) < self._grasp_slack(stage.grasp_obj)
            if getattr(stage, "contact", "pinch") == "press":
                # Press stages advance on completed closure rather than pinch certification.
                return near and self.sensor.closed()
            held = self._payload_held(stage.grasp_obj)
            if self._grasp_advance_on_hold:
                # A confirmed hold already identifies the grasped object.
                return held
            return near and held
        if stage.gripper == "hold" and stage.payload is not None:
            if getattr(stage, "advance_on_done", False):  # Task-state completion.
                return bool(stage.done())
            if plan_auth:
                # PLAN-AUTHORITATIVE: the stage's own sub-goal, and never the z-test below. The
                # plan states what "lifted" means; the runtime does not get a second opinion.
                return bool(stage.done())
            if self.advance_mode == "subgoal":  # Subgoal predicate.
                return bool(stage.done())
            # Legacy geometric test, retained for plans/compilers that author no sub-goal for a
            # hold stage. Self-referential whenever target() and the payload belief are two
            # readings of the SAME object (a surface keypoint vs its centre): the constant
            # centre-to-surface offset then makes this unsatisfiable at any height.
            return float(self._pos(stage.payload)[2]) >= float(stage.target()[2]) - self.lift_tol
        if stage.gripper == "place":
            certified = self._place_certified(stage, flags)
            if certified is not None:
                return certified
            if self.advance_mode == "subgoal":
                # A released payload may no longer satisfy a hover-based subgoal.
                return bool(self.sensor.released()) and self._place_seen is not None
            if self.advance_mode == "env_flags" and stage.done_flag is not None and stage.done_flag in flags:
                return bool(flags.get(stage.done_flag, False))
            seated = (bool(stage.done())
                      or (self.flag_fallback and stage.done_flag is not None
                          and bool(flags.get(stage.done_flag, False))))
            # Require a stable seat before advancing to the next object.
            settled = (self._place_since is not None
                       and self.stage_replans - self._place_since >= self._place_settle)
            return seated and settled and self.sensor.released()
        if plan_auth:
            # Move stages with no payload (gripper "open"): the plan's sub-goal decides, and an
            # env progress flag may not co-confirm it.
            return bool(stage.done())
        return bool(_should_advance(stage, flags, 0, self.commit_hold))
