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
from vlm_dp.world import GTWorld, SensedWorld
from vlm_dp.cost.base_cost import CompositeCost
from vlm_dp.cost.terms import grasp_slack
from vlm_dp.hold import payload_held
from vlm_dp.stage import _capture_held, _should_advance
from vlm_dp.cost import guard_cost
from vlm_dp.context import build_context
from vlm_dp.grasp_recovery import debounce_gripper, descent_stalled, probe_pattern

# Planner styles that pass tcp_pos instead of ee_pos.
_TCP_STYLES = ("ref_style", "explore", "grasp_flow", "capsule_flow")


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
        self.stall_margin = float(adv.get("stall_margin", 0.15))
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
        # Re-perceive after repeated closed-empty grasps.
        self._regrasp_perceive = bool(adv.get("regrasp_perceive", False))
        # Allow stale beliefs to bypass appearance matching.
        self._relax_identity_when_stale = bool(adv.get("relax_identity_when_stale", False))
        self._ground_err_debug = bool(adv.get("ground_error_debug", False))
        self._ground_err_every = int(adv.get("ground_error_every", 5))
        # Suppress brief open commands while holding; zero disables the filter.
        self._grip_debounce = int(adv.get("gripper_open_debounce", 0))
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
        self.seat_shift = bool(cost_cfg.get("grounding", {}).get("seat_shift", True))
        self.local_grasp = bool(cost_cfg.get("grounding", {}).get("local_grasp", False))
        self.local_grasp_radius = float(cost_cfg.get("grounding", {}).get("local_grasp_radius", 0.05))
        self.rotate_grasp_offset = bool(cost_cfg.get("grounding", {}).get("rotate_grasp_offset", False))
        self.lift_latch_xy = bool(cost_cfg.get("grounding", {}).get("lift_latch_xy", False))
        self.seat_from_plane = bool(cost_cfg.get("grounding", {}).get("seat_from_plane", False))
        self.support_extents = bool(cost_cfg.get("grounding", {}).get("support_extents", False))
        self.kp_source = cost_cfg.get("grounding", {}).get("kp_source", "perception")
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
        self._grasp_probe = np.zeros(3)
        self._grasp_probe_idx = 0
        self._closed_empty = 0
        self._open_run = 0
        self._close_val = 1.0
        self._z_hist = []                    # Never stage-reset; contact detection spans stage changes.
        self._contact_prev = False
        self._last_cmd_close = False
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
                         contact_criterion=self.contact_criterion,
                         rotate_grasp_offset=self.rotate_grasp_offset,
                         lift_latch_xy=self.lift_latch_xy, seat_from_plane=self.seat_from_plane,
                         open_half=float(self.geom.get("open_half", 0.04)), **self.roles)
        self.world = self._build_world(raw_env, percep, src)
        self.grounding = src.ground(self.env, self.world)
        # Advance checks use grounding estimates rather than simulator poses.
        self._obj_pos = {o.name: o.pos for o in self.grounding.objects}
        self._extents = {o.name: o.extents for o in self.grounding.objects}
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
        self._contact_seen = False
        self._released_latch = False
        self._grasp_probe = np.zeros(3)
        self._grasp_probe_idx = 0
        self._closed_empty = 0
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
            if self.sensor.closed_on_air():
                if not self._reopen:
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
        # Record TCP motion and release state for retreat terms.
        eef_now = np.asarray(ctx["eef_pos"], dtype=np.float64)[:3]
        if getattr(self, "_eef_prev", None) is not None and int(executed_steps) > 0:
            ctx["eef_step_motion"] = float(np.linalg.norm(eef_now - self._eef_prev)) / int(executed_steps)
        self._eef_prev = eef_now
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
        return ctx

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
        if self._grip_debounce <= 0:
            return actions, []
        st = self.stage()
        name = st.payload or st.grasp_obj
        held = bool(name) and self._payload_held(name)
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
        if self.sensor_cadence == "replan":
            return                                   # observe_plan already sampled this chunk
        commanded_close = bool(float(torch.as_tensor(action_step).reshape(-1)[7]) > 0.5)
        self._last_cmd_close = commanded_close
        self._observe_sensors(commanded_close)

    def _observe_sensors(self, commanded_close: bool) -> None:
        """Record one shared world and aperture-sensor observation."""
        # Route both state modes through the world to preserve latched holds.
        st = self.stage()
        cand = {n for n in (st.grasp_obj, st.payload) if n}
        observe = getattr(self.world, "observe", None)
        if callable(observe):
            observe(self.env, commanded_close, candidates=cand or None)
        else:
            self.sensor.observe(self.env, commanded_close)

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
        """Return whether the configured hold authority reports the payload held."""
        return payload_held(payload, self.hold_authority, self.world, self.sensor,
                            self.env.tcp(), self._pos(payload))

    def _grip_half_width(self, payload):
        """Return the half-width used to predict the payload stall angle."""
        obj = next((o for o in self.grounding.objects if o.name == payload), None)
        ge = getattr(obj, "grasp_extent", None) if obj is not None else None
        if ge is not None:
            return float(ge)
        ext = self._extents.get(payload)
        return float(ext[0]) if ext is not None else None

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
        if self.sensor.closed_on_air():
            self._log_grip("empty hand", stage.payload)
            return "empty hand"
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

    def _stage_reached(self, stage, flags):
        """Return whether the current stage target is satisfied."""
        if stage.gripper == "close" and stage.grasp_obj is not None:
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
            if self.advance_mode == "subgoal":  # Subgoal predicate.
                return bool(stage.done())
            return float(self._pos(stage.payload)[2]) >= float(stage.target()[2]) - self.lift_tol
        if stage.gripper == "place":
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
        return bool(_should_advance(stage, flags, 0, self.commit_hold))
