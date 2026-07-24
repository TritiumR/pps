"""Attaches the VLM grounding and its CompositeCost to eval_steering's sim_free_mpc planner."""
from __future__ import annotations

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
from vlm_dp.stage import _capture_held, _should_advance
from vlm_dp.cost import guard_cost
from vlm_dp.context import build_context
from vlm_dp.grasp_recovery import probe_pattern

# Styles the planner calls with tcp_pos=. CompositeCost only accepts ee_pos=, which is style priority.
_TCP_STYLES = ("ref_style", "explore", "grasp_flow", "capsule_flow")


class VlmDpBridge:
    """Swaps the planner's cost and feeds it grounding-derived context each replan."""

    def __init__(self, ground, roles, cost_cfg, *, task_key=None, device="cuda:0",
                 commit_hold=5, lift_tol=0.01, grasp_confirm=10, rise_confirm=0.01,
                 grasp_eps=0.02, state="gt", track="fk", segment="groundedsam",
                 vocab=None, fixtures=(), reperceive_every=8):
        """Args:
          ground: grounding source, one of gt, rekep_fake, rekep_real (and _vlm variants).
          roles: {grasp_obj, place_obj, grasp_objs} from the task_prompts.json entry.
          cost_cfg: parsed cost YAML (cost.terms and cost.geometry).
          task_key: short task key for the rekep sources. Required, no task default.
          commit_hold: forwarded to _should_advance.
          lift_tol: metres below the lift target that still counts as lifted.
          grasp_confirm: replans a payload may stay unrisen before it counts as lost (a cage).
          rise_confirm: metres the payload must rise from stage entry to count as carried.
          grasp_eps: metres from the grasp target that counts as reached.
          state: gt reads object poses from the sim, real from a SensedWorld (perception and
            FK-while-held). track, segment and reperceive_every configure it.
          vocab: task_prompts objects entry mapping name to detector text. Required for real.
          fixtures: objects whose geometry is workcell calibration, not perception.
        """
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
        self._geom_ns = types.SimpleNamespace(**self.geom)   # terms.grasp_slack reads attributes
        self.task_key = task_key
        self.device = device
        self.commit_hold = int(commit_hold)
        self.lift_tol = float(lift_tol)
        self.grasp_confirm = int(grasp_confirm)
        self.rise_confirm = float(rise_confirm)
        self.grasp_eps = float(grasp_eps)
        # Ablation switches (config advance and grounding sections). Defaults are the standard loop.
        adv = cost_cfg.get("advance", {})
        self.flag_fallback = bool(adv.get("flag_fallback", True))   # env flag may co-confirm a place
        self.backtrack_enabled = bool(adv.get("backtrack", True))
        self.advance_mode = adv.get("mode", "sensed")               # sensed or env_flags
        # Grasp recovery on a closed-empty: reopen re-closes on the same point (the absorbing state that
        # hammers), search steps a bounded feel-around so each retry probes a nearby pose. Default reopen.
        self._grasp_recovery = adv.get("grasp_recovery", "reopen")
        self._probe_pts = probe_pattern(float(adv.get("grasp_search_radius", 0.045)))
        self.stall_margin = float(adv.get("stall_margin", 0.15))    # thin grips need a tighter band
        self.settle_eps = float(adv.get("settle_eps", 0.01))        # light objects jitter in the grip
        self.seat_shift = bool(cost_cfg.get("grounding", {}).get("seat_shift", True))
        self.local_grasp = bool(cost_cfg.get("grounding", {}).get("local_grasp", False))
        self.local_grasp_radius = float(cost_cfg.get("grounding", {}).get("local_grasp_radius", 0.05))
        self.kp_source = cost_cfg.get("grounding", {}).get("kp_source", "perception")
        # feasibility (default) decides pinch versus press by measured geometry. plan restores the old
        # plan-structural inference, which cannot see whether the gripper actually fits.
        self.contact_criterion = cost_cfg.get("grounding", {}).get("contact_criterion", "feasibility")
        if self.state == "real" and (self.flag_fallback or self.advance_mode == "env_flags"):
            print("[vlm_dp] WARNING: --vlm_state real with env-flag advance (flag_fallback or env_flags) "
                  "lets privileged flags co-confirm stages. Use the flag-free advance config for an "
                  "honest sensed-state run.", flush=True)
        # Per-episode state, set in reset():
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
        self._grasp_probe = np.zeros(3)      # current feel-around offset in search mode, zero otherwise
        self._grasp_probe_idx = 0

    def attach_cost(self, mpc):
        """Replace the planner's cost. Call once, the cost is stateless across episodes."""
        style = mpc.config.cost_style
        if style != "priority" or style in _TCP_STYLES:
            raise ValueError(
                f"--vlm_cost requires --mpc_cost priority, got cost_style={style!r}. CompositeCost "
                f"has no tcp_pos= parameter, and sim_free_mpc/planner.py calls {_TCP_STYLES} "
                f"with tcp_pos=.")
        mpc.cost = guard_cost(CompositeCost(self.terms, self.geom))

    _SETTLE_SECONDS = 1.0   # objects are still falling and settling right after env.reset()

    def _settle(self, raw_env):
        """Let the scene come to rest before anything looks at it. Grounding on a mid-fall frame
        mis-places every capture (keypoints, masks, tracker seeds, grasp_z0)."""
        for _ in range(int(round(self._SETTLE_SECONDS / raw_env.physics_dt))):
            raw_env.sim.step(render=False)
            raw_env.scene.update(dt=raw_env.physics_dt)
        for _ in range(3):                  # camera annotators lag a frame, render until current
            raw_env.sim.render()
            raw_env.scene.update(dt=raw_env.physics_dt)

    def reset(self, raw_env):
        """Re-bind and re-ground. Must run after every ``env.reset()``."""
        self._settle(raw_env)
        self.env = DroidEnv.attach(raw_env, device=self.device)
        self.sensor = ApertureGraspSensor(stall_margin=self.stall_margin, settle_eps=self.settle_eps)
        # Sensed state -> sensed grounding too: rekep sources take their masks/extents from the
        # same Perception instead of sim instance segmentation (gt grounding ignores it).
        percep = self._build_perception() if self.state == "real" else None
        src = get_source(self.ground_name, task_key=self.task_key, perception=percep,
                         seat_shift=self.seat_shift, local_grasp=self.local_grasp,
                         local_grasp_radius=self.local_grasp_radius, kp_source=self.kp_source,
                         contact_criterion=self.contact_criterion,
                         open_half=float(self.geom.get("open_half", 0.04)), **self.roles)
        self.world = self._build_world(raw_env, percep, src)
        self.grounding = src.ground(self.env, self.world)
        # Object positions for advance checks come from the grounding's own closures (its estimate),
        # so a non-GT grounding is not silently judged against simulator poses.
        self._obj_pos = {o.name: o.pos for o in self.grounding.objects}
        self._extents = {o.name: o.extents for o in self.grounding.objects}
        self.stage_idx = 0
        self.held_offset = _capture_held(self.env, self.grounding,
                                         self.grounding.stages[0].held_idx)
        self.plan_ref = None
        self.grasp_z0 = {}
        self._enter_stage()

    def _build_perception(self):
        from vlm_dp.perception import Perception           # deferred: ~3GB of vision weights
        if not self.vocab:
            raise SystemExit("[vlm_dp] --vlm_state real needs the task's 'objects' vocabulary "
                             "in task_prompts.json")
        return Perception(self.vocab, fixtures=self.fixtures, segment=self.segment,
                          support=self.roles.get("support"),
                          support_names=self.roles.get("grasp_objs") or ())

    def _build_world(self, raw_env, percep, src):
        """The object-state source: the sim (gt), or perception and FK-while-held (real)."""
        if self.state != "real":
            return GTWorld(raw_env)                        # raw env, grounding needs the wrapper
        world = SensedWorld(percep, self.sensor, place_obj=self.roles.get("place_obj"),
                            tcp_offset=ROBOTIQ_GRASP_OFFSET, track=self.track,
                            reperceive_every=self.reperceive_every)
        percep.calibrate(self.env)
        world.refresh(self.env)
        missing = [n for n in self.vocab if n not in world.names]
        required = {self.roles.get("grasp_obj"), self.roles.get("place_obj"),
                    self.roles.get("support"), *(self.roles.get("grasp_objs") or ())} - {None}
        fatal = [n for n in missing if n in required]
        if fatal:
            raise SystemExit(f"[vlm_dp] perception did not find {fatal}; refusing to run half-blind")
        if missing:
            # An unseen obstacle degrades the clear term, it does not invalidate the run.
            print(f"[vlm_dp] WARNING: obstacles not found this frame: {missing}; "
                  f"continuing without them", flush=True)
        # Workcell-calibration seeds (e.g. an articulation's lip) must exist before the visual
        # tracker is built: CoTracker registers its query points once, at priming.
        cal = getattr(src, "calibration_points", None)
        for name, pos in (cal(self.env) if cal else {}).items():
            world.seed(name, pos)
        if self.track == "visual":
            from vlm_dp.visual_tracker import VisualTracker
            init = {n: world.object_pose(n)[0] for n in world.names}
            world.visual = VisualTracker(self.env.cam, world.names, init)
        print(f"[vlm_dp] sensed world: {world.names} (track={self.track})", flush=True)
        return world

    def stage(self):
        return self.grounding.stages[self.stage_idx]

    def _enter_stage(self):
        """Stage-entry bookkeeping: reset the replan counter and the place-grace timer, and
        capture the never-rose datum (a grasp stage records the object's z, so the following lift
        can tell a real rise from a caged push)."""
        self.stage_replans = 0
        self._place_seen = None
        self._place_since = None
        self._reopen = False
        self._grasp_probe = np.zeros(3)      # each stage starts its grasp search at the estimate
        self._grasp_probe_idx = 0
        st = self.stage()
        if st.gripper == "close" and st.grasp_obj is not None:
            self.grasp_z0[st.grasp_obj] = float(self._pos(st.grasp_obj)[2])

    def context(self, raw_env, obs, executed_steps=0):
        """World-frame cost context for the current stage (one per replan). ``executed_steps``:
        how many steps of the previously recorded chunk actually ran (shifts the consistency ref)."""
        placed = frozenset(s.payload for s in self.grounding.stages[:self.stage_idx]
                           if s.gripper == "place" and s.payload is not None)
        ref = None
        if self.plan_ref is not None:
            k = min(max(int(executed_steps), 0), self.plan_ref.shape[0])
            ref = torch.cat([self.plan_ref[k:], self.plan_ref[-1:].expand(k, 7)], dim=0) if k else self.plan_ref
        ctx = build_context(raw_env, obs, self.grounding, self.stage(),
                            plan_ref=ref, held_offset=self.held_offset,
                            placed=placed)
        ctx["destination"] = self.roles.get("place_obj")
        # Reopen recovery: a hand that closed on air AT the grasp target is an absorbing state --
        # the close gate is ~1 there and nothing else ever reopens it. Command open until the
        # fingers actually release, then let the ordinary close gate retry.
        st = self.stage()
        # A PRESS contact is exempt: it deliberately closes on a thin/articulated part that need not read
        # as held, so the reopen recovery would keep prying the gripper off the thing it is levering.
        if st.gripper == "close" and self.sensor is not None and getattr(st, "contact", "pinch") != "press":
            if self.sensor.closed_on_air():
                if not self._reopen:
                    print("[vlm_dp] closed-empty at grasp target: commanding reopen", flush=True)
                    if self._grasp_recovery == "search":   # feel around, do not re-close on the same point
                        self._grasp_probe_idx += 1
                        self._grasp_probe = self._probe_pts[self._grasp_probe_idx % len(self._probe_pts)]
                        print(f"[vlm_dp] grasp search -> probe {self._grasp_probe_idx} "
                              f"{np.round(self._grasp_probe, 3)}", flush=True)
                self._reopen = True
            elif self.sensor.is_open():
                self._reopen = False
        else:
            self._reopen = False
        if self._reopen:
            ctx["gripper_intent"] = "open"
        # Offset the grasp target by the current feel-around probe, so the cost and the reached-check
        # below both chase the same shifted pose. Zero in reopen mode, so shipped configs are unchanged.
        if self._grasp_recovery == "search" and st.gripper == "close" \
                and getattr(st, "contact", "pinch") != "press":
            ctx["target"] = np.asarray(ctx["target"], dtype=np.float32) + self._grasp_probe.astype(np.float32)
        return ctx

    def observe_plan(self, actions):
        """Record the decoded chunk (unshifted) for the next replan's consistency reference."""
        self.plan_ref = torch.as_tensor(actions, dtype=torch.float32, device=self.device)[..., :7]
        commanded_close = bool(float(torch.as_tensor(actions)[0, 7]) > 0.5)
        if self.state == "real":                           # SensedWorld drives the shared sensor itself
            st = self.stage()
            cand = {n for n in (st.grasp_obj, st.payload) if n}
            self.world.observe(self.env, commanded_close, candidates=cand or None)
        else:
            self.sensor.observe(self.env, commanded_close)

    def _pos(self, name):
        """Estimated position of a grounding object (closure), with the world as fallback."""
        fn = self._obj_pos.get(name)
        return np.asarray(fn() if fn is not None else self.world.object_pose(name)[0], dtype=np.float64)

    def viz(self):
        """Drawable overlay state: live keypoints + the current stage's constraint geometry."""
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

    def _payload_held(self, payload):
        """The payload is really in the hand: a settled finger stall near it (a caged push is not)."""
        return self.sensor.held_object({payload: self._pos(payload)}, self.env.tcp()) == payload

    _PLACE_GRACE = 6     # replans a placed payload may read not-seated (flicker) before empty-hand re-arms
    _PLACE_SETTLE = 5    # replans a placement must hold stably before advancing to the next object
    _REPERCEIVE_EVERY = 6  # while an object is stale, re-segment every N replans (not every one)

    # Finger-angle <-> grip-width map, calibrated once for this gripper (pear/apple stall anchors).
    _AP_SLOPE = 9.3    # rad per metre of grip width
    _AP_BAND = 0.16    # rad: tolerated deviation from the width-predicted stall angle

    def _grip_ok(self, payload):
        """The settled stall angle matches the payload's width. A sliver-of-skin pinch reads far more
        closed than the object's equator and will not survive the transit, so re-grasp now instead of
        dropping mid-carry."""
        ext = self._extents.get(payload)
        if ext is None:
            return True
        predicted = self.sensor.q_free - self._AP_SLOPE * (2.0 * float(ext[0]))
        return abs(self.sensor.aperture() - predicted) <= self._AP_BAND

    def advance(self, flags):
        """One control-loop decision, in ReKep Algorithm-1 form: backtrack one stage if a stage
        invariant is violated (checked every replan, no debounce), else advance when the stage's
        target is reached. World-state reasoning lives only in the invariants."""
        stage = self.stage()
        self.stage_replans += 1
        # Advance and context read the held estimate below, and observe() only updates it at the end of
        # the previous replan, so re-carry it to the current joints first or it lags a whole chunk.
        self.world.sync_fk(self.env)
        # A stale belief (contact lost mid-carry) is re-searched until the object is re-found, throttled
        # to every _REPERCEIVE_EVERY replans: a full re-segmentation is seconds of wall time, and the
        # object does not move faster than that between looks.
        if self.world.stale() and self.stage_replans % self._REPERCEIVE_EVERY == 0:
            self.world.refresh(self.env)
        violated = self._invariant_violated(stage, flags) if self.backtrack_enabled else None
        if violated:
            if self.stage_idx > 0:
                self.stage_idx -= 1
                self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
                self._enter_stage()
                print(f"[vlm_dp] BACKTRACK -> {self.stage_idx}: {self.stage().name} ({violated})",
                      flush=True)
            return
        if self._stage_reached(stage, flags) and self.stage_idx + 1 < len(self.grounding.stages):
            self.stage_idx += 1
            self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
            self._enter_stage()
            print(f"[vlm_dp] stage -> {self.stage_idx}: {self.stage().name}", flush=True)

    def _invariant_violated(self, stage, flags):
        """Stage path-constraint check (sensed): empty hand while a payload should be held, or a
        payload that never rose within the confirm window (a caged push). Returns the reason or None."""
        if stage.payload is None:
            return None
        if getattr(stage, "contact", "pinch") == "press":
            # A press never certifies a hold. The aperture-held test is a pinch concept: a thin
            # articulated part reads as an empty hand even while it is being levered, so this invariant
            # would backtrack the press forever (measured on the capsule lid: 12 advances, 12 backtracks).
            return None
        placed_now = stage.gripper == "place" and (
            bool(stage.done())
            or (self.flag_fallback and stage.done_flag is not None
                and bool(flags.get(stage.done_flag, False))))
        # An open, empty hand on the seat is the deliberate release, not a drop, so do not backtrack.
        # But this is a live check plus a short grace, not a permanent latch: if the object then leaves
        # the seat (bounced or rolled off), the grace lapses and the empty-hand recovery re-arms, so the
        # stage cannot get stuck servicing a fallen payload it can never re-place.
        if placed_now:
            self._place_seen = self.stage_replans
            if self._place_since is None:     # start the settle timer at the first continuous placement
                self._place_since = self.stage_replans
            return None
        self._place_since = None              # left the seat: the placement was not stable, re-time it
        if self._place_seen is not None and self.stage_replans - self._place_seen < self._PLACE_GRACE:
            return None                       # brief flicker of the seat check, not an actual departure
        if self.advance_mode == "env_flags":
            # Privileged-signal ablation: the env grasp flag drops -> the hold is lost (her signal).
            return None if flags.get(f"grasp_{stage.payload}", False) else "flag dropped"
        if self.sensor.closed_on_air():
            return "empty hand"
        if self.sensor.holding() and not self._grip_ok(stage.payload):
            return "poor grip"
        # never-rose imposes a fixed lift the VLM never specified. Under sub-goal advance the place
        # sub-goal owns the height, and empty-hand and poor-grip already catch a failed caged grasp.
        z0 = self.grasp_z0.get(stage.payload)
        if (self.advance_mode != "subgoal" and z0 is not None
                and self.stage_replans >= self.grasp_confirm
                and float(self._pos(stage.payload)[2]) < z0 + self.rise_confirm):
            return "never rose"
        return None

    def _grasp_slack(self, name):
        """The cost's own at-the-grasp-pose tolerance for this object (terms.grasp_slack).

        Shared on purpose: the stage machine must not advance on a proximity at which the cost would
        still command the gripper open. Falls back to grasp_eps only when the object has no measured
        geometry to derive from.
        """
        if name is None:
            return self.grasp_eps
        obj = next((o for o in self.grounding.objects if o.name == name), None)
        radius = getattr(obj, "grasp_extent", None) if obj is not None else None
        if radius is None:
            ext = self._extents.get(name)
            radius = ext[1] if ext is not None else None
        return self.grasp_eps if radius is None else grasp_slack(self._geom_ns, float(radius))

    def _stage_reached(self, stage, flags):
        """The stage's target is reached (pose/geometry proximity, as in Algorithm 1)."""
        if stage.gripper == "close" and stage.grasp_obj is not None:
            if getattr(stage, "advance_on_done", False):    # task-state goal (e.g. a lid angle), not a hold
                return bool(stage.done())
            if self.advance_mode == "env_flags":
                return bool(flags.get(f"grasp_{stage.grasp_obj}", False))
            # subgoal mode falls through: grasp is execution (proximity to the body centre plus grasp
            # sensed), not the VLM surface-keypoint sub-goal.
            # target plus the feel-around probe: the arm grasps at the shifted pose, so reached must
            # measure to it too, or a successful off-estimate grasp would hold but never advance. Zero
            # in reopen mode.
            tgt = np.asarray(stage.target()) + self._grasp_probe
            near = float(np.linalg.norm(np.asarray(self.env.tcp()) - tgt)) < self._grasp_slack(stage.grasp_obj)
            if getattr(stage, "contact", "pinch") == "press":
                # A press contact: advance on reaching plus the close having completed, not on holding(),
                # whose stall band a thin part never satisfies. closed() is the weakest honest claim the
                # angle supports here, and the next stage's sub-goal owns what the contact achieves.
                return near and self.sensor.closed()
            return near and self._payload_held(stage.grasp_obj)
        if stage.gripper == "hold" and stage.payload is not None:
            if getattr(stage, "advance_on_done", False):    # e.g. an articulated pull: done is joint-based
                return bool(stage.done())
            if self.advance_mode == "subgoal":              # a move/hold sub-goal is its own predicate
                return bool(stage.done())
            return float(self._pos(stage.payload)[2]) >= float(stage.target()[2]) - self.lift_tol
        if stage.gripper == "place":
            if self.advance_mode == "subgoal":
                # The VLM place sub-goal was satisfied (grace-tracked in _place_seen) and the gripper has
                # opened. Do not require the sub-goal to still hold post-release: a hover-then-release
                # sub-goal stops holding the instant the object leaves the hand.
                return bool(self.sensor.released()) and self._place_seen is not None
            if self.advance_mode == "env_flags" and stage.done_flag is not None and stage.done_flag in flags:
                return bool(flags.get(stage.done_flag, False))
            seated = (bool(stage.done())
                      or (self.flag_fallback and stage.done_flag is not None
                          and bool(flags.get(stage.done_flag, False))))
            # Settle-confirm: the payload must stay seated for _PLACE_SETTLE replans before we advance to
            # the next object. A marginal placement that rolls off within the window resets the timer
            # (see _invariant_violated), so we do not commit to the next object on an unstable one.
            settled = (self._place_since is not None
                       and self.stage_replans - self._place_since >= self._PLACE_SETTLE)
            return seated and settled and self.sensor.released()
        return bool(_should_advance(stage, flags, 0, self.commit_hold))

