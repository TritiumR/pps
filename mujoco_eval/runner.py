"""Run MuJoCo evaluation rollouts with optional sampling, steering, and perturbation mechanisms."""
from __future__ import annotations

import functools
import sys
import time
import types

import numpy as np
import torch
import yaml

from . import paths, record
paths.ensure_repo_on_path()


import sim_common.envs
_droid_stub = types.ModuleType("sim_common.envs.droid")
_droid_stub.DroidEnv = None
sys.modules.setdefault("sim_common.envs.droid", _droid_stub)

from sim_free_mpc.action_space import decode_model_action_chunks
from sim_free_mpc.fk import PandaFK
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig
from vlm_dp.bridge import VlmDpBridge
from vlm_dp.cost import guard_cost
from vlm_dp.grasp_sensor import ApertureGraspSensor
from vlm_dp.stage import _capture_held

from .env.mujoco_env import MuJoCoEnv, MGWorld
from .grounding.gt import MGGroundingSource
from .sampling.beam import Beam, BeamConfig
from .sampling.keypose import KPConfig, KPCost, KPPlanner

LOG = "[mujoco-eval]"


class _Stats:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)


class ChunkDecodePolicy:
    """Provide action normalization metadata for SimFreeMPC decoding."""

    def __init__(self, action_std):
        one = np.ones(8, dtype=np.float32)
        act_std = np.asarray(action_std, dtype=np.float32)
        # [D] (one scale for every chunk row) or [H, D] (the proxy's per-row scales).
        act_mean = np.zeros_like(act_std)
        act_mean[..., 7] = 0.5
        self._metadata = {
            "output_norm_stats": {"actions": _Stats(act_mean, act_std),
                                  "state": _Stats(np.zeros(8, np.float32), one)},
            "use_quantile_norm": False,
            "output_norm_stats_source": "demo_delta_stats",
        }


def demo_delta_stats(hdf5, horizon, n_demos=100):
    """Estimate action-delta standard deviations from demonstration data."""
    import h5py
    with h5py.File(hdf5, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:n_demos]
        deltas = []
        for n in names:
            q = np.asarray(f[f"data/{n}/obs/robot0_joint_pos"])
            for h in range(1, horizon + 1):
                deltas.append(q[h:] - q[:-h])
    std = np.concatenate(deltas, 0).std(0)
    return np.concatenate([std, [0.5]]).astype(np.float32)


_SENSOR = dict(q_free=0.078, stall_margin=0.012, q_touch=0.008,
               settle_eps=0.004, settle_steps=12, close_steps=12, proximity=0.10)


class MGBridge(VlmDpBridge):
    """Adapt VlmDpBridge to the MuJoCo evaluation environment."""

    _AP_SLOPE = 1.0
    _AP_BAND = 0.02

    def __init__(self, source, cost_cfg, **kw):
        super().__init__("gt", source.roles, cost_cfg, **kw)
        self._source = source

    def reset(self, raw_env):
        """Initialize bridge state for one MuJoCo episode."""
        self.env = raw_env

        self.sensor = ApertureGraspSensor(**_SENSOR, stall_margin_enter=self.hold_enter,
                                          stall_margin_exit=self.hold_exit)
        self.world = MGWorld(raw_env, sensor=self.sensor, names=self._source.movable)
        self.grounding = self._source.ground(self.env, self.world)
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
        self._eef_prev = None
        self._reset_churn()
        self._enter_stage()


_PLANNER_KEYS = ("delta_clip", "interpolate", "interpolate_frequency",
                 "interpolate_high_frequency", "interpolation_method")


def apply_planner_config(args, cfg):
    """Apply supported planner overrides from the configuration."""
    block = cfg.get("planner")
    if not block:
        return
    unknown = sorted(set(block) - set(_PLANNER_KEYS))
    if unknown:
        raise SystemExit(f"unknown planner config keys {unknown}; allowed {list(_PLANNER_KEYS)}")
    for key, value in block.items():
        if key == "interpolate" and isinstance(value, bool):
            value = "on" if value else "off"
        setattr(args, key, value)
    print(f"{LOG} planner overrides: {dict(block)}", flush=True)


def _proxy_action_std(checkpoint, horizon):
    """Return the proxy checkpoint's own [H, D] action std, or None if it ships none."""
    import json, pathlib as _pl
    f = _pl.Path(checkpoint) / "action_norm_stats.json"
    if not f.exists():
        return None
    std = np.asarray(json.loads(f.read_text())["std"], dtype=np.float32)
    return std[:horizon] if std.shape[0] >= horizon else None


def build_planner(args, fit, kp_cfg=None):
    action_std = demo_delta_stats(args.hdf5, args.horizon)
    if getattr(args, "align_proxy_norm", "off") == "on":
        # PPS requires the base and proxies to share one action representation. Ours do not: the
        # planner uses a single std per dim for every chunk row, the proxy a per-(row, dim) std
        # that grows ~13x across the chunk. Mapping a noisy planner x_t into proxy space then
        # lands 4.7x off at row 0 and 2.8x off at row 14, so the proxy scores an input that does
        # not match the level it is conditioned on. Adopting its stats makes the bridge identity.
        std = _proxy_action_std(args.proxy_checkpoint, args.horizon)
        if std is None:
            raise SystemExit("--align_proxy_norm needs action_norm_stats.json in --proxy_checkpoint")
        action_std = std
        print(f"{LOG} align_proxy_norm: planner adopts the proxy's [H, D] action std", flush=True)
    print(f"{LOG} demo delta std (model-space scale): {np.round(action_std, 4)}", flush=True)
    make = (lambda p, c: KPPlanner(p, c, kp_cfg)) if kp_cfg is not None else SimFreeMPC
    cfg_kw = dict(
        task_name=args.task, num_samples=args.candidates, iterations=1,
        noise=args.noise, temperature=args.temperature, action_dims=8,
        joint_delta_clip=args.delta_clip, cost_style="priority",
        optimize_space="action", sampler="base", grad_calc="mbd",
        control_frequency=20.0,
        rank_mode=args.rank_mode, prior_weight=args.prior_weight,
        prior_weight_high=args.prior_weight_high,
        prior_weight_schedule=args.prior_weight_schedule)
    if args.interpolate == "on":

        cfg_kw.update(interpolate=True, interpolation_method=args.interpolation_method,
                      interpolate_frequency=args.interpolate_frequency,
                      control_frequency=args.interpolate_high_frequency)
    planner = make(ChunkDecodePolicy(action_std), SimFreeMPCConfig(**cfg_kw))
    if args.interpolate == "on":
        print(f"{LOG} interpolate: {args.interpolation_method} "
              f"{args.interpolate_frequency}/{args.interpolate_high_frequency} Hz -> "
              f"{planner._interpolation_knot_count(args.horizon)} knots of {args.horizon} rows",
              flush=True)

    q_off = fit["orientation"][fit["stored_quat_convention"]]["R_off_quat_wxyz"]
    planner.fk = PandaFK(ee_offset=tuple(fit["tcp_offset_link8"]),
                         ee_offset_quat_wxyz=tuple(q_off))
    return planner


def infer_chunk(planner, env, ctx, args, stage_key=None, steer=None, x0_init=None, out=None):
    """Run one denoising chain and decode the resulting action chunk."""
    if steer is not None and steer.mode == "expert":
        steer.begin_replan(env, args.num_steps + 1)
        return steer.expert_chunk(), {}, None

    q0 = env.q0()
    state = torch.zeros(8, dtype=torch.float32)
    state[:7] = q0
    inputs = {"state": state}
    kp = isinstance(planner, KPPlanner)
    rows = args.horizon + (planner.kp.k if kp else 0)
    x_t = torch.randn(1, rows, 8) if x0_init is None else x0_init.clone()
    w_plan = None
    if kp:
        w0 = planner.kp_begin_replan(ctx, stage_key)
        x_t[0, args.horizon:, 3:] = 0.0
        if w0 is not None:
            x_t[0, args.horizon:, :3] = w0
    if steer is not None:
        steer.begin_replan(env, args.num_steps + 1)
        if steer.mode == "additive":
            ctx["score_addend"] = steer.score_addend

    planner.begin_inference()
    stats = {}
    for it in range(args.num_steps + 1):
        if steer is not None and steer.mode == "tilt":
            t = steer.tilt_for(it, args.num_steps + 1, planner.policy, args.tilt_lambda,
                               args.temperature, args.noise, dims=args.tilt_dims,
                               ess_cap=args.tilt_ess_cap,
                               discrimination=args.tilt_discrimination == "on")
            if t is None:
                ctx.pop("task_tilt", None)
            else:
                ctx["task_tilt"] = t
        elif steer is not None and steer.mode != "additive":
            inj = steer.inject_for(it, args.num_steps + 1, planner.policy)
            if inj is None:
                ctx.pop("inject", None)
            else:
                ctx["inject"] = inj
        x_t, stats = planner.step_mbd_score_action_prox(
            x_t, inputs, ctx, iteration=it, num_iterations=args.num_steps + 1)

    if steer is not None:
        ctx.pop("inject", None)
        ctx.pop("score_addend", None)
        ctx.pop("task_tilt", None)
    if kp:
        w_plan = planner.kp_finish_replan(x_t)
        x_t = x_t[:, :args.horizon]
    if out is not None:
        out["x_final"] = x_t.detach().clone()
    dec = decode_model_action_chunks(planner.policy, inputs, x_t, apply_clamp=True,
                                     current_joint_pos=q0, max_joint_delta=args.delta_clip)
    return dec.real_actions[0].detach().cpu().numpy(), stats, w_plan


def select_chunk(planner, env, ctx, args, stage_key, steer):
    """Proxy-ranked selection: draw M base plans, execute the one the proxy most agrees with.

    Deployable steering, unlike best-of-M over rollouts: one episode, one reality, M denoise
    chains inside a single replan. The proxy RANKS rather than proposes, so the base cost --
    whose prior terms drive injected candidates to zero softmax weight -- never gets a veto.
    """
    steer.begin_replan(env, args.num_steps + 1)
    ref = steer.expert_chunk()                       # the proxy's own clean chunk, real joints
    plans, stats_all, dists = [], [], []
    for _ in range(args.select_m):
        plan, st, _ = infer_chunk(planner, env, ctx, args, stage_key=stage_key)
        h = min(len(ref), len(plan))
        plans.append(plan)
        stats_all.append(st)
        dists.append(float(np.abs(np.asarray(plan)[:h, :7] - ref[:h, :7]).mean()))
    best = int(np.argmin(dists))
    rec = {"select_m": args.select_m, "select_best": best,
           "select_dist_best": round(dists[best], 4),
           "select_dist_worst": round(max(dists), 4),
           "select_dist_mean": round(float(np.mean(dists)), 4)}
    return plans[best], stats_all[best], rec


def _build_source(args):
    if args.ground in ("rekep", "rekep_vlm"):
        from .grounding.rekep import MGRekepGroundingSource, MGRekepVlmGroundingSource
        cls = MGRekepVlmGroundingSource if args.ground == "rekep_vlm" else MGRekepGroundingSource
        return cls(args.task, args.rekep_context, args.rekep_constraints)
    return MGGroundingSource(args.task)


def _build_keypose(args, cfg):
    if not args.kp:
        return None
    if args.interpolate == "on":
        raise SystemExit("--interpolate does not support --kp (waypoint rows are not action rows)")
    kc = dict(cfg.get("keypose", {}))
    if args.kp_align is not None:
        kc["align"] = args.kp_align
    kp_cfg = KPConfig(**kc)
    print(f"{LOG} keypose: {kp_cfg}", flush=True)
    return kp_cfg


def _build_beam(args, planner):
    if not args.beam:
        return None
    if args.kp:
        raise SystemExit("--beam does not support --kp (the waypoint rows are not warm plans)")
    if args.steer != "off":
        raise SystemExit("--beam is base self-improvement: keep it off the proxy arms")
    beam = Beam(BeamConfig(k=args.beam, warm=args.beam_warm,
                           resample_every=args.beam_resample, ema=args.beam_ema,
                           w_rate=args.beam_w_rate, w_subgoal=args.beam_w_subgoal,
                           stage_reset=args.beam_stage_reset == "on"),
                planner, horizon=args.horizon, spi=args.spi)
    print(f"{LOG} beam: {beam.cfg}", flush=True)
    return beam


def _build_steering(args, planner):
    """Construct proxy steering and its client when enabled."""
    if args.steer == "off":
        return None, None
    if args.kp and args.steer in ("inject", "proxy_only"):
        raise SystemExit(f"--steer {args.steer} does not support --kp: it writes whole candidates "
                         "into the sampler, which would overwrite the waypoint rows")
    if not args.proxy_checkpoint:
        raise SystemExit("--steer needs --proxy_checkpoint")
    from .steering.proxy import (AdditiveScoreSteering, ProxyScoreClient, ProxySteering,
                                 TASK_PROMPTS)
    prompt = args.proxy_prompt or TASK_PROMPTS.get(args.task)
    if prompt is None:
        raise SystemExit(f"No trained-proxy prompt for task {args.task!r}; pass --proxy_prompt")
    client = ProxyScoreClient(args.proxy_checkpoint, prompt, device=args.proxy_device,
                              prediction_mode=args.proxy_prediction_mode,
                              kv_cache=args.proxy_kv_cache == "on",
                              fp16=args.proxy_fp16 == "on").start()
    print(f"{LOG} proxy: {args.steer} on {args.proxy_device} "
          f"({args.proxy_prediction_mode})", flush=True)
    import atexit
    atexit.register(client.close)
    if args.steer == "additive":
        steer = AdditiveScoreSteering(client, gamma=args.steer_gamma,
                                      policy=planner.policy, horizon=args.horizon,
                                      score_cap=args.proxy_score_cap, seed=args.seed,
                                      at=args.proxy_score_at, ref=args.steer_ref)
    else:
        steer = ProxySteering(args.steer, client, rho=args.inject_rho,
                              schedule=args.inject_schedule,
                              ddim_train_timesteps=planner.config.ddim_num_train_timesteps,
                              horizon=args.horizon, base_seed=args.seed)
    return steer, client


def _build_perturb(args, source):
    if args.perturb == "none":
        return None
    from .perturb.protocol import Perturbation
    print(f"{LOG} perturb: {args.perturb} at {args.perturb_at} "
          f"mag {args.perturb_mag} obj {args.perturb_obj}", flush=True)
    return Perturbation(args.perturb, args.perturb_at, args.perturb_mag,
                        args.perturb_obj, args.seed, source.movable)


def _setup(args):
    """Build all episode components in dependency order."""
    env = MuJoCoEnv(args.hdf5, args.fk_fit, visual_only_render=args.visual_only_render == "on")
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    apply_planner_config(args, cfg)

    kp_cfg = _build_keypose(args, cfg)
    source = _build_source(args)
    bridge = MGBridge(source, cfg, task_key=args.task, device="cpu")
    planner = build_planner(args, env.fk_fit, kp_cfg)
    bridge.attach_cost(planner)
    if kp_cfg is not None:
        planner.cost = guard_cost(KPCost(planner.cost, kp_cfg))
    beam = _build_beam(args, planner)
    steer, client = _build_steering(args, planner)


    torch.manual_seed(args.seed if args.sampler_seed is None else args.sampler_seed)
    env.reset(seed=args.seed)
    planner.reset_episode()
    if beam is not None:
        beam.reset_episode()
    bridge.reset(env)
    ensemble = None
    if getattr(args, "expert_ensemble", 0) and steer is not None and steer.mode == "expert":
        from .steering.ensemble import ChunkEnsemble
        ensemble = ChunkEnsemble(decay=args.ensemble_decay, keep=args.expert_ensemble)
        print(f"{LOG} temporal ensembling: keep {args.expert_ensemble} chunks, "
              f"decay {args.ensemble_decay}", flush=True)
    return types.SimpleNamespace(env=env, source=source, bridge=bridge, planner=planner,
                                 kp_cfg=kp_cfg, beam=beam, steer=steer, client=client,
                                 ensemble=ensemble, perturb=_build_perturb(args, source))


def _episode_summary(args, s, replan_s, episode_s, success, stage_max):
    final = {
        "kind": "episode", "task": args.task, "seed": args.seed, "success": success,
        "env_steps": s.env.n_steps, "stage_final": s.bridge.stage_idx, "stage_max": stage_max,
        "stage_name": s.bridge.stage().name, "replans": len(replan_s),
        "replan_wall_median_s": round(float(np.median(replan_s)), 3),
        "replan_wall_mean_s": round(float(np.mean(replan_s)), 3),
        "episode_wall_s": round(episode_s, 1),
        "candidates": args.candidates, "delta_clip": args.delta_clip,
        "objects_final": record.object_snapshot(s.env, s.source.movable),
        "held_final": s.bridge.world.held(),
    }
    if args.interpolate == "on":
        final["interpolate"] = {"method": args.interpolation_method,
                                "low": args.interpolate_frequency,
                                "high": args.interpolate_high_frequency}
    if args.sampler_seed is not None:
        final["sampler_seed"] = args.sampler_seed
    if s.perturb is not None:
        final["perturb"] = s.perturb.record()
        final["recovery"] = s.perturb.summary(success)
    if s.beam is not None:
        final["beam"] = {"k": s.beam.cfg.k, "warm": s.beam.cfg.warm,
                         "resample_every": s.beam.cfg.resample_every, "ema": s.beam.cfg.ema,
                         "w_rate": s.beam.cfg.w_rate, "w_subgoal": s.beam.cfg.w_subgoal,
                         "stage_reset": s.beam.cfg.stage_reset}
    if s.kp_cfg is not None:
        final["kp"] = {"k": s.kp_cfg.k, "align": s.kp_cfg.align, "w_scale": s.kp_cfg.w_scale,
                       "warm_start": s.kp_cfg.warm_start}
    if s.steer is not None:
        mech = ({"gamma": args.steer_gamma} if args.steer == "additive"
                else {"rho": args.inject_rho, "schedule": args.inject_schedule})
        final["steer"] = {"mode": args.steer, **mech,
                          "checkpoint": args.proxy_checkpoint,
                          "visual_only_render": args.visual_only_render,
                          "proxy_kv_cache": args.proxy_kv_cache,
                          "proxy_fp16": args.proxy_fp16}
    return final


def _replan(args, s, step, log, dt):
    """Advance the stage, infer a plan, filter it, and record telemetry."""
    prev_stage = s.bridge.stage_idx
    s.bridge.advance({})
    if s.bridge.stage_idx != prev_stage:
        log.stage(step, prev_stage, s.bridge)
    ctx = s.bridge.context(s.env, {}, dt)
    stage_key = (s.bridge.stage_idx, s.bridge.stage().name)

    t0 = time.perf_counter()
    beam_rec, w_plan, select_rec = None, None, None
    if s.steer is not None and s.steer.mode == "select":
        plan, stats, select_rec = select_chunk(s.planner, s.env, ctx, args, stage_key, s.steer)
    elif s.steer is not None and s.steer.mode == "verify":
        from .steering.verify import verify_chunk
        s.steer.begin_replan(s.env, args.num_steps + 1)
        plan, stats, _ = infer_chunk(s.planner, s.env, ctx, args, stage_key=stage_key)
        q0 = s.env.q0()
        state = torch.zeros(8, dtype=torch.float32)
        state[:7] = q0
        plan, select_rec = verify_chunk(s.planner, s.env, ctx, args, plan, s.steer, {"state": state})
    elif s.beam is None:
        plan, stats, w_plan = infer_chunk(s.planner, s.env, ctx, args, stage_key=stage_key,
                                          steer=s.steer)
    else:

        infer = functools.partial(infer_chunk, s.planner, s.env, ctx, args, stage_key=stage_key)
        plan, stats, beam_rec = s.beam.replan(infer, s.env, ctx, stage_key)
    wall_s = time.perf_counter() - t0

    if s.steer is None or s.steer.mode not in ("expert", "verify"):
        plan, _ = s.bridge.filter_plan(plan, args.spi)
    s.bridge.observe_plan(plan)

    rec = log.replan(step, s.bridge, s.env, stats, wall_s, s.source.movable)
    if beam_rec is not None:
        rec["beam"] = beam_rec
    if select_rec is not None:
        rec.update(select_rec)
    if s.steer is not None:
        rec.update(record.steer_fields(s.steer, stats, plan, s.env))
    subgoal = record.subgoal_residual(s.bridge, s.env)
    if subgoal is not None:
        rec["rekep_subgoal_now"] = subgoal
    if w_plan is not None:
        rec.update(record.keypose_fields(w_plan, stats))
    log.write(rec)
    if s.perturb is not None:
        log.perturb(s.perturb.on_replan(step, s.env, s.bridge))
    return plan, wall_s


def rollout(args):
    out_dir = paths.results_dir(args.task, args.exp)
    jsonl_path, video_path = out_dir / f"{args.seed}.jsonl", out_dir / f"{args.seed}.mp4"
    s = _setup(args)

    t_ep = time.perf_counter()
    log = record.Recorder(jsonl_path, prefix=LOG)
    states = [s.env.get_state()]
    actions, start = None, 0
    success, replan_s, stage_max = False, [], 0
    for step in range(args.max_steps):
        if actions is None or step - start >= args.spi:
            plan, wall_s = _replan(args, s, step, log, dt=0 if actions is None else args.spi)
            replan_s.append(wall_s)
            stage_max = max(stage_max, s.bridge.stage_idx)
            actions, start = plan[: args.spi], step
            if s.ensemble is not None:      # the chunk _replan just fetched, not a second call
                s.ensemble.push(step, plan)
        a = s.ensemble.action(step) if s.ensemble is not None else actions[step - start]
        if s.perturb is not None:
            log.perturb(s.perturb.on_step(step, s.env, s.bridge))
            a = s.perturb.filter_action(a, s.env, s.bridge)
        s.bridge.observe_step(a)
        s.env.apply_arm(a[:7], grip_close=float(a[7]) > 0.5)
        states.append(s.env.get_state())
        if s.env.success():
            success = True
            break
    episode_s = time.perf_counter() - t_ep

    final = _episode_summary(args, s, replan_s, episode_s, success, stage_max)
    log.episode(final)
    log.close()
    if s.client is not None:
        s.client.close()

    record.save_video(s.env, states, video_path)
    print(f"{LOG} log:   {jsonl_path}", flush=True)
    print(f"{LOG} video: {video_path}", flush=True)
    return final
