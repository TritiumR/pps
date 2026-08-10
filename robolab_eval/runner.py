"""Run one RoboLab evaluation rollout of the ReKep-MBD base.

The RoboLab twin of mujoco_eval/runner.py, reduced to the base arm: no proxy steering, no beam,
no keypose sampler, no perturbation protocol. What is kept is exactly the loop those mechanisms
plug into -- advance the stage, denoise a chunk under the composite cost, filter it, execute
`spi` rows, sense -- so adding an arm later is a call site, not a rewrite.
"""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import torch
import yaml

from . import paths
paths.ensure_repo_on_path()

# vlm_dp.bridge imports sim_common.envs.droid at module scope, which drags in the whole IsaacLab
# task registry for a class this harness never constructs. mujoco_eval stubs it for the same
# reason; here it also keeps a second, unrelated env package out of the process.
import sim_common.envs                                                          # noqa: E402
_droid_stub = types.ModuleType("sim_common.envs.droid")
_droid_stub.DroidEnv = None
sys.modules.setdefault("sim_common.envs.droid", _droid_stub)

from mujoco_eval import record                                                  # noqa: E402
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig                   # noqa: E402
from vlm_dp.bridge import VlmDpBridge                                           # noqa: E402
from vlm_dp.grasp_sensor import ApertureGraspSensor                             # noqa: E402
from vlm_dp.stage import _capture_held                                          # noqa: E402

from .env.robolab_env import FREE_CLOSE, RoboLabEnv, RoboLabWorld, planner_fk    # noqa: E402
from .grounding.rekep import RoboLabRekepVlmGrounding                            # noqa: E402
from .tasks import spec                                                          # noqa: E402

LOG = "[robolab-eval]"

CONTROL_HZ = 15.0                     # sim.dt 1/120 x decimation 8

# Franka actuator velocity limits as RoboLab declares them (robolab/robots/droid.py): 2.175 rad/s
# on panda_joint1-4, 2.61 rad/s on panda_joint5-7. They set both the per-step motion a command can
# physically realise and the scale of the planner's action space (see `action_std`).
JOINT_VEL_LIMIT = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)

# ApertureGraspSensor thresholds in RoboLab's gripper units. The Robotiq 2F-85 `finger_joint` is
# commanded to 0.0 to open and pi/4 to close, and stalls short of pi/4 on whatever is between the
# fingers -- which is exactly the sensor's "aperture" convention (0 = wide open, q_free = met).
#   q_touch      0.03 rad: below this the fingers have barely left open.
#   stall_margin 0.05 rad: a hold is certified when the joint stalls at least this far short of
#                free close. The 2F-85's ~85 mm stroke over pi/4 is ~108 mm/rad, so 0.05 rad is
#                about a 5 mm object -- thinner than anything either task grasps.
#   settle       6 steps at 15 Hz = 0.4 s of a steady joint before a close counts as settled.
#   proximity    a permissive backstop only: RoboLabWorld already restricts the candidate set to
#                bodies the contact sensor reports the fingers touching.
_SENSOR = dict(q_free=FREE_CLOSE, stall_margin=0.05, q_touch=0.03,
               settle_eps=0.004, settle_steps=6, close_steps=6, proximity=0.15)


class _Stats:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)


class ChunkDecodePolicy:
    """Provide action normalization metadata for SimFreeMPC decoding."""

    def __init__(self, action_std):
        one = np.ones(8, dtype=np.float32)
        act_std = np.asarray(action_std, dtype=np.float32)
        act_mean = np.zeros_like(act_std)
        act_mean[..., 7] = 0.5
        self._metadata = {
            "output_norm_stats": {"actions": _Stats(act_mean, act_std),
                                  "state": _Stats(np.zeros(8, np.float32), one)},
            "use_quantile_norm": False,
            "output_norm_stats_source": "velocity_limit_scale",
        }


def action_std(horizon, speed_frac):
    """Per-dimension scale of the planner's action space, in joint radians.

    `decode_model_action_chunks` builds a candidate row as `q_now + std * x`, so `std` IS the
    typical displacement a proposal explores from the current pose. mujoco_eval measures it from
    demonstrations; RoboLab ships none here, so it is derived from the arm instead: a chunk that
    spans `horizon / CONTROL_HZ` seconds at `speed_frac` of each joint's velocity limit covers
    `speed_frac * v * H / rate` radians, and the average row covers half of that.

    The number is a SCALE, not a bound -- `--delta_clip` bounds execution -- so it only has to be
    the right order of magnitude, and it is printed so an arm's exploration width is recorded.
    """
    span = np.asarray(JOINT_VEL_LIMIT, dtype=np.float32) * float(speed_frac) * horizon / CONTROL_HZ
    return np.concatenate([span / 2.0, [0.5]]).astype(np.float32)


class RoboLabBridge(VlmDpBridge):
    """Adapt VlmDpBridge to the RoboLab environment."""

    # Aperture-consistency band for `_grip_ok`, on the sensor's own (radian) scale.
    _AP_SLOPE = 1.0
    _AP_BAND = 0.15

    def __init__(self, source, cost_cfg, **kw):
        super().__init__("gt", source.roles, cost_cfg, **kw)
        self._source = source

    def reset(self, raw_env):
        """Initialize bridge state for one RoboLab episode."""
        self.env = raw_env
        self.sensor = ApertureGraspSensor(**_SENSOR, stall_margin_enter=self.hold_enter,
                                          stall_margin_exit=self.hold_exit,
                                          lost_on_free_close=self.hold_free_close)
        self.world = RoboLabWorld(raw_env, sensor=self.sensor, names=self._source.movable,
                                  slip_margin=self.slip_margin)
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


def build_planner(args, env):
    """Build the MBD sampling planner and give it the calibrated FK."""
    std = action_std(args.horizon, args.action_std_speed)
    print(f"{LOG} action std (model-space scale, rad): {np.round(std, 4)}", flush=True)
    cfg_kw = dict(
        task_name=args.task, num_samples=args.candidates, iterations=1,
        noise=args.noise, temperature=args.temperature, action_dims=8,
        joint_delta_clip=args.delta_clip, cost_style="priority",
        optimize_space="action", sampler="base", grad_calc="mbd",
        control_frequency=CONTROL_HZ, rank_mode="total", prior_weight=1.0,
        prior_weight_high=1.0, prior_weight_schedule="flat", estimator="mean")
    if args.interpolate == "on":
        cfg_kw.update(interpolate=True, interpolation_method=args.interpolation_method,
                      interpolate_frequency=args.interpolate_frequency,
                      control_frequency=args.interpolate_high_frequency)
    planner = SimFreeMPC(ChunkDecodePolicy(std), SimFreeMPCConfig(**cfg_kw))
    planner.fk = planner_fk(env)
    if args.interpolate == "on":
        print(f"{LOG} interpolate: {args.interpolation_method} {args.interpolate_frequency}/"
              f"{args.interpolate_high_frequency} Hz -> "
              f"{planner._interpolation_knot_count(args.horizon)} knots of {args.horizon} rows",
              flush=True)
    return planner


def infer_chunk(planner, env, ctx, args):
    """Run one denoising chain and decode the resulting action chunk."""
    from sim_free_mpc.action_space import decode_model_action_chunks

    q0 = env.q0()
    state = torch.zeros(8, dtype=torch.float32)
    state[:7] = q0
    inputs = {"state": state}
    x_t = torch.randn(1, args.horizon, 8)
    planner.begin_inference()
    stats, levels = {}, []
    for it in range(args.num_steps + 1):
        x_t, stats = planner.step_mbd_score_action_prox(
            x_t, inputs, ctx, iteration=it, num_iterations=args.num_steps + 1)
        if stats:
            levels.append({"it": int(it),
                           "ess": round(float(stats.get("weight_ess", float("nan"))), 2),
                           "cost_min": round(float(stats.get("cost_min", float("nan"))), 5),
                           "cost_std": round(float(stats.get("cost_std", float("nan"))), 5)})
    if levels:
        stats = dict(stats)
        stats["base_levels"] = levels
    dec = decode_model_action_chunks(planner.policy, inputs, x_t, apply_clamp=True,
                                     current_joint_pos=q0, max_joint_delta=args.delta_clip)
    return dec.real_actions[0].detach().cpu().numpy(), stats


def _setup(args):
    """Build all episode components in dependency order."""
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    apply_planner_config(args, cfg)

    env = RoboLabEnv(args.task, device=args.device, num_envs=args.num_envs, seed=args.seed)
    source = RoboLabRekepVlmGrounding(args.task, context_path=args.rekep_context)
    bridge = RoboLabBridge(source, cfg, task_key=args.task, device="cpu")
    planner = build_planner(args, env)
    bridge.attach_cost(planner)

    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    planner.reset_episode()
    bridge.reset(env)
    return types.SimpleNamespace(env=env, source=source, bridge=bridge, planner=planner, cfg=cfg)


def _live_terms(step, bridge, env, stats):
    """One line per replan saying what the plan is actually scoring right now.

    A stage whose `constraint` is None scores exactly zero on rekep_keypose, and `path=0` makes
    rekep_path inert -- the silent fall-through this grounding exists to avoid. Reporting it every
    replan (rather than once at ground time) also catches the case where a stage advances into a
    weaker one mid-episode.
    """
    st = bridge.stage()
    residual = record.subgoal_residual(bridge, env)
    done = None
    try:
        done = bool(st.done())
    except Exception:                      # a predicate must never cost the rollout
        pass
    print(f"{LOG} live terms step={step} stage={bridge.stage_idx}:{st.name!r} "
          f"subgoal={'yes' if st.constraint is not None else 'NONE'} path={len(st.path_fns)} "
          f"held={list(st.held_idx)} residual={residual} subgoal_done={done} "
          f"grip={env.gripper_q():.3f} contact={list(getattr(bridge.world, 'contact', ()))} "
          f"world_held={bridge.world.held()!r} "
          f"cost_min={stats.get('cost_min', float('nan')):.4f} "
          f"ess={stats.get('weight_ess', float('nan')):.1f}", flush=True)
    return {"subgoal_residual": residual, "subgoal_done": done,
            "path_rules": len(st.path_fns), "held_idx": list(st.held_idx)}


def _replan(args, s, step, log, dt):
    """Advance the stage, infer a plan, and record telemetry."""
    prev_stage = s.bridge.stage_idx
    s.bridge.advance({})
    if s.bridge.stage_idx != prev_stage:
        log.stage(step, prev_stage, s.bridge)
    ctx = s.bridge.context(s.env, {}, dt)

    t0 = time.perf_counter()
    plan, stats = infer_chunk(s.planner, s.env, ctx, args)
    wall_s = time.perf_counter() - t0

    plan, _ = s.bridge.filter_plan(plan, args.spi)
    s.bridge.observe_plan(plan)

    rec = log.replan(step, s.bridge, s.env, stats, wall_s, s.source.movable)
    rec.update(_live_terms(step, s.bridge, s.env, stats))
    log.write(rec)
    return plan, wall_s


def _episode_summary(args, s, replan_s, episode_s, success, stage_max):
    return {
        "kind": "episode", "task": args.task, "seed": args.seed, "success": success,
        "env_steps": s.env.n_steps, "stage_final": s.bridge.stage_idx, "stage_max": stage_max,
        "stage_name": s.bridge.stage().name,
        "stages": [st.name for st in s.bridge.grounding.stages],
        "replans": len(replan_s),
        "replan_wall_median_s": round(float(np.median(replan_s)), 3),
        "replan_wall_mean_s": round(float(np.mean(replan_s)), 3),
        "episode_wall_s": round(episode_s, 1),
        "candidates": args.candidates, "delta_clip": args.delta_clip,
        "horizon": args.horizon, "spi": args.spi, "num_steps": args.num_steps,
        "control_hz": CONTROL_HZ,
        "fk_residual": s.env.fk_fit.get("residual"),
        "objects_final": record.object_snapshot(s.env, s.source.movable),
        "held_final": s.bridge.world.held(),
    }


def save_video(frames, path, fps=CONTROL_HZ):
    """Write the frames captured during the rollout.

    Live capture, not a state replay: RoboLab has no cheap set-state, so mujoco_eval's
    reset_to-per-frame renderer has no counterpart here.
    """
    import imageio

    if not frames:
        print(f"{LOG} no frames captured; skipping video", flush=True)
        return None
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


def rollout(args):
    """Run one episode and file its trace and video by outcome."""
    out_dir = paths.results_dir(args.task, args.exp)
    jsonl_path = paths.pending_path(out_dir, args.seed, "jsonl")
    s = _setup(args)

    t_ep = time.perf_counter()
    log = record.Recorder(jsonl_path, prefix=LOG)
    frames, actions, start = [], None, 0
    success, replan_s, stage_max = False, [], 0
    max_steps = args.max_steps or spec(args.task)["max_steps"]
    for step in range(max_steps):
        if actions is None or step - start >= args.spi:
            plan, wall_s = _replan(args, s, step, log, dt=0 if actions is None else args.spi)
            replan_s.append(wall_s)
            stage_max = max(stage_max, s.bridge.stage_idx)
            actions, start = plan[: args.spi], step
        a = actions[step - start]
        s.bridge.observe_step(a)
        s.env.apply_arm(a[:7], grip_close=float(a[7]) > 0.5)
        frame = s.env.rgb()
        if frame is not None:
            frames.append(frame)
        if s.env.success():
            success = True
            break
    episode_s = time.perf_counter() - t_ep

    final = _episode_summary(args, s, replan_s, episode_s, success, stage_max)
    log.episode(final)
    log.close()
    trace_path = paths.episode_path(out_dir, args.seed, success, "trace", "jsonl")
    jsonl_path.replace(trace_path)
    paths.prune_pending(out_dir)
    video_path = save_video(frames, paths.episode_path(out_dir, args.seed, success, "videos",
                                                       "mp4"))
    print(f"{LOG} log:   {trace_path}", flush=True)
    print(f"{LOG} video: {video_path}", flush=True)
    s.env.close()
    return final
