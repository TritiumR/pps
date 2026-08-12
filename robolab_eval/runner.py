"""Run one RoboLab evaluation rollout of the ReKep-MBD base.

The RoboLab twin of mujoco_eval/runner.py, reduced to the base arm: no proxy steering, no beam,
no keypose sampler, no perturbation protocol. What is kept is exactly the loop those mechanisms
plug into -- advance the stage, denoise a chunk under the composite cost, filter it, execute
`spi` rows, sense -- so adding an arm later is a call site, not a rewrite.
"""
from __future__ import annotations

import hashlib
import json
import os
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
from sim_free_mpc.fk import transform_points_wxyz                              # noqa: E402
from vlm_dp.bridge import VlmDpBridge                                           # noqa: E402
from vlm_dp.grasp_sensor import ApertureGraspSensor                             # noqa: E402
from vlm_dp.grounding import get_source                                          # noqa: E402
from vlm_dp.stage import _capture_held                                          # noqa: E402

from .env.robolab_env import FREE_CLOSE, RoboLabEnv, RoboLabWorld, planner_fk    # noqa: E402
from .grounding.rekep import (RoboLabRekepVlmGrounding,
                              attach_standard_insertion)                         # noqa: E402
from .tasks import spec                                                          # noqa: E402

LOG = "[robolab-eval]"


def _install_runtime_profiler(raw_env):
    """Time calls nested inside ``raw_env.step`` without changing their arguments/results.

    This is deliberately opt-in.  It mirrors the accepted Isaac/Weight profiler and avoids
    wrapping individual camera objects, whose bound-method reference cycles can outlive the
    Replicator teardown.  The returned dictionary is updated in place by the wrappers.
    """
    totals, originals = {}, []

    def wrap(owner, name, bucket):
        if owner is None or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        originals.append((owner, name, original))

        def timed(*call_args, **call_kwargs):
            started = time.perf_counter()
            try:
                return original(*call_args, **call_kwargs)
            finally:
                totals[bucket] = totals.get(bucket, 0.0) + time.perf_counter() - started

        setattr(owner, name, timed)

    wrap(getattr(raw_env, "sim", None), "step", "sim.physx")
    wrap(getattr(raw_env, "sim", None), "render", "sim.render")
    wrap(getattr(raw_env, "scene", None), "update", "scene.update")
    wrap(getattr(raw_env, "observation_manager", None), "compute", "obs.compute")
    recorder = getattr(raw_env, "recorder_manager", None)
    wrap(recorder, "record_pre_step", "recorder.pre_step")
    wrap(recorder, "record_post_step", "recorder.post_step")
    wrap(recorder, "record_pre_reset", "recorder.pre_reset")
    def restore():
        # Break owner -> wrapper -> bound-owner cycles before Replicator camera teardown.
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        originals.clear()

    return totals, restore

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

    def __init__(self, action_std, *, source="demo_delta_stats"):
        one = np.ones(8, dtype=np.float32)
        act_std = np.asarray(action_std, dtype=np.float32)
        act_mean = np.zeros_like(act_std)
        act_mean[..., 7] = 0.5
        self._metadata = {
            "output_norm_stats": {"actions": _Stats(act_mean, act_std),
                                  "state": _Stats(np.zeros(8, np.float32), one)},
            "use_quantile_norm": False,
            "output_norm_stats_source": source,
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


def demo_delta_stats(hdf5, horizon, n_demos=50):
    """Measure the exact action scale from achieved rows in Spoon demonstrations.

    ``joint_actions[t]`` is the achieved next configuration.  The planner's DeltaActions surface
    predicts future absolute rows relative to ``joint_pos[t]``, so this pools
    ``q[t+h]-q[t]`` for h=1..H, exactly like ``mujoco_eval.runner.demo_delta_stats``.
    """
    import h5py
    with h5py.File(hdf5, "r") as f:
        names = sorted(f["data"], key=lambda s: int(s.split("_")[1]))[:n_demos]
        if not names:
            raise ValueError(f"{hdf5}: no demonstrations")
        deltas = []
        for name in names:
            q = np.asarray(f["data"][name]["obs/joint_pos"], dtype=np.float64)[:, :7]
            for h in range(1, int(horizon) + 1):
                deltas.append(q[h:] - q[:-h])
    std = np.concatenate(deltas, axis=0).std(axis=0)
    if not np.isfinite(std).all() or np.any(std <= 1e-6):
        raise ValueError(f"invalid demo delta std from {hdf5}: {std}")
    return np.concatenate([std, [0.5]]).astype(np.float32), len(names)


def reverse_timestep_grid(num_steps, num_train_timesteps=100):
    """Accepted Weight grid: reserve the t=0 level but do not execute it."""
    num_iterations = int(num_steps) + 1
    ratio = int(num_train_timesteps) // num_iterations
    return [(num_iterations - 1 - it) * ratio for it in range(int(num_steps))]


class RoboLabBridge(VlmDpBridge):
    """Adapt the shared bridge to RoboLab without changing its control semantics.

    ``artifact`` retains the original RoboLab bring-up path.  ``perception`` is the reference
    VLM-DP path: named perception builds a sensed world, ReKep proposes/tracks keypoints, the VLM
    format compiles stages, and the shared bridge supplies progression and costs.
    """

    # Aperture-consistency band for `_grip_ok`, on the sensor's own (radian) scale.
    _AP_SLOPE = 1.0
    _AP_BAND = 0.15

    def __init__(self, task, cost_cfg, *, grounding="artifact", vlm="fake",
                 context_path=None, **kw):
        task_spec = spec(task)
        self._grounding_mode = str(grounding)
        self._vlm_mode = str(vlm)
        self._source = (RoboLabRekepVlmGrounding(task, context_path=context_path, vlm=vlm)
                        if self._grounding_mode == "artifact" else None)
        roles = {"grasp_obj": task_spec["grasp_objs"][0],
                 "grasp_objs": list(task_spec["grasp_objs"]),
                 "place_obj": task_spec["place_obj"]}
        if self._source is not None:
            roles = self._source.roles
        ground_name = "gt" if self._source is not None else f"rekep_{vlm}_vlm"
        super().__init__(ground_name, roles, cost_cfg, task_key=task,
                         vocab=task_spec.get("vocab"),
                         fixtures=tuple(n for n in task_spec.get("fixtures", ()) if n != "table"),
                         **kw)
        self.movable = list(self._source.movable if self._source is not None
                            else scene_objects_for_receipt(task_spec))

    def reset(self, raw_env):
        """Initialize bridge state for one RoboLab episode."""
        self._eef_last = None
        self._env_steps = 0
        self._payload_name = None
        self._payload_acq = None
        self.env = raw_env
        self.sensor = ApertureGraspSensor(**_SENSOR, stall_margin_enter=self.hold_enter,
                                          stall_margin_exit=self.hold_exit,
                                          lost_on_free_close=self.hold_free_close)
        if self._source is not None:
            self.world = RoboLabWorld(raw_env, sensor=self.sensor, names=self._source.movable,
                                      slip_margin=self.slip_margin)
            self.grounding = self._source.ground(self.env, self.world)
        else:
            percep = self._build_perception() if self.state == "real" else None
            source = get_source(
                self.ground_name, task_key=self.task_key, perception=percep,
                seat_shift=self.seat_shift, local_grasp=self.local_grasp,
                local_grasp_radius=self.local_grasp_radius, kp_source=self.kp_source,
                contact_criterion=self.contact_criterion, subgoal_eps=self.subgoal_eps,
                rotate_grasp_offset=self.rotate_grasp_offset,
                lift_latch_xy=self.lift_latch_xy, seat_from_plane=self.seat_from_plane,
                open_half=float(self.geom.get("open_half", 0.04)), geom=self.geom,
                sensor_cfg={"q_free": float(self.sensor.q_free),
                            "stall_margin": float(self.sensor.stall_margin),
                            "q_touch": float(self.sensor.q_touch)},
                **self.roles)
            # The adapter itself already presents the camera, TCP and FK interfaces expected by
            # SensedWorld; unlike the shared Isaac task bridge, there is no DroidEnv.attach step.
            self.world = self._build_world(raw_env, percep, source)
            self.grounding = source.ground(self.env, self.world)
            self.grounding = attach_standard_insertion(self.grounding)
            self._source = source
        self._obj_pos = {o.name: o.pos for o in self.grounding.objects}
        self._extents = {o.name: o.extents for o in self.grounding.objects}
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
        self._eef_prev = None
        self._pred_hist.reset()
        self.pred_shadow = None
        self._pred_transition = None
        self._reset_churn()
        self._enter_stage()


def scene_objects_for_receipt(task_spec):
    """Names whose physical state is recorded, excluding the support plane."""
    return [n for n in (*task_spec.get("movable", ()), *task_spec.get("fixtures", ()))
            if n != "table"]


_PLANNER_KEYS = ("delta_clip", "interpolate", "interpolate_frequency",
                 "interpolate_high_frequency", "interpolation_method", "noise",
                 "temperature", "cost_executable_actions", "replan_period_s")


def apply_planner_config(args, cfg):
    """Apply supported planner overrides from the configuration."""
    block = cfg.get("planner")
    if not block:
        return
    unknown = sorted(set(block) - set(_PLANNER_KEYS))
    if unknown:
        raise SystemExit(f"unknown planner config keys {unknown}; allowed {list(_PLANNER_KEYS)}")
    for key, value in block.items():
        if key in ("interpolate", "cost_executable_actions") and isinstance(value, bool):
            value = "on" if value else "off"
        setattr(args, key, value)
    if getattr(args, "replan_period_s", None) is not None:
        requested = float(args.replan_period_s)
        if requested <= 0.0:
            raise SystemExit("planner.replan_period_s must be positive")
        args.spi = max(1, int(round(requested * CONTROL_HZ)))
        args.replan_period_requested_s = requested
        args.replan_period_effective_s = args.spi / CONTROL_HZ
    print(f"{LOG} planner overrides: {dict(block)}", flush=True)


def apply_policy_execution_contract(args):
    """Apply policy-specific cadence after base-planner YAML overrides.

    ``spoon_base.yaml`` describes the live MBD controller and therefore carries its short
    replanning period.  A standalone H21 key-pose proxy has a different, already-established
    execution contract: rows 0--14 are the action chunk and all 15 are executed before the next
    inference.  Keeping this override explicit prevents an unrelated base YAML field from
    silently changing the learned-policy evaluation.
    """
    if args.policy != "keypose_proxy":
        return
    if int(args.horizon) != 15:
        raise SystemExit("standalone Spoon proxy requires --horizon 15 (H21 = 15 action + 5 wp + 1 kp)")
    spi = int(args.proxy_spi)
    if not 1 <= spi <= 15:
        raise SystemExit("--proxy_spi must be in [1,15]")
    args.spi = spi
    args.replan_period_requested_s = None
    args.replan_period_effective_s = spi / CONTROL_HZ
    print(f"{LOG} standalone proxy cadence: execute {spi}/15 trained action rows per replan "
          f"({spi / CONTROL_HZ:.3f} s)", flush=True)


def build_planner(args, env):
    """Build the MBD sampling planner and give it the calibrated FK."""
    if not args.hdf5:
        raise SystemExit(f"{LOG} --hdf5 is required; refusing velocity-derived action scale")
    std, n_demo = demo_delta_stats(args.hdf5, args.horizon)
    norm_source = "spoon_50_demo_executed_row_delta"
    print(f"{LOG} demo delta std ({n_demo} demos, exact executed-row contract): "
          f"{np.round(std, 6)}", flush=True)
    cfg_kw = dict(
        task_name=args.task, num_samples=args.candidates, iterations=1,
        noise=args.noise, temperature=args.temperature, action_dims=8,
        joint_delta_clip=args.delta_clip, cost_style="priority",
        optimize_space="action", sampler="base", grad_calc="mbd",
        control_frequency=CONTROL_HZ, rank_mode="total", prior_weight=1.0,
        prior_weight_high=1.0, prior_weight_schedule="flat", estimator="mean",
        cost_executable_actions=args.cost_executable_actions == "on")
    if args.interpolate == "on":
        cfg_kw.update(interpolate=True, interpolation_method=args.interpolation_method,
                      interpolate_frequency=args.interpolate_frequency,
                      control_frequency=args.interpolate_high_frequency)
    planner = SimFreeMPC(ChunkDecodePolicy(std, source=norm_source), SimFreeMPCConfig(**cfg_kw))
    planner.fk = planner_fk(env)
    planner.action_norm_receipt = {
        "source": norm_source, "dataset": str(args.hdf5), "dataset_sha256": _sha256(args.hdf5),
        "num_demos": int(n_demo), "horizon": int(args.horizon),
        "std": np.asarray(std).tolist(),
        "contract": "std(q[t+h]-q[t]), h=1..H; achieved arm rows + continuous gripper",
    }
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
    stats, levels, diffusion_tcp = {}, [], []
    # Match the accepted Weight operator exactly: num_iterations remains 11, but the default path
    # executes iterations 0..9 and therefore the literal [90,81,...,9] reverse grid.  Spoon's old
    # loop also executed iteration 10/t=0 and was a different sampler despite the same CLI label.
    num_iterations = args.num_steps + 1
    reverse_grid = reverse_timestep_grid(args.num_steps)
    for it, timestep in enumerate(reverse_grid):
        x_t, stats = planner.step_mbd_score_action_prox(
            x_t, inputs, ctx, iteration=it, num_iterations=num_iterations)
        if stats:
            levels.append({"it": int(it),
                           "timestep": int(timestep),
                           "ess": round(float(stats.get("weight_ess", float("nan"))), 2),
                           "cost_min": round(float(stats.get("cost_min", float("nan"))), 5),
                           "cost_std": round(float(stats.get("cost_std", float("nan"))), 5)})
        level_dec = decode_model_action_chunks(
            planner.policy, inputs, x_t, apply_clamp=True, current_joint_pos=q0,
            max_joint_delta=args.delta_clip).real_actions[0]
        fk = planner.fk.forward(level_dec[..., :7])
        level_tcp = transform_points_wxyz(
            torch.as_tensor(env.base_pos, dtype=fk.ee_pos.dtype),
            torch.as_tensor(env.base_quat_wxyz, dtype=fk.ee_pos.dtype), fk.ee_pos)
        diffusion_tcp.append(level_tcp.detach().cpu().numpy())
    if levels:
        stats = dict(stats)
        stats["base_levels"] = levels
        stats["reverse_timestep_grid"] = [x["timestep"] for x in levels]
    dec = decode_model_action_chunks(planner.policy, inputs, x_t, apply_clamp=True,
                                     current_joint_pos=q0, max_joint_delta=args.delta_clip)
    fk = planner.fk.forward(dec.real_actions[0, ..., :7])
    plan_tcp = transform_points_wxyz(
        torch.as_tensor(env.base_pos, dtype=fk.ee_pos.dtype),
        torch.as_tensor(env.base_quat_wxyz, dtype=fk.ee_pos.dtype), fk.ee_pos,
    ).detach().cpu().numpy()
    viz = {"plan_tcp": plan_tcp, "diffusion_tcp": np.asarray(diffusion_tcp)}
    return dec.real_actions[0].detach().cpu().numpy(), stats, viz


def _setup(args):
    """Build all episode components in dependency order."""
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    apply_planner_config(args, cfg)
    apply_policy_execution_contract(args)

    # ReKep writes its rendered constraints and resolved placeholders as a run receipt.  The
    # evaluation source tree is intentionally read-only in the container, so route only those
    # generated artifacts to the writable result root.
    os.environ.setdefault("VLMDP_VLM_QUERY_ROOT", str(paths.RESULTS / "_vlm_query"))
    # OpenPI's tokenizer loader first consults OPENPI_DATA_HOME.  The RoboLab image intentionally
    # remains immutable/minimal (and does not carry gcsfs), so point it at the verified local
    # tokenizer cache mounted with the experiment data.  Missing assets then fail visibly in the
    # local proxy subprocess rather than causing an implicit network/package dependency.
    os.environ.setdefault("OPENPI_DATA_HOME", str(paths.DATA / "openpi_cache"))

    perception = args.grounding == "perception"
    env = RoboLabEnv(args.task, device=args.device, num_envs=args.num_envs, seed=args.seed,
                     perception_camera=perception, runtime_profile=args.runtime_profile)
    bridge = RoboLabBridge(
        args.task, cfg, grounding=args.grounding, vlm=args.rekep_vlm,
        context_path=args.rekep_context, device="cpu", state=args.vlm_state,
        track=args.vlm_track, segment=args.vlm_segment)
    planner = build_planner(args, env)
    bridge.attach_cost(planner)

    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    planner.reset_episode()
    bridge.reset(env)
    proxy = None
    proxy_receipt = None
    if args.policy == "keypose_proxy":
        if args.task != "spoon_insertion":
            raise SystemExit("keypose_proxy is currently contracted only for spoon_insertion")
        from mujoco_eval.steering.proxy import ProxyScoreClient
        checkpoint = os.path.abspath(args.proxy_checkpoint)
        norm_path = os.path.join(checkpoint, "action_norm_stats.json")
        with open(norm_path, encoding="utf-8") as fh:
            norm = json.load(fh)
        expected = {"action_horizon": 21, "action_dim": 8, "awe_waypoints": 5,
                    "keypose_tail": True, "action_offset": 0}
        mismatch = {k: (norm.get(k), v) for k, v in expected.items() if norm.get(k) != v}
        if mismatch:
            raise SystemExit(f"Spoon proxy checkpoint layout mismatch: {mismatch}")
        proxy = ProxyScoreClient(
            checkpoint, args.proxy_prompt, config=args.proxy_config,
            device=args.proxy_device, prediction_mode=args.proxy_prediction_mode,
            kv_cache=args.proxy_kv_cache == "on", fp16=args.proxy_fp16 == "on",
            local_process=True, checkpoint_is_container=True,
        ).start()
        ready = proxy.ready_info
        if ready.get("action_horizon") != 21 or ready.get("action_dim") != 8:
            raise SystemExit(f"proxy server returned incompatible layout: {ready}")
        if ready.get("prediction_type") != "x0" or ready.get("action_norm") != "demo_delta":
            raise SystemExit(f"proxy server returned incompatible model contract: {ready}")
        proxy_receipt = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(os.path.join(checkpoint, "model.safetensors")),
            "action_norm_sha256": _sha256(norm_path),
            "layout": expected,
            "prompt": args.proxy_prompt,
            "config": args.proxy_config,
            "prediction_mode": args.proxy_prediction_mode,
            "ready": ready,
        }
    return types.SimpleNamespace(env=env, source=bridge, bridge=bridge, planner=planner, cfg=cfg,
                                 proxy=proxy, proxy_receipt=proxy_receipt)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _plan_provenance(args):
    """Record the semantic prompt and exact rendered ReKep program used by this process."""
    query_dir = paths.RESULTS / "_vlm_query" / f"vlm_query_{args.task}_p{os.getpid()}"
    query_files = {}
    if query_dir.is_dir():
        query_files = {p.name: _sha256(p) for p in sorted(query_dir.iterdir()) if p.is_file()}
    return {
        "prompt": spec(args.task)["prompt"],
        "grounding": args.grounding,
        "state": args.vlm_state,
        "tracker": args.vlm_track,
        "segmenter": args.vlm_segment,
        "vlm": args.rekep_vlm,
        "plan_authority": ("live_vlm" if args.rekep_vlm == "real"
                           else "canned_ReKep_format_response"),
        "query_dir": str(query_dir),
        "query_files_sha256": query_files,
        "config": str(args.config),
    }


def dry_receipt(args):
    """Prove the live stack produces one valid H8 plan without changing simulator state."""
    out_dir = paths.results_dir(args.task, args.exp)
    s = _setup(args)
    try:
        before_steps = int(s.env.n_steps)
        if s.proxy is not None:
            plan, stats, viz, _ = infer_proxy_chunk(s, args)
        else:
            ctx = s.bridge.context(s.env, {}, executed_steps=0)
            plan, stats, viz = infer_chunk(s.planner, s.env, ctx, args)
            plan, _ = s.bridge.filter_plan(plan, args.spi)
        plan = np.asarray(plan, dtype=np.float64)
        if plan.shape != (args.horizon, 8):
            raise SystemExit(f"{LOG} dry receipt expected {(args.horizon, 8)}, got {plan.shape}")
        if not np.isfinite(plan).all():
            raise SystemExit(f"{LOG} dry receipt contains non-finite actions")
        if float(plan[:, 7].min()) < -1e-6 or float(plan[:, 7].max()) > 1.0 + 1e-6:
            raise SystemExit(f"{LOG} dry receipt gripper leaves [0,1]: "
                             f"{plan[:, 7].min():.4f}..{plan[:, 7].max():.4f}")
        if int(s.env.n_steps) != before_steps:
            raise SystemExit(f"{LOG} dry receipt stepped the environment")

        fields = dict(s.bridge.grounding.plan_fields or {})
        keypoints = (np.asarray(s.bridge.grounding.keypoints(), dtype=np.float64)
                     if callable(s.bridge.grounding.keypoints) else np.empty((0, 3)))
        sensed = {name: np.round(s.bridge.world.object_pose(name)[0], 6).tolist()
                  for name in s.bridge.world.names}
        receipt = {
            "kind": "dry_receipt", "task": args.task, "seed": int(args.seed),
            "provenance": _plan_provenance(args),
            "observation": {
                "joint_dim": int(np.asarray(s.env.q0()).size),
                "tcp_world": np.round(s.env.tcp(), 6).tolist(),
                "sensed_objects_world": sensed,
                "keypoints_world_shape": list(keypoints.shape),
                "keypoints_world": np.round(keypoints, 6).tolist(),
            },
            "plan": {
                "stages": [{"name": st.name, "gripper": st.gripper,
                            "payload": st.payload, "place_target": st.place_target,
                            "place_mode": st.place_mode, "insert": st.insert is not None,
                            "subgoal": st.constraint is not None,
                            "path_rules": len(st.path_fns)}
                           for st in s.bridge.grounding.stages],
                "resolved_fields": fields,
            },
            "mbd": {
                "action_shape": list(plan.shape), "finite": True,
                "first_action": np.round(plan[0], 6).tolist(),
                "joint_min": float(plan[:, :7].min()),
                "joint_max": float(plan[:, :7].max()),
                "gripper_min": float(plan[:, 7].min()),
                "gripper_max": float(plan[:, 7].max()),
                "candidates": int(args.candidates), "num_steps": int(args.num_steps),
                "cost_min": (float(stats["cost_min"])
                             if np.isfinite(stats.get("cost_min", np.nan)) else None),
                "weight_ess": (float(stats["weight_ess"])
                               if np.isfinite(stats.get("weight_ess", np.nan)) else None),
                "reverse_timestep_grid": (stats.get("reverse_timestep_grid")
                                           or stats.get("native_reverse_grid")),
                "action_norm": s.planner.action_norm_receipt,
                "viz_plan_tcp_shape": list(np.asarray(viz["plan_tcp"]).shape),
                "viz_diffusion_tcp_shape": list(np.asarray(viz["diffusion_tcp"]).shape),
                "proxy": s.proxy_receipt,
                "proxy_input": getattr(s, "proxy_input_receipt", None),
                "proxy_physical_action_contract": stats.get("physical_action_contract"),
                "proxy_waypoint_tcp_shape": list(np.asarray(viz.get("waypoint_tcp", [])).shape),
                "proxy_keypose_tcp_shape": list(np.asarray(viz.get("keypose_tcp", [])).shape),
            },
            "environment_steps": int(s.env.n_steps),
        }
        path = out_dir / f"dry_receipt_seed{args.seed}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2, allow_nan=False)
        print(f"{LOG} DRY RECEIPT PASS: {path}", flush=True)
        return receipt
    finally:
        if s.proxy is not None:
            s.proxy.close()
        s.env.close()


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


def _tcp_for_rows(planner, env, rows):
    """FK absolute joint rows of any leading shape into world-frame TCP points."""
    real = torch.as_tensor(rows, dtype=torch.float32)
    fk = planner.fk.forward(real[..., :7])
    return transform_points_wxyz(
        torch.as_tensor(env.base_pos, dtype=fk.ee_pos.dtype),
        torch.as_tensor(env.base_quat_wxyz, dtype=fk.ee_pos.dtype), fk.ee_pos,
    ).detach().cpu().numpy()


def infer_proxy_chunk(s, args):
    """Run the frozen standalone H21 Spoon proxy and expose action/goal blocks separately."""
    from sim_free_mpc.action_space import clamp_real_action_chunk

    obs = s.env.proxy_observation()
    s.proxy_input_receipt = {
        "table_shape": list(np.asarray(obs["table"]).shape),
        "table_dtype": str(np.asarray(obs["table"]).dtype),
        "table_sha256": hashlib.sha256(np.ascontiguousarray(obs["table"]).tobytes()).hexdigest(),
        "wrist_shape": list(np.asarray(obs["wrist"]).shape),
        "wrist_dtype": str(np.asarray(obs["wrist"]).dtype),
        "wrist_sha256": hashlib.sha256(np.ascontiguousarray(obs["wrist"]).tobytes()).hexdigest(),
        "joint_pos": np.asarray(obs["joint_pos"], dtype=np.float64).tolist(),
        "gripper_pos_normalized": float(obs["gripper_pos"]),
    }
    if not hasattr(s, "proxy_input_receipts"):
        s.proxy_input_receipts = []
    s.proxy_input_receipts.append(dict(s.proxy_input_receipt))
    replan_idx = int(getattr(s, "proxy_replan_idx", 0))
    seed = int(args.seed) * 100003 + replan_idx
    t0 = time.perf_counter()
    chain, server_s = s.proxy.chain(
        seed=seed, num_iterations=11, joint_pos=obs["joint_pos"],
        gripper_pos=obs["gripper_pos"], table=obs["table"], wrist=obs["wrist"])
    wall_s = time.perf_counter() - t0
    chain = np.asarray(chain, dtype=np.float32)
    if chain.shape != (11, 21, 8) or not np.isfinite(chain).all():
        raise ValueError(f"proxy native chain must be finite [11,21,8], got {chain.shape}")
    # Server output is already denormalized and rebased to q0. Apply only physical joint limits
    # and continuous-gripper [0,1]; no MBD delta clamp or cost/filter is allowed in standalone.
    clean = clamp_real_action_chunk(torch.as_tensor(chain), max_joint_delta=None).numpy()
    action = clean[-1, :15].copy()                 # ONLY the trained action block is executable
    q0 = np.asarray(obs["joint_pos"], dtype=np.float32)
    diffusion_tcp = _tcp_for_rows(s.planner, s.env, clean[:, :15])
    goal_tcp = _tcp_for_rows(s.planner, s.env, clean[-1, 15:21])
    s.proxy_replan_idx = replan_idx + 1
    stats = {
        "policy": "keypose_proxy", "proxy_seed": seed, "proxy_server_s": float(server_s),
        "weight_ess": float("nan"), "cost_min": float("nan"),
        "native_reverse_grid": s.proxy.ready_info.get("native_reverse_grid_11"),
        "chunk_layout": {"action": [0, 15], "waypoint": [15, 20], "keypose": [20, 21]},
        "physical_action_contract": {
            "first_joint_delta_from_observation_max_abs": float(np.abs(action[0, :7] - q0).max()),
            "chunk_joint_delta_from_observation_max_abs": float(np.abs(action[:, :7] - q0).max()),
            "chunk_sequential_joint_step_max_abs": float(np.abs(np.diff(action[:, :7], axis=0)).max()),
            "gripper_range": [float(action[:, 7].min()), float(action[:, 7].max())],
            "postprocess": "joint limits + continuous gripper [0,1]; no MBD cost/filter/delta clip",
        },
    }
    viz = {
        "plan_tcp": diffusion_tcp[-1], "diffusion_tcp": diffusion_tcp,
        "waypoint_tcp": goal_tcp[:5], "keypose_tcp": goal_tcp[5],
        "diffusion_kind": "proxy clean x0 prediction",
    }
    return action, stats, viz, wall_s


def _replan(args, s, step, log, dt):
    """Advance the stage, infer a plan, and record telemetry."""
    prev_stage = s.bridge.stage_idx
    s.bridge.advance({})
    if s.bridge.stage_idx != prev_stage:
        log.stage(step, prev_stage, s.bridge)
    if s.proxy is not None:
        plan, stats, viz, wall_s = infer_proxy_chunk(s, args)
    else:
        ctx = s.bridge.context(s.env, {}, dt)
        t0 = time.perf_counter()
        plan, stats, viz = infer_chunk(s.planner, s.env, ctx, args)
        wall_s = time.perf_counter() - t0
        plan, _ = s.bridge.filter_plan(plan, args.spi)
    s.bridge.observe_plan(plan)

    rec = log.replan(step, s.bridge, s.env, stats, wall_s, s.source.movable)
    if s.proxy is None:
        rec.update(_live_terms(step, s.bridge, s.env, stats))
    else:
        rec.update({"policy": "keypose_proxy", "proxy_server_s": stats["proxy_server_s"],
                    "proxy_seed": stats["proxy_seed"], "proxy_layout": stats["chunk_layout"]})
        # These MBD-only fields are intentionally undefined for a standalone proxy.  JSON NaN is
        # non-standard and breaks strict receipt consumers, so preserve the schema with nulls.
        for key in ("cost_min", "cost_weighted", "weight_ess"):
            if not np.isfinite(rec.get(key, np.nan)):
                rec[key] = None
    log.write(rec)
    return plan, wall_s, stats, viz


def _episode_summary(args, s, replan_s, episode_s, success, stage_max):
    sensed = {name: np.round(s.bridge.world.object_pose(name)[0], 4).tolist()
              for name in s.bridge.world.names}
    return {
        "kind": "episode", "task": args.task, "seed": args.seed, "success": success,
        "policy": args.policy,
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
        "noise": args.noise, "temperature": args.temperature,
        "cost_executable_actions": args.cost_executable_actions,
        "replan_period_s": args.spi / CONTROL_HZ,
        "runtime_profile": args.runtime_profile,
        "artifact_mode": args.artifact_mode,
        "video_stride": int(args.video_stride),
        "runtime_breakdown": getattr(s, "runtime_breakdown", None),
        "reverse_timestep_grid": (getattr(s, "last_stats", {}).get("reverse_timestep_grid")
                                  or getattr(s, "last_stats", {}).get("native_reverse_grid")),
        "action_norm": s.planner.action_norm_receipt,
        "proxy": s.proxy_receipt,
        "proxy_runtime": (None if s.proxy is None else {
            "last_input": getattr(s, "proxy_input_receipt", None),
            "all_inputs": getattr(s, "proxy_input_receipts", []),
            "last_physical_action_contract": getattr(s, "last_stats", {}).get(
                "physical_action_contract"),
        }),
        "visualization": getattr(s, "viz_receipt", None),
        "mechanical_action_receipt": getattr(s, "motion_receipt", None),
        "plan_provenance": _plan_provenance(args),
        "success_predicate": "InsertSpaghettiSpoonTask/object_retained_in_container"
                             if args.task == "spoon_insertion" else "task_termination",
        "fk_residual": s.env.fk_fit.get("residual"),
        "objects_final": record.object_snapshot(s.env, s.source.movable),
        "sensed_objects_final": sensed,
        "plan_fields": s.bridge.grounding.plan_fields,
        "held_final": s.bridge.world.held(),
    }


def _motion_receipt(commanded, measured, tcp_world, replan_steps):
    """Summarize target continuity and realized motion in physical-time units.

    This deliberately distinguishes controller targets from achieved joint/TCP state. A target
    can satisfy the per-row delta contract yet still jump at a receding-horizon boundary, while
    actuator lag can make the realized trajectory look jerky for a different reason.
    """
    cmd = np.asarray(commanded, dtype=np.float64)
    obs = np.asarray(measured, dtype=np.float64)
    tcp = np.asarray(tcp_world, dtype=np.float64)
    if cmd.ndim != 2 or obs.shape != cmd.shape or tcp.shape != (len(cmd), 3):
        return {"valid": False, "command_shape": list(cmd.shape),
                "measured_shape": list(obs.shape), "tcp_shape": list(tcp.shape)}

    def _p95_max(values):
        if values.size == 0:
            return {"p95": None, "max": None}
        magnitude = np.linalg.norm(values, axis=-1)
        return {"p95": float(np.percentile(magnitude, 95)),
                "max": float(magnitude.max())}

    dcmd = np.diff(cmd[:, :7], axis=0)
    dobs = np.diff(obs[:, :7], axis=0)
    dtcp = np.diff(tcp, axis=0)
    replan_edges = [int(step) for step in replan_steps if 0 < int(step) < len(cmd)]
    boundary = np.asarray([cmd[i, :7] - cmd[i - 1, :7] for i in replan_edges],
                          dtype=np.float64).reshape(-1, 7)
    tracking = cmd[:, :7] - obs[:, :7]
    return {
        "valid": True, "num_control_steps": int(len(cmd)), "control_hz": CONTROL_HZ,
        "replan_steps": replan_edges,
        "command_joint_step_rad": _p95_max(dcmd),
        "command_joint_step_max_component_rad": (
            float(np.abs(dcmd).max()) if dcmd.size else None),
        "command_replan_boundary_rad": _p95_max(boundary),
        "command_replan_boundary_max_component_rad": (
            float(np.abs(boundary).max()) if boundary.size else None),
        "measured_joint_speed_rad_s": _p95_max(dobs * CONTROL_HZ),
        "measured_joint_accel_rad_s2": _p95_max(np.diff(dobs, axis=0) * CONTROL_HZ**2),
        "measured_joint_jerk_rad_s3": _p95_max(np.diff(dobs, n=2, axis=0) * CONTROL_HZ**3),
        "tcp_speed_m_s": _p95_max(dtcp * CONTROL_HZ),
        "tcp_accel_m_s2": _p95_max(np.diff(dtcp, axis=0) * CONTROL_HZ**2),
        "tcp_jerk_m_s3": _p95_max(np.diff(dtcp, n=2, axis=0) * CONTROL_HZ**3),
        "joint_tracking_error_rad": _p95_max(tracking),
        "command_gripper_step": _p95_max(np.diff(cmd[:, 7:8], axis=0)),
        "measured_gripper_step": _p95_max(np.diff(obs[:, 7:8], axis=0)),
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


def _active_keypoints(bridge, target, keypoints):
    """Resolve the plan-authored indices active in the current stage."""
    fields = bridge.grounding.plan_fields or {}
    idx = set(int(i) for i in bridge.stage().held_idx if int(i) >= 0)
    for key in (("g",) if bridge.stage_idx < 2 else ("h", "mouth")):
        value = fields.get(key)
        if isinstance(value, (int, np.integer)):
            idx.add(int(value))
    if target is not None and len(keypoints):
        idx.add(int(np.linalg.norm(keypoints - np.asarray(target)[None], axis=1).argmin()))
    return sorted(i for i in idx if 0 <= i < len(keypoints))


def _render_controller_frame(args, s, step, replan_idx, viz_plan, stats):
    """Build synchronized policy-input and perception-debug receipt panels."""
    from .viz import compose_policy_debug_view, render_overlay

    snapshot = s.env.rekep_camera_frame()
    if snapshot is None:
        return s.env.rgb(), {"error": "no_rekep_camera"}, None
    policy_frames = s.env.policy_camera_frames()
    controller = s.bridge.viz()
    keypoints = np.asarray(controller.get("keypoints", np.empty((0, 3))), dtype=np.float64)
    metadata_fn = getattr(s.bridge.grounding, "keypoint_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else [{} for _ in keypoints]
    target = controller.get("target")
    active = _active_keypoints(s.bridge, target, keypoints)
    status = (f"PROXY H21=15A+5W+1KP native-DDIM11 server={stats.get('proxy_server_s', float('nan')):.2f}s"
              if s.proxy is not None else
              f"MBD N={args.candidates} noise={args.noise:g} T={args.temperature:g} "
              f"clip={args.delta_clip:g} exec_cost={args.cost_executable_actions} "
              f"ESS={stats.get('weight_ess', float('nan')):.1f}")
    frame, metrics = render_overlay(
        snapshot, keypoints=keypoints, keypoint_metadata=metadata, active_indices=active,
        target=target, tcp=s.env.tcp(), plan_tcp=viz_plan.get("plan_tcp"),
        diffusion_tcp=viz_plan.get("diffusion_tcp"),
        waypoint_tcp=viz_plan.get("waypoint_tcp"), keypose_tcp=viz_plan.get("keypose_tcp"),
        stage=controller.get("stage", "?"),
        holding=controller.get("holding", False), step=step, replan=replan_idx,
        planner_status=status,
    )
    composite = compose_policy_debug_view(policy_frames["table"], policy_frames["wrist"], frame)
    metrics.update({
        "composite_shape": list(composite.shape),
        "policy_input_cameras": ["front_wide_camera", "wrist_cam"],
        "overlay_camera": "rekep_cam",
        "camera_frames_synchronized": True,
    })
    raw_frames = {
        "front_policy": policy_frames["table"],
        "wrist_policy": policy_frames["wrist"],
        "rekep_raw": snapshot["rgb"],
    }
    return composite, metrics, raw_frames


def rollout(args):
    """Run one episode and file its trace and video by outcome."""
    out_dir = paths.results_dir(args.task, args.exp)
    jsonl_path = paths.pending_path(out_dir, args.seed, "jsonl")
    s = _setup(args)
    nested_runtime, restore_runtime_profiler = (
        _install_runtime_profiler(s.env.env)
        if getattr(args, "profile_runtime", False) else ({}, lambda: None))

    t_ep = time.perf_counter()
    log = record.Recorder(jsonl_path, prefix=LOG)
    if args.video_stride is None:
        args.video_stride = 2 if args.runtime_profile == "weight" else 1
    if int(args.video_stride) < 1:
        raise SystemExit("--video_stride must be >= 1")
    frames, actions, start, viz_plan, last_stats = [], None, 0, {}, {}
    commanded, measured, tcp_world, replan_steps = [], [], [], []
    viz_samples = []
    success, replan_s, stage_max = False, [], 0
    env_step_s, frame_s = [], []
    max_steps = args.max_steps or spec(args.task)["max_steps"]
    for step in range(max_steps):
        if actions is None or step - start >= args.spi:
            replan_steps.append(step)
            plan, wall_s, last_stats, viz_plan = _replan(
                args, s, step, log, dt=0 if actions is None else args.spi)
            replan_s.append(wall_s)
            stage_max = max(stage_max, s.bridge.stage_idx)
            actions, start = plan[: args.spi], step
        a = actions[step - start]
        s.bridge.observe_step(a)
        t_step = time.perf_counter()
        s.env.apply_arm(a[:7], grip_command=float(a[7]))
        env_step_s.append(time.perf_counter() - t_step)
        commanded.append(np.asarray(a, dtype=np.float64).copy())
        measured.append(np.concatenate((np.asarray(s.env.q0(), dtype=np.float64),
                                        [float(s.env.gripper_q())])))
        tcp_world.append(np.asarray(s.env.tcp(), dtype=np.float64).copy())
        capture_frame = (args.artifact_mode == "full"
                         and step % int(args.video_stride) == 0)
        t_frame = time.perf_counter()
        if capture_frame and args.viz_overlay == "on":
            frame, viz_metrics, raw_frames = _render_controller_frame(
                args, s, step, len(replan_s), viz_plan, last_stats)
            if len(viz_samples) < 3 or step % max(args.spi * 20, 1) == 0:
                viz_samples.append({"step": int(step), **viz_metrics})
            if step == 0 and frame is not None and raw_frames is not None:
                import imageio.v3 as iio
                receipt_dir = out_dir / "visual_receipt"
                receipt_dir.mkdir(parents=True, exist_ok=True)
                iio.imwrite(receipt_dir / f"seed{args.seed}_front_policy.png",
                            raw_frames["front_policy"])
                iio.imwrite(receipt_dir / f"seed{args.seed}_wrist_policy.png",
                            raw_frames["wrist_policy"])
                iio.imwrite(receipt_dir / f"seed{args.seed}_rekep_raw.png",
                            raw_frames["rekep_raw"])
                iio.imwrite(receipt_dir / f"seed{args.seed}_three_panel.png", frame)
        elif capture_frame:
            frame = s.env.rgb()
        else:
            frame = None
        frame_s.append(time.perf_counter() - t_frame)
        if capture_frame and frame is not None:
            frames.append(frame)
        if s.env.success():
            success = True
            break
    episode_s = time.perf_counter() - t_ep

    s.last_stats = last_stats
    s.viz_receipt = {
        "enabled": args.viz_overlay == "on",
        "video_layout": "front policy input | wrist policy input | ReKep debug overlay",
        "policy_input_cameras": ["front_wide_camera", "wrist_cam"],
        "overlay_camera": "rekep_cam",
        "camera_frames_synchronized_per_control_step": True,
        "samples": viz_samples,
        "all_samples_have_keypoints": bool(viz_samples) and all(x.get("keypoints", 0) > 0 for x in viz_samples),
        "all_samples_have_plan": bool(viz_samples) and all(x.get("has_plan") for x in viz_samples),
        "all_samples_have_denoise": bool(viz_samples) and all(x.get("denoise_levels", 0) > 0 for x in viz_samples),
        "all_proxy_samples_have_waypoints": (None if s.proxy is None else
            bool(viz_samples) and all(x.get("waypoint_count", 0) == 5 for x in viz_samples)),
        "all_proxy_samples_have_keypose": (None if s.proxy is None else
            bool(viz_samples) and all(x.get("has_keypose_ghost") for x in viz_samples)),
    }
    s.motion_receipt = _motion_receipt(commanded, measured, tcp_world, replan_steps)

    s.runtime_breakdown = {
        "env_step_mean_s": float(np.mean(env_step_s)) if env_step_s else None,
        "env_step_median_s": float(np.median(env_step_s)) if env_step_s else None,
        "frame_build_mean_s": float(np.mean(frame_s)) if frame_s else None,
        "frame_build_median_s": float(np.median(frame_s)) if frame_s else None,
        "proxy_replan_total_s": float(np.sum(replan_s)),
        "episode_steps_per_wall_s": float(s.env.n_steps / episode_s) if episode_s else None,
        "env_step_internal_total_s": {k: float(v) for k, v in nested_runtime.items()},
        "env_step_internal_mean_s": {
            k: float(v / max(s.env.n_steps, 1)) for k, v in nested_runtime.items()},
        "disabled_contact_sensors": list(getattr(s.env, "disabled_contact_sensors", ())),
        "command_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(commanded, dtype=np.float64)).tobytes()).hexdigest(),
        "measured_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(measured, dtype=np.float64)).tobytes()).hexdigest(),
        "tcp_world_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(tcp_world, dtype=np.float64)).tobytes()).hexdigest(),
    }
    final = _episode_summary(args, s, replan_s, episode_s, success, stage_max)
    log.episode(final)
    log.close()
    trace_path = paths.episode_path(out_dir, args.seed, success, "trace", "jsonl")
    jsonl_path.replace(trace_path)
    paths.prune_pending(out_dir)
    video_path = save_video(frames, paths.episode_path(out_dir, args.seed, success, "videos",
                                                       "mp4"),
                            fps=CONTROL_HZ / int(args.video_stride))
    print(f"{LOG} log:   {trace_path}", flush=True)
    print(f"{LOG} video: {video_path}", flush=True)
    if s.proxy is not None:
        s.proxy.close()
    restore_runtime_profiler()
    s.env.close()
    return final
