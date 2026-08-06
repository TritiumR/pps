"""Run MuJoCo evaluation rollouts with optional sampling, steering, and perturbation mechanisms."""
from __future__ import annotations

import functools
import math
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
from sim_free_mpc.ddim import ddim_iteration_alphas
from sim_free_mpc.fk import PandaFK
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig, task_tilt_weight
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


# Raw robosuite demos store joint angles under robot0_joint_pos; the 224-render conversion the
# proxies train on writes joint_pos. Same quantity, and the stats must match whichever file the
# caller has, so accept both rather than making them pass the right one.
_JOINT_POS_KEYS = ("obs/robot0_joint_pos", "obs/joint_pos")


def demo_delta_stats(hdf5, horizon, n_demos=100):
    """Estimate action-delta standard deviations from demonstration data."""
    import h5py
    with h5py.File(hdf5, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:n_demos]
        key = next((k for k in _JOINT_POS_KEYS if k in f[f"data/{names[0]}"]), None)
        if key is None:
            raise KeyError(f"{hdf5}: no joint positions under any of {_JOINT_POS_KEYS}")
        deltas = []
        for n in names:
            q = np.asarray(f[f"data/{n}/{key}"])
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
                                          stall_margin_exit=self.hold_exit,
                                          lost_on_free_close=self.hold_free_close)
        self.world = MGWorld(raw_env, sensor=self.sensor, names=self._source.movable,
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
    # F9: default is now "on", so it must no-op when there is no proxy at all -- the base path
    # calls this too, and an unguarded flip made every un-steered rollout crash on None.
    if getattr(args, "align_proxy_norm", "off") == "on" and getattr(args, "proxy_checkpoint", None):
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
    n_cand = args.candidates
    if int(getattr(args, "fk_particles", 0)) > 1:
        # Budget neutral by construction: K chains x N/K candidates spend the base's N. That makes
        # --fk_particles 1 the exact matched control, unlike select, whose control needed M x N.
        k = int(args.fk_particles)
        n_cand = max(1, args.candidates // k)
        print(f"{LOG} fk: {k} particles x {n_cand} candidates = {k * n_cand} total "
              f"(base spends {args.candidates})", flush=True)
    cfg_kw = dict(
        task_name=args.task, num_samples=n_cand, iterations=1,
        noise=args.noise, temperature=args.temperature, action_dims=8,
        joint_delta_clip=args.delta_clip, cost_style="priority",
        optimize_space="action", sampler="base", grad_calc="mbd",
        control_frequency=20.0,
        rank_mode=args.rank_mode, prior_weight=args.prior_weight,
        prior_weight_high=args.prior_weight_high,
        prior_weight_schedule=args.prior_weight_schedule,
        estimator=getattr(args, "estimator", "mean"),
        draw_below=float(getattr(args, "draw_below", float("inf"))))
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


def _blend_blocks(x_pol, x_mbd, args):
    """Blend the policy chain toward the MBD step, per token block.

    Rows partition into [action | trajectory | keypose], each with its own coefficient, mirroring
    Cory's `_blend_policy_mbd_flows`. A block coefficient of 0 keeps the policy, 1 takes the MBD
    step; a single --steer_gamma applied to all three reproduces the uniform blend.

    The keypose defaults to the LAST row -- the final-action proxy his eval uses for a checkpoint
    that has no dedicated keypose token, which ours does not.
    """
    rows = x_pol.shape[1]
    keypose = rows - 1 if args.keypose_row is None else int(args.keypose_row)
    traj_start = rows // 2 if args.traj_start_row is None else int(args.traj_start_row)
    if not 0 <= traj_start <= keypose < rows:
        raise SystemExit(f"bad policy_base blocks: traj_start={traj_start} keypose={keypose} "
                         f"rows={rows}")
    g = float(args.steer_gamma)
    g_act = g if args.block_gamma_action is None else float(args.block_gamma_action)
    g_traj = g if args.block_gamma_traj is None else float(args.block_gamma_traj)
    g_kp = g if args.block_gamma_keypose is None else float(args.block_gamma_keypose)
    # lerp, NOT `x_pol + g*(x_mbd - x_pol)`: the latter is algebraically equal but loses the
    # endpoints to cancellation, and MBD is winner-take-all, so a 1e-7 drift at level 0 becomes a
    # different trajectory. (1-g)*a + g*b is exact at g=0 and g=1, which the identity tests need.
    out = x_pol.clone()
    for lo, hi, g_blk in ((0, traj_start, g_act),
                          (traj_start, keypose, g_traj),
                          (keypose, keypose + 1, g_kp)):
        if lo < hi:
            out[:, lo:hi] = (1.0 - g_blk) * x_pol[:, lo:hi] + g_blk * x_mbd[:, lo:hi]
    return out


def infer_chunk(planner, env, ctx, args, stage_key=None, steer=None, x0_init=None, out=None,
                vls=None):
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
        w = float(getattr(args, "warm_start_proxy", 0.0))
        if w > 0.0 and hasattr(steer, "expert_chunk") and hasattr(steer, "_to_model_space"):
            # Start the chain at the proxy's chunk instead of pure noise, noised to the level-0
            # scale so the sampler still sees an in-distribution input. Unlike inject this spends
            # no candidates: injection replaces rho of the pool with proposals that carry ~0
            # softmax weight, measured as ESS -> (1-rho)*N for no gain.
            x0 = steer._to_model_space(steer.expert_chunk(), planner.policy)
            x0 = torch.as_tensor(x0, dtype=x_t.dtype)[: args.horizon]
            x_t[0, : args.horizon] = (math.sqrt(w) * x0
                                      + math.sqrt(1.0 - w) * x_t[0, : args.horizon])

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
        elif steer is not None and steer.mode not in ("additive", "policy_base",
                                                      "proxy_pair", "vls"):
            inj = steer.inject_for(it, args.num_steps + 1, planner.policy)
            if inj is None:
                ctx.pop("inject", None)
            else:
                ctx["inject"] = inj
        if steer is not None and steer.mode == "proxy_pair":
            # Both operands learned: no MBD step at all.
            x_t = steer.policy_pair_step(x_t, it, args.num_steps + 1)
        elif steer is not None and steer.mode == "vls":
            # VLS: the trained policy denoises; the geometric objective enters as a NORMALISED
            # gradient on the policy's own clean prediction, gated by task progress. No second
            # score field, so nothing has to be commensurate; no fixed gamma, so the coefficient
            # is state-dependent. Guide x0, then re-enter the chain through the proxy's own DDIM
            # operator so the correction is integrated in the parameterisation it was applied in.
            vls.target = np.asarray(ctx["target"], dtype=np.float32)
            _st = steer.policy_step_full(x_t, it, args.num_steps + 1)
            _x0 = _st["x0_hat"]
            _dec = decode_model_action_chunks(planner.policy, inputs, _x0, apply_clamp=False,
                                              current_joint_pos=q0)
            # The gradient is taken in REAL joint units (FK needs them) and applied in MODEL
            # units, so it carries the chain-rule factor d(real)/d(model) = action_std.
            _std = torch.as_tensor(
                planner.policy._metadata["output_norm_stats"]["actions"].std,
                device=_x0.device, dtype=_x0.dtype)
            _g, _r, _gate, _p = vls.gradient(_dec.real_actions, planner.fk)
            _x0 = _x0 + (args.vls_scale * _gate) * _g * _std
            x_t = steer.redo_ddim(_x0)
        elif steer is not None and steer.mode == "policy_base":
            # Inverted direction: the trained policy denoises and the MBD step is the steering
            # signal. Both branches map x_t -> next x_t in planner space, so they blend directly.
            # gamma=0 is the policy alone, gamma=1 is the base -- both already-measured arms.
            x_pol = steer.policy_step(x_t, it, args.num_steps + 1)
            x_mbd, stats = planner.step_mbd_score_action_prox(
                x_t, inputs, ctx, iteration=it, num_iterations=args.num_steps + 1)
            x_t = _blend_blocks(x_pol, x_mbd, args)
        else:
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


def keypose_fk_chunk(planner, env, ctx, args, stage_key, steer, out=None):
    """Cory's keypose FK steering: the trained policy denoises, a keypose-row cost steers it.

    Inverted direction, like --steer policy_base, but the steering signal is no longer a whole
    MBD step: it is a cost-weighted mean over a proposal cloud around the KEYPOSE ROW only,
    resampled across particles by a Feynman-Kac potential and KL-capped before it is applied.
    See steering/keypose_fk.py for the per-step algorithm and the three adaptations.
    """
    from .steering import keypose_fk as kfk
    from .steering.chunk_cost import chunk_costs

    del stage_key
    q0 = env.q0()
    state = torch.zeros(8, dtype=torch.float32)
    state[:7] = q0
    inputs = {"state": state}
    rows = args.horizon
    p = int(args.kfk_particles)
    keypose_row = rows - 1 if args.keypose_row is None else int(args.keypose_row)
    if not 0 < keypose_row < rows:
        raise SystemExit(f"--keypose_row must be in (0, {rows}); got {keypose_row}")
    # Executable rows, when the chunk also carries AWE waypoint rows between the actions and
    # the keypose. Without this, --horizon 21 on an AWE proxy either steered W1 (keypose_row
    # 15, the measured bug) or executed the waypoint rows as if they were actions.
    action_rows = int(args.kfk_action_rows) if args.kfk_action_rows else keypose_row
    if not 0 < action_rows <= keypose_row:
        raise SystemExit(f"--kfk_action_rows must be in (0, {keypose_row}]; got {action_rows}")
    # Which rows the proposal cloud perturbs. Cory's method estimates the WHOLE goal block
    # (AWE waypoints + keypose) jointly -- 6 rows in his 22-row chunk -- not the keypose
    # alone. Perturbing one row of sixteen barely moves a cost that reduces over rows.
    goal_slice = (slice(action_rows, keypose_row + 1) if args.kfk_goal_block == "on"
                  else slice(keypose_row, keypose_row + 1))

    steer.begin_replan(env, args.num_steps + 1)
    x = torch.randn(p, rows, 8)
    gen = torch.Generator().manual_seed(int(args.seed) * 7919 + steer.replan_idx)
    # q0 in MODEL space, so the straight-line path the action rows are pulled along is expressed
    # in the same units as the chunk. demo_delta centres the arm at 0, so q0 maps to the origin.
    q0_model = torch.zeros(8, dtype=torch.float32)

    ess_trace, kl_trace, n_resample = [], [], 0
    spread, rel_spread = [], []
    stats = {}
    for it in range(args.num_steps + 1):
        alpha, _ = ddim_iteration_alphas(
            iteration=it, num_iterations=args.num_steps + 1,
            num_train_timesteps=planner.config.ddim_num_train_timesteps)
        # Proposal width. "sqrt_beta" is what we shipped: sigma * sqrt(1-alpha_bar), floored.
        # "flow_matched" is Cory's: a clean-space perturbation of t/(1-t) induces exactly t in
        # x_t, so the cloud matches the model's own noise schedule at every level instead of
        # being constant. We chased discrimination by inflating sigma to 6 (relspread 4.2 ->
        # 26.6%) and success did not move, because a fixed-width cloud goes off-manifold at low
        # noise -- which is the failure this schedule is designed to avoid.
        if args.kfk_proposal_schedule == "flow_matched":
            t = float(np.clip(math.sqrt(max(1.0 - float(alpha), 0.0)), 1e-3, 1.0 - 1e-3))
            sigma = float(args.kfk_proposal_std) * t / (1.0 - t)
        else:
            sigma = float(args.kfk_proposal_std) * max(
                math.sqrt(max(1.0 - float(alpha), 0.0)), 0.1)

        guided_chunks, potentials = [], []
        for i in range(p):
            # C4: rank on the model's clean prediction, not on x_{t-1}. Under --kfk_clean_rank
            # off this reproduces the old (buggy) behaviour bitwise for A/B.
            if getattr(args, "kfk_clean_rank", "on") == "on":
                _st = steer.policy_step_full(x[i : i + 1], it, args.num_steps + 1)
                x_pol = _st["x0_hat"][0]
            else:
                x_pol = steer.policy_step(x[i : i + 1], it, args.num_steps + 1)[0]
            proposals = kfk.sample_proposals(
                x_pol[goal_slice], sigma, int(args.kfk_proposals), gen)
            # Each proposal is a full chunk that differs from the policy's only in the GOAL rows,
            # so the cost sees a complete, decodable trajectory. Under --kfk_goal_block those are
            # the AWE waypoints AND the keypose, which is what Cory's method perturbs; the default
            # is the keypose row alone, as shipped.
            cand = x_pol.unsqueeze(0).repeat(int(args.kfk_proposals), 1, 1)
            cand[:, goal_slice] = proposals
            dec = decode_model_action_chunks(planner.policy, inputs, cand, apply_clamp=True,
                                             current_joint_pos=q0, max_joint_delta=args.delta_clip)
            costs = chunk_costs(planner, dec.real_actions.detach().cpu().numpy(), ctx,
                                bucket=args.kfk_cost_bucket, ranker=args.kfk_ranker,
                                keypose_row=keypose_row, action_rows=action_rows)
            guided_kp, potential = kfk.guided_from_costs(proposals, costs, args.kfk_temperature)
            # Does the cost actually separate the cloud? A near-zero spread means the softmax is
            # uniform and the "guided" keypose is just the cloud's mean -- guidance in name only.
            c = torch.as_tensor(costs, dtype=torch.float32).reshape(-1)
            spread.append(float(c.max() - c.min()))
            rel_spread.append(float((c.std() / c.abs().mean().clamp(min=1e-6))))

            delta, applied_kl = kfk.kl_capped(
                guided_kp - x_pol[goal_slice], sigma, float(args.kfk_max_kl))
            guided = x_pol.clone()
            guided[goal_slice] = x_pol[goal_slice] + delta
            # H6: the pull is meant to move EXECUTABLE rows toward the goal; passing keypose_row
            # let it overwrite the AWE waypoint rows that sit between actions and key-pose.
            guided = kfk.action_l1_pull(guided, keypose_row, q0_model,
                                        float(args.kfk_action_l1_step), action_rows=action_rows)
            # C4: both siblings re-enter the chain through the proxy's own DDIM operator, so the
            # guidance is integrated in the parameterisation it was applied in.
            if getattr(args, "kfk_clean_rank", "on") == "on":
                pure_next = steer.redo_ddim(x_pol.unsqueeze(0))[0]
                guided_next = steer.redo_ddim(guided.unsqueeze(0))[0]
                guided_chunks.append(torch.stack([pure_next, guided_next]))
            else:
                guided_chunks.append(torch.stack([x_pol, guided]))
            potentials.append(potential)
            kl_trace.append(applied_kl)

        ess_now = kfk.ess_of(potentials)
        ess_trace.append(round(ess_now, 2))
        parents = kfk.resample_parents(potentials, p, gen)
        n_resample += 1

        # Paired children: half continue the pure policy, half the guided chunk, so resampling
        # cannot collapse the population onto guidance alone.
        nxt = torch.empty_like(x)
        slot_potential, slot_parent = [], []
        if args.kfk_paired == "on":
            # F4. Slots used to draw parents INDEPENDENTLY and then let slot parity decide
            # pure-vs-guided, so slot 2i and 2i+1 were usually descendants of different particles
            # and the "paired" comparison confounded the intervention with a different base sample.
            # Draw P//2 parents and give each one both children.
            half = kfk.resample_parents(potentials, max(p // 2, 1), gen).tolist()
            for j in range(p):
                parent = half[(j // 2) % len(half)]
                nxt[j] = guided_chunks[parent][0 if j % 2 == 0 else 1]
                slot_potential.append(potentials[parent])
                slot_parent.append(parent)
        else:
            for slot, parent in enumerate(parents.tolist()):
                nxt[slot] = guided_chunks[parent][1]
                slot_potential.append(potentials[parent])
                slot_parent.append(parent)
        x = nxt

    # F5. Selection used the potential INHERITED from the parent, so pure and guided siblings
    # carried the same score even when their final chunks differed. Re-cost the actual final
    # candidates and pick among those.
    _fin = decode_model_action_chunks(planner.policy, inputs, x, apply_clamp=True,
                                      current_joint_pos=q0, max_joint_delta=args.delta_clip)
    _fc = chunk_costs(planner, _fin.real_actions.detach().cpu().numpy(), ctx,
                      bucket=args.kfk_cost_bucket, ranker=args.kfk_ranker,
                      keypose_row=keypose_row, action_rows=action_rows)
    _fc = np.asarray(_fc, dtype=np.float64).reshape(-1)
    best = int(np.argmin(_fc)) if np.isfinite(_fc).all() else int(np.argmax(slot_potential))
    x_t = x[best : best + 1]
    if out is not None:
        out["x_final"] = x_t.detach().clone()
    dec = decode_model_action_chunks(planner.policy, inputs, x_t, apply_clamp=True,
                                     current_joint_pos=q0, max_joint_delta=args.delta_clip)
    # The keypose is a PLANNING token, not an action: drop it before the controller sees the
    # plan, or a large spi would command the arm straight to the phase-end pose.
    plan = dec.real_actions[0, :action_rows].detach().cpu().numpy()
    rec = {"kfk_particles": p, "kfk_best": best, "kfk_resamples": n_resample,
           "kfk_ess_mean": round(float(np.mean(ess_trace)), 2),
           "kfk_ess_trace": ess_trace,
           "kfk_kl_mean": round(float(np.mean(kl_trace)), 4),
           "kfk_kl_max": round(float(np.max(kl_trace)), 4),
           "kfk_cost_spread": round(float(np.mean(spread)), 4),
           "kfk_cost_relspread": round(float(np.mean(rel_spread)), 4)}
    return plan, stats, rec, None


def fk_chunk(planner, env, ctx, args, stage_key, steer, out=None):
    """Feynman-Kac steering: denoise K chains in lockstep, resampling them by proxy potential.

    The only mode whose selection acts ACROSS chains. Each level steps every particle, scores it
    against the proxy's clean chunk for that level, accumulates FK log-weights and -- when the
    particle ESS falls below --fk_ess_frac -- resamples, so chains the proxy likes are duplicated
    and chains it dislikes die. --candidates is split K ways upstream, so K=1 is the plain base.
    """
    from .steering import fk as fk_mod

    q0 = env.q0()
    state = torch.zeros(8, dtype=torch.float32)
    state[:7] = q0
    inputs = {"state": state}
    kp = isinstance(planner, KPPlanner)
    rows = args.horizon + (planner.kp.k if kp else 0)
    k = int(args.fk_particles)

    steer.begin_replan(env, args.num_steps + 1)
    # Initial noise comes from the GLOBAL stream, exactly as infer_chunk draws it, so K=1 is
    # bit-identical to the unsteered base. Only resampling uses its own generator.
    x = torch.randn(k, rows, 8)
    gen = torch.Generator().manual_seed(int(args.seed) * 7919 + steer.replan_idx)
    if kp:
        w0 = planner.kp_begin_replan(ctx, stage_key)
        x[:, args.horizon:, 3:] = 0.0
        if w0 is not None:
            x[:, args.horizon:, :3] = w0

    planner.begin_inference()
    log_w = torch.zeros(k, dtype=torch.float64)
    g_prev = torch.zeros(k)
    ess_trace, n_resample, stats = [], 0, {}
    for it in range(args.num_steps + 1):
        nxt, per_particle = torch.empty_like(x), []
        for i in range(k):
            step, stats = planner.step_mbd_score_action_prox(
                x[i : i + 1], inputs, ctx, iteration=it, num_iterations=args.num_steps + 1)
            nxt[i : i + 1] = step
            per_particle.append(stats)
        x = nxt

        # Score the CLEAN estimate, not the noisy iterate: the planner's own proposal center is
        # x_t / sqrt(alpha_bar), and the proxy's per-level chunk lives in clean space.
        a_prev = max(float(stats.get("alpha_bar_prev", 1.0)), 1e-6)
        target, abar = steer.fk_target(it, args.num_steps + 1, planner.policy)
        if args.fk_signal == "cost":
            # Cory's potential: weight a particle by the MBD reward of the region it sits in.
            # This is NOT steering -- the cost is our base, so no external proxy signal enters.
            # It is the machinery test: can resampling alone move a base that scores 0/25?
            g = -torch.nan_to_num(
                torch.tensor([float(st.get("cost_weighted", 0.0)) for st in per_particle]),
                nan=0.0, posinf=0.0, neginf=0.0)
        else:
            g = fk_mod.log_potential(x / math.sqrt(a_prev), target, 1.0, dims=args.tilt_dims)
        if args.fk_norm == "on":
            # Standardize the SHAPE first, then apply lambda. Scaling before standardizing
            # divides lambda straight back out and leaves the knob inert -- measured, lambda 1
            # and lambda 4 gave identical particle ESS (3.99/6). alpha_bar keeps the SNR temper:
            # muted early, where the proxy's implied clean target is least reliable.
            g = fk_mod.normalize_logits(g) * float(args.fk_lambda) * abar
        else:
            g = g * task_tilt_weight(args.fk_lambda, args.temperature, args.noise, abar)
        log_w = log_w + (g - g_prev if args.fk_potential == "diff" else g).double()
        g_prev = g

        now = fk_mod.ess(log_w)
        ess_trace.append(round(now, 2))
        if (it + 1) % max(1, int(args.fk_resample_every)) == 0 and now < args.fk_ess_frac * k:
            idx = fk_mod.systematic_resample(log_w, gen)
            x, g_prev = x[idx].clone(), g_prev[idx].clone()
            log_w = torch.zeros(k, dtype=torch.float64)
            n_resample += 1

    best = int(torch.argmax(log_w))
    x_t = x[best : best + 1]
    if kp:
        w_plan = planner.kp_finish_replan(x_t)
        x_t = x_t[:, : args.horizon]
    else:
        w_plan = None
    if out is not None:
        out["x_final"] = x_t.detach().clone()
    dec = decode_model_action_chunks(planner.policy, inputs, x_t, apply_clamp=True,
                                     current_joint_pos=q0, max_joint_delta=args.delta_clip)
    rec = {"fk_particles": k, "fk_best": best, "fk_resamples": n_resample,
           "fk_ess_first": ess_trace[0], "fk_ess_last": ess_trace[-1],
           "fk_ess_mean": round(sum(ess_trace) / len(ess_trace), 2),
           "fk_ess_trace": ess_trace}
    return dec.real_actions[0].detach().cpu().numpy(), stats, rec, w_plan


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
        return cls(args.task, args.rekep_context, args.rekep_constraints,
                   vlm=getattr(args, "rekep_vlm", "fake"))
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
    if args.steer == "fk" and int(args.fk_particles) < 1:
        raise SystemExit("--steer fk needs --fk_particles >= 1 (1 is the budget-matched control)")
    if args.steer == "keypose_fk":
        # keypose_fk drops the keypose row before the controller sees the plan (it is a planning
        # token), so a chunk of H rows executes H-1. Without this the overrun surfaces as an
        # IndexError mid-rollout, after the first chunk is exhausted.
        rows = (int(args.kfk_action_rows) if args.kfk_action_rows
                else int(args.horizon) - 1)
        if int(args.spi) > rows:
            raise SystemExit(
                f"--steer keypose_fk executes only {rows} of --horizon {args.horizon} rows (the "
                f"keypose row is a planning token, not an action); --spi {args.spi} overruns it. "
                f"Use --spi <= {rows}, or --horizon {int(args.spi) + 1}.")
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
    ref_client = None
    if args.steer == "proxy_pair" and not args.ref_checkpoint:
        # H13: proxy_pair composes two LEARNED fields; without a reference it dereferenced None
        # deep inside score_addend instead of failing at setup.
        raise SystemExit("--steer proxy_pair needs --ref_checkpoint (it composes the task proxy "
                         "with a reference proxy; there is no MBD operand in this mode)")
    if args.steer_ref == "proxy" or args.steer == "proxy_pair":
        if not args.ref_checkpoint:
            raise SystemExit("--steer_ref proxy needs --ref_checkpoint (a proxy distilled from "
                             "the base; see grounding/../train_mpc_proxy_score_pytorch.py "
                             "generate-cache + train)")
        ref_client = ProxyScoreClient(args.ref_checkpoint, prompt, device=args.ref_device,
                                      prediction_mode=args.proxy_prediction_mode,
                                      kv_cache=args.proxy_kv_cache == "on",
                                      fp16=args.proxy_fp16 == "on").start()
        atexit.register(ref_client.close)
        # C1, CRITICAL. The server silently falls back to DROID quantile normalisation when a
        # checkpoint ships no action_norm_stats.json, and reports which one it used. Reference
        # checkpoints (ref_square_mbd, ref_can_mbd) ship none while task checkpoints do, so
        # s_task - s_ref was being differenced ACROSS TWO ACTION COORDINATE SYSTEMS. Confirmed
        # from the addref_g050 and pair_g050 logs: one server reported demo_delta, the other
        # droid_quantile, in the same rollout. Gamma cannot repair that, and the arms that used
        # it (addref 0/100, pair 0/100 and 8/100) are confounded, not evidence about composition.
        t_norm = str(client.ready_info.get("action_norm", "?"))
        r_norm = str(ref_client.ready_info.get("action_norm", "?"))
        if t_norm != r_norm:
            raise SystemExit(
                f"Task and reference proxies disagree on action normalisation: task={t_norm!r} "
                f"ref={r_norm!r}. Their scores are vector fields over different variables, so "
                f"s_task - s_ref is meaningless. Regenerate the reference with the SAME "
                f"--action_norm as the task proxy so it ships action_norm_stats.json, or pass "
                f"--ref_action_norm_stats explicitly.")
        print(f"{LOG} reference proxy on {args.ref_device} (action_norm={r_norm}, matched)",
              flush=True)
    if args.steer in ("additive", "policy_base", "keypose_fk", "proxy_pair", "vls"):
        # policy_base reuses this class for its space bridge and per-level score access; the
        # gamma here is the blend toward MBD, not a score addend.
        steer = AdditiveScoreSteering(client, gamma=args.steer_gamma,
                                      policy=planner.policy, horizon=args.horizon,
                                      score_cap=args.proxy_score_cap, seed=args.seed,
                                      at=args.proxy_score_at, ref=args.steer_ref,
                                      ref_client=ref_client,
                                      last_level=args.steer_last_level)
        steer.mode = args.steer
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
    # A bug Codex did not report. Measured on square with --ground rekep: the template attaches
    # `constraint` only to the PLACE stage, so rekep_subgoal falls through to zeros on grasp and
    # lift -- cost_min 0.0 and weight_ess 512.0/512, a perfectly uniform softmax, for the whole
    # part of the episode where the task is actually decided. --ground rekep_vlm emits stages from
    # the ReKep metadata and carries the constraint on every one of them.
    _rk_w = sum(float(v or 0.0)
                for k, v in ((cfg.get("cost") or {}).get("terms") or {}).items()
                if str(k).startswith("rekep_"))
    if _rk_w > 0.0 and args.ground == "rekep":
        print(f"{LOG} WARNING: this config weights rekep_* terms ({_rk_w:g} total) but "
              f"--ground rekep attaches the constraint to the place stage only, so grasp and "
              f"lift score exactly 0. Use --ground rekep_vlm for a live constraint everywhere.",
              flush=True)
    vls = None
    if args.steer == "vls":
        from .steering.vls import VLSGuidance
        # Objective = the stage's own target, the one thing the VLM specifies. Feasibility,
        # collision, smoothness and the gripper stay with the policy -- that division is what VLS
        # gets right and our 26-term cost does not (measured: 26 terms anti-align at -0.687, the
        # 1-term ReKep constraint at -0.208).
        def _objective(ee_pos, _v=None):
            # The stage target is already in the ctx infer_chunk receives; VLSGuidance carries it
            # per replan rather than re-deriving it from the bridge.
            tgt = torch.as_tensor(vls.target, device=ee_pos.device, dtype=ee_pos.dtype)
            return -((ee_pos - tgt) ** 2).sum(-1).mean()
        vls = VLSGuidance(_objective, scale=args.vls_scale, sigmoid_k=args.vls_sigmoid_k,
                          sigmoid_x0=args.vls_sigmoid_x0, action_rows=args.horizon)
        print(f"{LOG} vls: scale={args.vls_scale} k={args.vls_sigmoid_k} x0={args.vls_sigmoid_x0}",
              flush=True)
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
                                 kp_cfg=kp_cfg, beam=beam, steer=steer, client=client, vls=vls,
                                 ensemble=ensemble, perturb=_build_perturb(args, source))


def _episode_summary(args, s, replan_s, episode_s, success, stage_max, handoff=None):
    final = {
        "kind": "episode", "task": args.task, "seed": args.seed, "success": success,
        "env_steps": s.env.n_steps, "stage_final": s.bridge.stage_idx, "stage_max": stage_max,
        "stage_name": s.bridge.stage().name, "replans": len(replan_s),
        "replan_wall_median_s": round(float(np.median(replan_s)), 3),
        "replan_wall_mean_s": round(float(np.mean(replan_s)), 3),
        "episode_wall_s": round(episode_s, 1),
        **({"handoff_step": handoff} if handoff is not None else {}),
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
        if args.steer == "additive":
            mech = {"gamma": args.steer_gamma}
        elif args.steer == "policy_base":
            # Record the coefficient each block actually used, not the flags: an arm whose gamma
            # is not in its own telemetry cannot be attributed later.
            rows = args.horizon
            mech = {"gamma": args.steer_gamma,
                    "block_gamma": {
                        "action": args.block_gamma_action if args.block_gamma_action is not None
                        else args.steer_gamma,
                        "traj": args.block_gamma_traj if args.block_gamma_traj is not None
                        else args.steer_gamma,
                        "keypose": args.block_gamma_keypose if args.block_gamma_keypose is not None
                        else args.steer_gamma},
                    "keypose_row": rows - 1 if args.keypose_row is None else args.keypose_row,
                    "traj_start_row": rows // 2 if args.traj_start_row is None
                    else args.traj_start_row}
        else:
            mech = {"rho": args.inject_rho, "schedule": args.inject_schedule}
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
    if (s.steer is not None and getattr(args, "steer_gamma_stages", None)
            and s.steer.mode in ("additive", "policy_base")):
        # Stage-gated authority: gamma follows the bridge stage. additive reads steer.gamma at
        # call time; policy_base's _blend_blocks reads args.steer_gamma -- set both.
        gammas = [float(x) for x in args.steer_gamma_stages.split(",")]
        s.steer.gamma = args.steer_gamma = gammas[min(s.bridge.stage_idx, len(gammas) - 1)]

    t0 = time.perf_counter()
    beam_rec, w_plan, select_rec = None, None, None
    if s.steer is not None and s.steer.mode == "keypose_fk":
        plan, stats, select_rec, w_plan = keypose_fk_chunk(
            s.planner, s.env, ctx, args, stage_key, s.steer)
    elif s.steer is not None and s.steer.mode == "fk":
        plan, stats, select_rec, w_plan = fk_chunk(s.planner, s.env, ctx, args, stage_key, s.steer)
    elif s.steer is not None and s.steer.mode == "select":
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
                                      vls=getattr(s, 'vls', None),
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


def _handoff_due(args, s, step, fired):
    """True when the incumbent has stalled and the other policy should take over.

    Measured on can (base 72 / proxy 74 / oracle 93): the two policies fail on largely DISJOINT
    episodes -- 19 base-only, 21 proxy-only -- and in both groups the loser never grasps and burns
    the full 300-step budget, while winners finish in ~100-130. So "no grasp yet" at a deadline is
    a sufficient stall signal, and it needs no classifier, no outcome labels and no privileged
    features: `world.held()` is the same aperture latch the rollout already senses.

    This is the deployable form of the supervised router. The router decides once at t=0 from the
    object pose; this decides mid-episode from execution state, which is also what a recovery
    argument requires.
    """
    if not args.handoff_at or fired or s.steer is None:
        return False
    if step < int(args.handoff_at):
        return False
    return s.bridge.world.held() is None          # never got the object


def rollout(args):
    out_dir = paths.results_dir(args.task, args.exp)
    jsonl_path, video_path = out_dir / f"{args.seed}.jsonl", out_dir / f"{args.seed}.mp4"
    s = _setup(args)

    t_ep = time.perf_counter()
    log = record.Recorder(jsonl_path, prefix=LOG)
    states = [s.env.get_state()]
    ee_trace = [np.asarray(s.env.tcp(), dtype=np.float32)]
    goal_trace = []                       # (step, goal rows) per replan, for the ghost overlay
    actions, start = None, 0
    success, replan_s, stage_max = False, [], 0
    handoff_fired, handoff_step = False, None
    if args.handoff_from and s.steer is not None:
        s.steer.mode = args.handoff_from
    for step in range(args.max_steps):
        if _handoff_due(args, s, step, handoff_fired):
            # Swap which policy drives. Nothing else is reset: the scene, the stage ladder and
            # the hold latch carry over, so the successor inherits the incumbent's progress.
            was, s.steer.mode = s.steer.mode, args.handoff_to
            handoff_fired, handoff_step = True, step
            actions = None                              # force an immediate replan
            log.write({"kind": "handoff", "step": step, "from": was, "to": s.steer.mode})
            print(f"{LOG} handoff at step {step}: {was} -> {s.steer.mode} (nothing held)",
                  flush=True)
        if actions is None or step - start >= args.spi:
            plan, wall_s = _replan(args, s, step, log, dt=0 if actions is None else args.spi)
            replan_s.append(wall_s)
            stage_max = max(stage_max, s.bridge.stage_idx)
            actions, start = plan[: args.spi], step
            if s.ensemble is not None:      # the chunk _replan just fetched, not a second call
                s.ensemble.push(step, plan)
            _goals = (s.steer.goal_rows()
                      if s.steer is not None and hasattr(s.steer, "goal_rows") else None)
            if _goals is not None and len(_goals):
                goal_trace.append((step, np.asarray(_goals)[:, :7]))
        a = s.ensemble.action(step) if s.ensemble is not None else actions[step - start]
        if s.perturb is not None:
            log.perturb(s.perturb.on_step(step, s.env, s.bridge))
            a = s.perturb.filter_action(a, s.env, s.bridge)
        # H14. observe_step() reads live aperture and object poses, so calling it BEFORE
        # apply_arm pairs each command with the state that preceded it -- hold latching and stage
        # transitions run one control step (50 ms at 20 Hz) out of phase. sensor_cadence defaults
        # to "step", so this path is live. Default stays legacy on purpose: flipping it shifts
        # every baseline at once, so it needs a matched A/B, not a silent change.
        if getattr(args, "sensor_order", "legacy") == "apply_first":
            s.env.apply_arm(a[:7], grip_close=float(a[7]) > 0.5)
            s.bridge.observe_step(a)
        else:
            s.bridge.observe_step(a)
            s.env.apply_arm(a[:7], grip_close=float(a[7]) > 0.5)
        states.append(s.env.get_state())
        ee_trace.append(np.asarray(s.env.tcp(), dtype=np.float32))
        if s.env.success():
            success = True
            break
    episode_s = time.perf_counter() - t_ep

    final = _episode_summary(args, s, replan_s, episode_s, success, stage_max,
                             handoff=handoff_step)
    log.episode(final)
    log.close()
    if s.client is not None:
        s.client.close()

    # Overlay the grounding on every frame: keypoints (re-read per frame, so points riding moving
    # objects follow them), the active stage target, and the executed EE trail. Only drawn when the
    # grounding actually supplies keypoints, so gt-grounded runs are unchanged.
    _kp = getattr(s.bridge.grounding, "keypoints", None)
    _stage = s.bridge.stage()
    overlay = None
    if _kp is not None or goal_trace:
        overlay = {"keypoints": _kp,
                   "subgoal": (_stage.target if callable(getattr(_stage, "target", None)) else None),
                   "ee_path": ee_trace,
                   "goals": goal_trace,
                   "lines": lambda i: [f"{args.task} | {args.exp}",
                                       f"step {i}/{len(states) - 1}"]}
    record.save_video(s.env, states, video_path, overlay=overlay)
    print(f"{LOG} log:   {jsonl_path}", flush=True)
    print(f"{LOG} video: {video_path}", flush=True)
    return final
