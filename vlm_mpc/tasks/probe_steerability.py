"""Steerability diagnostic for the DIAL base: sampler-vs-optimizer, mode-commitment, latent smoothness.

The base is only PPS/DSRL-steerable if it expresses behaviour as a navigable noise->action map and score
field rather than collapsing to a point. Three probes measure that at a frozen state:

  P1 output diversity     spread of independent plans -> sampler (spread) vs optimizer (collapse)
  P2 ESS vs denoise step  effective sample size + plan change over the reverse loop -> commitment timing
  P3 latent interpolation terminal EE as the initial noise is interpolated -> behaviour-manifold smoothness

Steerability is state-dependent, so the probes run across regimes -- ``far_fresh`` (approach, fresh noise),
``near_fresh`` (driven up to the object, fresh noise), ``near_warm`` (up close, warm-started from the
previous plan, the normal operating point) -- and across the reverse-update modes the diffusion/flow
literature flags as the collapse knobs (``score_space`` vs ``ddim``, the flow-matching Euler sampler).
Nothing is executed during a probe; the state is frozen. Metrics JSON + a figure land in
``results/vlm_mpc/probe/``.

    /isaac-sim/python.sh -m vlm_mpc.main --task probe_steerability --grasp_obj pear
"""
import os

NAME = "probe_steerability"

# Fixed engine choices, matching the base task (faithful decode + B-spline smoother).
_REAL_STATS = True
_INTERPOLATE = True


def add_args(ap):
    ap.add_argument("--exp_name", type=str, default="probe")
    ap.add_argument("--grasp_obj", type=str, default="pear", help="object whose centre is the probe target")
    ap.add_argument("--regimes", type=str, default="far_fresh,near_fresh,near_warm",
                    help="operating points to probe (far/near state x fresh/warm init)")
    ap.add_argument("--updates", type=str, default="score_space,ddim",
                    help="reverse-update modes to compare (ddim = flow-matching Euler sampler)")
    ap.add_argument("--temperature", type=float, default=0.15, help="DIAL softmax temperature")
    ap.add_argument("--n_plans", type=int, default=12, help="independent plans for the diversity probe")
    ap.add_argument("--n_beta", type=int, default=7, help="interpolation points for the smoothness probe")
    ap.add_argument("--n_reps", type=int, default=2, help="repeats per interpolation point (average out MC noise)")
    ap.add_argument("--settle_chunks", type=int, default=30, help="max chunks to drive toward the object for near regimes")
    ap.add_argument("--near_thresh", type=float, default=0.06, help="stop driving once TCP within this of the object (m)")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--exec_knot", type=int, default=8)
    ap.add_argument("--warm_steps", type=int, default=6, help="warm regime: run only the last N reverse steps (SDEdit)")
    ap.add_argument("--denoise_iters", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--score_scale", type=float, default=0.3)
    ap.add_argument("--joint_delta_clip", type=float, default=0.05)
    ap.add_argument("--basis", type=str, default="bspline")
    ap.add_argument("--knots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)


def run(args):
    import json

    import numpy as np
    import torch

    from vlm_mpc import sim_free_core as core
    from vlm_mpc.minimal_base_cost import MinimalBaseCost, usd_extents
    from vlm_mpc.droid_env import DroidEnv

    DEV = "cuda:0"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.manual_seed(args.seed)
    H, k = args.horizon, min(args.exec_knot, args.horizon)

    E = DroidEnv(device=DEV)
    scene_objects = list(getattr(E.env.scene, "rigid_objects", {}) or {})
    extents = usd_extents(E, scene_objects)
    root_pos = E.robot.data.body_pos_w[0, E.l0].detach()
    root_quat = E.robot.data.body_quat_w[0, E.l0].detach()

    policy, state_stats = core.build_policy(_REAL_STATS, 0.1)
    mpc, cfg = core.build_mpc(policy, num_samples=args.num_samples, iterations=args.iterations,
                              noise=args.noise, temperature=args.temperature,
                              joint_delta_clip=args.joint_delta_clip, interpolate=_INTERPOLATE)
    core.apply_horizon_basis(mpc, args.basis, args.knots)
    mpc.cost = core.guard_cost(MinimalBaseCost(extents))

    def state_ctx(ref=None):
        """Context at the current physical state; ``ref`` sets the warm-start consistency reference."""
        objs = {}
        for n in scene_objects:
            try:
                pos, _ = E.object_pose(n)
            except Exception:
                continue
            objs[n] = {"pos": torch.as_tensor(pos, device=DEV, dtype=torch.float32)}
        target = E.object_pose(args.grasp_obj)[0].copy()
        zs = [float(objs[n]["pos"][2]) - extents.get(n, (0.05, 0.05, 0.05))[2] for n in objs]
        return {"objects": objs, "joint_pos": E.q0(), "robot_root_pos": root_pos, "robot_root_quat": root_quat,
                "target": target, "grasp_obj": args.grasp_obj, "payload": None,
                "z_table": (min(zs) if zs else None), "plan_ref": ref}

    def noise():
        return torch.randn(1, H, 8, device=DEV, dtype=torch.float32)

    def slerp(a, b, t):
        """Spherical interpolation between two Gaussian noise tensors (keeps the norm on-prior)."""
        af, bf = a.flatten(), b.flatten()
        omega = torch.arccos(torch.clamp(((af / af.norm()) * (bf / bf.norm())).sum(), -1.0, 1.0))
        if float(omega) < 1e-4:
            return ((1 - t) * af + t * bf).view_as(a)
        so = torch.sin(omega)
        return ((torch.sin((1 - t) * omega) / so) * af + (torch.sin(t * omega) / so) * bf).view_as(a)

    def reverse(pin, ctx, update, x_init, it_start=0, log=None):
        """One reverse (denoise) trajectory from ``it_start``; optionally records the plan per step."""
        x = x_init
        for it in range(it_start, args.denoise_iters):
            if update == "ddim":
                x, _ = mpc.step_ddim(x, pin, ctx, iteration=it, num_iterations=args.denoise_iters,
                                     step_scale=args.score_scale)
            else:
                x, _ = mpc.step_score_space(x, pin, ctx, step_scale=args.score_scale)
            if log is not None:
                log.append(x.detach().clone())
        return x

    def decode_ee(pin, x0):
        real = core.decode_model_action_chunks(policy, pin, x0, current_joint_pos=E.q0(),
                                               max_joint_delta=args.joint_delta_clip).real_actions
        joints = real[..., :7]
        return joints.detach(), mpc.fk.forward(joints).ee_pos.detach()  # base frame; spread is frame-invariant

    def warm_init_fn(carry):
        """Build the SDEdit x_init generator for a regime; returns ``(fn(noise=None), it_start)``."""
        if carry is None:
            return (lambda z=None: z if z is not None else noise()), 0
        it_start = max(0, args.denoise_iters - args.warm_steps)
        ab, _ = core.ddim_iteration_alphas(iteration=it_start, num_iterations=args.denoise_iters,
                                           num_train_timesteps=cfg.ddim_num_train_timesteps)
        ab_t = torch.tensor(float(ab), device=DEV, dtype=torch.float32)
        return (lambda z=None: torch.sqrt(ab_t) * carry
                + torch.sqrt(1.0 - ab_t) * (z if z is not None else noise())), it_start

    def run_probes(update, x_init_fn, ctx, it_start):
        pin = core.policy_inputs(E, state_stats, _REAL_STATS)

        # P1 -- diversity of independent plans from the same state.
        x0 = torch.cat([reverse(pin, ctx, update, x_init_fn(), it_start=it_start) for _ in range(args.n_plans)], 0)
        joints, ee = decode_ee(pin, x0)
        term = ee[:, -1]
        p1 = {"p1_terminal_spread_m": float(torch.linalg.vector_norm(term - term.mean(0, keepdim=True), dim=-1).mean()),
              "p1_traj_spread_m": float(ee.std(0).norm(dim=-1).mean()),
              "p1_excursion_m": float(torch.linalg.vector_norm(ee[:, -1] - ee[:, 0], dim=-1).mean()),
              "p1_joint_spread_rad": float(joints.std(0).mean())}

        # P2 -- ESS + plan change across the reverse loop (weights captured from either sampler path).
        wlog, patched = [], {}
        for m in ("optimize", "optimize_with_noise_scale"):
            orig = getattr(mpc.sampler, m, None)
            if orig is None:
                continue
            patched[m] = orig

            def _wrap(*a, _orig=orig, **kw):
                res = _orig(*a, **kw)
                wlog.append(res.weights.detach())
                return res
            setattr(mpc.sampler, m, _wrap)
        xs = []
        reverse(pin, ctx, update, x_init_fn(), it_start=it_start, log=xs)
        for m, orig in patched.items():
            setattr(mpc.sampler, m, orig)
        p2 = {"p2_ess_frac": [float(1.0 / w.pow(2).sum()) / args.num_samples for w in wlog],
              "p2_plan_delta": [float((xs[i] - xs[i - 1]).norm() / (xs[i].norm() + 1e-9)) for i in range(1, len(xs))]}

        # P3 -- terminal EE traced as the initial noise is interpolated (behaviour-manifold navigability).
        za, zb = noise(), noise()
        path = []
        for b in np.linspace(0.0, 1.0, args.n_beta):
            reps = [decode_ee(pin, reverse(pin, ctx, update, x_init_fn(slerp(za, zb, float(b))), it_start=it_start))[1][0, -1]
                    for _ in range(args.n_reps)]
            path.append(torch.stack(reps).mean(0))
        path = torch.stack(path)
        span = float(torch.linalg.vector_norm(path[1:] - path[:-1], dim=-1).sum())
        chord = float(torch.linalg.vector_norm(path[-1] - path[0]))
        p3 = {"p3_span_m": span, "p3_chord_m": chord, "p3_smoothness": float(chord / (span + 1e-9))}
        return {"update": update, **p1, **p2, **p3}

    def settle():
        """Drive the arm toward the object (operating config) until close; return warm carry + reference."""
        carry, ref = None, None
        d = float(np.linalg.norm(E.tcp() - E.object_pose(args.grasp_obj)[0]))
        for _ in range(args.settle_chunks):
            pin = core.policy_inputs(E, state_stats, _REAL_STATS)
            x_init_fn, it_start = warm_init_fn(carry)
            ctx = state_ctx(ref=ref)
            x0 = reverse(pin, ctx, "score_space", x_init_fn(), it_start=it_start)
            carry = torch.cat([x0[:, k:], x0[:, -1:].expand(1, k, 8)], dim=1).detach()
            real = core.decode_model_action_chunks(policy, pin, x0, current_joint_pos=E.q0(),
                                                   max_joint_delta=args.joint_delta_clip).real_actions[0]
            rj = real[:, :7].detach()
            ref = torch.cat([rj[k:], rj[-1:].expand(k, 7)], dim=0)
            for t in range(k):
                E.apply_arm(real[t][:7], grip_open=True)
            d = float(np.linalg.norm(E.tcp() - E.object_pose(args.grasp_obj)[0]))
            if d < args.near_thresh:
                break
        return carry, ref, d

    regimes = [r.strip() for r in args.regimes.split(",")]
    updates = [u.strip() for u in args.updates.split(",")]
    results = {}

    def do_regime(tag, carry, ref, warm):
        x_init_fn, it_start = warm_init_fn(carry if warm else None)
        ctx = state_ctx(ref=ref if warm else None)
        d = float(np.linalg.norm(E.tcp() - ctx["target"]))
        for update in updates:
            key = f"{tag}/{update}"
            m = run_probes(update, x_init_fn, ctx, it_start)
            m["regime"], m["state_dist_m"] = tag, d
            results[key] = m
            ess = m["p2_ess_frac"]
            verdict = ("OPTIMIZER" if m["p1_terminal_spread_m"] < 0.005 else
                       "SAMPLER" if m["p1_terminal_spread_m"] > 0.02 else "PARTIAL")
            print(f"[probe] {key:26s} d={d:.3f}m | P1 term={m['p1_terminal_spread_m'] * 1e3:5.1f}mm "
                  f"traj={m['p1_traj_spread_m'] * 1e3:5.1f}mm | P2 ESS {ess[0] * 100:.0f}%->{ess[-1] * 100:.0f}% | "
                  f"P3 span={m['p3_span_m'] * 1e3:5.1f}mm smooth={m['p3_smoothness']:.2f} -> {verdict}", flush=True)

    if "far_fresh" in regimes:
        do_regime("far_fresh", carry=None, ref=None, warm=False)
    carry = ref = None
    if any(r.startswith("near") for r in regimes):
        carry, ref, reached = settle()
        print(f"[probe] driven to TCP->{args.grasp_obj}={reached:.3f}m (near regimes)", flush=True)
    if "near_fresh" in regimes:
        do_regime("near_fresh", carry=None, ref=None, warm=False)
    if "near_warm" in regimes:
        do_regime("near_warm", carry=carry, ref=ref, warm=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "probe")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"grasp_obj": args.grasp_obj, "scene": scene_objects, "temperature": args.temperature,
            "denoise_iters": args.denoise_iters, "num_samples": args.num_samples, "n_plans": args.n_plans,
            "results": results}
    with open(os.path.join(out_dir, f"{args.exp_name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    _figure(out_dir, args.exp_name, results)
    print(f"[probe] DONE -> {out_dir}/{args.exp_name}.json ({len(results)} configs)", flush=True)


def _figure(out_dir, name, results):
    """Three panels: P1 diversity bars, P2 ESS-vs-step curves, P3 latent-span bars (one entry per config)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[probe] figure skipped ({exc})", flush=True)
        return
    keys = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].bar(range(len(keys)), [results[k]["p1_terminal_spread_m"] * 1e3 for k in keys], color="#4477aa")
    ax[0].axhline(5, ls="--", c="r", lw=1); ax[0].axhline(20, ls="--", c="g", lw=1)
    ax[0].set_ylabel("terminal EE spread (mm)"); ax[0].set_title("P1 diversity  (<5 optimizer, >20 sampler)")
    for key in keys:
        e = results[key]["p2_ess_frac"]
        ax[1].plot(range(len(e)), [x * 100 for x in e], marker="o", ms=3, label=key)
    ax[1].set_xlabel("reverse step"); ax[1].set_ylabel("ESS (% of samples)")
    ax[1].set_title("P2 mode-commitment"); ax[1].legend(fontsize=6)
    ax[2].bar(range(len(keys)), [results[k]["p3_span_m"] * 1e3 for k in keys], color="#aa7744")
    ax[2].set_ylabel("interp behaviour span (mm)"); ax[2].set_title("P3 latent navigability  (flat = un-steerable)")
    for a in (ax[0], ax[2]):
        a.set_xticks(range(len(keys))); a.set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=110)
    print(f"[probe] figure -> {out_dir}/{name}.png", flush=True)
