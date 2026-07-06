"""Run the collaborator's SimFreeMPC in our IsaacLab weight task with a checkpoint-free decode.

The planner runs unchanged from ``sim_free_mpc`` (DIAL sampler, reverse update, knot interpolation,
PriorityStateCost); the model-space-to-action decode that normally needs the pi0.5 checkpoint is served
by a mock policy carrying only the decode norm-stats. This is the faithful reproduction harness, used to
demonstrate the vetted position-space recipe on the real pipeline: B-spline knot smoothing, a consistency
term, warm-start, a joint-delta rate limit, and a NaN guard. Phase flags come from the env's own
``subtask_terms`` group.

    /isaac-sim/python.sh -m vlm_mpc.main --task sim_free_mbd --real_stats --interpolate --guard --w_consist 30
"""
import os

NAME = "sim_free_mbd"


def add_args(ap):
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--mode", type=str, default="denoise", choices=["denoise", "mean"])
    ap.add_argument("--update", type=str, default="score_space", choices=["mbd_score", "score_space", "ddim", "flow"])
    ap.add_argument("--init", type=str, default="warm", choices=["noise", "warm"])
    ap.add_argument("--warm_steps", type=int, default=6, help="denoise+warm: run only the last N reverse steps")
    ap.add_argument("--score_scale", type=float, default=0.3)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--exec_knot", type=int, default=8, help="chunk steps executed before re-planning")
    ap.add_argument("--denoise_iters", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--joint_delta_clip", type=float, default=0.05, help="post-decode per-step joint-motion cap")
    ap.add_argument("--real_stats", action="store_true", help="faithful pi05_droid_jointpos quantile decode")
    ap.add_argument("--action_std", type=float, default=0.1, help="stand-in action-norm std when --real_stats is off")
    ap.add_argument("--interpolate", action="store_true", help="coarse-knot horizon interpolation")
    ap.add_argument("--basis", type=str, default="bspline", choices=["linear", "cubic", "bspline", "rbf"],
                    help="knot-interpolation basis (bspline: approximating C2, no overshoot)")
    ap.add_argument("--knots", type=int, default=4)
    ap.add_argument("--w_consist", type=float, default=0.0, help="consistency weight toward the previous plan; 0 disables")
    ap.add_argument("--latch", action="store_true", help="monotonic phase flags (a stationary target per phase)")
    ap.add_argument("--guard", action="store_true", help="NaN-guard the cost before the sampler softmax")
    ap.add_argument("--max_chunks", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)


def run(args):
    import numpy as np
    import torch

    from vlm_mpc import sim_free_core as core
    from vlm_mpc import minimal_base_cost
    from vlm_mpc.droid_env import DroidEnv
    from vlm_mpc import overlay
    from rekep.video import write_video_h264

    DEV = "cuda:0"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.manual_seed(args.seed)
    exp_name = args.exp_name or f"{args.mode}_{args.update}_{args.init}_ek{args.exec_knot}"

    plan_ref = {"v": None}  # warm-started previous plan, used as the consistency reference

    class _ConsistencyCost:
        """Base cost plus a consistency term biasing the plan toward the previous one."""

        def __init__(self, base, weight):
            self._base = base
            self._weight = weight
            if hasattr(base, "weights"):
                self.weights = base.weights

        def target(self, *args_, **kwargs_):
            return self._base.target(*args_, **kwargs_)

        def __call__(self, *, real_actions, ee_pos, ee_quat=None, context):
            base = self._base(real_actions=real_actions, ee_pos=ee_pos, ee_quat=ee_quat, context=context)
            return base + minimal_base_cost._consistency_cost(real_actions, context, self._weight)

    policy, state_stats = core.build_policy(args.real_stats, args.action_std)
    mpc, cfg = core.build_mpc(policy, num_samples=args.num_samples, iterations=args.iterations,
                              noise=args.noise, temperature=args.temperature,
                              joint_delta_clip=args.joint_delta_clip, interpolate=args.interpolate)
    core.apply_horizon_basis(mpc, args.basis, args.knots)
    if args.w_consist > 0:
        mpc.cost = _ConsistencyCost(mpc.cost, args.w_consist)
    if args.guard:
        mpc.cost = core.guard_cost(mpc.cost)

    E = DroidEnv(device=DEV)
    scene_objects = list(getattr(E.env.scene, "rigid_objects", {}) or {})
    root_pos = E.robot.data.body_pos_w[0, E.l0].detach()
    root_quat = E.robot.data.body_quat_w[0, E.l0].detach()
    print(f"[sim-free-mbd] mode={args.mode} update={args.update} init={args.init} basis={args.basis} "
          f"w_consist={args.w_consist} scene={scene_objects}", flush=True)

    latched = {}

    def read_flags():
        try:
            group = E.env.observation_manager.compute_group("subtask_terms")
            raw = {key: bool(val.detach().flatten()[0].item()) for key, val in group.items()}
        except Exception:
            raw = {}
        if not args.latch:
            return raw
        for key, val in raw.items():  # once True, stays True, so the target does not flip back
            latched[key] = latched.get(key, False) or val
        return dict(latched)

    def build_context():
        objects = {}
        for name in scene_objects:
            try:
                pos, _ = E.object_pose(name)
            except Exception:
                continue
            objects[name] = {"pos": torch.as_tensor(pos, device=DEV, dtype=torch.float32)}
        ctx = {"objects": objects, "joint_pos": E.q0(), "robot_root_pos": root_pos,
               "robot_root_quat": root_quat, "plan_ref": plan_ref["v"]}
        ctx.update(read_flags())
        return ctx

    def on_scale(name):
        try:
            obj, _ = E.object_pose(name)
            scale, _ = E.object_pose("scale")
        except Exception:
            return False
        return float(np.linalg.norm(obj[:2] - scale[:2])) < 0.10

    H = args.horizon
    k = min(args.exec_knot, H)
    frames, q_hist, dist_hist = [], [], []
    obj_p0 = {n: E.object_pose(n)[0].copy() for n in scene_objects}
    x_carry = None
    for chunk in range(args.max_chunks):
        ctx = build_context()
        pin = core.policy_inputs(E, state_stats, args.real_stats)
        target = mpc.cost.target(ctx, DEV, torch.float32).detach().cpu().numpy()
        it_start = 0
        if args.init == "warm" and x_carry is not None:
            if args.mode == "denoise":  # SDEdit-style: forward-diffuse the carry, run only the last warm_steps
                it_start = max(0, args.denoise_iters - args.warm_steps)
                alpha_bar, _ = core.ddim_iteration_alphas(iteration=it_start, num_iterations=args.denoise_iters,
                                                          num_train_timesteps=cfg.ddim_num_train_timesteps)
                a = torch.tensor(float(alpha_bar), device=DEV, dtype=torch.float32)
                x_init = torch.sqrt(a) * x_carry + torch.sqrt(1.0 - a) * torch.randn(1, H, 8, device=DEV)
            else:
                x_init = x_carry
        else:
            x_init = torch.randn(1, H, 8, device=DEV, dtype=torch.float32)
        x0 = core.plan_chunk(mpc, x_init, pin, ctx, mode=args.mode, update=args.update,
                             denoise_iters=args.denoise_iters, score_scale=args.score_scale,
                             dt=E.dt, it_start=it_start)
        x_carry = torch.cat([x0[:, k:], x0[:, -1:].expand(1, k, 8)], dim=1).detach()
        real = core.decode_model_action_chunks(policy, pin, x0, current_joint_pos=E.q0(),
                                               max_joint_delta=args.joint_delta_clip).real_actions[0]
        joints = real[:, :7].detach()
        plan_ref["v"] = torch.cat([joints[k:], joints[-1:].expand(k, 7)], dim=0)
        flags = {}
        dist = 0.0
        for t in range(min(args.exec_knot, H)):
            action = real[t]
            E.apply_arm(action[:7], grip_open=float(action[7]) < 0.5)  # gripper from the decoded channel
            q_hist.append(E.q0().detach().cpu().numpy())
            dist = float(np.linalg.norm(E.tcp() - target))
            dist_hist.append(dist)
            flags = read_flags()
            frames.append(overlay.plain_frame(
                E.rgb(),
                f"SimFreeMPC[{args.mode}/{args.update}/{args.init}] ch{chunk}.{t} d={dist:.2f}m "
                f"pear={int(flags.get('grasp_pear', 0))} onScale={int(flags.get('pear_on_scale', 0))} "
                f"apple={int(flags.get('grasp_apple', 0))}"))
        if chunk % 10 == 0:
            print(f"[sim-free-mbd]   chunk {chunk}: dist_to_target={dist:.3f}m flags={flags}", flush=True)
        if on_scale("pear") and on_scale("apple"):
            print(f"[sim-free-mbd]   both objects on scale at chunk {chunk}", flush=True)
            break

    print("[sim-free-mbd] --- RESULT ---", flush=True)
    q = np.array(q_hist)
    if len(q) > 2:
        dq = np.diff(q, axis=0)
        step = np.linalg.norm(dq, axis=1)
        cos = (dq[1:] * dq[:-1]).sum(axis=1) / (step[1:] * step[:-1] + 1e-9)
        jerk = np.linalg.norm(q[2:] - 2 * q[1:-1] + q[:-2], axis=1).mean()
        tv = np.abs(dq).sum(axis=0)
        print(f"[sim-free-mbd] smoothness(cos)={cos.mean():.3f} jerk(2nd-diff)={jerk * 1e3:.2f}e-3 "
              f"speed={step.mean() * 1e3:.1f}e-3 over {len(q)} steps", flush=True)
        print("[sim-free-mbd] per-joint TV(rad): " + " ".join(f"j{i}={t:.2f}" for i, t in enumerate(tv)), flush=True)
    d = np.array(dist_hist)
    if len(d):
        print(f"[sim-free-mbd] reach(TCP->target): min={d.min():.3f}m final={d[-1]:.3f}m", flush=True)
    obstacles = [n for n in ("board", "mango", "cabbage") if n in scene_objects]
    moved = [float(np.linalg.norm(E.object_pose(n)[0] - obj_p0[n])) for n in obstacles]
    if moved:
        print(f"[sim-free-mbd] scene_disturbance({obstacles}): sum={sum(moved) * 100:.1f}cm "
              f"max={max(moved) * 100:.1f}cm", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "sim_free_mbd")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[sim-free-mbd] DONE -> {out} ({len(frames)} frames)", flush=True)
