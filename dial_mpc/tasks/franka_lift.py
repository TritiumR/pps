"""Franka Lift-Cube DIAL-MPC experiments (the early single-arm rungs), unified behind --mode.

Two modes on the shared `LiftEnv` (isaac_env.py):
  reach -- accel-MPPI + reach cost drives the EE to a hover point above the cube, closed-loop, no grasp.
           Validates sampler + cost + loop (the first controller rung).
  pick  -- full pick: HOVER -> DESCEND -> CLOSE (grasp cost) -> LIFT (carry cost, gripper closed), then a
           cube-z success check. Mirrors hydrax's predefined pick-place.

    python -m dial_mpc.main --task franka_lift --mode pick ...

Outputs -> results/vlm_mpc/<reach|pick>.mp4 (kept names, matches the franka_lift_rekep milestone dir).
"""
import os

NAME = "franka_lift"

# sampler defaults differ per mode (reach was tuned soft/greedy; pick uses the DIAL grasp settings)
_DEFAULTS = {
    "reach": dict(accel_std=2.5, accel_clip=7.0, temperature=5.0),
    "pick": dict(accel_std=6.0, accel_clip=15.0, temperature=0.2),
}


def add_args(ap):
    ap.add_argument("--mode", type=str, default="pick", choices=["reach", "pick"])
    # reach
    ap.add_argument("--steps", type=int, default=200, help="reach: loop length")
    ap.add_argument("--hover_reach", type=float, default=0.15, help="reach: hover height above the cube")
    ap.add_argument("--tol", type=float, default=None, help="tolerance (default 0.025 reach / 0.03 pick)")
    # pick phases
    ap.add_argument("--steps_hover", type=int, default=80)
    ap.add_argument("--steps_descend", type=int, default=70)
    ap.add_argument("--close_steps", type=int, default=25)
    ap.add_argument("--steps_lift", type=int, default=100)
    ap.add_argument("--hover", type=float, default=0.12)
    ap.add_argument("--lift_h", type=float, default=0.15)
    ap.add_argument("--lift_thresh", type=float, default=0.10)
    ap.add_argument("--r_cube", type=float, default=0.02)
    ap.add_argument("--beta_opt_iter", type=float, default=1.0, help="DIAL trajectory annealing (>=1e6 = plain MPPI)")
    ap.add_argument("--beta_horizon", type=float, default=1.0, help="DIAL action (horizon) annealing")
    # sampler (mode-conditional defaults resolved in run())
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--accel_std", type=float, default=None)
    ap.add_argument("--accel_clip", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--exec_knot", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)


def run(args):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as Rot

    from rekep.video import write_video_h264
    from sim_common.isaac_env import LiftCubeEnv, GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_reach_cost, make_grasp_cost, make_lift_cost
    from sim_common import overlay

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    frames = []
    mode = args.mode
    d = _DEFAULTS[mode]
    accel_std = args.accel_std if args.accel_std is not None else d["accel_std"]
    accel_clip = args.accel_clip if args.accel_clip is not None else d["accel_clip"]
    temperature = args.temperature if args.temperature is not None else d["temperature"]
    tol = args.tol if args.tol is not None else (0.025 if mode == "reach" else 0.03)

    E = LiftCubeEnv(device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)

    def sampler(cost):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=args.horizon, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=accel_std, accel_clip=accel_clip,
                                  temperature=temperature, beta_opt_iter=args.beta_opt_iter,
                                  beta_horizon=args.beta_horizon, device=DEV)

    def t(v):
        return torch.tensor(v, device=DEV, dtype=torch.float32)

    def record(label):
        frames.append(overlay.plain_frame(E.rgb(), label))

    if mode == "reach":
        target_np = E.cube_pos() + np.array([0.0, 0.0, args.hover_reach])
        target = t(target_np)
        print(f"[reach] dt={E.dt:.4f} target={target_np.round(3).tolist()} K={args.num_samples} "
              f"temp={temperature} astd={accel_std}", flush=True)
        plan = sampler(make_reach_cost(E.fk, z_floor=0.03))
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        reached, err = False, float("nan")
        for step in range(args.steps):
            mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, target, gen)
            E.apply_arm(q_traj[exec_knot], grip_open=True)
            err = float(np.linalg.norm(E.ee_pos() - target_np))
            record(f"reach MPC {step}: err={err*100:.1f}cm cost={score:.1f} ess={ess:.0f}")
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            if step % 20 == 0:
                print(f"[reach] step {step:3d}: err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}", flush=True)
            if err < tol:
                reached = True
                print(f"[reach] REACHED at step {step} (err={err*100:.2f}cm)", flush=True)
                break
        print(f"[reach] reached={reached} final_err={err*100:.2f}cm", flush=True)
        out = os.path.join(_REPO, "results", "vlm_mpc", "reach.mp4")

    else:  # pick (HOVER -> DESCEND -> CLOSE -> LIFT)
        cmd_log = []
        DOWN = np.array([0.0, 0.0, -1.0])
        R_down = Rot.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
        a_local = R_down.T @ DOWN
        print(f"[pick] dt={E.dt:.4f} cube={E.cube_pos().round(3).tolist()} a_local={a_local.round(3).tolist()} "
              f"K={args.num_samples} temp={temperature} astd={accel_std}", flush=True)
        plan_grasp = sampler(make_grasp_cost(E.fk, a_local, grasp_offset=GRASP_OFFSET, z_floor=0.03,
                                             r_cube=args.r_cube, device=DEV))

        def run_phase(plan, ctx_fn, grip_open, done, max_steps, label):
            mean_a = torch.zeros(args.horizon, 7, device=DEV)
            qd0 = torch.zeros(7, device=DEV)
            for step in range(max_steps):
                mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx_fn(), gen)
                q_cmd = q_traj[exec_knot]
                E.apply_arm(q_cmd, grip_open)
                cmd_log.append(q_cmd.detach().cpu().numpy())
                record(f"{label} {step}: cost={score:.1f} ess={ess:.0f}")
                mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
                if done():
                    print(f"[pick] {label}: sub-goal reached at step {step}", flush=True)
                    return True
                if step % 20 == 0:
                    print(f"[pick] {label} {step:3d}: tcp={E.tcp().round(3).tolist()} "
                          f"cube_z={E.cube_pos()[2]:.3f} cost={score:.2f} ess={ess:.0f}", flush=True)
            print(f"[pick] {label}: hit step cap {max_steps}", flush=True)
            return False

        def hold(grip_open, n, label):
            q = E.q0()
            for _ in range(n):
                E.apply_arm(q, grip_open)
                cmd_log.append(q.detach().cpu().numpy())
                record(label)

        hover_off = np.array([0.0, 0.0, args.hover])
        run_phase(plan_grasp, lambda: (t(E.cube_pos() + hover_off), t(E.cube_pos())), True,
                  lambda: float(np.linalg.norm(E.tcp() - (E.cube_pos() + hover_off))) < tol,
                  args.steps_hover, "HOVER")
        cube_frozen = E.cube_pos().copy()
        cfz = t(cube_frozen)
        run_phase(plan_grasp, lambda: (cfz, cfz), True,
                  lambda: float(np.linalg.norm(E.tcp() - cube_frozen)) < tol, args.steps_descend, "DESCEND")
        hold(False, args.close_steps, "CLOSE")

        cube_at_grasp = E.cube_pos().copy()
        carry_offset = cube_at_grasp - E.tcp()
        lift_target_np = cube_at_grasp + np.array([0.0, 0.0, args.lift_h])
        print(f"[pick] grasp landed: cube={cube_at_grasp.round(3).tolist()} "
              f"carry_off={carry_offset.round(3).tolist()} lift_target={lift_target_np.round(3).tolist()}", flush=True)
        plan_lift = sampler(make_lift_cost(E.fk, a_local, carry_offset, grasp_offset=GRASP_OFFSET,
                                           z_floor=0.03, device=DEV))
        lift_target = t(lift_target_np)
        z_thresh = cube_at_grasp[2] + args.lift_thresh
        run_phase(plan_lift, lambda: lift_target, False,
                  lambda: E.cube_pos()[2] > z_thresh, args.steps_lift, "LIFT")
        hold(False, 15, "LIFT-hold")

        cube = E.cube_pos()
        slip = float(np.linalg.norm(cube - E.tcp()))
        p = np.asarray(cmd_log)
        rough = float(np.mean(np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)) * 1e3) if len(p) >= 3 else 0.0
        print(f"[pick] RESULT lifted={bool(cube[2] > z_thresh)} cube_z={cube[2]:.3f} "
              f"(start {cube_at_grasp[2]:.3f}, +{(cube[2]-cube_at_grasp[2])*100:.1f}cm) "
              f"slip={slip*100:.2f}cm roughness={rough:.2f}", flush=True)
        out = os.path.join(_REPO, "results", "vlm_mpc", "pick.mp4")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[{mode}] DONE -> {out} ({len(frames)} frames)", flush=True)
