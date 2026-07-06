"""ReKep-cost mug grasp on DIAL-MPC (rungs 2-3), unified behind --ground.

  gt  -- a hand-written ReKep-form relational cost (align TCP with the handle keypoint) + a privileged
         GT handle keypoint. Tests the cost->DIAL bridge and the relational cost's steerability.
  vlm -- the full VLM-DP rung: camera-proposed keypoints (tracked live) + a GPT-authored constraint
         (np->torch shim) drive DIAL. Grasp cost is the GPT stage-1 subgoal constraint.

Both: LiftEnv mug task -> HOVER -> DESCEND (relational cost) -> CLOSE -> fixed-target LIFT probe.

    python -m dial_mpc.main --task rekep_pick --ground vlm ...

Outputs -> results/vlm_mpc/rekep_pick.mp4 (gt) / rekep_vlm_pick_<exp_name>.mp4 (vlm).
"""
import json
import os

NAME = "rekep_pick"


def add_args(ap):
    ap.add_argument("--ground", type=str, default="vlm", choices=["gt", "vlm"])
    ap.add_argument("--task", type=str, default="Isaac-Lift-Mug-Franka-v0")
    ap.add_argument("--obj_z", type=float, default=0.12)
    ap.add_argument("--obj_yaw", type=float, default=180.0, help="object yaw (deg); 180 faces the handle toward the robot")
    # gt
    ap.add_argument("--handle", type=float, nargs=3, default=[0.04, 0.035, 0.0], help="gt: handle keypoint offset (object frame)")
    # vlm
    ap.add_argument("--exp_name", type=str, default="lift_mug_vlm_sideny")
    ap.add_argument("--prompt", type=str, default="Pick up the mug by the handle.")
    ap.add_argument("--cam_eye", type=float, nargs=3, default=[0.5, -0.95, 0.6])
    ap.add_argument("--cam_target", type=float, nargs=3, default=[0.55, 0.0, 0.08])
    ap.add_argument("--min_dist", type=float, default=0.025, help="keypoint_proposer.min_dist_bt_keypoints")
    ap.add_argument("--use_cached", action="store_true", help="reuse keypoints/constraints under results/vlm_mpc/rekep/<exp_name>")
    ap.add_argument("--grasp_cost", action="store_true", default=True,
                    help="vlm: DESCEND with grasp-yaw+straddle terms (--no-grasp_cost for plain-reach A/B)")
    ap.add_argument("--no-grasp_cost", dest="grasp_cost", action="store_false")
    # phases
    ap.add_argument("--steps_hover", type=int, default=80)
    ap.add_argument("--steps_descend", type=int, default=70)
    ap.add_argument("--close_steps", type=int, default=25)
    ap.add_argument("--steps_lift", type=int, default=60)
    ap.add_argument("--hover", type=float, default=0.10)
    ap.add_argument("--lift_h", type=float, default=0.12)
    ap.add_argument("--tol", type=float, default=0.03)
    # sampler
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--accel_std", type=float, default=6.0)
    ap.add_argument("--accel_clip", type=float, default=15.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--exec_knot", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)


def run(args):
    import numpy as np
    import torch
    import cv2
    from scipy.spatial.transform import Rotation as Rot

    from rekep.video import write_video_h264
    from sim_common.isaac_env import LiftEnv, GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_rekep_cost, make_rekep_grasp_cost, fixed_reach
    from sim_common import overlay

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DOWN = np.array([0.0, 0.0, -1.0])
    frames = []
    exec_knot = min(args.exec_knot, args.horizon - 1)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)

    def t(v):
        return torch.tensor(v, device=DEV, dtype=torch.float32)

    def record(label):
        frames.append(overlay.plain_frame(E.rgb(), label))

    def sampler(cost):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=args.horizon, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=args.accel_std,
                                  accel_clip=args.accel_clip, temperature=args.temperature, device=DEV)

    def run_phase(plan, ctx_fn, grip_open, target_fn, max_steps, label):
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        for step in range(max_steps):
            mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx_fn(), gen)
            E.apply_arm(q_traj[exec_knot], grip_open)
            record(f"{label} {step}: cost={score:.1f} ess={ess:.0f}")
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            err = float(np.linalg.norm(E.tcp() - target_fn()))
            if err < args.tol:
                print(f"[rekep-pick] {label}: reached (tcp_err={err*100:.2f}cm) at step {step}", flush=True)
                return
            if step % 20 == 0:
                print(f"[rekep-pick] {label} {step:3d}: tcp_err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}", flush=True)
        print(f"[rekep-pick] {label}: step cap (tcp_err={float(np.linalg.norm(E.tcp()-target_fn()))*100:.2f}cm)", flush=True)

    def hold(grip_open, n, label):
        q = E.q0()
        for _ in range(n):
            E.apply_arm(q, grip_open)
            record(label)

    R_down = Rot.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
    a_local = R_down.T @ DOWN

    if args.ground == "gt":
        E = LiftEnv(device=DEV, task=args.task, obj_z=args.obj_z, obj_yaw=args.obj_yaw)
        handle_np = E.keypoints_world([args.handle])[0]  # privileged GT handle keypoint
        print(f"[rekep-pick] gt obj={E.object_pose()[0].round(3).tolist()} handle_kp={handle_np.round(3).tolist()} "
              f"a_local={a_local.round(3).tolist()}", flush=True)

        def grasp_handle(tcp, keypoints):  # ReKep grasp constraint: align TCP with the handle keypoint
            return torch.linalg.norm(tcp - keypoints[0], dim=-1)

        plan = sampler(make_rekep_cost(E.fk, grasp_handle, a_local, grasp_offset=GRASP_OFFSET, device=DEV))
        hover_np = handle_np + np.array([0.0, 0.0, args.hover])
        run_phase(plan, lambda: t([hover_np]), True, lambda: hover_np, args.steps_hover, "HOVER")
        run_phase(plan, lambda: t([handle_np]), True, lambda: handle_np, args.steps_descend, "DESCEND")
        hold(False, args.close_steps, "CLOSE")
        grasp_err = float(np.linalg.norm(E.tcp() - handle_np))
        obj_before = E.object_pose()[0]
        lift_np = handle_np + np.array([0.0, 0.0, args.lift_h])
        run_phase(plan, lambda: t([lift_np]), False, lambda: lift_np, args.steps_lift, "LIFT")
        hold(False, 12, "LIFT-hold")
        obj_after = E.object_pose()[0]
        print(f"[rekep-pick] RESULT grasp_tcp_err={grasp_err*100:.2f}cm (reached_handle={grasp_err < 0.04}) | "
              f"lift: obj_z {obj_before[2]:.3f}->{obj_after[2]:.3f} "
              f"(+{(obj_after[2]-obj_before[2])*100:.1f}cm, held={obj_after[2] > obj_before[2]+0.05})", flush=True)
        out = os.path.join(_REPO, "results", "vlm_mpc", "rekep_pick.mp4")

    else:  # vlm
        from rekep import grounding
        from rekep.constraint_generation import ConstraintGenerator
        from rekep.keypoint_tracking import KeypointTracker
        from rekep.utils import get_callable_grasping_cost_fn, load_default_config
        from sim_common.np_shim import TorchNumpyShim, load_torch_constraints, make_torch_constraint

        out_dir = os.path.join(_REPO, "results", "vlm_mpc", "rekep", args.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        E = LiftEnv(device=DEV, task=args.task, obj_z=args.obj_z, obj_yaw=args.obj_yaw,
                    rekep_cam={"eye": args.cam_eye, "target": args.cam_target})
        config = load_default_config()
        config["keypoint_proposer"]["min_dist_bt_keypoints"] = args.min_dist

        kp_path = os.path.join(out_dir, "keypoints.npy")
        if args.use_cached and os.path.exists(kp_path) and os.path.exists(os.path.join(out_dir, "metadata.json")):
            keypoints = np.load(kp_path)
            print(f"[rekep-pick] cached: {len(keypoints)} keypoints", flush=True)
        else:
            grounded = grounding.propose_keypoints(E.rekep_cam, E.env, config)
            keypoints = grounded["keypoints"]
            cv2.imwrite(os.path.join(out_dir, "keypoints.png"), grounded["projected"][..., ::-1])
            np.save(kp_path, keypoints)
            if len(keypoints) == 0:
                raise SystemExit("[rekep-pick] no keypoints proposed -- adjust --obj_yaw / camera")
            ConstraintGenerator(config["constraint_generator"]).generate(
                grounded["projected"], args.prompt,
                metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)}, task_dir=out_dir)

        with open(os.path.join(out_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        num_stages = metadata["num_stages"]
        grasp_keypoints = metadata["grasp_keypoints"]
        grasp_stage = next((s for s in range(1, num_stages + 1) if grasp_keypoints[s - 1] != -1), 1)
        grasp_idx = grasp_keypoints[grasp_stage - 1]
        print(f"[rekep-pick] num_stages={num_stages} grasp_stage={grasp_stage} grasp_keypoint={grasp_idx}", flush=True)

        tracker = KeypointTracker(E.env, keypoints)
        shim = TorchNumpyShim(device=DEV)
        get_grasp_fn = get_callable_grasping_cost_fn([])
        callables = load_torch_constraints(os.path.join(out_dir, f"stage{grasp_stage}_subgoal_constraints.txt"),
                                           get_grasp_fn, shim)
        gpt_constraint = make_torch_constraint(callables)

        cost_gpt = make_rekep_cost(E.fk, gpt_constraint, a_local, grasp_offset=GRASP_OFFSET, device=DEV)
        cost_probe = make_rekep_cost(E.fk, fixed_reach, a_local, grasp_offset=GRASP_OFFSET, device=DEV)
        if args.grasp_cost:
            cost_descend = make_rekep_grasp_cost(E.fk, gpt_constraint, a_local, grasp_idx,
                                                 grasp_offset=GRASP_OFFSET, device=DEV)
            print("[rekep-pick] DESCEND cost = GPT reach + grasp-yaw + straddle", flush=True)
        else:
            cost_descend = cost_gpt
            print("[rekep-pick] DESCEND cost = GPT reach only", flush=True)
        plan_gpt, plan_descend, plan_probe = sampler(cost_gpt), sampler(cost_descend), sampler(cost_probe)

        def grasp_kp_world(z_off=0.0):
            kp = tracker.get_positions()[grasp_idx].copy()
            kp[2] += z_off
            return kp

        def ctx_from_tracker(z_off=0.0):
            kps = tracker.get_positions().copy()
            kps[grasp_idx, 2] += z_off
            return torch.tensor(kps, device=DEV, dtype=torch.float32)

        run_phase(plan_gpt, lambda: ctx_from_tracker(args.hover), True, lambda: grasp_kp_world(args.hover),
                  args.steps_hover, "HOVER")
        run_phase(plan_descend, lambda: ctx_from_tracker(0.0), True, lambda: grasp_kp_world(0.0),
                  args.steps_descend, "DESCEND")
        hold(False, args.close_steps, "CLOSE")
        grasp_err = float(np.linalg.norm(E.tcp() - grasp_kp_world(0.0)))
        obj_before = E.object_pose()[0]
        lift_target = grasp_kp_world(0.0) + np.array([0.0, 0.0, args.lift_h])
        lift_t = t([lift_target])
        run_phase(plan_probe, lambda: lift_t, False, lambda: lift_target, args.steps_lift, "LIFT")
        hold(False, 12, "LIFT-hold")
        obj_after = E.object_pose()[0]
        print(f"[rekep-pick] RESULT grasp_tcp_err={grasp_err*100:.2f}cm (reached={grasp_err < 0.04}) | "
              f"lift: obj_z {obj_before[2]:.3f}->{obj_after[2]:.3f} "
              f"(+{(obj_after[2]-obj_before[2])*100:.1f}cm, held={obj_after[2] > obj_before[2]+0.05})", flush=True)
        out = os.path.join(_REPO, "results", "vlm_mpc", f"rekep_vlm_pick_{args.exp_name}.mp4")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[rekep-pick] DONE -> {out} ({len(frames)} frames)", flush=True)
