"""ReKep-faithful weight task, solved by our DIAL-MPC.

Mirrors `rekep/run_rekep_rollout.py`'s structure -- metadata (stages + grasp/release) + per-stage
relational keypoint constraints + KeypointTracker (movable = on the grasped body) -- but swaps ReKep's
scipy SubgoalSolver for our DIAL sampler (cost = the subgoal constraint via `make_rekep_cost`) and runs
on the Droid weight task. The VLM output is either stubbed (`weight_fake_vlm`) or generated live by
ReKep's `ConstraintGenerator` (`--vlm real`); both write the identical metadata + constraint files.

Per stage: build the DIAL cost from the loaded subgoal constraint, plan->execute->replan until the
subgoal constraint is SATISFIED on the tracked keypoints (the ReKep transition), then grasp/release per
the metadata flag.

    python -m vlm_mpc.main --task droid_weight [--vlm fake|real] ...
"""
import json
import os

NAME = "droid_weight"


def add_args(ap):
    ap.add_argument("--exp_name", type=str, default="weight_rekep_s0")
    ap.add_argument("--vlm", type=str, default="fake", choices=["fake", "real"],
                    help="fake: weight_fake_vlm writes constraints from GT masks; real: ReKep ConstraintGenerator (GPT-4o)")
    ap.add_argument("--instruction", type=str, default=None,
                    help="task instruction for the real VLM (default: task_prompts.json weight prompt)")
    ap.add_argument("--tol", type=float, default=0.015, help="subgoal-constraint value (m) below which a stage is done")
    ap.add_argument("--max_steps", type=int, default=120, help="max DIAL steps per stage")
    ap.add_argument("--close_steps", type=int, default=15, help="steps to hold the gripper toggle per grasp/release")
    ap.add_argument("--w_smooth", type=float, default=0.3, help="smoothness weight in the cost (damps jitter)")
    ap.add_argument("--w_local", type=float, default=0.5, help="trust-region weight (stay near current pose; damps cross-step jitter)")
    ap.add_argument("--w_clear", type=float, default=50.0, help="obstacle-clearance weight (avoid other objects -> up-and-over)")
    # Consistency cost reverted to 0 (the sub-goal solve fixes the root cause; the term is kept in
    # sampler.py and re-enables with --w_consist >0 if we want it back).
    ap.add_argument("--w_consist", type=float, default=0.0, help="consistency weight (plan close to previous); 0=off")
    ap.add_argument("--obstacle_r", type=float, default=0.04)
    ap.add_argument("--w_transit", type=float, default=60.0, help="transit-clearance weight on the carry/place (lift->transit-high->descend)")
    ap.add_argument("--descend_r", type=float, default=0.06, help="horizontal radius around the target where descent is allowed")
    ap.add_argument("--transit_margin", type=float, default=0.10, help="clearance above obstacle tops for the transit height")
    ap.add_argument("--obj_r", type=float, default=0.035)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    # Default = receding (the working mode: reaches targets, jittery). plan_once is smooth but a DENSE long
    # horizon under-optimizes (caps short); it needs a coarse-waypoint parameterization to be usable.
    ap.add_argument("--control_mode", type=str, default="receding", choices=["receding", "plan_once"],
                    help="receding: re-plan every step (per-step re-sampling); plan_once: plan a long trajectory + execute it")
    ap.add_argument("--horizon_once", type=int, default=30, help="horizon for plan_once mode (one plan covers the reach)")
    ap.add_argument("--accel_std", type=float, default=6.0)
    ap.add_argument("--accel_clip", type=float, default=15.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--exec_knot", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)


def run(args):
    import numpy as np
    import torch

    from rekep import grounding
    from rekep.keypoint_tracking import KeypointTracker
    from rekep.utils import get_callable_grasping_cost_fn, load_default_config
    from rekep.video import write_video_h264
    from vlm_mpc.droid_env import DroidEnv, ROBOTIQ_GRASP_OFFSET
    from vlm_mpc.sampler import make_accel_sampler
    from vlm_mpc.costs import make_grasp_cost, make_rekep_cost
    from vlm_mpc.np_shim import TorchNumpyShim, load_torch_constraints, make_torch_constraint
    from vlm_mpc import weight_fake_vlm, overlay, control

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # vlm_mpc/tasks -> repo root
    frames = []

    def resolve_subgoal(constraint_fn, tcp0, held_idx, held_off, keypoints, iters=200, lr=0.01):
        """Sub-goal solver: find the fixed TCP that satisfies the (relational) subgoal constraint.

        Gradient descent on the loaded constraint over the TCP, with held keypoints riding along
        (kp = TCP + offset). General (any differentiable constraint), no hardcoded target. Returns the
        resolved TCP [3] -- a fixed, stationary goal for the controller (ReKep's sub-goal solver).
        """
        kp0 = torch.tensor(np.asarray(keypoints), device=DEV, dtype=torch.float32)  # [N,3]
        off = {i: torch.tensor(o, device=DEV, dtype=torch.float32) for i, o in zip(held_idx, held_off)}
        tcp = torch.tensor(np.asarray(tcp0), device=DEV, dtype=torch.float32).clone().requires_grad_(True)
        opt = torch.optim.Adam([tcp], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            kp = torch.stack([tcp + off[i] if i in off else kp0[i] for i in range(kp0.shape[0])])
            constraint_fn(tcp.reshape(1, 1, 3), kp).sum().backward()
            opt.step()
        return tcp.detach().cpu().numpy()

    E = DroidEnv(device=DEV)
    out_dir = os.path.join(_REPO, "results", "vlm_mpc", "weight_rekep")
    # front-end artifacts (metadata + constraints); per-VLM dir so fake/real don't overwrite. mp4 stays flat.
    vlm_dir = os.path.join(out_dir, f"vlm_query_{args.vlm}")
    os.makedirs(vlm_dir, exist_ok=True)
    config = load_default_config()
    grounded = grounding.propose_keypoints(E.cam, E.env, config)
    keypoints = grounded["keypoints"]
    if len(keypoints) == 0:
        raise SystemExit("[weight-rekep] no keypoints proposed")
    print(f"[weight-rekep] {len(keypoints)} keypoints", flush=True)

    # ---- VLM front-end -> metadata.json + per-stage constraint files ----
    # fake: weight_fake_vlm writes them from GT masks. real: ReKep's ConstraintGenerator (GPT-4o)
    # writes the IDENTICAL artifacts from the keypoint-annotated image + instruction. The driver loads
    # them the same either way. The grasp center needs the object behind a keypoint: fake returns a role
    # table; real grounds the chosen keypoint to an object via GT masks (object_for_keypoint).
    if args.vlm == "fake":
        metadata, roles = weight_fake_vlm.generate(vlm_dir, keypoints, grounded, E.env)
        object_for_kp = lambda kp_idx: next((n for n, i in roles.items() if i == kp_idx), None)
    else:
        from rekep.constraint_generation import ConstraintGenerator
        if args.instruction:
            instruction = args.instruction
        else:
            with open(os.path.join(_REPO, "task_prompts.json"), "r", encoding="utf-8") as f:
                instruction = json.load(f)["weight"]["prompt"]
        print(f"[weight-rekep] real VLM (gpt-4o): instruction={instruction!r}", flush=True)
        ConstraintGenerator(config["constraint_generator"]).generate(
            grounded["projected"], instruction,
            metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)},
            task_dir=vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        object_for_kp = lambda kp_idx: weight_fake_vlm.object_for_keypoint(grounded, E.env, keypoints[kp_idx])
    print(f"[weight-rekep] metadata: num_stages={metadata['num_stages']} "
          f"grasp={metadata['grasp_keypoints']} release={metadata['release_keypoints']}", flush=True)

    tracker = KeypointTracker(E.env, keypoints)
    shim = TorchNumpyShim(DEV)
    z_floor = float(np.min(keypoints[:, 2])) - 0.05
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)

    # obstacle centers = the other scene objects' centroids (avoid sweeping them during reach/carry)
    fruit_c = {}
    for fname in ("pear", "apple", "mango", "cabbage"):
        try:
            c = weight_fake_vlm.local_centroid(grounded, E.env, fname)
        except Exception:
            c = None
        if c is not None:
            fruit_c[fname] = c
    # transit clearance height = above the tallest scene object (derived) + a physical margin
    z_clear = (max(c[2] for c in fruit_c.values()) + args.transit_margin) if fruit_c else (z_floor + 0.2)
    print(f"[weight-rekep] obstacle centers: {list(fruit_c)}; z_clear={z_clear:.3f}", flush=True)

    def sampler(cost, H):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=H, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=args.accel_std,
                                  accel_clip=args.accel_clip, temperature=args.temperature,
                                  w_consist=args.w_consist, device=DEV)

    def kps_t():
        return torch.tensor(tracker.get_positions(), device=DEV, dtype=torch.float32)

    def subgoal_value(constraint_fn):
        """Current subgoal-constraint value on the real tracked state (for the stage transition)."""
        tcp = torch.tensor(E.tcp(), device=DEV, dtype=torch.float32).reshape(1, 1, 3)
        return float(constraint_fn(tcp, kps_t())[0, 0])

    def record_frame(stage, label, cur, grip_closed):
        frames.append(overlay.camera_overlay_frame(E.cam, tracker, [
            f"DIAL+ReKep weight (stage {stage}/{num_stages}, {args.control_mode})",
            f"{label}: subgoal={cur*100:.1f}cm grip={'C' if grip_closed else 'O'}"]))

    def run_stage(plan, ctx_fn, transition, stage, label, grip_closed, H):
        """Run one stage's motion. Two modes (kept side-by-side, switch with --control_mode):
          receding  -- re-plan every control step, execute one knot (per-step re-sampling -> jitter).
          plan_once -- plan a long trajectory once and execute it knot-by-knot, re-planning only when
                       it's exhausted (no per-step re-sampling -> smoother). Closed-loop returns once
                       v_vlm=(target-x_t) is wired for steering; here we execute the plan directly.
        """
        qd0 = torch.zeros(7, device=DEV)
        if args.control_mode == "receding":
            mean_a = torch.zeros(H, 7, device=DEV)
            for step in range(args.max_steps):
                mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx_fn(), gen)
                E.apply_arm(q_traj[min(exec_knot, H - 1)], grip_open=not grip_closed)
                q_hist.append(E.q0().detach().cpu().numpy())
                cur = transition()
                record_frame(stage, label, cur, grip_closed)
                mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
                if cur < args.tol:
                    print(f"[weight-rekep]   subgoal satisfied ({cur*100:.2f}cm) at step {step}", flush=True)
                    return
        else:  # plan_once
            total = 0
            while total < args.max_steps:
                mean_a, q_traj, score, ess = plan(torch.zeros(H, 7, device=DEV), E.q0(), qd0, ctx_fn(), gen)
                for knot in range(q_traj.shape[0]):
                    E.apply_arm(q_traj[knot], grip_open=not grip_closed)
                    q_hist.append(E.q0().detach().cpu().numpy())
                    cur = transition()
                    total += 1
                    record_frame(stage, label, cur, grip_closed)
                    if cur < args.tol:
                        print(f"[weight-rekep]   subgoal satisfied ({cur*100:.2f}cm) at step {total}", flush=True)
                        return
                    if total >= args.max_steps:
                        break
        print(f"[weight-rekep]   stage cap (subgoal={transition()*100:.2f}cm)", flush=True)

    num_stages = metadata["num_stages"]
    grasped_body, grip_closed = None, False
    obj_z0 = {n: E.object_pose(n)[0][2] for n in ("pear", "apple")}
    q_hist = []  # executed joint configs during the DIAL motion (for the jitter metric)

    for stage in range(1, num_stages + 1):
        grasp_kp = metadata["grasp_keypoints"][stage - 1]
        release_kp = metadata["release_keypoints"][stage - 1]
        is_grasp, is_release = grasp_kp != -1, release_kp != -1
        held = [i for i, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body]
        get_grasp_fn = get_callable_grasping_cost_fn(held)
        sub = load_torch_constraints(os.path.join(vlm_dir, f"stage{stage}_subgoal_constraints.txt"),
                                     get_grasp_fn, shim)
        constraint_fn = make_torch_constraint(sub)

        if is_grasp:
            # ReKep keypoint SELECTS the object; the grasp targets its perception-derived local center
            # (not the surface keypoint) -- the grasp-module half ReKep keeps and we'd skipped.
            name = object_for_kp(grasp_kp)
            center = weight_fake_vlm.local_centroid(grounded, E.env, name, near=keypoints[grasp_kp])
            center_t = torch.tensor(center, device=DEV, dtype=torch.float32)
            obstacles = [c for f, c in fruit_c.items() if f != name]  # avoid the other objects
            # No transit term for the grasp: it must descend straight onto the object (top-down is
            # handled by the orient term); transit-clearance is a carry concern and would block descent.
            cost = make_grasp_cost(E.fk, E.a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET, r_cube=args.obj_r,
                                   open_half=0.0425, finger_r=0.012, w_local=args.w_local,
                                   w_smooth=args.w_smooth, w_clear=args.w_clear, obstacles=obstacles,
                                   obstacle_r=args.obstacle_r, z_floor=z_floor, device=DEV)
            ctx_fn = lambda: (center_t, center_t)
            transition = lambda: float(np.linalg.norm(E.tcp() - center))
            label = f"grasp {name} (kp{grasp_kp}->center)"
        else:
            rel_name = object_for_kp(release_kp)
            obstacles = [c for f, c in fruit_c.items() if f != rel_name]  # avoid others while carrying
            held_off = [(tracker.get_positions()[i] - E.tcp()).tolist() for i in held]  # captured at grasp
            # SUB-GOAL SOLVE: resolve the relational constraint into a FIXED TCP target, then reach that
            # (a stationary, well-posed problem -> smooth) instead of chasing the live moving constraint.
            # The transition still checks the REAL relational constraint on the tracked state.
            target = resolve_subgoal(constraint_fn, E.tcp(), held, held_off, tracker.get_positions())
            target_t = torch.tensor(target, device=DEV, dtype=torch.float32)
            print(f"[weight-rekep]   resolved place target={np.round(target,3).tolist()}", flush=True)
            fixed_reach = lambda tcp, kp: torch.linalg.norm(tcp - target_t, dim=-1)  # reach the fixed point
            cost = make_rekep_cost(E.fk, fixed_reach, E.a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET,
                                   w_local=args.w_local, w_smooth=args.w_smooth, w_clear=args.w_clear,
                                   obstacles=obstacles, obstacle_r=args.obstacle_r, transit_target=target,
                                   z_clear=z_clear, descend_r=args.descend_r, w_transit=args.w_transit,
                                   z_floor=z_floor, device=DEV)
            ctx_fn = lambda: kps_t()
            transition = lambda: subgoal_value(constraint_fn)
            label = f"place (release kp{release_kp})"
        H_mode = args.horizon if args.control_mode == "receding" else args.horizon_once
        plan = sampler(cost, H_mode)
        print(f"[weight-rekep] stage {stage}/{num_stages}: {label}; held={held} mode={args.control_mode}", flush=True)
        run_stage(plan, ctx_fn, transition, stage, label, grip_closed, H_mode)

        if is_grasp:
            grip_closed, grasped_body = True, tracker.owners[grasp_kp]
            control.hold_pose(E, args.close_steps, False,
                              lambda: frames.append(overlay.camera_overlay_frame(
                                  E.cam, tracker, [f"stage {stage}: CLOSE", ""])))
        if is_release:
            grip_closed, grasped_body = False, None
            control.hold_pose(E, args.close_steps, True,
                              lambda: frames.append(overlay.camera_overlay_frame(
                                  E.cam, tracker, [f"stage {stage}: RELEASE", ""])))

    print(f"[weight-rekep] --- RESULT ---", flush=True)
    q = np.array(q_hist)
    dq = np.diff(q, axis=0)
    nrm = np.linalg.norm(dq, axis=1)
    cos = (dq[1:] * dq[:-1]).sum(axis=1) / (nrm[1:] * nrm[:-1] + 1e-9)
    print(f"[weight-rekep] exec smoothness (mean cos of consecutive joint steps, 1=smooth)={cos.mean():.3f} "
          f"std={cos.std():.3f} over {len(q)} steps", flush=True)
    for name in ("pear", "apple"):
        p = E.object_pose(name)[0]
        print(f"[weight-rekep] {name}: final={np.round(p,3).tolist()} dz={(p[2]-obj_z0[name])*100:+.1f}cm",
              flush=True)
    out = os.path.join(out_dir, f"{args.exp_name}.mp4")
    os.makedirs(out_dir, exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[weight-rekep] DONE -> {out} ({len(frames)} frames)", flush=True)
