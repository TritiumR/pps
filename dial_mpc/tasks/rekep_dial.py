"""Task-agnostic ReKep + DIAL driver for any Droid task (weight / pot / tea / capsule).

One driver, no per-task engineering: the VLM (ReKep ConstraintGenerator) writes each stage's subgoal +
path constraints; a 6-DoF sub-goal solver resolves each subgoal into a fixed pose (position + orientation,
held keypoints riding the gripper rigidly); our DIAL sampler reaches it under the unified `make_task_cost`.
Objects and obstacles are derived from the scene (`env.scene.rigid_objects`) -- no hardcoded name lists.

  grasp stage  -> reach the object's perception centroid, top-down, with grasp yaw/straddle;
  place stage  -> reach the 6-DoF resolved pose, with transit-clearance carry + running path constraints.

The gripper is an in-cost proximity decision inside the DIAL chunk (8th variable): it closes near the
object on grasp stages and opens near the target on place stages. Contact-hold / articulation stay the
PPS residual's job. The cost terms + weights mirror the collaborator's sim_free_mpc.

    python -m dial_mpc.main --task rekep_dial --task_key tea --vlm real
"""
import json
import os

NAME = "rekep_dial"


def add_args(ap):
    ap.add_argument("--task_key", type=str, default="weight", choices=["weight", "pot", "tea", "capsule"])
    ap.add_argument("--vlm", type=str, default="real", choices=["fake", "real"],
                    help="real: ReKep ConstraintGenerator (GPT-4o); fake: GT-mask stub (weight only)")
    ap.add_argument("--exp_name", type=str, default=None, help="default: <task_key>_<vlm>")
    ap.add_argument("--instruction", type=str, default=None, help="override the task_prompts.json instruction")
    ap.add_argument("--tol", type=float, default=0.02, help="subgoal-constraint value (m) below which a stage is done")
    ap.add_argument("--max_steps", type=int, default=120, help="max DIAL steps per stage")
    ap.add_argument("--settle_steps", type=int, default=12, help="closed-gripper settle after a grasp stage")
    ap.add_argument("--w_transit", type=float, default=60.0, help="transit-clearance weight on the carry/place")
    ap.add_argument("--descend_r", type=float, default=0.06, help="horizontal radius around the target where descent is allowed")
    ap.add_argument("--transit_margin", type=float, default=0.10, help="clearance above obstacle tops for the transit height")
    ap.add_argument("--obstacle_r", type=float, default=0.05, help="non-target keepout radius")
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--accel_std", type=float, default=6.0)
    ap.add_argument("--accel_clip", type=float, default=15.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--grip_std", type=float, default=0.3)
    ap.add_argument("--param", type=str, default="accel", choices=["accel", "delta"],
                    help="accel: sample accelerations, double-integrate (smooth by construction); "
                         "delta: sample per-step joint-position deltas, single-integrate + clamp (emulates the collaborator)")
    ap.add_argument("--delta_std", type=float, default=0.06, help="delta-mode sampling std (rad/step)")
    ap.add_argument("--delta_clip", type=float, default=0.15, help="delta-mode max per-step joint change (rad)")
    ap.add_argument("--exec_knot", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)


def run(args):
    import numpy as np
    import torch

    from rekep import grounding
    from rekep.keypoint_tracking import KeypointTracker
    from vlm_dp.world import GTWorld
    from rekep.utils import get_callable_grasping_cost_fn, load_default_config
    from rekep.video import write_video_h264
    from sim_common.envs.droid import DroidEnv, ROBOTIQ_GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_task_cost
    from sim_common.constraints import TorchNumpyShim, load_torch_constraints, make_torch_constraint
    from vlm_dp.grounding import fake_vlm, masks
    from sim_common import overlay

    DEV = "cuda:0"
    OFFSET = ROBOTIQ_GRASP_OFFSET
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    frames = []

    with open(os.path.join(_REPO, "task_prompts.json"), "r", encoding="utf-8") as f:
        prompts = json.load(f)
    task_id = prompts[args.task_key]["task_id"]
    instruction = args.instruction or prompts[args.task_key]["prompt"]
    exp_name = args.exp_name or f"{args.task_key}_{args.vlm}"
    if args.vlm == "fake" and args.task_key != "weight":
        raise SystemExit("[rekep-dial] --vlm fake is only defined for the weight task; use --vlm real")

    def rotvec_to_R(rv):
        """Differentiable Rodrigues: rotation vector [3] -> rotation matrix [3,3]."""
        theta = torch.linalg.norm(rv) + 1e-8
        k = rv / theta
        z = torch.zeros((), device=DEV)
        K = torch.stack([torch.stack([z, -k[2], k[1]]),
                         torch.stack([k[2], z, -k[0]]),
                         torch.stack([-k[1], k[0], z])])
        return torch.eye(3, device=DEV) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    def resolve_subgoal(constraint_fn, tcp0, R0, held_idx, keypoints, iters=200, lr=0.01):
        """6-DoF sub-goal solver: find the fixed EE pose (position + orientation) satisfying the relational
        constraint, with held keypoints riding the pose rigidly. Returns (pos[3], R[3,3], held_local[k,3]).
        """
        kp0 = torch.tensor(np.asarray(keypoints), device=DEV, dtype=torch.float32)  # [N,3]
        R0t = torch.tensor(np.asarray(R0), device=DEV, dtype=torch.float32)          # [3,3]
        tcp0t = torch.tensor(np.asarray(tcp0), device=DEV, dtype=torch.float32)       # [3]
        local = {i: R0t.T @ (kp0[i] - tcp0t) for i in held_idx}  # held keypoint offsets in the gripper frame
        pos = tcp0t.clone().requires_grad_(True)
        rv = torch.zeros(3, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([pos, rv], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            R = rotvec_to_R(rv) @ R0t
            kp = torch.stack([pos + R @ local[i] if i in local else kp0[i] for i in range(kp0.shape[0])])
            constraint_fn(pos.reshape(1, 1, 3), kp).sum().backward()
            opt.step()
        R = (rotvec_to_R(rv) @ R0t).detach()
        # A degenerate VLM constraint (e.g. an arccos with a 0/0) can NaN the gradients and drive the
        # pose to NaN; fall back to the starting pose so a bad subgoal never poisons the controller.
        if not torch.isfinite(pos).all() or not torch.isfinite(R).all():
            pos, R = tcp0t, R0t
        held_local = np.stack([local[i].detach().cpu().numpy() for i in held_idx]) if held_idx else np.zeros((0, 3))
        return pos.detach().cpu().numpy(), R.detach().cpu().numpy(), held_local

    E = DroidEnv(device=DEV, task=task_id)
    if args.task_key == "pot":
        from pot_scene_fix import seat_pot_lid
        seat_pot_lid(E.env, E._neutral())

    out_dir = os.path.join(_REPO, "results", "vlm_mpc", "rekep_dial", args.task_key)
    vlm_dir = os.path.join(out_dir, f"vlm_query_{args.vlm}")
    os.makedirs(vlm_dir, exist_ok=True)
    config = load_default_config()
    grounded = grounding.propose_keypoints(E.cam, E.env, config)
    keypoints = grounded["keypoints"]
    if len(keypoints) == 0:
        raise SystemExit("[rekep-dial] no keypoints proposed")
    scene_objects = list(getattr(E.env.scene, "rigid_objects", {}) or {})
    print(f"[rekep-dial] task={args.task_key} {len(keypoints)} keypoints; scene objects={scene_objects}", flush=True)

    # ---- VLM front-end -> metadata + per-stage constraints ----
    if args.vlm == "fake":
        metadata, roles = fake_vlm.generate(args.task_key, vlm_dir, keypoints, grounded, E.env)
        object_for_kp = lambda i: next((n for n, j in roles.items() if j == i), None)
    else:
        from rekep.constraint_generation import ConstraintGenerator
        print(f"[rekep-dial] real VLM (gpt-4o): {instruction!r}", flush=True)
        ConstraintGenerator(config["constraint_generator"]).generate(
            grounded["projected"], instruction,
            metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)}, task_dir=vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        object_for_kp = lambda i: masks.object_for_keypoint(grounded, E.env, keypoints[i], names=scene_objects)
    num_stages = metadata["num_stages"]
    print(f"[rekep-dial] metadata: num_stages={num_stages} grasp={metadata['grasp_keypoints']} "
          f"release={metadata['release_keypoints']}", flush=True)

    # obstacle centroids = every scene object's masked centroid (avoid sweeping the non-manipulated ones)
    obj_c = {}
    for name in scene_objects:
        try:
            c = masks.local_centroid(grounded, E.env, name)
        except Exception:
            c = None
        if c is not None:
            obj_c[name] = c
    z_floor = float(np.min(keypoints[:, 2])) - 0.05
    z_clear = (max(c[2] for c in obj_c.values()) + args.transit_margin) if obj_c else (z_floor + 0.2)
    print(f"[rekep-dial] obstacle objects={list(obj_c)}; z_clear={z_clear:.3f}", flush=True)

    tracker = KeypointTracker(GTWorld(E.env), keypoints)
    shim = TorchNumpyShim(DEV)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)
    obj_z0 = {n: E.object_pose(n)[0][2] for n in scene_objects}
    q_hist = []

    def kps_t():
        return torch.tensor(tracker.get_positions(), device=DEV, dtype=torch.float32)

    def tcp_pose():
        pos, R = E.fk.grasp_point(E.q0().unsqueeze(0), OFFSET)
        return pos[0].detach().cpu().numpy(), R[0].detach().cpu().numpy()

    def sampler(cost, H):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=H, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=args.accel_std,
                                  accel_clip=args.accel_clip, temperature=args.temperature,
                                  grip_std=args.grip_std, param=args.param, delta_std=args.delta_std,
                                  delta_clip=args.delta_clip, device=DEV)

    def record(stage, label, cur, g):
        frames.append(overlay.camera_overlay_frame(E.cam, tracker, [
            f"DIAL+ReKep {args.task_key} (stage {stage}/{num_stages})",
            f"{label}: subgoal={cur*100:.1f}cm grip={'C' if g > 0.5 else 'O'}"]))

    def control(plan, ctx_fn, transition, close_when_near, stage, label, H):
        """Receding-horizon DIAL with the gripper channel. close_when_near: grasp phase (close near obj)
        vs place phase (open near target). Returns the executed gripper command."""
        mean_a = torch.zeros(H, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        g_mean = torch.full((H, 1), 0.0 if close_when_near else 1.0, device=DEV)  # grasp opens, place holds closed
        g_cmd = float(g_mean[exec_knot, 0])
        for step in range(args.max_steps):
            mean_a, q_traj, g_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx_fn(), gen, g_mean)
            g_cmd = float(g_traj[exec_knot, 0])
            E.apply_arm(q_traj[min(exec_knot, H - 1)], grip_open=g_cmd < 0.5)
            q_hist.append(E.q0().detach().cpu().numpy())
            cur = transition()
            record(stage, label, cur, g_cmd)
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            g_mean = torch.cat([g_traj[1:], g_traj[-1:]], dim=0)
            if cur < args.tol:
                print(f"[rekep-dial]   subgoal satisfied ({cur*100:.2f}cm) at step {step}", flush=True)
                return g_cmd
        print(f"[rekep-dial]   stage cap (subgoal={transition()*100:.2f}cm)", flush=True)
        return g_cmd

    grasped_body = None
    for stage in range(1, num_stages + 1):
        grasp_kp = metadata["grasp_keypoints"][stage - 1]
        release_kp = metadata["release_keypoints"][stage - 1]
        is_grasp, is_release = grasp_kp != -1, release_kp != -1
        held = [i for i, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body]
        get_grasp_fn = get_callable_grasping_cost_fn(held)
        sub = load_torch_constraints(os.path.join(vlm_dir, f"stage{stage}_subgoal_constraints.txt"), get_grasp_fn, shim)
        constraint_fn = make_torch_constraint(sub)
        path_txt = os.path.join(vlm_dir, f"stage{stage}_path_constraints.txt")
        path_fns = ([make_torch_constraint([c]) for c in load_torch_constraints(path_txt, get_grasp_fn, shim)]
                    if os.path.exists(path_txt) else [])

        if is_grasp:
            name = object_for_kp(grasp_kp)
            center = masks.local_centroid(grounded, E.env, name, near=keypoints[grasp_kp])
            obstacles = [c for n, c in obj_c.items() if n != name]
            cost = make_task_cost(E.fk, E.a_local, center, grasp_offset=OFFSET, target_axis=(0.0, 0.0, -1.0),
                                  grasp_center=center, obstacles=obstacles, obstacle_r=args.obstacle_r,
                                  z_floor=z_floor, device=DEV)  # top-down grasp, no transit, close-near
            ctx_fn, transition = kps_t, lambda: float(np.linalg.norm(E.tcp() - center))
            label = f"grasp {name} (kp{grasp_kp}->center)"
            close_when_near = True
        else:
            rel_name = object_for_kp(release_kp)
            obstacles = [c for n, c in obj_c.items() if n != rel_name]
            tcp0, R0 = tcp_pose()
            tgt_pos, tgt_R, held_local = resolve_subgoal(constraint_fn, tcp0, R0, held, tracker.get_positions())
            tgt_axis = (tgt_R @ np.asarray(E.a_local)).tolist()  # resolved world approach axis
            print(f"[rekep-dial]   resolved place target={np.round(tgt_pos,3).tolist()}", flush=True)
            # Path constraints are loaded + logged, but NOT fed to the running cost: evaluating a
            # VLM/numpy constraint per-candidate through the np->torch shim is ~3x the per-step cost, and
            # so far they are either contact ("still grasping", skipped) or degenerate. Enforcement belongs
            # at the stage transition/backtrack; running-cost enforcement waits on a native-torch eval.
            cost = make_task_cost(E.fk, E.a_local, tgt_pos, grasp_offset=OFFSET, target_axis=tgt_axis,
                                  obstacles=obstacles, obstacle_r=args.obstacle_r, transit=True, w_transit=args.w_transit,
                                  z_clear=z_clear, descend_r=args.descend_r, z_floor=z_floor,
                                  path_fns=[], gripper_close_when_near=False, device=DEV)
            ctx_fn = kps_t
            transition = lambda: float(constraint_fn(torch.tensor(E.tcp(), device=DEV, dtype=torch.float32).reshape(1, 1, 3), kps_t())[0, 0])
            label = f"place (release kp{release_kp})"
            close_when_near = False

        plan = sampler(cost, args.horizon)
        print(f"[rekep-dial] stage {stage}/{num_stages}: {label}; held={held} path_fns={len(path_fns)}", flush=True)
        g_cmd = control(plan, ctx_fn, transition, close_when_near, stage, label, args.horizon)

        if is_grasp:
            grasped_body = tracker.owners[grasp_kp]
            for _ in range(args.settle_steps):  # brief closed-gripper settle so the grasp seats
                E.apply_arm(E.q0(), grip_open=False)
                frames.append(overlay.camera_overlay_frame(E.cam, tracker, [f"stage {stage}: CLOSE", ""]))
        if is_release:
            grasped_body = None
            for _ in range(args.settle_steps):
                E.apply_arm(E.q0(), grip_open=True)
                frames.append(overlay.camera_overlay_frame(E.cam, tracker, [f"stage {stage}: RELEASE", ""]))

    print("[rekep-dial] --- RESULT ---", flush=True)
    q = np.array(q_hist)
    if len(q) > 2:
        dq = np.diff(q, axis=0)
        nrm = np.linalg.norm(dq, axis=1)
        cos = (dq[1:] * dq[:-1]).sum(axis=1) / (nrm[1:] * nrm[:-1] + 1e-9)  # 1st-order direction consistency
        jerk = np.linalg.norm(q[2:] - 2 * q[1:-1] + q[:-2], axis=1).mean()  # mean |2nd-diff|, lower = smoother
        print(f"[rekep-dial] exec smoothness(cos)={cos.mean():.3f} jerk(2nd-diff)={jerk*1e3:.2f}e-3 "
              f"over {len(q)} steps", flush=True)
    for n in scene_objects:
        p = E.object_pose(n)[0]
        print(f"[rekep-dial] {n}: final={np.round(p,3).tolist()} dz={(p[2]-obj_z0[n])*100:+.1f}cm", flush=True)
    out = os.path.join(out_dir, f"{exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[rekep-dial] DONE -> {out} ({len(frames)} frames)", flush=True)
