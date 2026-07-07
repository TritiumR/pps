"""Droid single-object grasp experiments (the scaffold-free investigation), unified behind --mode.

Four modes, each a faithful version of its original standalone driver. They share the env + grounding +
target selection + sampler; only the control loop differs:

  cost_only    -- ONE grasp cost, ONE receding loop, gripper held OPEN, no phases / no early-exit.
                  Shows what the geometric cost alone drives (the close is the residual's job).
  cost_gripper -- cost + proximity-gripper: close LATCHES on proximity, then a coarse stage switch lifts.
  collab       -- the collaborator's sim_free_mpc cost + weights in our DIAL-execute-the-mean setup
                  (reach/terminal/smooth/delta/orient; --no-clamp to study the unclamped case).
  scaffold     -- the scripted HOVER -> DESCEND -> CLOSE -> LIFT grasp (the earliest D1 pear grasp).

    python -m dial_mpc.main --task droid_grasp --mode cost_only --object pear ...

Outputs -> results/vlm_mpc/droid_grasp/<mode>_<exp_name>.mp4.
"""
import argparse
import os

NAME = "droid_grasp"


def add_args(ap):
    ap.add_argument("--mode", type=str, default="cost_only",
                    choices=["cost_only", "cost_gripper", "collab", "scaffold"])
    ap.add_argument("--exp_name", type=str, default="s0")
    ap.add_argument("--object", type=str, default="pear")
    ap.add_argument("--ground", type=str, default="rekep", choices=["gt", "rekep"])
    ap.add_argument("--target", type=str, default=None, choices=["keypoint", "centroid"],
                    help="grasp target (default: centroid, except scaffold mode which defaults to keypoint)")
    ap.add_argument("--pear_r", type=float, default=None,
                    help="object half-extent for the straddle term (default: 0.03 cost_only, else 0.035)")
    # cost_only / collab loop length
    ap.add_argument("--steps", type=int, default=160, help="loop length for cost_only / collab")
    # cost_gripper
    ap.add_argument("--grip_thresh", type=float, default=0.012, help="proximity radius that triggers the close")
    ap.add_argument("--settle_steps", type=int, default=15, help="steps to hold the close before lifting")
    ap.add_argument("--max_steps", type=int, default=200, help="loop length for cost_gripper")
    ap.add_argument("--lift_h", type=float, default=0.15, help="post-grasp lift height (cost_gripper / scaffold)")
    # collab cost weights
    ap.add_argument("--clamp", action=argparse.BooleanOptionalAction, default=True,
                    help="clamp integrated configs to joint limits (--no-clamp to study the unclamped case)")
    ap.add_argument("--w_reach", type=float, default=25.0)
    ap.add_argument("--w_terminal", type=float, default=40.0)
    ap.add_argument("--w_smooth", type=float, default=0.03)
    ap.add_argument("--w_delta", type=float, default=0.005)
    ap.add_argument("--w_orient", type=float, default=0.25)
    # scaffold phases
    ap.add_argument("--hover", type=float, default=0.12)
    ap.add_argument("--steps_hover", type=int, default=90)
    ap.add_argument("--steps_descend", type=int, default=90)
    ap.add_argument("--close_steps", type=int, default=30)
    ap.add_argument("--steps_lift", type=int, default=70)
    ap.add_argument("--tol", type=float, default=0.02)
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
    import re

    import numpy as np
    import torch

    from rekep import grounding
    from rekep.keypoint_tracking import KeypointTracker
    from rekep.utils import load_default_config
    from rekep.video import write_video_h264
    from sim_common.envs.droid import DroidEnv, ROBOTIQ_GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_grasp_cost, make_rekep_cost, fixed_reach
    from sim_common import overlay
    from dial_mpc import control

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
    frames = []
    mode = args.mode

    # Mode-conditional defaults preserve each original driver's behavior; explicit flags still override.
    target_mode = args.target or ("keypoint" if mode == "scaffold" else "centroid")
    pear_r = args.pear_r if args.pear_r is not None else (0.03 if mode == "cost_only" else 0.035)

    def tt(v):
        return control.tt(v, DEV)

    def make_collab_cost(fk, grasp_offset, w_reach, w_terminal, w_smooth, w_delta, w_orient, device):
        """Faithful copy of sim_free_mpc/costs.py (arm terms; gripper term omitted)."""
        z_local = torch.tensor([0.0, 0.0, 1.0], device=device)   # tool axis = EE local +Z
        down = torch.tensor([0.0, 0.0, -1.0], device=device)

        def cost_fn(q_traj, q_cur, target):
            k, h, _ = q_traj.shape
            pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
            pos = pos.reshape(k, h, 3)
            rot = rot.reshape(k, h, 3, 3)
            dist2 = ((pos - target) ** 2).sum(dim=-1)                          # [k,h]
            reach = w_reach * dist2.mean(dim=1) + w_terminal * dist2[:, -1]
            dq = q_traj[:, 1:, :] - q_traj[:, :-1, :]
            smooth = w_smooth * (dq ** 2).sum(dim=(1, 2))
            delta = w_delta * ((q_traj - q_cur) ** 2).sum(dim=(1, 2))
            tool = torch.einsum("khij,j->khi", rot, z_local)                   # R @ [0,0,1]
            orient = w_orient * (1.0 - (tool * down).sum(dim=-1)).mean(dim=1)
            return reach + smooth + delta + orient

        return cost_fn

    # ---- env + grounding + target selection (shared) ----
    E = DroidEnv(device=DEV)
    a_local = E.a_local
    gt = E.object_pose(args.object)[0]
    tracker = None
    if args.ground == "rekep":
        grounded = grounding.propose_keypoints(E.cam, E.env, load_default_config())
        keypoints = grounded["keypoints"]
        if len(keypoints) == 0:
            raise SystemExit("[droid-grasp] no keypoints proposed")
        tracker = KeypointTracker(E.env, keypoints)
        if target_mode == "centroid":
            rel = re.sub(r"^/World/envs/env_[^/]*/", "", E.env.scene[args.object].cfg.prim_path)
            ids = [i for i, prim in grounded["id_to_prim"].items() if rel and rel in prim]
            sel = np.isin(grounded["masks"], ids) & np.isfinite(grounded["points"]).all(axis=-1)
            target = grounded["points"][sel].mean(axis=0)
        else:
            target = keypoints[int(np.argmin(np.linalg.norm(keypoints - gt, axis=1)))]
    else:
        target = gt
    z_floor = float(target[2]) - 0.05
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)
    print(f"[droid-grasp] mode={mode} target {args.object}={np.round(target,3).tolist()} "
          f"(GT {np.round(gt,3).tolist()}, err={np.linalg.norm(target-gt)*100:.1f}cm) "
          f"tcp0={np.round(E.tcp(),3).tolist()} z_floor={z_floor:.3f}", flush=True)

    def sampler(cost, clamp=True):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=args.horizon, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=args.accel_std,
                                  accel_clip=args.accel_clip, temperature=args.temperature,
                                  clamp_limits=clamp, device=DEV)

    def frame(title, lbl):
        frames.append(overlay.camera_overlay_frame(E.cam, tracker, [title, lbl]) if tracker is not None
                      else overlay.plain_frame(E.rgb()))

    obj_z0 = E.object_pose(args.object)[0][2]

    # ---------------- modes ----------------
    if mode == "cost_only":
        cost = make_grasp_cost(E.fk, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET, r_cube=pear_r,
                               open_half=0.0425, finger_r=0.012, z_floor=z_floor, device=DEV)
        plan = sampler(cost)
        ctx = (tt(target),) * 2
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        for step in range(args.steps):
            mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx, gen)
            E.apply_arm(q_traj[exec_knot], grip_open=True)            # gripper OPEN -- no scripted close
            err = float(np.linalg.norm(E.tcp() - target))
            frame("DIAL cost-only (no scaffold)", f"COST-ONLY {step}: tcp_err={err*100:.1f}cm cost={score:.1f} ess={ess:.0f}")
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            if step % 20 == 0:
                print(f"[droid-grasp] {step:3d}: tcp_err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}", flush=True)
        obj_z1 = E.object_pose(args.object)[0][2]
        print(f"[droid-grasp] FINAL tcp_err={float(np.linalg.norm(E.tcp()-target))*100:.2f}cm | "
              f"obj_z {obj_z0:.3f}->{obj_z1:.3f} (moved {(obj_z1-obj_z0)*100:.1f}cm; no close => expect ~0)", flush=True)

    elif mode == "cost_gripper":
        cost_grasp = make_grasp_cost(E.fk, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET, r_cube=pear_r,
                                     open_half=0.0425, finger_r=0.012, z_floor=z_floor, device=DEV)
        cost_reach = make_rekep_cost(E.fk, fixed_reach, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET,
                                     z_floor=z_floor, device=DEV)
        plan_grasp, plan_reach = sampler(cost_grasp), sampler(cost_reach)
        pear_t = tt(target)
        lift_tgt = tt(target + np.array([0.0, 0.0, args.lift_h]))
        grasped, lifting, grasp_step = False, False, None
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        for step in range(args.max_steps):
            if lifting:
                mean_a, q_traj, score, ess = plan_reach(mean_a, E.q0(), qd0, lift_tgt.unsqueeze(0), gen)
                phase = "LIFT"
            else:
                mean_a, q_traj, score, ess = plan_grasp(mean_a, E.q0(), qd0, (pear_t, pear_t), gen)
                phase = "HOLD" if grasped else "GRASP"
            E.apply_arm(q_traj[exec_knot], grip_open=not grasped)
            d = float(np.linalg.norm(E.tcp() - target))
            if not grasped and d < args.grip_thresh:                 # proximity close + latch
                grasped, grasp_step = True, step
                print(f"[droid-grasp] CLOSE (proximity) at step {step}, d={d*100:.2f}cm", flush=True)
            if grasped and not lifting and step >= grasp_step + args.settle_steps:
                lifting = True
                print(f"[droid-grasp] LIFT begins at step {step}", flush=True)
            frame("DIAL cost + proximity-gripper",
                  f"{phase} {step}: d={d*100:.1f}cm grip={'C' if grasped else 'O'} cost={score:.1f}")
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            if step % 20 == 0:
                print(f"[droid-grasp] {step:3d}: {phase} d={d*100:.2f}cm grip={'C' if grasped else 'O'} cost={score:.2f}", flush=True)
        obj_z1 = E.object_pose(args.object)[0][2]
        held = bool(obj_z1 > obj_z0 + 0.05)
        print(f"[droid-grasp] RESULT close={'yes' if grasped else 'NO'} lift={'yes' if lifting else 'NO'} | "
              f"obj_z {obj_z0:.3f}->{obj_z1:.3f} (+{(obj_z1-obj_z0)*100:.1f}cm, held={held})", flush=True)

    elif mode == "collab":
        print(f"[droid-grasp] weights: reach={args.w_reach} terminal={args.w_terminal} smooth={args.w_smooth} "
              f"delta={args.w_delta} orient={args.w_orient} (no yaw/straddle/floor; gripper open) "
              f"clamp_limits={args.clamp}", flush=True)
        cost = make_collab_cost(E.fk, ROBOTIQ_GRASP_OFFSET, args.w_reach, args.w_terminal, args.w_smooth,
                                args.w_delta, args.w_orient, DEV)
        plan = sampler(cost, clamp=args.clamp)
        ctx = tt(target)
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        for step in range(args.steps):
            mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx, gen)
            E.apply_arm(q_traj[exec_knot], grip_open=True)
            err = float(np.linalg.norm(E.tcp() - target))
            title = f"her cost+weights in our DIAL ({'CLAMP' if args.clamp else 'NO-CLAMP'})"
            frame(title, f"COLLAB-COST {step}: tcp_err={err*100:.1f}cm cost={score:.1f} ess={ess:.0f}")
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            if step % 20 == 0:
                print(f"[droid-grasp] {step:3d}: tcp_err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}", flush=True)
        print(f"[droid-grasp] FINAL tcp_err={float(np.linalg.norm(E.tcp()-target))*100:.2f}cm", flush=True)

    else:  # scaffold (HOVER -> DESCEND -> CLOSE -> LIFT)
        cost_grasp = make_grasp_cost(E.fk, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET, r_cube=pear_r,
                                     open_half=0.0425, finger_r=0.012, z_floor=z_floor, device=DEV)
        cost_reach = make_rekep_cost(E.fk, fixed_reach, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET,
                                     z_floor=z_floor, device=DEV)
        plan_grasp, plan_reach = sampler(cost_grasp), sampler(cost_reach)

        def record(label):
            if tracker is not None:
                frames.append(overlay.camera_overlay_frame(E.cam, tracker, [f"DIAL {args.object} grasp", label]))
            else:
                frames.append(overlay.plain_frame(E.rgb(), label))

        def run_phase(plan, ctx, grip_open, tgt, max_steps, label):
            mean_a = torch.zeros(args.horizon, 7, device=DEV)
            qd0 = torch.zeros(7, device=DEV)
            for step in range(max_steps):
                mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx, gen)
                E.apply_arm(q_traj[exec_knot], grip_open)
                record(f"{label} {step}: cost={score:.1f} ess={ess:.0f}")
                mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
                err = float(np.linalg.norm(E.tcp() - tgt))
                if err < args.tol:
                    print(f"[droid-grasp] {label}: reached (tcp_err={err*100:.2f}cm) at step {step}", flush=True)
                    return
                if step % 20 == 0:
                    print(f"[droid-grasp] {label} {step:3d}: tcp_err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}", flush=True)
            print(f"[droid-grasp] {label}: cap (tcp_err={float(np.linalg.norm(E.tcp()-tgt))*100:.2f}cm)", flush=True)

        hover_tgt = target + np.array([0.0, 0.0, args.hover])
        run_phase(plan_reach, tt([hover_tgt]), True, hover_tgt, args.steps_hover, "HOVER")
        run_phase(plan_grasp, (tt(target), tt(target)), True, target, args.steps_descend, "DESCEND")
        control.hold_pose(E, args.close_steps, False, lambda: record("CLOSE"))

        grasp_err = float(np.linalg.norm(E.tcp() - target))
        obj_before = E.object_pose(args.object)[0]
        lift_tgt = target + np.array([0.0, 0.0, args.lift_h])
        run_phase(plan_reach, tt([lift_tgt]), False, lift_tgt, args.steps_lift, "LIFT")
        control.hold_pose(E, 12, False, lambda: record("LIFT-hold"))
        obj_after = E.object_pose(args.object)[0]
        print(f"[droid-grasp] RESULT {args.object}_grasp_tcp_err={grasp_err*100:.2f}cm (reached={grasp_err<0.05}) | "
              f"lift: obj_z {obj_before[2]:.3f}->{obj_after[2]:.3f} (+{(obj_after[2]-obj_before[2])*100:.1f}cm, "
              f"held={obj_after[2] > obj_before[2]+0.05})", flush=True)

    out = os.path.join(_REPO, "results", "vlm_mpc", "droid_grasp", f"{mode}_{args.exp_name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[droid-grasp] DONE -> {out} ({len(frames)} frames)", flush=True)
