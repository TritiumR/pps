"""Shared base-controller driver: runs a Grounding on the sim_free engine and records a rollout video.

Grounding-agnostic: it consumes only the Grounding contract (obstacles + stages) plus an already
built sim_free MPC, so front-ends (ReKep / VoxPoser / MOKA / GT) and controllers vary independently. The
loop: SDEdit warm start, B-spline-smoothed chunks, a consistency reference, a proximity gripper, and
per-stage advance on a committed grasp.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from rekep.video import write_video_h264
from sim_common import overlay
from vlm_base import metrics
from vlm_base import sim_free_core as core
from sim_common.envs.droid import ROBOTIQ_GRASP_OFFSET


def _capture_held(env, grounding, held_idx):
    """Gripper-local offsets of held keypoints at stage entry (for rigid riding in the constraint cost)."""
    if not held_idx or grounding.keypoints is None:
        return None
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), ROBOTIQ_GRASP_OFFSET)
    tcp, rmat = pos[0].detach().cpu().numpy(), rot[0].detach().cpu().numpy()
    return np.stack([rmat.T @ (kps[i] - tcp) for i in held_idx])   # local = R^T (kp - tcp)


def _ctx_from_stage(env, grounding, stage, root_pos, root_quat, plan_ref, held_offset=None):
    """Cost context for the current stage: live target + obstacles + phase (+ ReKep constraint if present)."""
    dev = env.device
    objs = {o.name: {"pos": torch.as_tensor(o.pos(), device=dev, dtype=torch.float32), "extents": o.extents}
            for o in grounding.objects}
    z_bottoms = [float(o.pos()[2]) - o.extents[2] for o in grounding.objects]
    ctx = {"objects": objs, "joint_pos": env.q0(), "robot_root_pos": root_pos, "robot_root_quat": root_quat,
           "target": np.asarray(stage.target(), dtype=np.float32), "grasp_obj": stage.grasp_obj,
           "payload": stage.payload, "place_target": stage.place_target,
           "eef_pos": np.asarray(env.tcp(), dtype=np.float32),   # current TCP (grasp_flow cost reads this)
           "subtasks": core.read_subtask_flags(env),   # env grasp/place flags (grasp_flow cost re-grasps on slip)
           "z_table": (min(z_bottoms) if z_bottoms else None), "plan_ref": plan_ref}
    if stage.constraint is not None:              # ReKep constraint-as-cost stage
        ctx["keypoints"] = np.asarray(grounding.keypoints(), dtype=np.float32)
        ctx["constraint"] = stage.constraint
        ctx["path_fns"] = stage.path_fns
        ctx["held_idx"] = stage.held_idx
        ctx["held_offset"] = held_offset
    return ctx


def _should_advance(stage, flags, hold, commit_hold):
    """Advance on the stage's task-progress flag when it names one, else the held-grasp fallback."""
    if stage.done_flag is not None and stage.done_flag in flags:
        return flags[stage.done_flag]
    return (stage.gripper == "close" and hold >= commit_hold) or stage.done()


def _gripper_open(env, stage, target, seat_dist):
    """Gripper command for the stage: close near target, hold closed, open, or release when placed."""
    if stage.gripper == "hold":
        return False
    if stage.gripper == "open":
        return True
    if stage.gripper == "place":
        return bool(stage.done())   # carry closed, release once the placement constraint is satisfied
    return not (float(np.linalg.norm(env.tcp() - target)) < seat_dist)   # "close": shut when seated


def _frame(env, grounding, stage, chunk, t, dist, grip_open):
    """Rollout frame: live tracked keypoints + the stage's relational constraint drawn on the cam.

    ReKep grounding exposes live keypoints, so draw them (numbered, indices matching the VLM) plus the
    subgoal constraint as a line between its operands: TCP to the grasp keypoint for a grasp, or the held
    keypoint to the placement point for a place. Falls back to a plain labelled frame for keypoint-less (GT)
    grounding.
    """
    text = [stage.name, f"ch{chunk}.{t} d={dist:.2f}m grip={'open' if grip_open else 'closed'}"]
    if grounding.keypoints is None:
        return overlay.plain_frame(env.rgb(), " | ".join(text))
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    src = kps[stage.held_idx[0]] if stage.held_idx else np.asarray(env.tcp(), dtype=np.float64)
    tgt = np.asarray(stage.target(), dtype=np.float64)
    link = f"{np.linalg.norm(src - tgt) * 100:.0f}cm"
    return overlay.constraint_overlay_frame(env.cam, kps, [(src, tgt, link)], text)


def run_base(env, grounding, *, mpc, policy, state_stats, cfg, args, out_dir):
    """Drive the grounded stages with the sim_free MPC; write <exp_name>.mp4 and print metrics."""
    dev = env.device
    H = args.horizon
    k = min(args.exec_knot, H)
    root_pos = env.robot.data.body_pos_w[0, env.l0].detach()
    root_quat = env.robot.data.body_quat_w[0, env.l0].detach()

    plan_ref = {"v": None}                       # warm-started previous plan (the consistency reference)
    frames, q_hist, dist_hist = [], [], []
    obj_gt0 = {o.name: env.object_pose(o.name)[0].copy() for o in grounding.objects}   # GT initial poses (report)
    stage_idx, hold, x_carry, released, advance_streak = 0, 0, None, False, 0
    held_offset = _capture_held(env, grounding, grounding.stages[0].held_idx)

    for chunk in range(args.max_chunks):
        stage = grounding.stages[stage_idx]
        ctx = _ctx_from_stage(env, grounding, stage, root_pos, root_quat, plan_ref["v"], held_offset)
        pin = core.policy_inputs(env, state_stats, True)
        it_start = 0
        if args.init == "warm" and x_carry is not None:   # SDEdit warm start (temporal coherence)
            it_start = max(0, args.denoise_iters - args.warm_steps)
            x_init = core.sdedit_warm_start(x_carry, it_start, args.denoise_iters,
                                            cfg.ddim_num_train_timesteps, H, dev)
        else:
            x_init = torch.randn(1, H, 8, device=dev, dtype=torch.float32)
        x0 = core.plan_chunk(mpc, x_init, pin, ctx, mode=args.mode, update=args.update,
                             denoise_iters=args.denoise_iters, score_scale=args.score_scale,
                             dt=env.dt, it_start=it_start)
        x_carry = torch.cat([x0[:, k:], x0[:, -1:].expand(1, k, 8)], dim=1).detach()
        real = core.decode_model_action_chunks(policy, pin, x0, current_joint_pos=env.q0(),
                                               max_joint_delta=args.joint_delta_clip).real_actions[0]
        rj = real[:, :7].detach()
        plan_ref["v"] = torch.cat([rj[k:], rj[-1:].expand(k, 7)], dim=0)   # shift forward = warm reference

        target = ctx["target"]
        cost_gripper = getattr(args, "gripper_source", "latch") == "cost"   # grasp_flow cost drives the gripper
        for t in range(k):
            if cost_gripper or (getattr(args, "gripper_in_loop", False) and stage.gripper == "close"):
                grip_open = float(real[t][7]) < 0.5   # follow the decoded (cost-driven) gripper channel
            else:
                grip_open = _gripper_open(env, stage, stage.target(), args.seat_dist)  # live target per step
            if stage.gripper == "close" and not grip_open:
                hold += 1
            if stage.gripper == "place" and grip_open:   # gripper opened on a place stage -> object released
                released = True
            env.apply_arm(real[t][:7], grip_open=grip_open)
            q_hist.append(env.q0().detach().cpu().numpy())
            d_now = float(np.linalg.norm(env.tcp() - target))
            dist_hist.append(d_now)
            frames.append(_frame(env, grounding, stage, chunk, t, d_now, grip_open))

        flags = core.read_subtask_flags(env)
        signal = _should_advance(stage, flags, hold, args.commit_hold)
        if stage.gripper == "place":
            signal = signal and released   # don't advance a place until the object is actually released
        advance_streak = advance_streak + 1 if signal else 0   # debounce: a stable signal, not a one-chunk flicker
        if advance_streak >= getattr(args, "advance_persist", 2) and stage_idx + 1 < len(grounding.stages):
            stage_idx, hold, released, advance_streak = stage_idx + 1, 0, False, 0
            held_offset = _capture_held(env, grounding, grounding.stages[stage_idx].held_idx)
            print(f"[base] stage -> {grounding.stages[stage_idx].name} at chunk {chunk}", flush=True)
        gobj = stage.grasp_obj or stage.payload   # grasp/lift diagnostic: does the object rise (grasped) or stay/slip?
        gz = f" {gobj}_z={float(env.object_pose(gobj)[0][2]):.3f}" if gobj else ""
        grip = float(real[:k, 7].mean()) if real.shape[-1] > 7 else float("nan")   # decoded gripper channel (cost-driven)
        print(f"[base]   chunk {chunk}: stage={stage.name} dist={d_now:.3f}m grip={grip:.2f} hold={hold}{gz} "
              f"flags={flags}", flush=True)

    _report(env, grounding, obj_gt0, q_hist, dist_hist)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[base] DONE -> {out} ({len(frames)} frames)", flush=True)


def _report(env, grounding, obj_gt0, q_hist, dist_hist):
    """Print motion smoothness, reach, scene disturbance, and grasp lift, all from GT object poses.

    Reads ground-truth poses (env.object_pose + the GT initial snapshot obj_gt0) for both the
    disturbance and the grasp lift, so the two ends of every delta share one authoritative frame. The
    perception centroids the controller uses carry a top-surface bias that must not enter the metric.
    """
    print("[base] --- RESULT ---", flush=True)
    sm = metrics.motion_smoothness(q_hist)
    if sm:
        print(f"[base] jerk(2nd-diff)={sm['jerk'] * 1e3:.2f}e-3 speed={sm['speed'] * 1e3:.1f}e-3 "
              f"over {sm['n']} steps", flush=True)
    rs = metrics.reach_stats(dist_hist)
    if rs:
        print(f"[base] reach(TCP->target): min={rs['min']:.3f}m final={rs['final']:.3f}m", flush=True)
    obstacles = [o.name for o in grounding.objects if o.name not in grounding.manipulated]
    _, tot, mx = metrics.scene_disturbance(env, obj_gt0, obstacles)
    print(f"[base] scene_disturbance({obstacles}): sum={tot * 100:.1f}cm max={mx * 100:.1f}cm", flush=True)
    for g in sorted({s.grasp_obj for s in grounding.stages if s.grasp_obj}):   # any stage that grasps an object
        dz = (env.object_pose(g)[0][2] - obj_gt0[g][2]) * 100
        print(f"[base] grasp_obj {g}: dz={dz:+.1f}cm grasped={'Y' if dz > 3.0 else 'N'}", flush=True)
