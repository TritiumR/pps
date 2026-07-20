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
from vlm_base import geom_proxy
from vlm_base import metrics
from vlm_base import sim_free_core as core
from sim_common.envs.droid import ROBOTIQ_GRASP_OFFSET
from task_success import report_task_success


def _capture_held(env, grounding, held_idx):
    """Gripper-local offsets of held keypoints at stage entry (for rigid riding in the constraint cost)."""
    if not held_idx or grounding.keypoints is None:
        return None
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), ROBOTIQ_GRASP_OFFSET)
    tcp, rmat = pos[0].detach().cpu().numpy(), rot[0].detach().cpu().numpy()
    return np.stack([rmat.T @ (kps[i] - tcp) for i in held_idx])


def _ctx_snapshot(ctx, obj_names, stage_idx):
    """Geometry the base score reads at a chunk: target, per-object pose+extent, grasp target, stage.

    This is the conditioning a score-space reference proxy distills. The base is invariant to pixels, so it
    denoises against this geometry, not the camera. Keypoints are included only when the grounding carries
    them (ReKep grounding); under gt grounding the object positions are the conditioning.
    """
    snap = {
        "target": np.asarray(ctx["target"], dtype=np.float32),
        "obj_pos": np.stack([ctx["objects"][n]["pos"].detach().cpu().numpy() for n in obj_names]).astype(np.float32),
        "obj_ext": np.stack([np.asarray(ctx["objects"][n]["extents"], dtype=np.float32) for n in obj_names]),
        "grasp_idx": np.int32(obj_names.index(ctx["grasp_obj"]) if ctx.get("grasp_obj") in obj_names else -1),
        "stage": np.int32(stage_idx),
    }
    if "keypoints" in ctx:
        snap["keypoints"] = np.asarray(ctx["keypoints"], dtype=np.float32)
    return snap


def _ctx_from_stage(env, grounding, world, stage, root_pos, root_quat, plan_ref, held_offset=None):
    """Cost context for the current stage: live target + obstacles + phase (+ ReKep constraint if present)."""
    dev = env.device
    objs = {o.name: {"pos": torch.as_tensor(o.pos(), device=dev, dtype=torch.float32), "extents": o.extents}
            for o in grounding.objects}
    z_bottoms = [float(o.pos()[2]) - o.extents[2] for o in grounding.objects]
    ctx = {"objects": objs, "joint_pos": env.q0(), "robot_root_pos": root_pos, "robot_root_quat": root_quat,
           "target": np.asarray(stage.target(), dtype=np.float32), "grasp_obj": stage.grasp_obj,
           "payload": stage.payload, "place_target": stage.place_target,
           "eef_pos": np.asarray(env.tcp(), dtype=np.float32), "subtasks": world.flags(),
           "z_table": (min(z_bottoms) if z_bottoms else None), "plan_ref": plan_ref}
    if stage.constraint is not None:
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


# Metres the carried object must rise before a grasp counts as real (see _payload_lost).
_RISE_CONFIRM = 0.01


def _grasp_baseline(world, grounding, stage_idx, baselines):
    """Record the grasp object's height as its grasp stage begins: the datum for "has it actually risen".

    Its resting height would let anything already on a raised surface pass with an empty gripper; its height
    at lift-stage entry demands a second rise and condemns a good grasp. The height when the attempt began is
    right in both cases.
    """
    stage = grounding.stages[stage_idx]
    if stage.payload is None and stage.grasp_obj is not None:
        baselines[stage.grasp_obj] = float(world.object_pose(stage.grasp_obj)[0][2])


def _held(flags, obj):
    """Is ``obj`` reported in the gripper?"""
    return bool(obj is not None and flags.get(f"grasp_{obj}", False))


def _released_ok(stage, flags):
    """The stage let go on purpose, so a payload no longer in the gripper is success rather than a slip.

    True when the stage's task flag is met, or when a place stage's geometric ``done`` holds -- some envs
    expose no placement flag, so a flag-only check would read every deliberate release as a slip. This is
    about the stage's intent, not the loss evidence, so both monitors share it.
    """
    if stage.done_flag is not None and flags.get(stage.done_flag, False):
        return True
    return stage.gripper == "place" and stage.done()


def _payload_lost(stage, flags, stage_chunk=0, risen=False, confirm_chunks=10):
    """The stage's carried object is not really in the gripper, and the stage's goal is not met.

    Two tests, because a grasp signal may lie about an empty hand: the payload is lost when the signal has
    dropped (a slip), or when it has been carried long enough that a real grasp would have moved the object
    and it has not risen. The second test is what a weak, flickery grasp signal needs; a gripper that reports
    what is between its fingers only ever agrees with it. The rise is read from the world model, so it is
    never privileged. This is the proprioceptive monitor (``--monitor threshold``); ``_grasp_invariant_lost``
    is the geometric one.
    """
    if stage.payload is None:
        return False
    if _released_ok(stage, flags):
        return False
    if not _held(flags, stage.payload):
        return True
    return stage_chunk > confirm_chunks and not risen


def _grasp_invariant_lost(env, grounding, stage, flags, held_offset, tol):
    """The stage's grasp-hold invariant is violated: a grasped keypoint no longer rides the gripper.

    ReKep backtracks when a stage's PATH (invariant) constraint is violated. The VLM here emitted only
    subgoal constraints (empty path files), so the one invariant a carry stage needs -- "still grasping
    keypoint i" -- is synthesized from ``held_idx``: each held keypoint must stay at ``tcp + R_ee @ local``.
    Evaluated on the LIVE tracked keypoints, so it is a geometric, vision-based test wherever the tracker is
    independent of the arm (true poses under ``--state gt``, a point tracker under ``--track visual``); under
    ``--track fk`` the keypoint is arm-propagated and matches by construction, so the test defers to contact.
    """
    if stage.payload is None or _released_ok(stage, flags):
        return False
    # Nothing grasped to watch (e.g. a coarsened grounding) -- no opinion.
    if not stage.held_idx or held_offset is None or grounding.keypoints is None:
        return False
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), ROBOTIQ_GRASP_OFFSET)
    tcp, rmat = pos[0].detach().cpu().numpy(), rot[0].detach().cpu().numpy()
    return any(float(np.linalg.norm(kps[j] - (tcp + rmat @ local))) > tol
               for local, j in zip(held_offset, stage.held_idx))


def _grasp_stage_for(grounding, payload, from_idx):
    """Index of the stage that grasps ``payload`` (scanning back from ``from_idx``), or None if there is none.

    A coarsened grounding carries the payload on the grasping stage itself, so no such stage exists and the
    monitor never regresses. A cost using that grounding is expected to re-grasp internally.
    """
    for j in range(from_idx, -1, -1):
        s = grounding.stages[j]
        if s.payload is None and s.grasp_obj == payload:
            return j
    return None


def _gripper_open(env, stage, target, seat_dist):
    """Gripper command for the stage: close near target, hold closed, open, or release when placed."""
    if stage.gripper == "hold":
        return False
    if stage.gripper == "open":
        return True
    if stage.gripper == "place":
        return bool(stage.done())
    return not (float(np.linalg.norm(env.tcp() - target)) < seat_dist)


def _frame(env, grounding, stage, chunk, t, dist, grip_open):
    """Rollout frame: live tracked keypoints + the stage's relational constraint drawn on the cam.

    ReKep grounding exposes live keypoints, so draw them (numbered, indices matching the VLM) plus the
    subgoal constraint as a line between its operands. Falls back to a plain labelled frame for keypoint-less
    (GT) grounding.
    """
    text = [stage.name, f"ch{chunk}.{t} d={dist:.2f}m grip={'open' if grip_open else 'closed'}"]
    if grounding.keypoints is None:
        return overlay.plain_frame(env.rgb(), " | ".join(text))
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    src = kps[stage.held_idx[0]] if stage.held_idx else np.asarray(env.tcp(), dtype=np.float64)
    tgt = np.asarray(stage.target(), dtype=np.float64)
    link = f"{np.linalg.norm(src - tgt) * 100:.0f}cm"
    return overlay.constraint_overlay_frame(env.cam, kps, [(src, tgt, link)], text)


def run_base(env, grounding, world, *, mpc, policy, state_stats, cfg, args, out_dir, steer=None):
    """Drive the grounded stages with the sim_free MPC; write <exp_name>.mp4 and print metrics.

    ``steer``: a geom_proxy.GeomSteer for score-space PPS. Bound to each chunk's ctx geometry and passed to
    plan_chunk, which adds gamma * (s_task - s_ref) to the base score at every denoise step.
    """
    dev = env.device
    H = args.horizon
    k = min(args.exec_knot, H)
    root_pos = env.robot.data.body_pos_w[0, env.l0].detach()
    root_quat = env.robot.data.body_quat_w[0, env.l0].detach()

    if getattr(args, "monitor", "threshold") == "constraint" and grounding.keypoints is None:
        print("[base] NOTE: --monitor constraint watches tracked keypoints; this grounding has none "
              "(use --ground rekep_real/rekep_fake), so it will never backtrack.", flush=True)

    plan_ref = {"v": None}
    ref_rec = [] if getattr(args, "record_ref", None) else None
    ref_obs = []
    frames, q_hist, dist_hist = [], [], []
    obj_gt0 = {o.name: env.object_pose(o.name)[0].copy() for o in grounding.objects}
    obj_names = [o.name for o in grounding.objects]
    stage_idx, hold, x_carry, released, advance_streak, lost_streak = 0, 0, None, False, 0, 0
    ever_held = False   # payload was actually grasped at some point THIS stage; gates a genuine release
    stage_chunk = 0
    grasp_baselines = {}
    held_offset = _capture_held(env, grounding, grounding.stages[0].held_idx)
    _grasp_baseline(world, grounding, 0, grasp_baselines)

    for chunk in range(args.max_chunks):
        stage = grounding.stages[stage_idx]
        ctx = _ctx_from_stage(env, grounding, world, stage, root_pos, root_quat, plan_ref["v"], held_offset)
        pin = core.policy_inputs(env, state_stats, True)
        it_start = 0
        # Warm-start (SDEdit from the shifted previous plan) while carrying a payload, for cross-chunk
        # continuity: a fresh-noise replan every chunk jerks the arm and slips the payload. Grasp/reach keep
        # fresh noise (the tuned dwell-free init). General: gated on carrying a payload, not on the task.
        warm = args.init == "warm" or stage.payload is not None
        if warm and x_carry is not None:
            it_start = max(0, args.denoise_iters - args.warm_steps)
            x_init = core.sdedit_warm_start(
                x_carry, it_start, args.denoise_iters, cfg.ddim_num_train_timesteps, H, dev)
        else:
            x_init = torch.randn(1, H, 8, device=dev, dtype=torch.float32)
        chunk_labels = [] if ref_rec is not None else None
        if steer is not None:
            snap = _ctx_snapshot(ctx, obj_names, stage_idx)
            steer.set_chunk(geom_proxy.features_from_geom(
                snap["obj_pos"], snap["obj_ext"], snap["grasp_idx"], snap["target"],
                np.asarray(env.tcp()), snap["stage"], env.q0().detach().cpu().numpy()))
        x0 = core.plan_chunk(
            mpc, x_init, pin, ctx, mode=args.mode, update=args.update, denoise_iters=args.denoise_iters,
            score_scale=args.score_scale, dt=env.dt, it_start=it_start, record=chunk_labels, steer=steer)
        if ref_rec is not None:
            ref_obs.append({"joint_pos": env.q0().detach().cpu().numpy(), "eef_pos": np.asarray(env.tcp()),
                            "rgb": env.rgb(), "chunk": chunk,
                            "ctx": _ctx_snapshot(ctx, obj_names, stage_idx)})
            for lab in chunk_labels:
                lab["chunk"] = chunk
                ref_rec.append(lab)
        x_carry = torch.cat([x0[:, k:], x0[:, -1:].expand(1, k, 8)], dim=1).detach()
        # Tighter per-step joint cap while carrying: a large slew slips the payload; grasp/reach keep the looser
        # cap for fast convergence. Phase-gated on the payload, not the task.
        clip = args.joint_delta_clip
        if stage.payload is not None:
            clip = getattr(args, "carry_joint_delta_clip", 0.07)
        real = core.decode_model_action_chunks(
            policy, pin, x0, current_joint_pos=env.q0(), max_joint_delta=clip).real_actions[0]
        rj = real[:, :7].detach()
        plan_ref["v"] = torch.cat([rj[k:], rj[-1:].expand(k, 7)], dim=0)

        target = ctx["target"]
        cost_gripper = getattr(args, "gripper_source", "latch") == "cost"
        cmd_open = None
        for t in range(k):
            if cost_gripper or (getattr(args, "gripper_in_loop", False) and stage.gripper == "close"):
                grip_open = float(real[t][7]) < 0.5
            else:
                grip_open = _gripper_open(env, stage, stage.target(), args.seat_dist)
            if stage.gripper == "close" and not grip_open:
                hold += 1
            # `released` marks a deliberate let-go: it advances a place stage and latches the gripper open.
            # Gate it on `ever_held` -- a coarsened "transport" stage is gripper="place" from its first chunk,
            # before it has grasped anything, so an open gripper at the start would otherwise read as an
            # instant release and skip the object without grasping. A fine place stage is entered already
            # holding the payload (ever_held set by its first carry chunk), so this is a no-op there.
            if stage.gripper == "place" and grip_open and ever_held:
                released = True
            # A placement is one-way: once released over the target, stay open, even as the geometric `done`
            # flickers while a round object settles on the surface.
            if stage.gripper == "place" and released:
                grip_open = True
            cmd_open = grip_open
            env.apply_arm(real[t][:7], grip_open=grip_open)
            # The stage names what the controller is reaching for, which is what lets the fingers'
            # "something is in here" become "the pear is".
            world.observe(env, commanded_close=not grip_open,
                          candidates={o for o in (stage.grasp_obj, stage.payload) if o})
            q_hist.append(env.q0().detach().cpu().numpy())
            d_now = float(np.linalg.norm(env.tcp() - target))
            dist_hist.append(d_now)
            frames.append(_frame(env, grounding, stage, chunk, t, d_now, grip_open))

        flags = world.flags()
        ever_held = ever_held or (stage.payload is not None and _held(flags, stage.payload))
        # This chunk's replanning verdict. Recovery belongs here, not in the cost: regressing to the grasp
        # stage clears `payload`, which re-enables the cost's grasp terms (they gate on it). `lost_streak`
        # gives a noisy grasp signal some hysteresis.
        stage_chunk += 1
        risen = (stage.payload is not None and stage.payload in grasp_baselines
                 and float(world.object_pose(stage.payload)[0][2])
                 > grasp_baselines[stage.payload] + _RISE_CONFIRM)
        flag_gone = stage.payload is not None and not _held(flags, stage.payload)
        if getattr(args, "monitor", "threshold") == "constraint":
            lost = _grasp_invariant_lost(env, grounding, stage, flags, held_offset,
                                         getattr(args, "constraint_tol", 0.08))
            why = "invariant violated"
        else:
            lost = _payload_lost(stage, flags, stage_chunk, risen, getattr(args, "grasp_confirm", 10))
            why = "slipped" if flag_gone else "never rose"
        lost_streak = lost_streak + 1 if lost else 0
        placed_done = stage.gripper == "place" and released
        regress_to = (_grasp_stage_for(grounding, stage.payload, stage_idx)
                      if lost_streak >= getattr(args, "regress_persist", 2) and not placed_done else None)
        if regress_to is not None and regress_to != stage_idx:
            print(f"[base] REGRESS -> {grounding.stages[regress_to].name} at chunk {chunk} "
                  f"(lost {stage.payload}: {why})", flush=True)
            # The object was dropped, so where it is is no longer known: look again before grasping again.
            world.refresh(env)
            stage_idx, hold, released, advance_streak, lost_streak = regress_to, 0, False, 0, 0
            ever_held = False
            stage_chunk = 0
            _grasp_baseline(world, grounding, stage_idx, grasp_baselines)
            held_offset = _capture_held(env, grounding, grounding.stages[stage_idx].held_idx)
        else:
            signal = _should_advance(stage, flags, hold, args.commit_hold)
            # The env's placement flag is unreliable (it can stay False with the object squarely placed), so
            # a place stage advances on release instead.
            if stage.gripper == "place":
                signal = released
            advance_streak = advance_streak + 1 if signal else 0
            if advance_streak >= getattr(args, "advance_persist", 2) and stage_idx + 1 < len(grounding.stages):
                stage_idx, hold, released, advance_streak = stage_idx + 1, 0, False, 0
                ever_held = False
                stage_chunk = 0
                _grasp_baseline(world, grounding, stage_idx, grasp_baselines)
                held_offset = _capture_held(env, grounding, grounding.stages[stage_idx].held_idx)
                print(f"[base] stage -> {grounding.stages[stage_idx].name} at chunk {chunk}", flush=True)
        gopen = "open" if cmd_open else "shut"
        # Per-object xy-drift from the start (GT), so a placed object being knocked -- or the movable scale
        # pushed out from under it -- shows even while another stage is active.
        _manip = ({s.grasp_obj for s in grounding.stages if s.grasp_obj}
                  | {s.place_target for s in grounding.stages if s.place_target})
        gz = "".join(f" {g}@{np.linalg.norm(env.object_pose(g)[0][:2] - obj_gt0[g][:2]) * 100:.0f}cm"
                     for g in sorted(_manip) if g in obj_gt0)
        grip = float(real[:k, 7].mean()) if real.shape[-1] > 7 else float("nan")
        # Place diagnostic: how far the carried object is from its seat (why placed()/release does or doesn't fire).
        seat_dbg = ""
        if stage.gripper == "place" and stage.payload and stage.place_target:
            _ext = {o.name: o.extents for o in grounding.objects}
            _pp, _sp = world.object_pose(stage.payload)[0], world.object_pose(stage.place_target)[0]
            _seat_z = float(_sp[2]) + float(_ext.get(stage.place_target, (0, 0, 0.05))[2]) \
                + float(_ext.get(stage.payload, (0, 0, 0.03))[2])
            seat_dbg = (f" seat[dz={(float(_pp[2]) - _seat_z) * 100:+.1f}cm "
                        f"xy={float(np.linalg.norm(_pp[:2] - _sp[:2])) * 100:.0f}cm]")
        print(f"[base]   chunk {chunk}: stage={stage.name} dist={d_now:.3f}m grip={grip:.2f}/{gopen} "
              f"hold={hold}{gz}{seat_dbg} flags={flags}", flush=True)

    _report(env, grounding, obj_gt0, q_hist, dist_hist, getattr(args, "task", None))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[base] DONE -> {out} ({len(frames)} frames)", flush=True)

    if ref_rec is not None:
        # On-policy reference-distillation labels for score-space PPS (paper Eq. 5). Arrays: x/score [M,H,8];
        # it/num_iters/chunk [M]; per-chunk obs [C]: joint, eef, rgb, plus the ctx geometry the score reads
        # (obs_target [C,3], obs_obj_pos/ext [C,K,3], obs_grasp_idx/stage [C], names in obj_names).
        os.makedirs(os.path.dirname(os.path.abspath(args.record_ref)) or ".", exist_ok=True)
        snaps = [o["ctx"] for o in ref_obs]
        save_kw = dict(
            x=np.stack([r["x"][0] for r in ref_rec]),
            score=np.stack([r["score"][0] for r in ref_rec]),
            it=np.array([r["it"] for r in ref_rec], dtype=np.int32),
            num_iters=np.array([r["N"] for r in ref_rec], dtype=np.int32),
            chunk=np.array([r["chunk"] for r in ref_rec], dtype=np.int32),
            obs_joint=np.stack([o["joint_pos"] for o in ref_obs]),
            obs_eef=np.stack([o["eef_pos"] for o in ref_obs]),
            obs_rgb=np.stack([o["rgb"] for o in ref_obs]),
            obs_chunk=np.array([o["chunk"] for o in ref_obs], dtype=np.int32),
            obs_target=np.stack([s["target"] for s in snaps]),
            obs_obj_pos=np.stack([s["obj_pos"] for s in snaps]),
            obs_obj_ext=np.stack([s["obj_ext"] for s in snaps]),
            obs_grasp_idx=np.array([s["grasp_idx"] for s in snaps], dtype=np.int32),
            obs_stage=np.array([s["stage"] for s in snaps], dtype=np.int32),
            obj_names=np.array(obj_names),
        )
        if all("keypoints" in s for s in snaps) and len({s["keypoints"].shape for s in snaps}) == 1:
            save_kw["obs_kp"] = np.stack([s["keypoints"] for s in snaps])
        np.savez_compressed(args.record_ref, **save_kw)
        print(f"[base] recorded {len(ref_rec)} on-policy reference labels + {len(ref_obs)} obs "
              f"(ctx: {obj_names}{', +keypoints' if 'obs_kp' in save_kw else ''}) "
              f"-> {args.record_ref}.npz", flush=True)


def _report(env, grounding, obj_gt0, q_hist, dist_hist, task_key=None):
    """Print motion smoothness, reach, scene disturbance, grasp lift, and the task's own success verdict.

    The verdict is the task's ``task_done_<task>`` termination -- the same predicate the eval harness scores.
    Reads ground-truth poses (never the perception centroids, which carry a top-surface bias) so both ends of
    every delta share one authoritative frame.
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
    for g in sorted({s.grasp_obj for s in grounding.stages if s.grasp_obj}):
        dz = (env.object_pose(g)[0][2] - obj_gt0[g][2]) * 100
        print(f"[base] grasp_obj {g}: dz={dz:+.1f}cm grasped={'Y' if dz > 3.0 else 'N'}", flush=True)
    if task_key is not None:
        report_task_success(env.env, task_key)
