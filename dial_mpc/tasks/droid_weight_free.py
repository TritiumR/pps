"""Full weight task on the scaffold-free pipeline: grasp pear -> place on scale -> grasp apple -> place.

Generalizes the cost+proximity-gripper grasp to an N-stage machine. ONE continuous DIAL loop; the only
scripting is the coarse task-stage index advancing on gripper events (the collaborator's pickup->placement
staging), NOT a fine HOVER/DESCEND/CLOSE/LIFT scaffold. Per stage:
  pickup: target = object GT-mask centroid, cost = make_grasp_cost, gripper PROXIMITY-CLOSE + latch.
  place:  target = scale placement point, cost = make_rekep_cost (reach), gripper PROXIMITY-OPEN + latch.

Grounding is GT object selection (the object's GT instance mask) + real table_cam depth centroid -- no
VLM. The carry HOLD is contact-limited (the residual's job), so expect mid-carry slips; this validates
the *pipeline structure*, not task success.

    python -m dial_mpc.main --task droid_weight_free ...
"""
import os

NAME = "droid_weight_free"


def add_args(ap):
    ap.add_argument("--exp_name", type=str, default="weight_task_full")
    ap.add_argument("--close_thresh", type=float, default=0.012, help="proximity radius to close on an object")
    ap.add_argument("--open_thresh", type=float, default=0.03, help="proximity radius to release over the scale")
    ap.add_argument("--settle_steps", type=int, default=15, help="steps to hold a gripper transition before advancing")
    ap.add_argument("--stage_budget", type=int, default=90, help="max steps in a stage before force-advancing")
    ap.add_argument("--obj_r", type=float, default=0.035, help="object half-extent for the straddle term")
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
    from sim_common.world import GTWorld
    from rekep.utils import load_default_config
    from rekep.video import write_video_h264
    from sim_common.envs.droid import DroidEnv, ROBOTIQ_GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_grasp_cost, make_rekep_cost, fixed_reach
    from sim_common import overlay
    from dial_mpc.control import tt as _tt

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
    frames = []

    # Scale placement geometry -- measured scene constants (from the collaborator's sim_free_mpc; verified
    # against this scene's `scale` GT). place_target = scale_root + center_off + [0,0, top + half + clear].
    SCALE_CENTER_OFF = np.array([-0.0470425, 0.0, 0.0272255])
    SCALE_TOP_Z = 0.0523800
    PLACE_CLEAR_Z = 0.0150000
    HALF_H = {"pear": 0.0620635, "apple": 0.0376650}

    def tt(v):
        return _tt(v, DEV)

    def gt_centroid(env, grounded, name):
        """Real table_cam depth centroid of an object's GT instance mask (GT selection, perceived position).

        `env` is the DroidEnv; the IsaacLab scene is `env.env.scene`.
        """
        rel = re.sub(r"^/World/envs/env_[^/]*/", "", env.env.scene[name].cfg.prim_path)
        ids = [i for i, prim in grounded["id_to_prim"].items() if rel and rel in prim]
        sel = np.isin(grounded["masks"], ids) & np.isfinite(grounded["points"]).all(axis=-1)
        if int(sel.sum()) == 0:
            return None
        return grounded["points"][sel].mean(axis=0)

    E = DroidEnv(device=DEV)
    a_local = E.a_local
    scale_root = E.object_pose("scale")[0]
    print("[weight] grounding...", flush=True)
    grounded = grounding.propose_keypoints(E.cam, E.env, load_default_config())
    keypoints = grounded["keypoints"]
    print(f"[weight] propose_keypoints done ({len(keypoints)} kp)", flush=True)
    if len(keypoints) == 0:
        raise SystemExit("[weight] no keypoints proposed")
    tracker = KeypointTracker(GTWorld(E.env), keypoints)
    print("[weight] tracker built", flush=True)

    # GT-mask centroids for the manipulated objects; scale placement targets from measured geometry.
    obj_centroid = {}
    for name in ("pear", "apple"):
        c = gt_centroid(E, grounded, name)
        if c is None:
            raise SystemExit(f"[weight] no masked depth points for {name}")
        gt = E.object_pose(name)[0]
        obj_centroid[name] = c
        print(f"[weight] {name} centroid={np.round(c,3).tolist()} (GT {np.round(gt,3).tolist()}, "
              f"err={np.linalg.norm(c-gt)*100:.1f}cm)", flush=True)

    def place_target(name):
        return scale_root + SCALE_CENTER_OFF + np.array([0.0, 0.0, SCALE_TOP_Z + HALF_H[name] + PLACE_CLEAR_Z])

    place_pear, place_apple = place_target("pear"), place_target("apple")
    print(f"[weight] scale_root={np.round(scale_root,3).tolist()} place_pear={np.round(place_pear,3).tolist()} "
          f"place_apple={np.round(place_apple,3).tolist()}", flush=True)

    z_floor = float(min(obj_centroid["pear"][2], obj_centroid["apple"][2])) - 0.05
    cost_grasp = make_grasp_cost(E.fk, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET, r_cube=args.obj_r,
                                 open_half=0.0425, finger_r=0.012, z_floor=z_floor, device=DEV)
    cost_reach = make_rekep_cost(E.fk, fixed_reach, a_local, grasp_offset=ROBOTIQ_GRASP_OFFSET,
                                 z_floor=z_floor, device=DEV)

    def sampler(cost):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=args.horizon, num_samples=args.num_samples,
                                  iterations=args.iterations, accel_std=args.accel_std,
                                  accel_clip=args.accel_clip, temperature=args.temperature, device=DEV)

    plan_grasp, plan_reach = sampler(cost_grasp), sampler(cost_reach)

    # The 4-stage task machine (coarse staging only; costs/gripper are state-driven).
    stages = [
        {"name": "grasp_pear", "kind": "pickup", "target": obj_centroid["pear"], "obj": "pear"},
        {"name": "place_pear", "kind": "place", "target": place_pear, "obj": "pear"},
        {"name": "grasp_apple", "kind": "pickup", "target": obj_centroid["apple"], "obj": "apple"},
        {"name": "place_apple", "kind": "place", "target": place_apple, "obj": "apple"},
    ]

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)
    mean_a = torch.zeros(args.horizon, 7, device=DEV)
    qd0 = torch.zeros(7, device=DEV)

    stage_i = 0
    grip_closed = False           # latched gripper state (open at start)
    event_step = None             # when this stage's gripper transition fired
    stage_start = 0
    max_steps = len(stages) * args.stage_budget
    z0 = {n: E.object_pose(n)[0][2] for n in ("pear", "apple")}

    for step in range(max_steps):
        stage = stages[stage_i]
        target = stage["target"]
        d = float(np.linalg.norm(E.tcp() - target))
        if stage["kind"] == "pickup":
            mean_a, q_traj, score, ess = plan_grasp(mean_a, E.q0(), qd0, (tt(target), tt(target)), gen)
            if not grip_closed and d < args.close_thresh:           # proximity close + latch (commit)
                grip_closed, event_step = True, step
                print(f"[weight] {stage['name']}: CLOSE at step {step}, d={d*100:.2f}cm", flush=True)
            done = grip_closed and event_step is not None and step - event_step >= args.settle_steps
        else:  # place
            mean_a, q_traj, score, ess = plan_reach(mean_a, E.q0(), qd0, tt(target).unsqueeze(0), gen)
            if grip_closed and d < args.open_thresh:                # proximity open + latch (release)
                grip_closed, event_step = False, step
                print(f"[weight] {stage['name']}: RELEASE at step {step}, d={d*100:.2f}cm", flush=True)
            done = not grip_closed and event_step is not None and step - event_step >= args.settle_steps

        E.apply_arm(q_traj[exec_knot], grip_open=not grip_closed)
        lbl = f"{stage['name']} {step}: d={d*100:.1f}cm grip={'C' if grip_closed else 'O'}"
        frames.append(overlay.camera_overlay_frame(E.cam, tracker, ["DIAL weight task (scaffold-free)", lbl]))
        mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)

        timed_out = step - stage_start >= args.stage_budget
        if (done or timed_out) and stage_i < len(stages) - 1:
            if timed_out and not done:
                print(f"[weight] {stage['name']}: stage timeout at step {step} (advancing)", flush=True)
            stage_i += 1
            event_step = None
            stage_start = step + 1
        elif (done or timed_out) and stage_i == len(stages) - 1:
            break
        if step % 20 == 0:
            print(f"[weight] {step:3d}: {stage['name']} d={d*100:.2f}cm grip={'C' if grip_closed else 'O'} "
                  f"cost={score:.2f}", flush=True)

    # Faithfulness readout: where did each object end up vs its placement target?
    print(f"[weight] --- RESULT (stage reached: {stages[stage_i]['name']}) ---", flush=True)
    for name, ptgt in (("pear", place_pear), ("apple", place_apple)):
        p = E.object_pose(name)[0]
        on = bool(p[2] > z0[name] + 0.03 and np.linalg.norm(p[:2] - ptgt[:2]) < 0.08)
        print(f"[weight] {name}: final={np.round(p,3).tolist()} dz={(p[2]-z0[name])*100:+.1f}cm "
              f"xy_to_scale={np.linalg.norm(p[:2]-ptgt[:2])*100:.1f}cm on_scale={on}", flush=True)

    out = os.path.join(_REPO, "results", "vlm_mpc", "droid_weight", f"{args.exp_name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[weight] DONE -> {out} ({len(frames)} frames)", flush=True)
