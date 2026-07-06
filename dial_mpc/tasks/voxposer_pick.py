"""V1: VoxPoser-style affordance REGION as the DIAL cost, grasping the mug. The affordance map is a
ball region on the GT handle (the 'fake' analog of VoxPoser's set_voxel_by_radius); its EDT-smoothed
cost field (voxposer_bridge.build_costmap) is sampled by DIAL at the candidate TCPs. Tests whether a
REGION grasps better than ReKep's single keypoint. Records camera | affordance-map + DIAL-path panel.

    python -m dial_mpc.main --task voxposer_pick ...
"""
import argparse
import json
import os

NAME = "voxposer_pick"


def add_args(ap):
    ap.add_argument("--task", type=str, default="Isaac-Lift-Mug-Franka-v0")
    ap.add_argument("--exp_name", type=str, default="lift_mug_vox")
    ap.add_argument("--obj_z", type=float, default=0.12)
    ap.add_argument("--obj_yaw", type=float, default=180.0, help="handle toward robot (easy grasp; GT region)")
    ap.add_argument("--handle", type=float, nargs=3, default=[0.04, 0.035, 0.0],
                    help="handle keypoint offset in the object frame (m); region is centered here")
    ap.add_argument("--radius", type=float, default=0.03, help="affordance region radius (m)")
    ap.add_argument("--map_size", type=int, default=100)
    ap.add_argument("--load_map", type=str, default=None,
                    help="dir with maps_0.npz + bounds.json from the VoxPoser front-end (V2); else GT region (V1)")
    ap.add_argument("--hover", type=float, default=0.10, help="fixed-reach hover height above the target before descent")
    ap.add_argument("--steps_hover", type=int, default=80)
    ap.add_argument("--grasp_cost", action=argparse.BooleanOptionalAction, default=True,
                    help="APPROACH with grasp-yaw + straddle terms (--no_grasp_cost for the plain-reach A/B)")
    ap.add_argument("--obj_r", type=float, default=0.012, help="object half-extent for the straddle term (m)")
    ap.add_argument("--steps_approach", type=int, default=110)
    ap.add_argument("--close_steps", type=int, default=25)
    ap.add_argument("--steps_lift", type=int, default=60)
    ap.add_argument("--lift_h", type=float, default=0.12)
    ap.add_argument("--tol", type=float, default=0.02)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from rekep.video import write_video_h264
    from sim_common.isaac_env import LiftEnv, GRASP_OFFSET
    from dial_mpc.sampler import make_accel_sampler
    from dial_mpc.costs import make_rekep_cost
    from dial_mpc.voxposer_bridge import (affordance_region, build_costmap, make_voxposer_cost,
                                         make_voxposer_grasp_cost)

    DEV = "cuda:0"
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DOWN = np.array([0.0, 0.0, -1.0])
    BMIN = [0.20, -0.40, 0.0]
    BMAX = [0.90, 0.40, 0.60]
    frames = []

    def fixed_reach(tcp, keypoints):
        return torch.linalg.norm(tcp - keypoints[0], dim=-1)

    def make_panel(region_pts):
        """Side panel renderer: affordance region (green) + TCP (red) + DIAL mean path (orange)."""
        sub = region_pts[:: max(1, len(region_pts) // 400)]
        fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        def render(tcp, path_world, label):
            ax.cla()
            ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c="tab:green", s=6, alpha=0.25)
            ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], c="red", s=45)
            if path_world is not None:
                ax.plot(path_world[:, 0], path_world[:, 1], path_world[:, 2], c="orange", lw=2)
            ax.set_xlim(BMIN[0], BMAX[0]); ax.set_ylim(BMIN[1], BMAX[1]); ax.set_zlim(BMIN[2], BMAX[2])
            ax.set_title(f"affordance region + DIAL path\n{label}", fontsize=10)
            ax.view_init(elev=22, azim=-60)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            return np.ascontiguousarray(img)

        return render

    E = LiftEnv(device=DEV, task=args.task, obj_z=args.obj_z, obj_yaw=args.obj_yaw)
    R_down = Rot.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
    a_local = R_down.T @ DOWN

    if args.load_map:  # V2: affordance authored by the VoxPoser LMP
        d = np.load(os.path.join(args.load_map, "maps_0.npz"), allow_pickle=True)
        grid = np.asarray(d["affordance"], dtype=np.float64)
        with open(os.path.join(args.load_map, "bounds.json"), encoding="utf-8") as f:
            b = json.load(f)
        BMIN, BMAX = b["min"], b["max"]
        D = grid.shape[0]
        vidx = np.argwhere(grid > 0)
        region_pts = vidx / (D - 1) * (np.array(BMAX) - np.array(BMIN)) + np.array(BMIN)
        target = region_pts.mean(0)
        costmap = build_costmap(grid)
        print(f"[vox-pick] load_map={args.load_map} target_voxels={len(vidx)} "
              f"target={target.round(3).tolist()} bounds={np.round(BMIN,3).tolist()}..{np.round(BMAX,3).tolist()}",
              flush=True)
    else:  # V1: hand-built GT region
        target = E.keypoints_world([args.handle])[0]
        grid, region_pts = affordance_region(target, args.radius, BMIN, BMAX, args.map_size)
        costmap = build_costmap(grid)
        print(f"[vox-pick] fake target={target.round(3).tolist()} region_voxels={int(grid.sum())} "
              f"bounds={BMIN}..{BMAX}", flush=True)

    cost_probe = make_rekep_cost(E.fk, fixed_reach, a_local, grasp_offset=GRASP_OFFSET, device=DEV)
    # APPROACH cost: affordance reach + (optionally) the grasp-yaw + straddle terms so the gripper encloses.
    if args.grasp_cost:
        cost_approach = make_voxposer_grasp_cost(E.fk, costmap, BMIN, BMAX, a_local, target,
                                                 grasp_offset=GRASP_OFFSET, obj_r=args.obj_r, device=DEV)
        print("[vox-pick] APPROACH cost = affordance reach + grasp-yaw + straddle", flush=True)
    else:
        cost_approach = make_voxposer_cost(E.fk, costmap, BMIN, BMAX, a_local, grasp_offset=GRASP_OFFSET,
                                           device=DEV)
        print("[vox-pick] APPROACH cost = affordance reach only (--no_grasp_cost A/B)", flush=True)

    def sampler(cost):
        return make_accel_sampler(cost, E.q_lo, E.q_hi, E.dt, H=args.horizon,
                                  num_samples=args.num_samples, iterations=args.iterations,
                                  accel_std=args.accel_std, accel_clip=args.accel_clip,
                                  temperature=args.temperature, device=DEV)

    plan_vox, plan_probe = sampler(cost_approach), sampler(cost_probe)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    exec_knot = min(args.exec_knot, args.horizon - 1)
    panel = make_panel(region_pts)

    def record(label, path_world=None):
        img = np.ascontiguousarray(E.rgb())
        cv2.putText(img, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
        side = cv2.cvtColor(panel(E.tcp(), path_world, label), cv2.COLOR_RGB2BGR)  # match camera order
        frames.append(np.hstack([img, side]))

    def path_of(q_best):
        p, _ = E.fk.grasp_point(q_best, GRASP_OFFSET)
        return p.detach().cpu().numpy()

    def run(plan, ctx, grip_open, target, max_steps, label):
        mean_a = torch.zeros(args.horizon, 7, device=DEV)
        qd0 = torch.zeros(7, device=DEV)
        for step in range(max_steps):
            mean_a, q_traj, score, ess = plan(mean_a, E.q0(), qd0, ctx, gen)
            E.apply_arm(q_traj[exec_knot], grip_open)
            record(f"{label} {step}: cost={score:.1f} ess={ess:.0f}", path_of(q_traj))
            mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=DEV)], dim=0)
            err = float(np.linalg.norm(E.tcp() - target))
            if err < args.tol:
                print(f"[vox-pick] {label}: reached (tcp_err={err*100:.2f}cm) at step {step}", flush=True)
                return
            if step % 20 == 0:
                print(f"[vox-pick] {label} {step:3d}: tcp_err={err*100:.2f}cm cost={score:.2f} ess={ess:.0f}",
                      flush=True)
        print(f"[vox-pick] {label}: step cap (tcp_err={float(np.linalg.norm(E.tcp()-target))*100:.2f}cm)",
              flush=True)

    def hold(grip_open, n, label):
        q = E.q0()
        for _ in range(n):
            E.apply_arm(q, grip_open)
            record(label)

    # --- HOVER (fixed-reach into bounds), DESCEND the affordance field, CLOSE, LIFT probe ---
    hover_target = target + np.array([0.0, 0.0, args.hover])
    run(plan_probe, torch.tensor([hover_target], device=DEV, dtype=torch.float32), True,
        hover_target, args.steps_hover, "HOVER")
    run(plan_vox, None, True, target, args.steps_approach, "APPROACH")
    hold(False, args.close_steps, "CLOSE")
    grasp_err = float(np.linalg.norm(E.tcp() - target))
    obj_before = E.object_pose()[0]
    lift_target = target + np.array([0.0, 0.0, args.lift_h])
    lift_t = torch.tensor([lift_target], device=DEV, dtype=torch.float32)
    run(plan_probe, lift_t, False, lift_target, args.steps_lift, "LIFT")
    hold(False, 12, "LIFT-hold")
    obj_after = E.object_pose()[0]

    print(f"[vox-pick] RESULT target={target.round(3).tolist()} grasp_tcp_err={grasp_err*100:.2f}cm "
          f"(reached={grasp_err < 0.04}) | lift: obj_z {obj_before[2]:.3f}->{obj_after[2]:.3f} "
          f"(+{(obj_after[2]-obj_before[2])*100:.1f}cm, held={obj_after[2] > obj_before[2]+0.05})", flush=True)

    out = os.path.join(_REPO, "results", "vlm_mpc", f"voxposer_pick_{args.exp_name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[vox-pick] DONE -> {out} ({len(frames)} frames)", flush=True)
