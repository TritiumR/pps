"""Run the SimFreeMPC engine on the IsaacLab weight task with a checkpoint-free decode.

The planner runs unchanged from sim_free_mpc (DIAL sampler, reverse update, knot interpolation, builtin
PriorityStateCost); the model-to-action decode that normally needs the pi0.5 checkpoint is served by a
mock policy carrying only the decode norm-stats. Exercises the position-space recipe end to end: B-spline
knot smoothing, an optional consistency term, warm-start, a joint-delta rate limit, and a NaN guard. Phase
flags come from the env's subtask_terms group.

    python vlm_base/diagnostics/sim_free_mbd.py --real_stats --interpolate --guard --w_consist 30
"""
import os
import random
import sys

import numpy as np
import torch

NAME = "sim_free_mbd"


def add_args(ap):
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--mode", type=str, default="denoise", choices=["denoise", "mean"])
    ap.add_argument("--update", type=str, default="score_space", choices=["mbd_score", "score_space", "ddim", "flow"])
    ap.add_argument("--cost_style", type=str, default="priority",
                    choices=["priority", "explore", "ref_style", "grasp_flow"],
                    help="builtin sim_free cost (grasp_flow = the full grasp+lift+place weight cost)")
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
    ap.add_argument("--real_stats", action="store_true", help="exact pi05_droid_jointpos quantile decode")
    ap.add_argument("--action_std", type=float, default=0.1, help="stand-in action-norm std when --real_stats is off")
    ap.add_argument("--interpolate", action="store_true", help="coarse-knot horizon interpolation")
    ap.add_argument("--basis", type=str, default="bspline", choices=["linear", "cubic", "bspline", "rbf"],
                    help="knot-interpolation basis (bspline: approximating C2, no overshoot)")
    ap.add_argument("--knots", type=int, default=4)
    ap.add_argument("--arm_only_smooth", action="store_true",
                    help="smooth the arm control points but keep the gripper channel sharp (full-res close)")
    ap.add_argument("--grip_scale", type=float, default=1.0,
                    help="scale grasp_flow gripper weights (close/lift/place) for a firmer hold; 1.0 = her defaults")
    ap.add_argument("--w_consist", type=float, default=0.0, help="consistency weight toward the previous plan; 0 disables")
    ap.add_argument("--latch", action="store_true", help="monotonic phase flags (a stationary target per phase)")
    ap.add_argument("--guard", action="store_true", help="NaN-guard the cost before the sampler softmax")
    ap.add_argument("--max_chunks", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)


def run(args):
    # Repo-local + Isaac imports: need the bootstrapped sys.path + a booted app (see runtime.run_standalone).
    from vlm_base import sim_free_core as core
    from vlm_base import metrics
    from vlm_base.cost_terms import TERMS, CostInputs
    from sim_common.envs.droid import DroidEnv
    from sim_common import overlay
    from rekep.video import write_video_h264
    from sim_free_mpc.costs_ref_style import _WEIGHT_OBJECT_HORIZONTAL_RADIUS

    DEV = "cuda:0"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    random.seed(args.seed)          # match eval_steering: seed all three RNGs before the env reset
    np.random.seed(args.seed)       # so the scene (object poses) is deterministic, like her seed-1 run
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
            inputs = CostInputs(real_actions, ee_pos, ee_quat, context, {}, None)
            return base + self._weight * TERMS["consistency"](inputs)

    policy, state_stats = core.build_policy(args.real_stats, args.action_std)
    mpc, cfg = core.build_mpc(policy, num_samples=args.num_samples, iterations=args.iterations,
                              noise=args.noise, temperature=args.temperature, task_name="weight",
                              cost_style=args.cost_style, joint_delta_clip=args.joint_delta_clip,
                              interpolate=(True if args.arm_only_smooth else args.interpolate))
    if args.arm_only_smooth:                       # arm knots smoothed, gripper channel kept sharp
        core.apply_arm_only_smoothing(mpc)
    else:
        core.apply_horizon_basis(mpc, args.basis, args.knots)
    if args.cost_style == "grasp_flow" and args.grip_scale != 1.0:   # firmer hold: scale the gripper weights
        import dataclasses
        from sim_free_mpc.costs_grasp_flow import GraspFlowStateCost, GraspFlowCostWeights
        bw = GraspFlowCostWeights()
        w = dataclasses.replace(bw, close_gripper=bw.close_gripper * args.grip_scale,
                                lift_gripper=bw.lift_gripper * args.grip_scale,
                                place_gripper=bw.place_gripper * args.grip_scale)
        mpc.cost = GraspFlowStateCost("weight", weights=w)
        print(f"[sim-free-mbd] grip_scale={args.grip_scale}: close={w.close_gripper:.1f} "
              f"lift={w.lift_gripper:.1f} place={w.place_gripper:.1f}", flush=True)
    grasp_cost = mpc.cost if args.cost_style == "grasp_flow" else None  # raw cost: read last_stage after each plan
    if args.w_consist > 0:
        mpc.cost = _ConsistencyCost(mpc.cost, args.w_consist)
    if args.guard:
        mpc.cost = core.guard_cost(mpc.cost)

    E = DroidEnv(device=DEV)
    scene_objects = list(getattr(E.env.scene, "rigid_objects", {}) or {})
    root_pos = E.robot.data.body_pos_w[0, E.l0].detach()
    root_quat = E.robot.data.body_quat_w[0, E.l0].detach()
    print(f"[sim-free-mbd] cost={args.cost_style} mode={args.mode} update={args.update} init={args.init} "
          f"basis={args.basis} w_consist={args.w_consist} scene={scene_objects}", flush=True)

    latched = {}

    def read_flags():
        raw = core.read_subtask_flags(E)
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
        flags = read_flags()
        ctx = {"objects": objects, "joint_pos": E.q0(), "robot_root_pos": root_pos,
               "robot_root_quat": root_quat, "plan_ref": plan_ref["v"], "subtasks": flags,
               "eef_pos": torch.as_tensor(E.tcp(), device=DEV, dtype=torch.float32)}
        ctx.update(flags)   # also expose flags at top level (PriorityStateCost / _flag fallback)
        return ctx

    def readout_target(flags):
        """A live target for the distance readout only (grasp_flow exposes no single cost target)."""
        for flag, name in (("grasp_pear", "pear"), ("pear_on_scale", "scale"), ("grasp_apple", "apple")):
            if not flags.get(flag, False) and name in scene_objects:
                try:
                    return E.object_pose(name)[0]
                except Exception:
                    break
        return E.tcp()

    def on_scale(name):
        try:
            obj, _ = E.object_pose(name)
            scale, _ = E.object_pose("scale")
        except Exception:
            return False
        return float(np.linalg.norm(obj[:2] - scale[:2])) < 0.10

    def close_gate(tcp, obj_pos, radius):
        """Grasp-stage close_gate for the executed tip; mirrors GraspFlowStateCost._grasp_terms.

        gate = exp(-center_excess^2 / xy_scale^2 - z_excess^2 / z_scale^2), with center_err the 3D
        tip-to-object distance and z_off the vertical offset. Returns (gate, center_err, z_off).
        """
        d = np.asarray(tcp, dtype=np.float64) - np.asarray(obj_pos, dtype=np.float64)
        center_err = float(np.linalg.norm(d))
        z_off = float(abs(d[2]))
        center_radius = 0.40 * radius            # _GRASP_FLOW_CENTER_REGION_RADIUS_SCALE
        center_excess = max(center_err - center_radius, 0.0)
        z_excess = max(z_off - 0.025, 0.0)       # gripper_close_z_scale
        xy_scale = max(center_radius, 1e-3)
        gate = float(np.exp(-center_excess ** 2 / xy_scale ** 2 - z_excess ** 2 / 0.025 ** 2))
        return gate, center_err, z_off

    H = args.horizon
    k = min(args.exec_knot, H)
    frames, q_hist, dist_hist = [], [], []
    obj_p0 = {n: E.object_pose(n)[0].copy() for n in scene_objects}
    x_carry = None
    for chunk in range(args.max_chunks):
        ctx = build_context()
        pin = core.policy_inputs(E, state_stats, args.real_stats)
        try:
            target = mpc.cost.target(ctx, DEV, torch.float32).detach().cpu().numpy()
        except AttributeError:                     # grasp_flow etc. expose no single target
            target = np.asarray(readout_target(ctx.get("subtasks", {})), dtype=np.float32)
        it_start = 0
        if args.init == "warm" and x_carry is not None:
            if args.mode == "denoise":  # SDEdit-style: forward-diffuse the carry, run only the last warm_steps
                it_start = max(0, args.denoise_iters - args.warm_steps)
                x_init = core.sdedit_warm_start(x_carry, it_start, args.denoise_iters,
                                                cfg.ddim_num_train_timesteps, H, DEV)
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
        gate_best = None   # closest approach to the pick object: (center_err, gate, z_off, gripper, pick)
        for t in range(min(args.exec_knot, H)):
            action = real[t]
            E.apply_arm(action[:7], grip_open=float(action[7]) < 0.5)  # gripper from the decoded channel
            q_hist.append(E.q0().detach().cpu().numpy())
            dist = float(np.linalg.norm(E.tcp() - target))
            dist_hist.append(dist)
            flags = read_flags()
            pick = "pear" if not flags.get("grasp_pear", False) else "apple"   # current grasp-stage target
            if pick in scene_objects:
                gate, cerr, zoff = close_gate(E.tcp(), E.object_pose(pick)[0],
                                              _WEIGHT_OBJECT_HORIZONTAL_RADIUS.get(pick, 0.05))
                if gate_best is None or cerr < gate_best[0]:
                    gate_best = (cerr, gate, zoff, float(action[7]), pick)
            frames.append(overlay.plain_frame(
                E.rgb(),
                f"SimFreeMPC[{args.mode}/{args.update}/{args.init}] ch{chunk}.{t} d={dist:.2f}m "
                f"pear={int(flags.get('grasp_pear', 0))} onScale={int(flags.get('pear_on_scale', 0))} "
                f"apple={int(flags.get('grasp_apple', 0))}"))
        g = real[:min(args.exec_knot, H), 7]   # decoded gripper channel over the executed steps
        gate_str = ""
        if gate_best is not None:   # localize the grasp-stall: low gate = arm imprecise, high gate + low grip = weak weight
            cerr, gate, zoff, grip_at, pick = gate_best
            gate_str = (f" | {pick}@closest cerr={cerr * 1000:.0f}mm zoff={zoff * 1000:.0f}mm "
                        f"gate={gate:.2f} grip={grip_at:.2f}")
        stage = getattr(grasp_cost, "last_stage", "?") if grasp_cost is not None else "?"
        pear_z = float(E.object_pose("pear")[0][2]) if "pear" in scene_objects else float("nan")
        print(f"[sim-free-mbd]   chunk {chunk}: dist={dist:.3f}m stage={stage} pear_z={pear_z:.3f} "
              f"grip[mn/mx]={float(g.min()):.2f}/{float(g.max()):.2f}{gate_str} flags={flags}", flush=True)
        if on_scale("pear") and on_scale("apple"):
            print(f"[sim-free-mbd]   both objects on scale at chunk {chunk}", flush=True)
            break

    print("[sim-free-mbd] --- RESULT ---", flush=True)
    sm = metrics.motion_smoothness(q_hist)
    if sm:
        print(f"[sim-free-mbd] smoothness(cos)={sm['cos']:.3f} jerk(2nd-diff)={sm['jerk'] * 1e3:.2f}e-3 "
              f"speed={sm['speed'] * 1e3:.1f}e-3 over {sm['n']} steps", flush=True)
        print("[sim-free-mbd] per-joint TV(rad): " + " ".join(f"j{i}={t:.2f}" for i, t in enumerate(sm['tv'])), flush=True)
    rs = metrics.reach_stats(dist_hist)
    if rs:
        print(f"[sim-free-mbd] reach(TCP->target): min={rs['min']:.3f}m final={rs['final']:.3f}m", flush=True)
    obstacles = [n for n in ("board", "mango", "cabbage") if n in scene_objects]
    moved, tot, mx = metrics.scene_disturbance(E, obj_p0, obstacles)
    if moved:
        print(f"[sim-free-mbd] scene_disturbance({obstacles}): sum={tot * 100:.1f}cm max={mx * 100:.1f}cm", flush=True)

    out_dir = os.path.join(repo, "results", "vlm_mpc", "sim_free_mbd")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[sim-free-mbd] DONE -> {out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from sim_common import runtime

    runtime.run_standalone(add_args, run, "sim_free_mbd: run the SimFreeMPC engine on the weight task")
