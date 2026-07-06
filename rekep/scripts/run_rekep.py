"""Run ReKep on an IsaacLab task: propose keypoints, generate GPT-4o constraints, and
-- unless ``--plan-only`` -- solve each stage's subgoal pose and drive the arm there.

Per stage the solver (rekep.solvers) optimizes the next end-effector pose from the relational
constraints, and the arm reaches it via IK-Rel deltas, grasping/releasing per the metadata;
the rollout is recorded with the live keypoint overlay. Execution is position-primary with a
small orientation gain (the solver omits ReKep's collision/IK terms -- IsaacLab's IK handles
reachability). ``--plan-only`` stops after grounding; ``--use_cached`` reuses a prior run.

    /isaac-sim/python.sh rekep/scripts/run_rekep.py --task Isaac-Tea-Droid-Visuomotor-IK-Rel-v0 --task_key tea
"""

import argparse
import json
import os
import sys

# Put the repo root + the bundled IsaacLab packages on sys.path (self-contained).
_REKEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rekep/ (this file is in rekep/scripts/)
_REPO_DIR = os.path.dirname(_REKEP_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _pkg_src = os.path.join(_ISAACLAB_DIR, "source", _pkg)
    if _pkg_src not in sys.path:
        sys.path.insert(0, _pkg_src)

from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401  (side-effect import)


def parse_args():
    parser = argparse.ArgumentParser(description="Ground + optionally roll out ReKep on an IsaacLab task.")
    parser.add_argument("--task", type=str, default="Isaac-Tea-Droid-Visuomotor-IK-Rel-v0")
    parser.add_argument("--task_key", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="config yaml (default: rekep/configs/default.yaml)")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--max_steps_per_stage", type=int, default=80)
    parser.add_argument("--grasp_steps", type=int, default=12)
    parser.add_argument("--pos_tol", type=float, default=0.03)
    parser.add_argument("--rot_gain", type=float, default=0.3)
    parser.add_argument("--grasp_rot_gain", type=float, default=1.0, help="orientation gain during grasp approach")
    parser.add_argument("--hover_h", type=float, default=0.12, help="pre-grasp hover height above keypoint (m)")
    parser.add_argument("--grasp_z", type=float, default=0.0, help="z offset at grasp (m, +up)")
    parser.add_argument("--lift_h", type=float, default=0.15, help="post-grasp lift height (m)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--use_cached", action="store_true", help="reuse keypoints/constraints from results/rekep/<key>")
    parser.add_argument("--plan-only", dest="plan_only", action="store_true",
                        help="stop after grounding (keypoints + constraints); skip the solve + rollout")
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()

with open(os.path.join(_REPO_DIR, "task_prompts.json"), "r", encoding="utf-8") as f:
    TASK_PROMPTS = json.load(f)
task_key = args.task_key or next((k for k in TASK_PROMPTS if k.lower() in args.task.lower()), None)
if task_key is None:
    raise SystemExit(f"Pass --task_key (one of {list(TASK_PROMPTS)})")
instruction = TASK_PROMPTS[task_key]["prompt"]
exp_name = args.exp_name or task_key
out_dir = os.path.join(_REPO_DIR, "results", "rekep", exp_name)
os.makedirs(out_dir, exist_ok=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imported after the sim app launches (gym-task registration + torch/CUDA setup need it).
import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import isaaclab_tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pot_scene_fix import seat_pot_lid

from rekep import grounding, isaaclab_helpers
from rekep.constraint_generation import ConstraintGenerator
from rekep.keypoint_tracking import KeypointTracker
from rekep.rekep_viz import camera_overlay_frame
from rekep.solvers import SubgoalSolver
from rekep.utils import get_callable_grasping_cost_fn, get_config, load_default_config, load_functions_from_txt
from rekep.video import write_video_h264

GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 1.0


def ee_pose7(env, env_index=0):
    ee = env.scene["ee_frame"]
    pos = ee.data.target_pos_w[env_index, 0].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = ee.data.target_quat_w[env_index, 0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return np.concatenate([pos, quat_xyzw])


def robot_base(env, env_index=0):
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_index].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = robot.data.root_quat_w[env_index].detach().cpu().numpy().astype(np.float64)
    rot = Rot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return pos, rot


def drive_to_pose(env, target_pose7, gripper_cmd, record_fn, max_steps, pos_tol, rot_gain):
    """Position-primary IK-Rel control toward target_pose7 (base-frame deltas)."""
    for _ in range(max_steps):
        cur = ee_pose7(env)
        world_pos_err = target_pose7[:3] - cur[:3]
        if np.linalg.norm(world_pos_err) < pos_tol:
            break
        base_pos, base_rot = robot_base(env)
        arm_cmd = np.zeros(6)
        arm_cmd[:3] = np.clip(base_rot.inv().apply(world_pos_err) / 0.5, -0.2, 0.2)
        if rot_gain > 0:
            world_rotvec = (Rot.from_quat(target_pose7[3:]) * Rot.from_quat(cur[3:]).inv()).as_rotvec()
            arm_cmd[3:] = np.clip(rot_gain * base_rot.inv().apply(world_rotvec) / 0.5, -0.2, 0.2)
        action = np.concatenate([arm_cmd, [gripper_cmd]]).astype(np.float32)
        env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
        record_fn()


def hold_gripper(env, gripper_cmd, record_fn, steps):
    """Hold the current EE pose while toggling the gripper (zero arm command)."""
    for _ in range(steps):
        action = np.concatenate([np.zeros(6), [gripper_cmd]]).astype(np.float32)
        env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
        record_fn()


def grasp(env, grasp_xyz, approach_quat, record_fn, hover, grasp_z, lift, max_steps, pos_tol, rot_gain):
    """Top-down grasp: hover above the point, descend (open), close, then lift."""
    g = np.asarray(grasp_xyz, dtype=np.float64)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + hover], approach_quat]),
                  GRIPPER_OPEN, record_fn, max_steps, pos_tol, rot_gain)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + grasp_z], approach_quat]),
                  GRIPPER_OPEN, record_fn, max_steps, pos_tol, rot_gain)
    hold_gripper(env, GRIPPER_CLOSE, record_fn, 60)
    drive_to_pose(env, np.concatenate([[g[0], g[1], g[2] + lift], approach_quat]),
                  GRIPPER_CLOSE, record_fn, max_steps, pos_tol, rot_gain)


def _stage_constraints(task_dir, stage, get_grasp_fn):
    subgoal = load_functions_from_txt(os.path.join(task_dir, f"stage{stage}_subgoal_constraints.txt"), get_grasp_fn)
    path = load_functions_from_txt(os.path.join(task_dir, f"stage{stage}_path_constraints.txt"), get_grasp_fn)
    return subgoal, path


def _run(env_name, env_cfg):
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()
    obs_dict, _ = env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(args.settle_steps):
        env.step(hold)
    # Reseat the pot lid (pot task only; no-op elsewhere) -- it spawns detached and must be
    # fixed after the scene settles.
    seat_pot_lid(env, hold)

    camera = env.scene["table_cam"]
    config = get_config(args.config) if args.config else load_default_config()

    # ---- ground: keypoints + constraints ----
    if args.use_cached and os.path.exists(os.path.join(out_dir, "keypoints.npy")):
        keypoints = np.load(os.path.join(out_dir, "keypoints.npy"))
        bounds = isaaclab_helpers.workspace_bounds_from_scene(env)
        print(f"[rekep] using cached keypoints ({len(keypoints)})")
    else:
        grounded = grounding.propose_keypoints(camera, env, config)
        keypoints, bounds = grounded["keypoints"], grounded["bounds"]
        cv2.imwrite(os.path.join(out_dir, "rgb.png"), grounded["rgb"][..., ::-1])
        cv2.imwrite(os.path.join(out_dir, "keypoints.png"), grounded["projected"][..., ::-1])
        np.save(os.path.join(out_dir, "keypoints.npy"), keypoints)
        with open(os.path.join(out_dir, "id_to_prim.json"), "w", encoding="utf-8") as f:
            json.dump(grounded["id_to_prim"], f, indent=2)
        ConstraintGenerator(config["constraint_generator"]).generate(
            grounded["projected"], instruction,
            metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)}, task_dir=out_dir)
    print(f"[rekep] {len(keypoints)} keypoints")
    if args.plan_only or len(keypoints) == 0:
        print(f"[rekep] DONE (plan-only) -> {out_dir}" if args.plan_only else "[rekep] no keypoints; aborting")
        env.close()
        return

    with open(os.path.join(out_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    num_stages = metadata["num_stages"]
    grasp_keypoints = metadata["grasp_keypoints"]
    release_keypoints = metadata["release_keypoints"]

    tracker = KeypointTracker(env, keypoints)
    bounds_min, bounds_max = bounds
    solver_cfg = {"bounds_min": bounds_min.tolist(), "bounds_max": bounds_max.tolist(),
                  **config["subgoal_solver"]}
    solver = SubgoalSolver(solver_cfg)

    frames = []
    grasped_body = None
    gripper_cmd = GRIPPER_OPEN
    ee0 = ee_pose7(env)
    top_down_quat = ee0[3:]   # reset orientation (gripper z-axis points world-down)
    _, base0_rot = robot_base(env)
    debug = {"action_space": str(env.action_space.shape), "num_keypoints": int(len(keypoints)),
             "grasp_keypoints": grasp_keypoints, "owners": tracker.owners,
             "ee_init_pos": ee0[:3].tolist(),
             "ee_init_rotmat": np.round(Rot.from_quat(ee0[3:]).as_matrix(), 3).tolist(),
             "base_rotmat": np.round(base0_rot.as_matrix(), 3).tolist(),
             "stages": []}

    def record(stage, label):
        frames.append(camera_overlay_frame(camera, tracker.get_positions(), [
            f"ReKep: {instruction}", f"stage {stage}/{num_stages}: {label}"]))

    for stage in range(1, num_stages + 1):
        is_grasp = grasp_keypoints[stage - 1] != -1
        is_release = release_keypoints[stage - 1] != -1
        # movable keypoints = those on the currently grasped body
        movable = np.array([owner is not None and owner == grasped_body for owner in tracker.owners])

        held_indices = [i for i, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body]
        get_grasp_fn = get_callable_grasping_cost_fn(held_indices)
        subgoal_cons, path_cons = _stage_constraints(out_dir, stage, get_grasp_fn)

        ee = ee_pose7(env)
        kp_full = np.concatenate([[ee[:3]], tracker.get_positions()], axis=0)
        mask_full = np.concatenate([[True], movable])
        subgoal_pose7, dbg = solver.solve(ee, kp_full, mask_full, subgoal_cons, path_cons,
                                          is_grasp, from_scratch=(stage == 1))
        frames_before = len(frames)
        target_pos = subgoal_pose7[:3]
        grasp_kp_world = (tracker.get_positions()[grasp_keypoints[stage - 1]].tolist()
                          if is_grasp else None)

        if is_grasp:
            gripper_cmd = GRIPPER_CLOSE
            grasped_body = tracker.owners[grasp_keypoints[stage - 1]]
            grasp(env, target_pos, top_down_quat, lambda: record(stage, "grasp"),
                  args.hover_h, args.grasp_z, args.lift_h,
                  args.max_steps_per_stage, args.pos_tol, args.grasp_rot_gain)
        else:
            drive_to_pose(env, subgoal_pose7, gripper_cmd, lambda: record(stage, "move"),
                          args.max_steps_per_stage, args.pos_tol, args.rot_gain)
            if is_release:
                gripper_cmd = GRIPPER_OPEN
                grasped_body = None
                hold_gripper(env, gripper_cmd, lambda: record(stage, "release"), args.grasp_steps)

        ee_after = ee_pose7(env)
        debug["stages"].append({
            "stage": stage, "is_grasp": is_grasp, "is_release": is_release,
            "ee_before": ee[:3].tolist(), "target_pos": target_pos.tolist(),
            "ee_after": ee_after[:3].tolist(), "grasp_kp_world": grasp_kp_world,
            "solver_cost": dbg["cost"], "move_frames": len(frames) - frames_before,
            "init_pos_err": float(np.linalg.norm(target_pos - ee[:3])),
            "reached_pos_err": float(np.linalg.norm(target_pos - ee_after[:3])),
            "grasped_body": grasped_body,
        })

    with open(os.path.join(out_dir, "rollout_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    out_path = os.path.join(out_dir, f"{exp_name}_rekep_rollout.mp4")
    write_video_h264(frames, out_path, args.fps)
    print(f"[rekep] DONE -> {out_path} ({len(frames)} frames)")
    env.close()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_name = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    isaaclab_helpers.augment_table_cam_with_depth_and_seg(env_cfg)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    try:
        _run(env_name, env_cfg)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
