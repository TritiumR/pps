"""Open-loop MOKA rollout on a PPS / IsaacLab task (Milestone B).

Pipeline: launch the IK-Rel task -> run MOKA's mark-based front-end on the table-camera
frame (decompose -> GroundedSAM -> P/Q marks + 5x5 grid -> GPT-4o select -> 3D lift) ->
faithfully execute MOKA's FrankaMarkPolicy phase sequence
(reach_pre_grasp -> reach_grasp -> grip -> lift -> [pre_contact -> target -> post_contact]
-> release), driving each phase's absolute goal pose with the reused IK-Rel
``drive_to_pose`` -> record the rollout with the projected grasp/function/target/waypoint
marks overlaid.

Two faithful adaptations to the sim: (1) MOKA's absolute table heights (safe_z=0.25,
min_z=0.12) are real-robot coords -- the PPS scenes sit at world z~1.2, so hover/lift
heights are taken relative to the grasp z; (2) the grasp/target poses keep the verified
top-down gripper orientation (reset EE z-axis points world-down) plus the grasp yaw,
while ``transform_gripper_position`` (verbatim from franka_policy) carries the function
point onto the target.

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh \
        moka/run_moka_rollout.py --task Isaac-Tea-Droid-Visuomotor-IK-Rel-v0 --task_key tea
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _pkg_src = os.path.join(_ISAACLAB_DIR, "source", _pkg)
    if _pkg_src not in sys.path:
        sys.path.insert(0, _pkg_src)

from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop MOKA rollout on a PPS task.")
    parser.add_argument("--task", type=str, default="Isaac-Tea-Droid-Visuomotor-IK-Rel-v0")
    parser.add_argument("--task_key", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--camera", choices=["overhead", "table"], default="overhead",
                        help="Planner camera: 'overhead' (top-down, matches MOKA) or 'table'. The "
                             "rollout video is always rendered from table_cam.")
    parser.add_argument("--max_steps_per_phase", type=int, default=90)
    parser.add_argument("--grasp_steps", type=int, default=14)
    parser.add_argument("--pos_tol", type=float, default=0.03)
    parser.add_argument("--rot_gain", type=float, default=0.5)
    parser.add_argument("--hover_h", type=float, default=0.12, help="pre-grasp hover above grasp keypoint (m)")
    parser.add_argument("--grasp_z", type=float, default=0.0, help="z offset at grasp (m, +up)")
    parser.add_argument("--lift_h", type=float, default=0.15, help="post-grasp lift height (m)")
    parser.add_argument("--contact_h", type=float, default=0.10, help="clearance for 'above' contact waypoints (m)")
    parser.add_argument("--fps", type=int, default=15)
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
out_dir = os.path.join(_REPO_DIR, "results", "moka", exp_name)
os.makedirs(out_dir, exist_ok=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from moka.frontend import compute_moka_context
from moka.isaac_control import ee_pose7, robot_base, drive_to_pose, hold_gripper, grasp_at, GRIPPER_OPEN, GRIPPER_CLOSE
from moka.overhead_cam import add_overhead_camera, place_overhead_camera
from moka.policies.franka_policy import transform_gripper_position
from rekep.rekep_viz import draw_keypoints, draw_text_lines, world_to_pixel
from rekep.video import write_video_h264

# MOKA's grasp/current gripper eulers (franka_policy): top-down with the grasp yaw,
# carried "forward" (straight down) during manipulation.
_FORWARD_EULER = np.array([np.pi, 0.0, 0.0])


def _augment_table_cam(env_cfg):
    cam = env_cfg.scene.table_cam
    data_types = list(cam.data_types)
    for needed in ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast"):
        if needed not in data_types:
            data_types.append(needed)
    cam.data_types = data_types
    cam.colorize_instance_id_segmentation = False


def _yawed_top_down(top_down_quat, yaw):
    """Reset (world-down) gripper orientation rotated by yaw about world z."""
    return (Rot.from_rotvec([0.0, 0.0, float(yaw)]) * Rot.from_quat(top_down_quat)).as_quat()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_name = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    _augment_table_cam(env_cfg)  # always present: it renders the rollout video
    if args.camera == "overhead":
        add_overhead_camera(env_cfg)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    try:
        _run(env_name, env_cfg)
    finally:
        simulation_app.close()


def _run(env_name, env_cfg):
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()
    obs_dict, _ = env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(args.settle_steps):
        env.step(hold)
    # Seat the lid on the pot (pot task only; no-op elsewhere) -- the dynamic lid otherwise
    # spawns detached, floating to the side. Must run post-settle, not in a reset event.
    from pot_scene_fix import seat_pot_lid
    seat_pot_lid(env, hold)

    # The rollout video is always rendered from the oblique table_cam; the planner uses
    # the top-down overhead camera (matching MOKA) unless --camera table is given.
    camera = env.scene["table_cam"]
    plan_pose_override = None
    if args.camera == "overhead":
        center, height, eye, quat = place_overhead_camera(env)
        print(f"[moka-rollout] overhead cam over {np.round(center, 2)} at +{height:.2f}m")
        for _ in range(4):
            env.step(hold)
        plan_camera = env.scene["moka_cam"]
        plan_pose_override = (eye, quat)
    else:
        plan_camera = camera

    # ---- MOKA front-end (kit alive) -> selected 3D context ----
    context, obs, camera_params, points, masks, dense_kp, _ = compute_moka_context(
        plan_camera, instruction, out_dir, task_key, pose_override=plan_pose_override)
    grasp_w = context["keypoints_3d"]["grasp"]
    function_w = context["keypoints_3d"]["function"]
    target_w = context["keypoints_3d"]["target"]
    grasp_yaw = float(context.get("grasp_yaw") or 0.0)
    pre_wps = [w for w in context["waypoints_3d"].get("pre_contact", []) if w is not None]
    post_wps = [w for w in context["waypoints_3d"].get("post_contact", []) if w is not None]
    print(f"[moka-rollout] grasp={None if grasp_w is None else np.round(grasp_w,3)} "
          f"function={None if function_w is None else np.round(function_w,3)} "
          f"target={None if target_w is None else np.round(target_w,3)} yaw={grasp_yaw:.2f}")

    # ---- overlay marks (static world points projected each frame) ----
    overlay_pts, overlay_lbl = [], []
    for name, p in [("grasp", grasp_w), ("function", function_w), ("target", target_w)]:
        if p is not None:
            overlay_pts.append(p)
            overlay_lbl.append(name)
    for w in pre_wps + post_wps:
        overlay_pts.append(w)
        overlay_lbl.append("wp")
    overlay_pts = np.array(overlay_pts, dtype=np.float64) if overlay_pts else np.zeros((0, 3))

    frames = []

    def record(label):
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        pos = camera.data.pos_w[0].detach().cpu().numpy()
        quat = camera.data.quat_w_ros[0].detach().cpu().numpy()
        intr = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
        if len(overlay_pts):
            pixels, visible = world_to_pixel(overlay_pts, pos, quat, intr)
            frame = draw_keypoints(rgb, pixels, visible)
        else:
            frame = rgb
        frame = draw_text_lines(frame, [f"MOKA: {instruction}", label])
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    ee0 = ee_pose7(env)
    top_down_quat = ee0[3:]  # reset EE z-axis points world-down (verified in ReKep port)
    debug = {"task": args.task, "instruction": instruction,
             "grasp": None if grasp_w is None else np.round(grasp_w, 4).tolist(),
             "function": None if function_w is None else np.round(function_w, 4).tolist(),
             "target": None if target_w is None else np.round(target_w, 4).tolist(),
             "grasp_yaw": grasp_yaw, "phases": []}

    def go(label, pos, quat, grip, tol=None, rot_gain=None):
        target = np.concatenate([np.asarray(pos, dtype=np.float64), quat])
        before = ee_pose7(env)[:3]
        drive_to_pose(env, target, grip, lambda: record(label),
                      args.max_steps_per_phase, tol or args.pos_tol, args.rot_gain if rot_gain is None else rot_gain)
        after = ee_pose7(env)[:3]
        debug["phases"].append({"phase": label, "goal": np.round(target[:3], 4).tolist(),
                                "reached_err": float(np.linalg.norm(target[:3] - after)),
                                "moved": float(np.linalg.norm(after - before))})

    def finish():
        with open(os.path.join(out_dir, "rollout_debug.json"), "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)
        out_path = os.path.join(out_dir, f"{exp_name}_moka_rollout.mp4")
        write_video_h264(frames, out_path, args.fps)
        print(f"[moka-rollout] DONE -> {out_path} ({len(frames)} frames)")
        env.close()

    # ---- no-grasp subtask (MOKA picks no grasp keypoint, e.g. push/open a lid):
    # move the gripper to the target keypoint (reach_pre_target -> reach_target). ----
    if grasp_w is None:
        if target_w is not None:
            tz = float(target_w[2])
            go("reach_pre_target", [target_w[0], target_w[1], tz + args.contact_h], top_down_quat, GRIPPER_OPEN)
            go("reach_target", [target_w[0], target_w[1], tz], top_down_quat, GRIPPER_OPEN,
               tol=args.pos_tol * 0.6, rot_gain=1.0)
        else:
            for _ in range(args.grasp_steps):
                record("idle")
        hold_gripper(env, GRIPPER_OPEN, lambda: record("release"), args.grasp_steps)
        finish()
        return

    grasp_quat = _yawed_top_down(top_down_quat, grasp_yaw)
    safe_z = float(grasp_w[2]) + args.lift_h

    # ---- grasp: contact-aware (hover -> descend to contact -> close -> lift) ----
    # The grasp keypoint sits on the object's visible top surface, which is above the gripper's
    # contact height; descending to contact (instead of closing at a fixed Z) lands the fingers on
    # the object so the close actually captures it. See moka.isaac_control.grasp_at.
    grasp_at(env, grasp_w, grasp_quat, lambda: record("grip"), hover=args.lift_h, lift=args.lift_h)

    # ---- manipulation: carry function point onto the target (faithful transform) ----
    def carried_goal(target_point, raise_above):
        """EE goal that places the function point at target_point (MOKA geometry)."""
        tgt = np.asarray(target_point, dtype=np.float64).copy()
        if raise_above:
            tgt[2] = max(tgt[2], grasp_w[2]) + args.contact_h
        if function_w is None:
            return tgt  # no function point -> move the grasp point itself to the target
        return transform_gripper_position(
            grasp_point=np.asarray(grasp_w, dtype=np.float64),
            function_point=np.asarray(function_w, dtype=np.float64),
            target_point=tgt,
            grasp_euler=np.array([np.pi, 0.0, grasp_yaw]),
            current_euler=_FORWARD_EULER)

    pre_above = (context.get("pre_contact_height") == "above")
    post_above = (context.get("post_contact_height") == "above")

    for i, wp in enumerate(pre_wps):
        go(f"reach_pre_contact_{i}", carried_goal(wp, pre_above), grasp_quat, GRIPPER_CLOSE)
    if target_w is not None:
        go("reach_target", carried_goal(target_w, pre_above), grasp_quat, GRIPPER_CLOSE)
    for i, wp in enumerate(post_wps):
        go(f"reach_post_contact_{i}", carried_goal(wp, post_above), grasp_quat, GRIPPER_CLOSE)

    hold_gripper(env, GRIPPER_OPEN, lambda: record("release"), args.grasp_steps)
    from task_success import report_task_success
    report_task_success(env, task_key)
    finish()


if __name__ == "__main__":
    main()
