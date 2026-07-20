"""Record a ReKep keypoint-tracking video on an IsaacLab task.

Proposes keypoints at reset, registers them to the scene's rigid bodies, then steps the sim
with a gentle arm wave while overlaying the live-tracked keypoints (numbered dots) on the
table camera. Useful for checking the projection / tracking / overlay before a full rollout.

    python rekep/scripts/visualize_keypoints.py --task Isaac-Tea-Droid-Visuomotor-v0 --task_key tea
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Record a ReKep keypoint-tracking video.")
    parser.add_argument("--task", type=str, default="Isaac-Tea-Droid-Visuomotor-v0")
    parser.add_argument("--task_key", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="config yaml (default: rekep/configs/default.yaml)")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--motion_amp", type=float, default=0.15, help="arm wave amplitude (rad); 0 to hold still")
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()

exp_name = args.exp_name or (args.task_key or args.task)
out_dir = os.path.join(_REPO_DIR, "results", "rekep", exp_name)
os.makedirs(out_dir, exist_ok=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imported after the sim app launches (gym-task registration + torch/CUDA setup need it).
import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pot_scene_fix import seat_pot_lid

from rekep import grounding, isaaclab_helpers
from rekep.keypoint_tracking import KeypointTracker
from sim_common.world import GTWorld
from rekep.rekep_viz import camera_overlay_frame
from rekep.utils import get_config, load_default_config
from rekep.video import write_video_h264


def _run(env_name, env_cfg):
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    obs_dict, _ = env.reset()
    joint_pos = obs_dict["policy"]["joint_pos"]
    hold_action = joint_pos[:, :8].to(dtype=torch.float32, device=env.device).clone()
    for _ in range(args.settle_steps):
        env.step(hold_action)
    # Reseat the pot lid (pot task only; no-op elsewhere) -- it spawns detached and must be
    # fixed after the scene settles.
    seat_pot_lid(env, hold_action)

    camera = env.scene["table_cam"]
    config = get_config(args.config) if args.config else load_default_config()
    grounded = grounding.propose_keypoints(camera, env, config)
    keypoints = grounded["keypoints"]
    print(f"[viz] proposed {len(keypoints)} keypoints")
    cv2.imwrite(os.path.join(out_dir, "keypoints.png"), grounded["projected"][..., ::-1])
    if len(keypoints) == 0:
        print("[viz] no keypoints; aborting")
        env.close()
        return

    tracker = KeypointTracker(GTWorld(env), keypoints)
    summary = tracker.summary()
    print(f"[viz] keypoints tracked-on-objects={summary['tracked']} static={summary['static']}")

    frames = []
    for step in range(args.num_steps):
        action = hold_action.clone()
        if args.motion_amp > 0:   # gently wave two joints so the tracking is visible
            phase = 2.0 * np.pi * step / max(args.num_steps - 1, 1)
            wave = args.motion_amp * np.sin(phase)
            action[:, 1] += wave
            action[:, 3] += wave
        env.step(action)

        frames.append(camera_overlay_frame(
            camera, tracker.get_positions(),
            [f"ReKep keypoints (tracked={summary['tracked']})", f"step {step + 1}/{args.num_steps}"]))

    out_path = os.path.join(out_dir, f"{exp_name}_keypoints.mp4")
    write_video_h264(frames, out_path, args.fps)
    print(f"[viz] DONE -> {out_path} ({len(frames)} frames)")
    env.close()


def main():
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
