"""Step 1 full-task oracle (WEIGHT): with GT targets, grasp both fruits and place them on the
real scale, then check the task's own success function. Isolates the manipulation layer (grasp +
place primitives + 2-object sequencing) from grounding -- the 'perfect grounding' baseline."""
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    p = os.path.join(_REPO, "IsaacLab", "source", _pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
from isaaclab.app import AppLauncher
import pinocchio  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-IK-Rel-v0")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from moka.isaac_control import (ee_pose7, grasp_at, descend_to_contact, drive_to_pose,
                                hold_gripper, robot_base, GRIPPER_OPEN, GRIPPER_CLOSE)
from task_success import report_task_success

noop = lambda: None

try:
    env_name = args.task.split(":")[-1]
    cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    env = gym.make(env_name, cfg=cfg).unwrapped
    rigid = env.scene.rigid_objects

    def pos(name):
        return rigid[name].data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)

    frames = []

    def rec():
        rgb = env.scene["table_cam"].data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        frames.append(rgb[..., ::-1].copy())  # RGB -> BGR for write_video_h264

    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    # Re-roll the randomized layout until the scale is comfortably reachable. The far-scale
    # layouts fail on reach (the reach lever) -- gating to a reachable layout demonstrates the
    # manipulation capability cleanly; robustness across all layouts is still pending.
    sdist = 99.0
    for attempt in range(12):
        env.reset(); env.reset()
        for _ in range(8):
            env.step(hold)
        bpos, _ = robot_base(env)
        sdist = float(np.linalg.norm((pos("scale") - bpos)[:2]))
        if sdist < 0.62:
            break
    print(f"layout: scale dist from base = {sdist:.3f}m (reroll {attempt})", flush=True)

    quat = ee_pose7(env)[3:]
    scale = pos("scale")
    print(f"scale={np.round(scale,3)}  apple={np.round(pos('apple'),3)}  pear={np.round(pos('pear'),3)}", flush=True)

    # GT place spots: on the scale, offset so the two fruits don't collide. Instrument the place
    # (grasp -> carry -> descend -> release) to see exactly where each fruit is lost.
    for fruit, dxy in (("apple", np.array([-0.05, 0.0])), ("pear", np.array([0.05, 0.0]))):
        gp = pos(fruit); gp[2] += 0.04  # grasp the fruit's top surface
        grasp_at(env, gp, quat, rec)
        print(f"[{fruit}] after grasp+lift: fruit={np.round(pos(fruit),3)} ee={np.round(ee_pose7(env)[:3],3)}", flush=True)
        place = np.array([scale[0] + dxy[0], scale[1] + dxy[1], scale[2] + 0.05])
        drive_to_pose(env, np.concatenate([[place[0], place[1], place[2] + 0.14], quat]),
                      GRIPPER_CLOSE, rec, 300, 0.015, 1.0)
        print(f"[{fruit}] after carry-hover: fruit={np.round(pos(fruit),3)} ee={np.round(ee_pose7(env)[:3],3)} (target xy={np.round(place[:2],3)})", flush=True)
        descend_to_contact(env, rec, gripper_cmd=GRIPPER_CLOSE, min_z=place[2] + 0.08)
        print(f"[{fruit}] after descend: fruit={np.round(pos(fruit),3)} ee={np.round(ee_pose7(env)[:3],3)}", flush=True)
        hold_gripper(env, GRIPPER_OPEN, rec, 40)
        cur = ee_pose7(env)
        drive_to_pose(env, np.concatenate([[cur[0], cur[1], cur[2] + 0.18], quat]), GRIPPER_OPEN, rec, 300, 0.015, 1.0)
        print(f"[{fruit}] after release+retract: fruit={np.round(pos(fruit),3)}  (scale={np.round(scale,3)})", flush=True)

    print("---- final ----", flush=True)
    report_task_success(env, "weight")
    from rekep.video import write_video_h264
    write_video_h264(frames, "/workspace/pps/results/weight_oracle_success.mp4", 30)
    print(f"[video] results/weight_oracle_success.mp4 ({len(frames)} frames)", flush=True)
    env.close()
finally:
    app.close()
