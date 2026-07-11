"""Phase 1 grasp-capture calibration (investigation, throwaway).

For the egg (small, correct-physics, easily reached) sweep the EE grasp height and, at each,
directly log the world position of the GRIPPER FINGER-PAD links vs the EE frame vs the egg, then
close + lift and measure whether the egg is captured. This pins down whether the grasp fails
because the pads close off the object (a fixable EE-frame->fingerpad offset) and finds the EE
height that actually grips.
"""
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
parser.add_argument("--task", default="Isaac-Pot-Droid-Visuomotor-IK-Rel-v0")
parser.add_argument("--object", default="egg")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from moka.isaac_control import ee_pose7, drive_to_pose, hold_gripper, GRIPPER_OPEN, GRIPPER_CLOSE

noop = lambda: None

try:
    env_name = args.task.split(":")[-1]
    cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    env = gym.make(env_name, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene.rigid_objects[args.object]

    bn = list(robot.data.body_names)
    # finger-pad links: the inner finger bodies that actually contact the object
    pad_idx = [i for i, n in enumerate(bn) if "inner_finger" in n.lower() and "knuckle" not in n.lower()]
    if not pad_idx:
        pad_idx = [i for i, n in enumerate(bn) if "finger" in n.lower()]
    print(f"all body names: {bn}", flush=True)
    print(f"pad links: {[bn[i] for i in pad_idx]}", flush=True)

    def pad_world():
        bp = robot.data.body_pos_w[0].detach().cpu().numpy()
        pads = bp[pad_idx]
        return pads.mean(0), pads  # mean pad pos, individual

    def egg_pos():
        return obj.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)

    def ee():
        return ee_pose7(env)

    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)

    def attempt(dz):
        env.reset(); env.reset()
        for _ in range(8):
            env.step(hold)
        e0 = egg_pos()
        quat = ee()[3:]
        tgt = e0.copy(); tgt[2] += dz
        hov = tgt.copy(); hov[2] += 0.12
        # approach open
        drive_to_pose(env, np.concatenate([hov, quat]), GRIPPER_OPEN, noop, max_steps=150, pos_tol=0.015, rot_gain=1.0)
        drive_to_pose(env, np.concatenate([tgt, quat]), GRIPPER_OPEN, noop, max_steps=250, pos_tol=0.012, rot_gain=1.0)
        padm, pads = pad_world()
        eez = ee()[2]
        print(f"\n=== dz={dz:+.2f}  EE_z={eez:.3f}  pad_z={padm[2]:.3f}  egg_z={e0[2]:.3f}  "
              f"pad_minus_egg={padm[2]-e0[2]:+.3f}  ee_minus_pad={eez-padm[2]:+.3f}", flush=True)
        print(f"    pads xy: L={np.round(pads[0][:2],3)} R={np.round(pads[1][:2],3)}  egg xy={np.round(e0[:2],3)}", flush=True)
        # close + lift
        hold_gripper(env, GRIPPER_CLOSE, noop, 60)
        lift = tgt.copy(); lift[2] += 0.20
        drive_to_pose(env, np.concatenate([lift, quat]), GRIPPER_CLOSE, noop, max_steps=150, pos_tol=0.012, rot_gain=1.0)
        gain = float(egg_pos()[2] - e0[2])
        print(f"    -> egg_z_gain={gain:+.3f}  CAPTURED={gain > 0.05}", flush=True)
        return gain

    for dz in (0.14, 0.10, 0.06, 0.02, -0.02):
        attempt(dz)

    env.close()
finally:
    app.close()
