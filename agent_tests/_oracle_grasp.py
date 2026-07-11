"""Oracle grasp test (investigation, throwaway).

Hand-specify a PERFECT top-down grasp on the pot lid (its known runtime position) and use the
SAME shared low-level control (moka.isaac_control.drive_to_pose / hold_gripper) the rollouts use,
to: hover -> descend onto the lid -> close gripper -> lift. Logs per-phase reach error and the
lid's pose throughout.

Purpose: isolate the low-level/bridge/physics layers from the high-level grounding. If even a
perfect oracle grasp can't (a) reach the lid, (b) close on it, or (c) lift it, the failure is
low-level/bridge/physics -- not the VLM grounding.
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
parser.add_argument("--object", default="cover", help="rigid-object name to grasp")
parser.add_argument("--grasp_dz", type=float, default=0.16, help="grasp height above object origin")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from moka.isaac_control import ee_pose7, drive_to_pose, hold_gripper, robot_base, GRIPPER_OPEN, GRIPPER_CLOSE
from pot_scene_fix import seat_pot_lid

noop = lambda: None

try:
    env_name = args.task.split(":")[-1]
    cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    env = gym.make(env_name, cfg=cfg).unwrapped
    env.reset()
    env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(8):
        env.step(hold)
    seat_pot_lid(env, hold)

    obj = env.scene.rigid_objects[args.object]
    robot = env.scene["robot"]
    jn = list(robot.data.joint_names)
    grip_idx = [i for i, n in enumerate(jn) if any(k in n.lower() for k in ("finger", "knuckle", "gripper"))]

    def grip_state():
        jp = robot.data.joint_pos[0].detach().cpu().numpy()
        return {jn[i]: round(float(jp[i]), 3) for i in grip_idx}

    print(f"gripper joints: {[jn[i] for i in grip_idx]}", flush=True)

    def cover_pos():
        return obj.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)

    def ee():
        return ee_pose7(env)

    lid0 = cover_pos()
    quat = ee()[3:]  # keep the reset top-down orientation (gripper approach axis = world -z)
    bpos, brot = robot_base(env)
    print(f"robot_base = {np.round(bpos, 3)}", flush=True)
    print(f"ee_start   = {np.round(ee()[:3], 3)}  quat(xyzw)={np.round(quat, 3)}", flush=True)
    print(f"obj_start  = {np.round(lid0, 3)}  ({args.object})", flush=True)
    print(f"obj dist from base (xy) = {np.linalg.norm((lid0 - bpos)[:2]):.3f} m", flush=True)

    handle = lid0.copy(); handle[2] += args.grasp_dz
    hover = handle.copy(); hover[2] += 0.10
    lift = handle.copy(); lift[2] += 0.25

    def reach(name, tgt, grip, total=400, chunk=80):
        """Drive toward tgt in chunks, logging EE z to expose a kinematic plateau vs step-limit."""
        done = 0
        err = 9.9
        while done < total:
            n = min(chunk, total - done)
            drive_to_pose(env, np.concatenate([tgt, quat]), grip, noop, max_steps=n, pos_tol=0.015, rot_gain=1.0)
            done += n
            e = ee()[:3]
            err = float(np.linalg.norm(e - tgt))
            print(f"  [{name} +{done:3d}] ee_z={e[2]:.3f} err={err:.3f}", flush=True)
            if err < 0.015:
                break
        print(f"[{name:8s}] target={np.round(tgt,3)} reached={np.round(ee()[:3],3)} reach_err={err:.3f}  lid={np.round(cover_pos(),3)}", flush=True)
        return err

    reach("hover", hover, GRIPPER_OPEN, total=200)
    reach("descend", handle, GRIPPER_OPEN, total=480)
    print(f"[pre-close ] gripper={grip_state()}  ee={np.round(ee()[:3],3)}  obj={np.round(cover_pos(),3)}  ee_above_obj={ee()[2]-cover_pos()[2]:+.3f}", flush=True)
    hold_gripper(env, GRIPPER_CLOSE, noop, 60)
    print(f"[post-close] gripper={grip_state()}  ee={np.round(ee()[:3],3)}  obj={np.round(cover_pos(),3)}", flush=True)
    reach("lift", lift, GRIPPER_CLOSE, total=250)

    lidf = cover_pos()
    dz = float(lidf[2] - lid0[2])
    print(f"lid_end    = {np.round(lidf,3)}  lid_z_gain={dz:+.3f}", flush=True)
    print(f"VERDICT: lid_lifted={dz > 0.05}  (z gain {dz:+.3f} m)", flush=True)
    env.close()
finally:
    app.close()
