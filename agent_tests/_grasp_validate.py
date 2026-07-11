"""Validate the contact-aware grasp_at primitive: does it capture the egg + teapot? (throwaway)"""
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
parser.add_argument("--task", required=True)
parser.add_argument("--object", required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from moka.isaac_control import ee_pose7, grasp_at

noop = lambda: None

try:
    env_name = args.task.split(":")[-1]
    cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    env = gym.make(env_name, cfg=cfg).unwrapped
    obj = env.scene.rigid_objects[args.object]

    def obj_pos():
        return obj.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)

    env.reset(); env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(8):
        env.step(hold)

    o0 = obj_pos()
    quat = ee_pose7(env)[3:]
    # Grasp point given on the object's TOP SURFACE (+0.04 above origin) -- exactly the kind of
    # too-high target the methods produce; the primitive must descend past it to contact.
    grasp_xyz = o0.copy(); grasp_xyz[2] += 0.04
    print(f"object={args.object} start={np.round(o0, 3)}  grasp_pt(surface)={np.round(grasp_xyz, 3)}", flush=True)

    grasp_at(env, grasp_xyz, quat, noop)

    of = obj_pos()
    dz = float(of[2] - o0[2])
    print(f"object end={np.round(of, 3)}  z_gain={dz:+.3f}  CAPTURED={dz > 0.05}", flush=True)
    env.close()
finally:
    app.close()
