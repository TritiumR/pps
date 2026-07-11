"""Read-only probe of the weight task to scope DroidEnv: base pose, panda joints/limits, ee_frame
(Robotiq) TCP, object poses + reachability, and the FK->ee_frame grasp-offset calibration. No DIAL.

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh agent_tests/_droid_weight_probe.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_IL = os.path.join(_REPO, "IsaacLab")
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _sp = os.path.join(_IL, "source", _p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from isaaclab.app import AppLauncher

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=str, default="Isaac-Weight-Droid-Visuomotor-v0")
ap.add_argument("--settle", type=int, default=20)
AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(enable_cameras=True, headless=True)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as Rot

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from sim_common.fk import FrankaFK


def q2R(q_wxyz):
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if hasattr(cfg, "terminations"):
        for t in list(vars(cfg.terminations).keys()):
            try:
                setattr(cfg.terminations, t, None)
            except Exception:
                pass
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()
    print(f"[probe] action_space={env.action_space.shape}", flush=True)
    hold = torch.zeros((1,) + tuple(env.action_space.shape[1:]), dtype=torch.float32, device=env.device)
    for _ in range(args.settle):
        env.step(hold)

    # --- joints ---
    jn = list(robot.data.joint_names)
    bn = list(robot.data.body_names)
    print(f"[probe] joint_names={jn}", flush=True)
    print(f"[probe] body_names={bn}", flush=True)
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    arm_ids = [jn.index(n) for n in arm if n in jn]
    print(f"[probe] arm_ids(panda_joint1-7)={arm_ids}  present={[n for n in arm if n in jn]}", flush=True)
    lim = getattr(robot.data, "joint_pos_limits", None)
    if lim is None:
        lim = robot.data.soft_joint_pos_limits
    print(f"[probe] arm limits=\n{np.round(lim[0, arm_ids].detach().cpu().numpy(), 3)}", flush=True)
    qarm = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
    print(f"[probe] current arm q={np.round(qarm, 3).tolist()}", flush=True)

    # --- base / FK frame (panda_link0) ---
    root_p = robot.data.root_pos_w[0].detach().cpu().numpy()
    root_q = robot.data.root_quat_w[0].detach().cpu().numpy()
    print(f"[probe] root_pos_w={np.round(root_p,3).tolist()} root_quat_w(wxyz)={np.round(root_q,4).tolist()}",
          flush=True)
    if "panda_link0" in bn:
        l0 = bn.index("panda_link0")
        l0_p = robot.data.body_pos_w[0, l0].detach().cpu().numpy()
        l0_q = robot.data.body_quat_w[0, l0].detach().cpu().numpy()
        print(f"[probe] panda_link0_pos_w={np.round(l0_p,3).tolist()} quat(wxyz)={np.round(l0_q,4).tolist()}",
              flush=True)
    else:
        l0_p, l0_q = root_p, root_q
        print("[probe] panda_link0 NOT in body_names; using root as FK base", flush=True)

    # --- ee_frame (Robotiq TCP) ---
    ee = env.scene["ee_frame"]
    ee_p = ee.data.target_pos_w[0, 0].detach().cpu().numpy()
    ee_q = ee.data.target_quat_w[0, 0].detach().cpu().numpy()
    print(f"[probe] ee_frame TCP world={np.round(ee_p,4).tolist()}", flush=True)

    # --- FK calibration: WorldFK(panda_hand) vs ee_frame TCP ---
    fk = FrankaFK(device=args.device)
    hand_pos_b, hand_R_b = fk.fk(torch.tensor(qarm[None], dtype=torch.float32, device=args.device))
    hand_pos_b = hand_pos_b[0].detach().cpu().numpy()
    hand_R_b = hand_R_b[0].detach().cpu().numpy()
    Rb = q2R(l0_q)
    hand_world = Rb @ hand_pos_b + l0_p          # panda_hand in world (base-composed FK)
    hand_R_world = Rb @ hand_R_b
    off_world = ee_p - hand_world                # world offset hand->TCP
    off_hand = hand_R_world.T @ off_world        # in the hand frame (the grasp_offset to use)
    print(f"[probe] FK panda_hand world={np.round(hand_world,4).tolist()}", flush=True)
    print(f"[probe] grasp_offset (hand frame) hand->Robotiq_TCP = {np.round(off_hand,4).tolist()} "
          f"|len={np.linalg.norm(off_hand)*100:.2f}cm| (panda hand was 0,0,0.107)", flush=True)

    # --- objects + reachability (from panda_link0) ---
    for name in ("pear", "apple", "scale", "mango", "cabbage", "board"):
        try:
            op = env.scene[name].data.root_pos_w[0].detach().cpu().numpy()
            reach = float(np.linalg.norm(op - l0_p))
            print(f"[probe] {name:7s} pos={np.round(op,3).tolist()}  dist_from_armbase={reach*100:.1f}cm",
                  flush=True)
        except Exception as e:
            print(f"[probe] {name}: {type(e).__name__}", flush=True)

    print("[probe] PROBE DONE", flush=True)
    env.close()


try:
    main()
finally:
    simulation_app.close()
