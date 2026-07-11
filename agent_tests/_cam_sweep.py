"""Throwaway camera-position sweep: in ONE Isaac boot, mount several candidate ReKep cameras at
different eye poses, run the front-end (propose_keypoints) from EACH independently, and measure how
close the nearest proposed keypoint lands to the GT mug handle. Goal: find a single camera that sees
the handle for a chosen (graspable) handle yaw -- without building the multi-view rig.

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh agent_tests/_cam_sweep.py --obj_yaw 180
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=str, default="Isaac-Lift-Mug-Franka-v0")
ap.add_argument("--obj_z", type=float, default=0.12)
ap.add_argument("--obj_yaw", type=float, default=180.0)
ap.add_argument("--min_dist", type=float, default=0.025)
ap.add_argument("--handle_local", type=float, nargs=3, default=[0.035, 0.064, 0.0],
                help="GT handle offset in the mug frame (derived from the yaw=-45 capture)")
ap.add_argument("--settle", type=int, default=15)
AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(enable_cameras=True, headless=True)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import cv2
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as Rot

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from rekep import grounding
from rekep.utils import load_default_config
import sim_common.envs.lift_mug  # noqa: F401  -- registers Isaac-Lift-Mug-Franka-v0
from sim_common.envs.lift import make_rekep_cam_cfg

OBJ_XY = [0.55, 0.0]
OBJ_CENTER = [0.55, 0.0, 0.08]
OUT = os.path.join(_REPO, "results", "vlm_mpc", "rekep", "cam_sweep")

# Candidate cameras (name, eye); all look at the mug center. Spread around the workspace, with an
# emphasis on views that might see a robot-facing (-x) handle: over-the-shoulder + sides.
CAMERAS = [
    ("front_diag", [1.0, -0.7, 0.7]),     # current rung-3 view (baseline; sees +x/-y)
    ("oshoulder_hi", [-0.1, 0.0, 1.15]),  # behind+above robot, looking forward-down at the -x face
    ("oshoulder_ny", [0.0, -0.6, 1.05]),  # behind-ish + to -y, high
    ("side_ny", [0.5, -0.95, 0.6]),       # pure -y side
    ("near_top", [0.3, -0.2, 1.1]),       # near-overhead, tilted from -x/-y
]


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if hasattr(cfg, "terminations") and hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    for name, eye in CAMERAS:
        setattr(cfg.scene, f"cam_{name}", make_rekep_cam_cfg(eye, OBJ_CENTER, name=f"cam_{name}"))

    env = gym.make(args.task, cfg=cfg).unwrapped
    obj = env.scene["object"]
    env.reset()
    neutral = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
    for _ in range(args.settle):
        env.step(neutral)
    # teleport mug to the test pose
    st = obj.data.root_state_w.clone()
    st[0, :3] = torch.tensor([OBJ_XY[0], OBJ_XY[1], args.obj_z], device=env.device) + env.scene.env_origins[0]
    phi = np.deg2rad(args.obj_yaw)
    st[0, 3:7] = torch.tensor([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], device=env.device)
    st[0, 7:] = 0.0
    obj.write_root_state_to_sim(st)
    for _ in range(args.settle):
        env.step(neutral)

    # GT handle world position for this pose
    pos = obj.data.root_pos_w[0].detach().cpu().numpy()
    quat = obj.data.root_quat_w[0].detach().cpu().numpy()  # wxyz
    R = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    handle_w = pos + R @ np.asarray(args.handle_local)
    print(f"[sweep] obj_yaw={args.obj_yaw} obj={pos.round(3).tolist()} "
          f"handle_GT={handle_w.round(3).tolist()}", flush=True)

    config = load_default_config()
    config["keypoint_proposer"]["min_dist_bt_keypoints"] = args.min_dist

    print(f"[sweep] {'camera':<14} {'#kp':>4} {'nearest_kp_to_handle(cm)':>26}", flush=True)
    for name, eye in CAMERAS:
        cam = env.scene[f"cam_{name}"]
        try:
            g = grounding.propose_keypoints(cam, env, config)
        except Exception as e:  # a view that sees nothing in-bounds can fail to cluster
            print(f"[sweep] {name:<14} {'ERR':>4}  {type(e).__name__}: {e}", flush=True)
            continue
        kps = g["keypoints"]
        cv2.imwrite(os.path.join(OUT, f"kp_{name}.png"), g["projected"][..., ::-1])
        if len(kps) == 0:
            print(f"[sweep] {name:<14} {0:>4} {'--':>26}", flush=True)
            continue
        d = np.linalg.norm(kps - handle_w, axis=1)
        j = int(np.argmin(d))
        print(f"[sweep] {name:<14} {len(kps):>4} {d[j]*100:>22.2f}  (kp{j}={kps[j].round(3).tolist()})",
              flush=True)

    print(f"[sweep] DONE -> {OUT}", flush=True)


try:
    main()
finally:
    simulation_app.close()
