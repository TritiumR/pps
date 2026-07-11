"""cuRobo cube-pick execute -- step 2 of the demo (Isaac Sim, warp 1.7.1, NO cuRobo import).

Loads the cube-pick joint configs solved by _curobo_solve.py and drives the Franka through
pre-grasp -> grasp -> close -> lift in Isaac-Lift-Cube-Franka-v0, recording an added camera
-> results/curobo/<exp>.mp4. Separate process from the solve step (warp version clash).

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh _curobo_solve.py --out /tmp/c.json
    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh _curobo_demo.py  --configs /tmp/c.json
"""
import argparse
import json
import os
import sys

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    _p = os.path.join(_ISAACLAB_DIR, "source", _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Execute a cuRobo cube pick in Lift-Cube-Franka.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--exp_name", type=str, default="cube_curobo_pick")
parser.add_argument("--configs", type=str, default="/tmp/curobo_cube_configs.json")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--steps_per_phase", type=int, default=45)
parser.add_argument("--settle", type=int, default=15)
parser.add_argument("--converge_max", type=int, default=150)
parser.add_argument("--converge_tol", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args = parser.parse_args()

with open(args.configs, "r", encoding="utf-8") as f:
    DATA = json.load(f)
CUBE = DATA["cube"]
WP = {w["lbl"]: w for w in DATA["waypoints"]}
print(f"[demo] loaded configs from {args.configs} | CUBE={CUBE}", flush=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from rekep.video import write_video_h264

CAM_EYE = [1.25, -0.85, 0.7]
CAM_TARGET = [0.45, -0.05, 0.1]
frames = []


def add_demo_cam(env_cfg, name="demo_cam"):
    setattr(env_cfg.scene, name, CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.0476, horizontal_aperture=1.05, vertical_aperture=0.59,
            clipping_range=(1e-4, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    ))


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # The joint-pos RL task uses low-PD franka gains -> the arm sags under gravity and never
    # reaches an arbitrary commanded config. The IK lift cfgs swap in HIGH_PD for precise
    # tracking; do the same so the arm actually reaches cuRobo's joint configs.
    env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.episode_length_s = 1.0e4
    if hasattr(env_cfg, "terminations"):  # don't reset on success/dropout mid-demo
        for _t in list(vars(env_cfg.terminations).keys()):
            try:
                setattr(env_cfg.terminations, _t, None)
            except Exception:
                pass
    add_demo_cam(env_cfg)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    cube = env.scene["object"]
    cam = env.scene["demo_cam"]

    env.reset()

    def neutral():
        a = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
        a[0, 7] = 1.0  # gripper open (>=0)
        return a

    for _ in range(args.settle):
        env.step(neutral())

    # teleport the cube to the fixed pose cuRobo planned around, then settle
    state = cube.data.root_state_w.clone()
    state[0, :3] = torch.tensor(CUBE, dtype=torch.float32, device=env.device) + env.scene.env_origins[0]
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    state[0, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    for _ in range(args.settle):
        env.step(neutral())

    # angled view of the workspace
    cam.set_world_poses_from_view(
        torch.tensor([CAM_EYE], dtype=torch.float32, device=env.device),
        torch.tensor([CAM_TARGET], dtype=torch.float32, device=env.device))

    default_arm = robot.data.default_joint_pos[0, :7].detach().cpu().numpy()
    robot_arm_names = list(robot.data.joint_names[:7])

    def ee_z():
        return float(env.scene["ee_frame"].data.target_pos_w[0, 0, 2].detach().cpu())

    def cube_z():
        return float(cube.data.root_state_w[0, 2].detach().cpu())

    _bn = list(robot.data.body_names)
    ph_idx = _bn.index("panda_hand") if "panda_hand" in _bn else None

    def ph_pos():
        if ph_idx is None:
            return [0.0, 0.0, 0.0]
        return [round(float(x), 3) for x in robot.data.body_pos_w[0, ph_idx]]

    print(f"[demo] cuRobo names={WP['grasp']['names']}", flush=True)
    print(f"[demo] robot_arm_names={robot_arm_names}", flush=True)
    print(f"[demo] default_arm={[round(float(x), 3) for x in default_arm]}", flush=True)

    def record(label):
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        img = np.ascontiguousarray(rgb.astype(np.uint8))
        cv2.putText(img, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (40, 255, 40), 2, cv2.LINE_AA)
        frames.append(img)

    def act(q_arm, grip_open, label):
        raw = 2.0 * (np.asarray(q_arm, dtype=np.float64) - default_arm)  # scale 0.5, default offset
        g = 1.0 if grip_open else -1.0
        a = torch.tensor([list(raw) + [g]], dtype=torch.float32, device=env.device)
        env.step(a)
        record(label)

    def cfg_arm(lbl):
        s = WP[lbl]
        qbn = dict(zip(s["names"], s["q"]))
        try:
            return np.array([qbn[n] for n in robot_arm_names], dtype=np.float64), s["ok"]
        except KeyError:
            return np.asarray(s["q"], dtype=np.float64), s["ok"]

    def move(cur, lbl, grip_open, text):
        q_goal, ok = cfg_arm(lbl)
        if not ok:
            print(f"[demo] {lbl} UNREACHABLE", flush=True)
            return cur
        tgt_z = WP[lbl]["pos"][2]
        for t in range(1, args.steps_per_phase + 1):
            a = t / args.steps_per_phase
            act((1 - a) * cur + a * q_goal, grip_open, text)
        # hold the target until the PD actually converges (a single interpolation pass lags
        # because the per-step target moves faster than the joint velocity limit)
        jerr = 9.0
        for _ in range(args.converge_max):
            act(q_goal, grip_open, text)
            jerr = float(np.max(np.abs(robot.data.joint_pos[0, :7].detach().cpu().numpy() - q_goal)))
            if jerr < args.converge_tol:
                break
        print(f"[demo] -> {lbl}: cmd_hand_z={tgt_z:.3f} converged_joint_err={jerr:.3f} "
              f"panda_hand={ph_pos()} ee_z={ee_z():.3f} cube_z={cube_z():.3f} "
              f"(grip {'open' if grip_open else 'closed'})", flush=True)
        return q_goal

    cur = robot.data.joint_pos[0, :7].detach().cpu().numpy().copy()
    print(f"[demo] start ee_z={ee_z():.3f} cube_z={cube_z():.3f}", flush=True)
    cur = move(cur, "pre-grasp", True, "cuRobo pick: approach")
    cur = move(cur, "grasp", True, "cuRobo pick: descend to cube")
    # close the gripper on the cube (hold the grasp config)
    print("[demo] closing gripper", flush=True)
    for _ in range(max(args.settle, 25)):
        act(cur, False, "cuRobo pick: grasp (close)")
    print(f"[demo] after close: ee_z={ee_z():.3f} cube_z={cube_z():.3f}", flush=True)
    cur = move(cur, "lift", False, "cuRobo pick: lift")
    for _ in range(args.settle):
        act(cur, False, "cuRobo pick: lift")
    print(f"[demo] FINAL: ee_z={ee_z():.3f} cube_z={cube_z():.3f} "
          f"(table top ~0.0; cube lifted if cube_z > ~0.1)", flush=True)

    out = os.path.join(_REPO_DIR, "results", "curobo", f"{args.exp_name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[demo] DONE -> {out} ({len(frames)} frames)", flush=True)


try:
    main()
finally:
    simulation_app.close()
