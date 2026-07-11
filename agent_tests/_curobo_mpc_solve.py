"""cuRobo MPC -- step 1 of the MPC demo (run WITHOUT Isaac Sim, warp 1.14.0).

Runs cuRobo's MPPI-style MPC closed-loop on its OWN kinematic model, tracking a MOVING goal
pose (a circle in front of the robot) so the reactivity is visible. Saves the resulting joint
trajectory to JSON; _curobo_mpc_demo.py replays it in Isaac Sim (separate process -- warp clash).

True live closed-loop (MPC <-> sim each step) would need an out-of-process IPC; this offline
form still exercises the real MPC optimizer and shows it tracking a moving target.

    /isaac-sim/python.sh _curobo_mpc_solve.py --out /tmp/curobo_mpc_traj.json
"""
import argparse
import json
import math

import numpy as np
import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import JointState, Pose

DEV = "cuda:0"
ACTIVE_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
# start at the Lift-Cube-Franka home so the sim replay starts from the same config
HOME = [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741]
# moving goal: a circle in the y-z (frontal) plane in front of the robot
CENTER = np.array([0.50, 0.0, 0.45])
RADIUS = 0.18
DOWN = [0.0, 1.0, 0.0, 0.0]  # EE pointing down (wxyz)


def goal_at(t):  # t in [0,1] -> one full circle
    ang = 2 * math.pi * t
    return [CENTER[0], CENTER[1] + RADIUS * math.sin(ang), CENTER[2] + RADIUS * math.cos(ang)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/curobo_mpc_traj.json")
    ap.add_argument("--robot", default="franka.yml")
    ap.add_argument("--steps", type=int, default=260)
    a = ap.parse_args()

    cfg = ModelPredictiveControlCfg.create(robot=a.robot, store_debug=False)
    mpc = ModelPredictiveControl(cfg)
    tool = mpc.tool_frames
    print(f"[mpc] ready | tool_frames={tool}", flush=True)

    state = JointState.from_position(torch.tensor([HOME], device=DEV, dtype=torch.float32))
    mpc.setup(state)

    traj = []
    for i in range(a.steps):
        g = goal_at(i / a.steps)
        pose = Pose(
            position=torch.tensor([g], device=DEV, dtype=torch.float32),
            quaternion=torch.tensor([DOWN], device=DEV, dtype=torch.float32))
        mpc.update_goal_tool_poses({tool[0]: pose})
        res = mpc.optimize_action_sequence(state)
        nxt = res.next_action
        q = nxt.position.reshape(-1)[:7].detach().cpu().numpy()
        traj.append([float(x) for x in q])
        state = nxt
        if i % 40 == 0:
            print(f"[mpc] step {i}/{a.steps} goal=[{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}] "
                  f"q0={q[0]:.2f}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"names": ACTIVE_NAMES, "home": HOME, "traj": traj}, f)
    print(f"[mpc] wrote {len(traj)} steps -> {a.out}", flush=True)


main()
