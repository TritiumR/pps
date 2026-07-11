"""cuRobo cube-pick IK solve -- step 1 of the demo (run WITHOUT Isaac Sim, warp 1.14.0).

Solves pre-grasp / grasp / lift joint configs for a top-down pick of a cube with the stock
franka.yml (the Lift-Cube-Franka robot IS a standard Franka, so franka.yml matches exactly).
Writes the configs + per-phase gripper state to JSON for the execute step (_curobo_demo.py),
which runs in a separate process because Isaac Sim's warp 1.7.1 can't share with cuRobo's 1.14.0.

    /isaac-sim/python.sh _curobo_solve.py --out /tmp/curobo_cube_configs.json
"""
import argparse
import json

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import Pose, GoalToolPose

DEV = "cuda:0"

# Cube + grasp geometry. World == robot base frame (the Franka base is at the origin).
# Must match the cube pose the execute step teleports the cube to.
CUBE = [0.55, 0.0, 0.0205]   # cube center (rests on the table top at z~=0)
HAND_OFF = 0.107             # franka.yml tool frame is panda_hand, which sits this far ABOVE
                             # the grasp point (along the down-pointing gripper z), so target
                             # panda_hand at cube_z + HAND_OFF to place fingertips at the cube.
HOVER = 0.12                 # pre-grasp standoff above the grasp (m)
LIFT = 0.20                  # lift height above the grasp (m)
DOWN = [0.0, 1.0, 0.0, 0.0]  # EE pointing straight down (wxyz)

WAYPOINTS = [
    ("pre-grasp", [CUBE[0], CUBE[1], CUBE[2] + HAND_OFF + HOVER], DOWN, "open"),
    ("grasp",     [CUBE[0], CUBE[1], CUBE[2] + HAND_OFF],         DOWN, "open"),
    ("lift",      [CUBE[0], CUBE[1], CUBE[2] + HAND_OFF + LIFT],  DOWN, "close"),
]

# Lift-Cube-Franka's actual home arm config (read from the env). cuRobo's franka.yml default
# seed is elsewhere, so we CHAIN: each waypoint picks the IK seed closest to the previous config
# (starting here) to keep the whole pick in one continuous branch -- otherwise linear joint
# interpolation between different IK branches swings the arm meters and the PD can't track it.
HOME_ARM = np.array([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])


ACTIVE_NAMES = [f"panda_joint{i}" for i in range(1, 8)]  # franka.yml locks the 2 fingers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/curobo_cube_configs.json")
    ap.add_argument("--robot", default="franka.yml")
    ap.add_argument("--num_seeds", type=int, default=64)
    ap.add_argument("--return_seeds", type=int, default=30)
    a = ap.parse_args()

    ik = InverseKinematics(InverseKinematicsCfg.create(robot=a.robot, num_seeds=a.num_seeds))
    tool = ik.tool_frames
    print(f"[solve] cuRobo IK ready | tool_frames={tool} | CUBE={CUBE}", flush=True)

    out = []
    q_prev = HOME_ARM.copy()  # chain from the Lift-Cube home (arm joints)
    for lbl, pos, quat, grip in WAYPOINTS:
        p = torch.tensor([pos], device=DEV, dtype=torch.float32)
        q = torch.tensor([quat], device=DEV, dtype=torch.float32)
        goal = GoalToolPose.from_poses(
            {tool[0]: Pose(position=p, quaternion=q).unsqueeze(1)}, ordered_tool_frames=tool)
        res = ik.solve_pose(goal_tool_poses=goal, return_seeds=a.return_seeds)
        cand = res.solution.reshape(-1, 7).detach().cpu().numpy()  # [n_returned, 7] active arm
        succ = res.success.reshape(-1).detach().cpu().numpy().astype(bool)
        errs = res.position_error.reshape(-1).detach().cpu().numpy()
        idxs = np.where(succ)[0]
        if len(idxs):
            # among returned solutions pick the one CLOSEST to the previous config -> one branch
            d = np.linalg.norm(cand[idxs] - q_prev[None, :], axis=1)
            pick = int(idxs[int(np.argmin(d))])
            ok = True
        else:
            pick = int(np.argmin(errs))
            ok = False
        qsol = cand[pick]
        err = float(errs[pick])
        step = float(np.linalg.norm(qsol - q_prev))
        q_prev = qsol.copy()
        out.append({
            "ok": ok, "names": ACTIVE_NAMES, "q": [float(x) for x in qsol],
            "err": err, "lbl": lbl, "pos": pos, "grip": grip,
        })
        print(f"[solve] '{lbl}' z={pos[2]:.3f}: ok={ok} pos_err={err:.4f}m "
              f"cand={cand.shape[0]} branch_step={step:.3f}rad", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"cube": CUBE, "waypoints": out}, f, indent=2)
    print(f"[solve] wrote {len(out)} configs -> {a.out}", flush=True)


main()
