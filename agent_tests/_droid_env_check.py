"""Validate DroidEnv before the grasp: (1) WorldFK TCP matches the env ee_frame across configs
(confirms base-pose SE3 composition + calibrated Robotiq offset), (2) apply_arm (absolute joint
targets) tracks to the commanded q. No DIAL.

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh agent_tests/_droid_env_check.py
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
AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(enable_cameras=True, headless=True)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from sim_common.envs.droid import DroidEnv

DEV = "cuda:0"


def main():
    E = DroidEnv(device=DEV)
    print(f"[droid-check] arm_ids={E.arm_ids} grip_id={E.grip_id} dt={E.dt:.4f} act_dim={E.act_dim}",
          flush=True)
    print(f"[droid-check] q_lo={np.round(E.q_lo.cpu().numpy(),2).tolist()}", flush=True)
    print(f"[droid-check] q_hi={np.round(E.q_hi.cpu().numpy(),2).tolist()}", flush=True)

    errs = []
    # (1) FK vs ee_frame at the rest pose
    fk_tcp, ee_tcp = E.tcp(), E.ee_frame_tcp()
    e0 = float(np.linalg.norm(fk_tcp - ee_tcp))
    errs.append(e0)
    print(f"[droid-check] rest: WorldFK tcp={np.round(fk_tcp,4).tolist()} ee_frame={np.round(ee_tcp,4).tolist()} "
          f"err={e0*100:.2f}cm", flush=True)

    # (2) command a few distinct arm configs; check FK-vs-ee_frame + arm tracking
    q0 = E.q0().cpu().numpy()
    for k, delta in enumerate([[0.2, 0, 0, 0, 0, 0, 0], [0, -0.3, 0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0.3, 0, 0.4]]):
        q_tgt = torch.tensor(q0 + np.array(delta), dtype=torch.float32, device=DEV)
        q_tgt = torch.clamp(q_tgt, E.q_lo, E.q_hi)
        for _ in range(40):
            E.apply_arm(q_tgt, grip_open=True)
        q_now = E.q0().cpu().numpy()
        track = float(np.linalg.norm(q_now - q_tgt.cpu().numpy()))
        fk_tcp, ee_tcp = E.tcp(), E.ee_frame_tcp()
        e = float(np.linalg.norm(fk_tcp - ee_tcp))
        errs.append(e)
        print(f"[droid-check] cfg{k}: arm-track_err={track*100:.2f}cm  FK-vs-ee={e*100:.2f}cm", flush=True)

    pear, _ = E.object_pose("pear")
    print(f"[droid-check] pear={np.round(pear,3).tolist()} tcp_now={np.round(E.tcp(),3).tolist()}", flush=True)

    ok = max(errs) < 0.02   # <2cm FK-vs-ee_frame across configs
    print(f"[droid-check] max FK-vs-ee err={max(errs)*100:.2f}cm  RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    print("[droid-check] CHECK DONE", flush=True)
    E.env.close()


try:
    main()
finally:
    simulation_app.close()
