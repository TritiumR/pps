import os, sys
_REPO=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_REPO)
for p in ("isaaclab","isaaclab_assets","isaaclab_tasks","isaaclab_rl","isaaclab_mimic"):
    sp=os.path.join(_REPO,"IsaacLab","source",p); sys.path.insert(0,sp)
from isaaclab.app import AppLauncher
import argparse
ap=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(enable_cameras=True, headless=True)
a=ap.parse_args()
app=AppLauncher(a).app
import gymnasium as gym, torch
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
try:
    cfg=parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device=a.device, num_envs=1)
    env=gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg).unwrapped
    env.reset()
    for _ in range(5): env.step(torch.zeros((1,)+env.action_space.shape[1:], device=env.device))
    print("PROBE scene_keys:", list(env.scene.keys()), flush=True)
    r=env.scene["robot"]
    print("PROBE robot_root_pose_w:", [round(float(x),3) for x in r.data.root_state_w[0,:7]], flush=True)
    print("PROBE robot_joint_names:", list(r.data.joint_names), flush=True)
    print("PROBE action_shape:", tuple(env.action_space.shape), flush=True)
    for k in ("object","cube"):
        if k in env.scene.keys():
            o=env.scene[k]
            print(f"PROBE {k}_pose_w:", [round(float(x),3) for x in o.data.root_state_w[0,:7]], flush=True)
    if "table_cam" in env.scene.keys(): print("PROBE has table_cam", flush=True)
    print("PROBE_OK", flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("PROBE_FAIL", repr(e), flush=True)
finally:
    app.close()
