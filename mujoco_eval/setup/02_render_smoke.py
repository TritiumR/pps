"""Run a headless EGL render smoke test for robosuite."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
)
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import pathlib
import time

import imageio
import numpy as np
import robosuite
from robosuite import load_controller_config

OUT_DIR = str(pathlib.Path(__file__).resolve().parent / "smoke_frames")
CAMERAS = ["agentview", "robot0_eye_in_hand"]
NUM_STEPS = 50
SAVE_STEPS = (0, 24, 49)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctrl_cfg = load_controller_config(default_controller="JOINT_POSITION")
    env = robosuite.make(
        "Stack",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERAS,
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    obs = env.reset()
    print("robosuite:", robosuite.__version__)
    print("controller type:", ctrl_cfg["type"])
    print("action_dim:", env.action_dim)
    print("obs keys:", sorted(obs.keys()))

    low, high = env.action_spec
    rng = np.random.default_rng(0)
    n_saved = 0
    t0 = time.time()

    for i in range(NUM_STEPS):
        action = rng.uniform(low, high)
        obs, _, _, _ = env.step(action)

        if i in SAVE_STEPS:
            for cam in CAMERAS:
                frame = obs[f"{cam}_image"][::-1]
                imageio.imwrite(
                    os.path.join(OUT_DIR, f"step{i:03d}_{cam}.png"),
                    frame,
                )
                n_saved += 1

    dt = time.time() - t0
    print(
        f"{NUM_STEPS} env steps in {dt:.2f}s -> "
        f"{NUM_STEPS / dt:.2f} steps/sec "
        f"(incl. 2 cameras @ 256x256 per step)"
    )
    print(f"saved {n_saved} frames to {OUT_DIR}")
    env.close()


if __name__ == "__main__":
    main()