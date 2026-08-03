"""Re-render path check: rebuild the env from env_args, reset_to a stored state, replay actions.

Verifies that stepping the stored actions reproduces the stored state sequence, i.e. that the
replay path is deterministic.

    python setup/06_replay_check.py [n_steps]
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import h5py
import mimicgen  # noqa: F401  (registers *_D0 envs with robosuite)
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

# robomimic needs the obs-modality registry initialized before env use
ObsUtils.initialize_obs_modality_mapping_from_dict(
    {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                 "robot0_joint_pos", "object"]}
)

from mujoco_eval import paths  # noqa: E402

DATA = paths.DATA
TASKS = ("stack_d0", "stack_three_d0", "square_d0")


def check(task, n_steps=20):
    path = f"{DATA}/{task}/demo.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False,
        use_image_obs=False,
    )
    with h5py.File(path, "r") as f:
        d0 = f["data/demo_0"]
        states = d0["states"][()]
        actions = d0["actions"][()]
        model_xml = d0.attrs["model_file"]

    env.reset()
    env.reset_to({"states": states[0], "model": model_xml})
    errs = []
    for t in range(min(n_steps, len(actions) - 1)):
        env.step(actions[t])
        sim_state = env.get_state()["states"]
        errs.append(np.abs(sim_state - states[t + 1]).max())
    print(f"{task}: env={env_meta['env_name']} "
          f"open-loop state err over {len(errs)} steps: "
          f"max={max(errs):.2e} final={errs[-1]:.2e}")

    # state-sync check: reset_to(states[t]) must reproduce stored obs
    with h5py.File(path, "r") as f:
        d0 = f["data/demo_0"]
        obs_eef = d0["obs/robot0_eef_pos"][()]
        obs_obj = d0["obs/object"][()]
        T = obs_eef.shape[0]
        sync_errs = []
        for t in (0, T // 2, T - 1):
            ob = env.reset_to({"states": d0["states"][t]})
            sync_errs.append(max(
                np.abs(ob["robot0_eef_pos"] - obs_eef[t]).max(),
                np.abs(ob["object"] - obs_obj[t]).max(),
            ))
    print(f"    reset_to(state[t]) obs err (t=0,mid,end): "
          f"{['%.1e' % e for e in sync_errs]}")
    return env


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for task in TASKS:
        check(task, n)
