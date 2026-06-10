# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
import cv2
from datetime import datetime
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--save_camera_videos",
    action="store_true",
    default=False,
    help="Save per-episode camera observations from replay into MP4 videos.",
)
parser.add_argument(
    "--camera_keys",
    type=str,
    nargs="+",
    default=["table_cam", "wrist_cam"],
    help="Camera observation keys to save when --save_camera_videos is enabled.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default="./videos/replay_demos",
    help="Directory where replay camera videos are stored.",
)
parser.add_argument("--video_fps", type=int, default=30, help="Frame rate for saved replay videos.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def sanitize_file_component(value: str) -> str:
    """Sanitize a string for safe file naming."""
    return value.replace(os.sep, "_").replace(" ", "_").replace(":", "_")


def prepare_video_frame(image: np.ndarray) -> np.ndarray:
    """Convert a camera observation into a uint8 BGR frame for OpenCV."""
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}.")

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(f"Expected image with 1, 3, or 4 channels, got shape {image.shape}.")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def close_video_recorders(recorder_state: dict):
    """Release any active writers for the current episode."""
    for camera_key, writer in recorder_state["writers"].items():
        writer.release()
        output_path = recorder_state["paths"][camera_key]
        frame_count = recorder_state["frame_counts"][camera_key]
        print(f"Saved {camera_key} video: {output_path} ({frame_count} frames)")
    recorder_state["writers"].clear()
    recorder_state["paths"].clear()
    recorder_state["frame_counts"].clear()
    recorder_state["episode_index"] = None
    recorder_state["episode_name"] = None


def record_camera_frames(
    obs: dict,
    env_id: int,
    recorder_state: dict,
    output_dir: str,
    fps: int,
    camera_keys: list[str],
):
    """Write one frame per requested camera from the current observation."""
    if not isinstance(obs, dict):
        return

    policy_obs = obs.get("policy", obs)
    if not isinstance(policy_obs, dict):
        return

    for camera_key in camera_keys:
        if camera_key not in policy_obs:
            continue

        image = policy_obs[camera_key][env_id]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        else:
            image = np.asarray(image)
        frame = prepare_video_frame(image)

        if camera_key not in recorder_state["writers"]:
            height, width = frame.shape[:2]
            episode_index = recorder_state["episode_index"]
            episode_name = sanitize_file_component(recorder_state["episode_name"])
            file_name = f"episode_{episode_index:04d}_{episode_name}_env_{env_id}_{camera_key}.mp4"
            output_path = os.path.join(output_dir, file_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            recorder_state["writers"][camera_key] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            recorder_state["paths"][camera_key] = output_path
            recorder_state["frame_counts"][camera_key] = 0

        recorder_state["writers"][camera_key].write(frame)
        recorder_state["frame_counts"][camera_key] += 1


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    print(f"Env name: {env_name}")
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    video_output_dir = None
    camera_recorders = None
    if args_cli.save_camera_videos:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"{sanitize_file_component(env_name)}_{timestamp}"
        video_output_dir = os.path.join(args_cli.video_output_dir, output_dir_name)
        os.makedirs(video_output_dir, exist_ok=True)
        camera_recorders = [
            {"episode_index": None, "episode_name": None, "writers": {}, "paths": {}, "frame_counts": {}}
            for _ in range(num_envs)
        ]
        print(f"Saving replay camera videos to: {video_output_dir}")

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            if args_cli.save_camera_videos:
                                close_video_recorders(camera_recorders[env_id])
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            if args_cli.save_camera_videos:
                                camera_recorders[env_id]["episode_index"] = next_episode_index
                                camera_recorders[env_id]["episode_name"] = episode_names[next_episode_index]
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                obs, _, _, _, _ = env.step(actions)

                if args_cli.save_camera_videos:
                    for env_id in range(num_envs):
                        if camera_recorders[env_id]["episode_index"] is None:
                            continue
                        record_camera_frames(
                            obs=obs,
                            env_id=env_id,
                            recorder_state=camera_recorders[env_id],
                            output_dir=video_output_dir,
                            fps=args_cli.video_fps,
                            camera_keys=args_cli.camera_keys,
                        )

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break
    if args_cli.save_camera_videos:
        for recorder_state in camera_recorders:
            close_video_recorders(recorder_state)
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
