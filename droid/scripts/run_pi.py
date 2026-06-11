# ruff: noqa

import contextlib
import csv
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Annotated
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import pickle

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    # left_camera_id: str = "31079775"  # e.g., "24259877"
    left_camera_id: str = "37414691"  # after 01/03/2026
    right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    wrist_camera_id: str = "16131644"  # e.g., "13062452"
    # instruction: str = "stack the red cube on the blue cube"
    # instruction: str = "stack the cubes"
    instruction: str | None = None
    # instruction: str = "pick up the cucumber toy and hit the pigs"
    # instruction: str = "fold the clothes"
    # instruction: str = "put the blue cube into the clear plastic cup"
    # instruction : str = "Pick up the flower and insert it in the vase." #failed. Policy picked up the flower head and force it into the vase.
    # instruction: str = "Pick up the flower by its stem and insert the stem into the vase." #failed. Instruction is not followed. Behavior is like previous one.
    rotate_camera: bool = True
    exp_name: str | None = None

    # Policy parameters
    external_camera: str | None = (
        "left"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )
    noise_std: float = 0.0

    # Rollout parameters
    max_timesteps: int = 1200
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    action_space: str = "joint_position"
    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )
    websocket_open_timeout: float = 60.0
    websocket_ping_interval: float = 20.0
    websocket_ping_timeout: float = 600.0

    visualize: bool = False

    print_gripper_action: bool = False

    fix_exposure: Annotated[
        bool,
        tyro.conf.arg(aliases=["--fix_exposure"]),
    ] = False


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert args.external_camera is not None and args.external_camera in [
        "left",
        "right",
    ], f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    if args.action_space != "joint_position":
        raise ValueError(
            "This OpenPI/DSRL DROID rollout path expects absolute joint-position "
            "actions. Use --action-space joint_position."
        )

    # Initialize the Panda environment. DSRL/OpenPI action chunks are absolute
    # joint positions, but use velocity commands for the gripper to match teleop
    # behavior under contact.
    fixed_exposure_camera_kwargs = {
        "16131644": {"auto_exposure_gain": False, "exposure": 60, "gain": 7},
        "37414691": {"auto_exposure_gain": False, "exposure": 60, "gain": 31},
    }
    camera_kwargs = fixed_exposure_camera_kwargs if args.fix_exposure else {}
    env = RobotEnv(
        action_space=args.action_space,
        gripper_action_space="velocity",
        camera_kwargs=camera_kwargs,
    )
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host,
        args.remote_port,
        open_timeout=args.websocket_open_timeout,
        ping_interval=args.websocket_ping_interval,
        ping_timeout=args.websocket_ping_timeout,
    )

    # df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    results_log = []
    visualize_steps = []
    log_columns = ["success", "duration", "video_filename"]

    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

    while True:
        # instruction = input("Enter instruction: ")
        instruction = args.instruction

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        # added to track how many steps were slept
        slept_steps = 0

        inference_steps = 0
        previous_action = np.zeros((8,))
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])
                visualize_step = dict()

                if args.visualize:
                    visualize_step["image"] = curr_obs["left_image"]

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    inference_steps += 1

                    if args.print_gripper_action:
                        print("gripper position: ", curr_obs["gripper_position"])

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    if instruction is not None:
                        request_data = {
                            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                                curr_obs[f"{args.external_camera}_image"], 224, 224
                            ),
                            "observation/wrist_image_left": image_tools.resize_with_pad(
                                curr_obs["wrist_image"], 224, 224
                            ),
                            "dsrl/observation/exterior_image_1_left": curr_obs[
                                f"{args.external_camera}_image"
                            ],
                            "dsrl/observation/wrist_image_left": curr_obs[
                                "wrist_image"
                            ],
                            "observation/joint_position": curr_obs["joint_position"],
                            "observation/gripper_position": curr_obs["gripper_position"],
                            "prompt": instruction,
                            "step": inference_steps,
                            "visualize": args.visualize,
                        }
                    else:
                        request_data = {
                            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                                curr_obs[f"{args.external_camera}_image"], 224, 224
                            ),
                            "observation/wrist_image_left": image_tools.resize_with_pad(
                                curr_obs["wrist_image"], 224, 224
                            ),
                            "dsrl/observation/exterior_image_1_left": curr_obs[
                                f"{args.external_camera}_image"
                            ],
                            "dsrl/observation/wrist_image_left": curr_obs[
                                "wrist_image"
                            ],
                            "observation/joint_position": curr_obs["joint_position"],
                            "observation/gripper_position": curr_obs["gripper_position"],
                            "step": inference_steps,
                            "visualize": args.visualize,
                        }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        results = policy_client.infer(request_data)
                        pred_action_chunk = results["actions"]
                        if args.print_gripper_action:
                            print("gripper action: ", pred_action_chunk[:, -1])
                        if args.visualize:
                            visualize_step["observation"] = request_data
                            visualize_vectors_step = results["visualize_vectors_step"]
                            visualize_step["num_steps"] = visualize_vectors_step["num_steps"]
                            visualize_step["inputs"] = visualize_vectors_step["inputs"]
                            visualize_step["vectors"] = visualize_vectors_step["vectors"]
                            visualize_step["base_vectors"] = visualize_vectors_step["base_vectors"]
                            visualize_step["steer_vectors"] = visualize_vectors_step["steer_vectors"]
                            visualize_step["actions"] = pred_action_chunk
                            if "steer_target" in visualize_vectors_step:
                                visualize_step["steer_target"] = visualize_vectors_step["steer_target"]
                            if "steer_target_idx" in visualize_vectors_step:
                                visualize_step["steer_target_idx"] = visualize_vectors_step["steer_target_idx"]
                            if "steer_masks" in visualize_vectors_step:
                                visualize_step["steer_masks"] = visualize_vectors_step["steer_masks"]
                            if "steer_delta_vectors" in visualize_vectors_step:
                                visualize_step["steer_delta_vectors"] = visualize_vectors_step["steer_delta_vectors"]
                            if "mimic_vectors" in visualize_vectors_step:
                                visualize_step["mimic_vectors"] = visualize_vectors_step["mimic_vectors"]
                            else:
                                visualize_step["mimic_vectors"] = None
                    assert pred_action_chunk.shape[1] == 8

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if args.visualize:
                    visualize_steps.append(visualize_step)

                # print("action: ", action - previous_action)
                # previous_action = action

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # print("gripper closing")
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], -np.ones((1,))])
                    # action = np.concatenate([action[:-1], np.zeros((1,))])

                # if action[-1] > 0.1:
                #     action = np.concatenate([action[:-1], np.array([action[-1] + 0.05])])
                # else:
                #     action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                if args.action_space == "joint_velocity":
                    action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                    slept_steps += 1
            except KeyboardInterrupt:
                break

        print(f"Slept for {slept_steps} steps")

        if input("save video? (enter y or n) ").lower() == "y":
            video = np.stack(video)
            if args.exp_name is not None:
                save_filename = "video_" + args.exp_name + "_" + timestamp
            else:
                save_filename = "video_" + timestamp
            ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        # success: str | float | None = None
        # while not isinstance(success, float):
        #     success = input(
        #         "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
        #     )
        #     if success == "y":
        #         success = 1.0
        #     elif success == "n":
        #         success = 0.0

        #     success = float(success) / 100
        #     if not (0 <= success <= 1):
        #         print(f"Success must be a number in [0, 100] but got: {success * 100}")

        # results_log.append({
        #     "success": success,
        #     "duration": t_step,
        #     "video_filename": save_filename,
        # })

        # df = df.append(
        #     {
        #         "success": success,
        #         "duration": t_step,
        #         "video_filename": save_filename,
        #     },
        #     ignore_index=True,
        # )

        # save visualize steps into a file
        if args.visualize:
            timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
            pkl_path = os.path.join(results_root, args.instruction or "none", args.exp_name or "none")
            os.makedirs(pkl_path, exist_ok=True)
            pkl_filename = os.path.join(pkl_path, f"visualize_steps_{timestamp}.pkl")
            with open(pkl_filename, "wb") as f:
                pickle.dump(visualize_steps, f)
            print(f"Saved visualize steps to {pkl_filename}")

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break

        env.reset()
        visualize_steps = []

    # timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    # csv_filename = os.path.join("results", f"eval_{timestamp}.csv")

    # with open(csv_filename, "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=log_columns)
    #     writer.writeheader()
    #     for data in results_log:
    #         writer.writerow(data)

    # # df.to_csv(csv_filename)
    # print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # print(key)
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension and convert to RGB
    assert left_image is not None or right_image is not None
    assert wrist_image is not None

    if left_image is not None:
        left_image = left_image[..., :3]
        left_image = left_image[..., ::-1]
        if args.rotate_camera:
            # rotate 180 degrees
            left_image = left_image[::-1, ::-1]
    if right_image is not None:
        right_image = right_image[..., :3]
        right_image = right_image[..., ::-1]
        if args.rotate_camera:
            right_image = right_image[::-1, ::-1]

    wrist_image = wrist_image[..., :3]
    wrist_image = wrist_image[..., ::-1]

    # Add Gaussian noise to images if noise_std is not 0
    if args.noise_std > 0:
        if args.external_camera == "left" and left_image is not None:
            noise = np.random.normal(0, args.noise_std, left_image.shape)
            left_image = np.clip(left_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif args.external_camera == "right" and right_image is not None:
            noise = np.random.normal(0, args.noise_std, right_image.shape)
            right_image = np.clip(right_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        noise = np.random.normal(0, args.noise_std, wrist_image.shape)
        wrist_image = np.clip(wrist_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        concatenated_image = None
        for image in [left_image, wrist_image, right_image]:
            if image is not None:
                if concatenated_image is None:
                    concatenated_image = image.copy()
                else:
                    concatenated_image = np.concatenate([concatenated_image, image], axis=1)
        combined_image = Image.fromarray(concatenated_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
