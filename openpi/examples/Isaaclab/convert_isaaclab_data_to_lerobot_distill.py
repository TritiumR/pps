"""
Minimal example script for converting a dataset collected in IsaacLab to LeRobot format.

Usage:
uv run examples/Isaaclab/convert_isaaclab_data_to_lerobot.py --data_file /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/Isaaclab/convert_isaaclab_data_to_lerobot.py --data_file /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import shutil
from typing import Any

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config

# from PIL import Image
from tqdm import tqdm
import tyro
import torch

import copy

ACTION_HORIZON = 10
ACTION_DIM = 32

def main(
    data_file: str,
    repo_name: str,
    *,
    push_to_hub: bool = False,
    model_name: str,
    checkpoint_dir: str,
    prompt: str,
    augment_factor: int = 1,
):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            # "exterior_image_2_left": {
            #     "dtype": "image",
            #     "shape": (180, 320, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "mic1_log_mel": {
                "dtype": "float32",
                "shape": (80, 198),
                "names": ["mel", "time"],
            },
            "mic2_log_mel": {
                "dtype": "float32",
                "shape": (80, 198),
                "names": ["mel", "time"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10, 8),
                "names": ["actions"],
            },
            "noise": {
                "dtype": "float32",
                "shape": (10, 8),
                "names": ["noise"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # load checkpoint
    config = _config.get_config(model_name)

    vla_policy = policy_config.create_trained_policy(config, checkpoint_dir)

    # # Load language annotations
    # # Note: we load the DROID language annotations for this example, but you can manually define them for your own data
    # with (data_dir / "aggregated-annotations-030724.json").open() as f:
    #     language_annotations = json.load(f)

    # Load the dataset
    file = h5py.File(data_file, "r")
    demos = file["data"]
    demo_name_list = sorted(demos.keys())

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for demo_name in tqdm(demo_name_list, desc="Converting episodes"):
        trajectory = demos[demo_name]

        # print("trajectory: ", trajectory.keys())

        # Assign a dummy language instruction since proxy model is not trained on language instructions
        language_instruction = prompt

        trajectory_length = len(trajectory["obs/table_cam"])
        assert trajectory_length == len(trajectory["obs/wrist_cam"])
        assert trajectory_length == len(trajectory["obs/joint_pos"])
        assert trajectory_length == len(trajectory["obs/gripper_pos"])
        assert trajectory_length == len(trajectory["obs/joint_actions"])
        has_sound = "obs/mic1_log_mel" in trajectory and "obs/mic2_log_mel" in trajectory

        # Write to LeRobot dataset
        for step_idx in tqdm(range(trajectory_length), desc="Converting steps"):
            exterior_image_1_left = trajectory["obs/table_cam"][step_idx]
            wrist_image_left = trajectory["obs/wrist_cam"][step_idx]
            joint_position = trajectory["obs/joint_pos"][step_idx][:7]
            gripper_position = trajectory["obs/gripper_pos"][step_idx][:1]
            mic1_log_mel = (
                np.asarray(trajectory["obs/mic1_log_mel"][step_idx], dtype=np.float32)
                if has_sound
                else np.zeros((80, 198), dtype=np.float32)
            )
            mic2_log_mel = (
                np.asarray(trajectory["obs/mic2_log_mel"][step_idx], dtype=np.float32)
                if has_sound
                else np.zeros((80, 198), dtype=np.float32)
            )

            obs: dict[str, Any] = {
                "observation/exterior_image_1_left": exterior_image_1_left,
                "observation/wrist_image_left": wrist_image_left,
                "observation/joint_position": joint_position,
                "observation/gripper_position": gripper_position,
                "observation/mic1_log_mel": mic1_log_mel,
                "observation/mic2_log_mel": mic2_log_mel,
                "prompt": prompt,
            }

            for augment_idx in range(augment_factor):
                # Generate the noise
                noise = np.random.randn(ACTION_HORIZON, ACTION_DIM).astype(
                    np.float32
                )
                with torch.no_grad():
                    # Get full action chunk from base policy
                    actions = vla_policy.infer(copy.deepcopy(obs), noise=noise)[
                        "actions"
                    ]
                    actions = actions.astype(np.float32)

                dataset.add_frame(
                    {
                        "exterior_image_1_left": exterior_image_1_left,
                        "wrist_image_left": wrist_image_left,
                        "joint_position": joint_position,
                        "gripper_position": gripper_position,
                        "mic1_log_mel": mic1_log_mel,
                        "mic2_log_mel": mic2_log_mel,
                        # Important: we use joint position actions here since it is hard for simulation to simulate joint velocity
                        "actions": actions,
                        "noise": noise[:, :8],
                        "task": language_instruction,
                    }
                )
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
