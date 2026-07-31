"""
Minimal example script for converting a dataset collected in IsaacLab to LeRobot format.

Usage:
uv run examples/Isaaclab/convert_isaaclab_data_to_lerobot.py --data_file /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/Isaaclab/convert_isaaclab_data_to_lerobot.py --data_file /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from PIL import Image
from tqdm import tqdm
import tyro

def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(
    data_file: str,
    repo_name: str,
    prompt: str,
    *,
    push_to_hub: bool = False,
    droid_action: bool = False,
    resume: bool = False,
):
    output_path = HF_LEROBOT_HOME / repo_name
    dataset = None
    if resume:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {output_path}")
        dataset = LeRobotDataset(repo_id=repo_name, root=output_path)
    elif output_path.exists():
        # A non-resume conversion starts from a clean output directory.
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = dataset or LeRobotDataset.create(
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
                "shape": (
                    8,
                ),  # We will use joint *position* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # # Load language annotations
    # # Note: we load the DROID language annotations for this example, but you can manually define them for your own data
    # with (data_dir / "aggregated-annotations-030724.json").open() as f:
    #     language_annotations = json.load(f)

    # Load the dataset
    file = h5py.File(data_file, "r")
    demos = file["data"]
    demo_name_list = sorted(demos.keys())
    if resume:
        completed_episodes = dataset.meta.total_episodes
        if completed_episodes > len(demo_name_list):
            raise ValueError(
                f"Existing dataset has {completed_episodes} episodes but source has only {len(demo_name_list)}"
            )
        completed_frames = sum(
            len(demos[name]["obs/table_cam"])
            for name in demo_name_list[:completed_episodes]
        )
        if completed_frames != dataset.meta.total_frames:
            raise ValueError(
                "Existing dataset is not a valid prefix of the source: "
                f"metadata has {dataset.meta.total_frames} frames, expected {completed_frames}"
            )
        demo_name_list = demo_name_list[completed_episodes:]
        print(
            f"Resuming {repo_name} at episode {completed_episodes}; "
            f"converting {len(demo_name_list)} remaining episodes"
        )
        dataset.start_image_writer(num_processes=5, num_threads=10)
        # Discard PNGs for an episode that was never committed to metadata.
        # Completed image episodes are embedded in parquet and already removed.
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.clear_episode_buffer()


    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for demo_name in tqdm(demo_name_list, desc="Converting episodes"):
        trajectory = demos[demo_name]

        # print("trajectory: ", trajectory.keys())

        language_instruction = prompt

        trajectory_length = len(trajectory["obs/table_cam"])
        assert trajectory_length == len(trajectory["obs/wrist_cam"])
        assert trajectory_length == len(trajectory["obs/joint_pos"])
        assert trajectory_length == len(trajectory["obs/gripper_pos"])
        assert trajectory_length == len(trajectory["obs/joint_actions"])
        has_sound = "obs/mic1_log_mel" in trajectory and "obs/mic2_log_mel" in trajectory

        # Write to LeRobot dataset
        for step_idx in range(trajectory_length):
            exterior_image_1_left = trajectory["obs/table_cam"][step_idx]
            wrist_image_left = trajectory["obs/wrist_cam"][step_idx]
            exterior_image_1_left = resize_image(exterior_image_1_left, (320, 180))
            wrist_image_left = resize_image(wrist_image_left, (320, 180))
            joint_position = np.asarray(
                trajectory["obs/joint_pos"][step_idx][:7], dtype=np.float32
            )
            gripper_position = np.asarray(
                trajectory["obs/gripper_pos"][step_idx][:1], dtype=np.float32
            )
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
            # print("gripper_position: ", gripper_position)
            if droid_action:
                actions = trajectory["actions"][step_idx]
            else:
                actions = trajectory["obs/joint_actions"][
                    min(step_idx + 1, len(trajectory["obs/joint_actions"]) - 1)
                ]  # use next step recorded last joint action as action

            # print("actions: ", actions)

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
                    "task": language_instruction,
                }
            )
        dataset.save_episode()

    dataset.stop_image_writer()

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
