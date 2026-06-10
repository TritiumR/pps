"""
Convert an IsaacLab thermal dataset to LeRobot format.

This variant keeps the original RGB camera streams and additionally stores thermal camera streams
at the same resolution.
"""

import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from PIL import Image
from tqdm import tqdm
import tyro


IMAGE_SIZE = (320, 180)
IMAGE_SHAPE = (180, 320, 3)


def resize_image(image, size=IMAGE_SIZE):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def _image_feature():
    return {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    }


def main(
    data_file: str,
    repo_name: str,
    prompt: str,
    *,
    push_to_hub: bool = False,
    droid_action: bool = False,
):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": _image_feature(),
            "wrist_image_left": _image_feature(),
            "thermal_exterior_image_1_left": _image_feature(),
            "thermal_wrist_image_left": _image_feature(),
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
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    file = h5py.File(data_file, "r")
    demos = file["data"]
    demo_name_list = sorted(demos.keys())

    for demo_name in tqdm(demo_name_list, desc="Converting episodes"):
        trajectory = demos[demo_name]
        language_instruction = prompt

        trajectory_length = len(trajectory["obs/table_cam"])
        assert trajectory_length == len(trajectory["obs/wrist_cam"])
        assert trajectory_length == len(trajectory["obs/thermal_table_cam"])
        assert trajectory_length == len(trajectory["obs/thermal_wrist_cam"])
        assert trajectory_length == len(trajectory["obs/joint_pos"])
        assert trajectory_length == len(trajectory["obs/gripper_pos"])
        assert trajectory_length == len(trajectory["obs/joint_actions"])

        for step_idx in range(trajectory_length):
            exterior_image_1_left = resize_image(trajectory["obs/table_cam"][step_idx])
            wrist_image_left = resize_image(trajectory["obs/wrist_cam"][step_idx])
            thermal_exterior_image_1_left = resize_image(
                trajectory["obs/thermal_table_cam"][step_idx]
            )
            thermal_wrist_image_left = resize_image(
                trajectory["obs/thermal_wrist_cam"][step_idx]
            )
            joint_position = np.asarray(
                trajectory["obs/joint_pos"][step_idx][:7], dtype=np.float32
            )
            gripper_position = np.asarray(
                trajectory["obs/gripper_pos"][step_idx][:1], dtype=np.float32
            )
            if droid_action:
                actions = trajectory["actions"][step_idx]
            else:
                actions = trajectory["obs/joint_actions"][
                    min(step_idx + 1, len(trajectory["obs/joint_actions"]) - 1)
                ]

            dataset.add_frame(
                {
                    "exterior_image_1_left": exterior_image_1_left,
                    "wrist_image_left": wrist_image_left,
                    "thermal_exterior_image_1_left": thermal_exterior_image_1_left,
                    "thermal_wrist_image_left": thermal_wrist_image_left,
                    "joint_position": joint_position,
                    "gripper_position": gripper_position,
                    "actions": actions,
                    "task": language_instruction,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
