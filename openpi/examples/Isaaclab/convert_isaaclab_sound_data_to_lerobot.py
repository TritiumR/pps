"""
Convert an IsaacLab RGB+sound dataset to LeRobot format.

Expected IsaacLab HDF5 keys per demo:
- obs/table_cam
- obs/wrist_cam
- obs/joint_pos
- obs/gripper_pos
- obs/joint_actions
- obs/mic1_log_mel
- obs/mic2_log_mel

Usage:
uv run examples/Isaaclab/convert_isaaclab_sound_data_to_lerobot.py \
    --data_file /path/to/generated_dataset_sound.hdf5 \
    --repo_name cn356/isaaclab_phone_sound \
    --prompt "pick up the phone"
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
SOUND_KEYS = ("obs/mic1_log_mel", "obs/mic2_log_mel")


def resize_image(image, size=IMAGE_SIZE):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def _image_feature():
    return {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    }


def _sound_feature(sound_shape: tuple[int, int]):
    return {
        "dtype": "float32",
        "shape": sound_shape,
        "names": ["mel", "time"],
    }


def _read_log_mel(value, *, key: str, step_idx: int) -> np.ndarray:
    log_mel = np.asarray(value, dtype=np.float32)

    # IsaacLab observations are normally (mel, time), but some recorders may keep
    # a singleton batch/channel dimension. Keep the converter strict after removing
    # harmless singleton axes so mismatched data fails early.
    while log_mel.ndim > 2 and 1 in (log_mel.shape[0], log_mel.shape[-1]):
        if log_mel.shape[0] == 1:
            log_mel = log_mel[0]
        elif log_mel.shape[-1] == 1:
            log_mel = log_mel[..., 0]

    if log_mel.ndim != 2:
        raise ValueError(
            f"Expected {key} at step {step_idx} to have shape (mel, time), got {log_mel.shape}."
        )
    if not np.isfinite(log_mel).all():
        log_mel = np.nan_to_num(log_mel, nan=-18.420680743952367, posinf=8.0349, neginf=-18.420680743952367)
    return log_mel.astype(np.float32, copy=False)


def _require_keys(trajectory, keys: tuple[str, ...], demo_name: str) -> None:
    missing = [key for key in keys if key not in trajectory]
    if missing:
        raise ValueError(f"Demo {demo_name} is missing required dataset keys: {missing}")


def main(
    data_file: str,
    repo_name: str,
    prompt: str,
    *,
    push_to_hub: bool = False,
    droid_action: bool = False,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    file = h5py.File(data_file, "r")
    demos = file["data"]
    demo_name_list = sorted(demos.keys())
    if not demo_name_list:
        raise ValueError(f"No demos found in {data_file}")

    first_demo_name = demo_name_list[0]
    first_trajectory = demos[first_demo_name]
    _require_keys(
        first_trajectory,
        (
            "obs/table_cam",
            "obs/wrist_cam",
            "obs/joint_pos",
            "obs/gripper_pos",
            "obs/joint_actions",
            *SOUND_KEYS,
        ),
        first_demo_name,
    )
    sound_shape = _read_log_mel(
        first_trajectory[SOUND_KEYS[0]][0],
        key=SOUND_KEYS[0],
        step_idx=0,
    ).shape

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": _image_feature(),
            "wrist_image_left": _image_feature(),
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
            "mic1_log_mel": _sound_feature(sound_shape),
            "mic2_log_mel": _sound_feature(sound_shape),
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for demo_name in tqdm(demo_name_list, desc="Converting episodes"):
        trajectory = demos[demo_name]
        _require_keys(
            trajectory,
            (
                "obs/table_cam",
                "obs/wrist_cam",
                "obs/joint_pos",
                "obs/gripper_pos",
                "obs/joint_actions",
                *SOUND_KEYS,
            ),
            demo_name,
        )

        trajectory_length = len(trajectory["obs/table_cam"])
        assert trajectory_length == len(trajectory["obs/wrist_cam"])
        assert trajectory_length == len(trajectory["obs/joint_pos"])
        assert trajectory_length == len(trajectory["obs/gripper_pos"])
        assert trajectory_length == len(trajectory["obs/joint_actions"])
        assert trajectory_length == len(trajectory[SOUND_KEYS[0]])
        assert trajectory_length == len(trajectory[SOUND_KEYS[1]])

        for step_idx in range(trajectory_length):
            exterior_image_1_left = resize_image(trajectory["obs/table_cam"][step_idx])
            wrist_image_left = resize_image(trajectory["obs/wrist_cam"][step_idx])
            joint_position = np.asarray(trajectory["obs/joint_pos"][step_idx][:7], dtype=np.float32)
            gripper_position = np.asarray(trajectory["obs/gripper_pos"][step_idx][:1], dtype=np.float32)

            mic1_log_mel = _read_log_mel(
                trajectory[SOUND_KEYS[0]][step_idx],
                key=SOUND_KEYS[0],
                step_idx=step_idx,
            )
            mic2_log_mel = _read_log_mel(
                trajectory[SOUND_KEYS[1]][step_idx],
                key=SOUND_KEYS[1],
                step_idx=step_idx,
            )
            if mic1_log_mel.shape != sound_shape or mic2_log_mel.shape != sound_shape:
                raise ValueError(
                    f"Sound shape changed in {demo_name} step {step_idx}: "
                    f"mic1={mic1_log_mel.shape}, mic2={mic2_log_mel.shape}, expected={sound_shape}."
                )

            if droid_action:
                actions = np.asarray(trajectory["actions"][step_idx], dtype=np.float32)[:8]
            else:
                actions = np.asarray(
                    trajectory["obs/joint_actions"][
                        min(step_idx + 1, len(trajectory["obs/joint_actions"]) - 1)
                    ],
                    dtype=np.float32,
                )[:8]

            dataset.add_frame(
                {
                    "exterior_image_1_left": exterior_image_1_left,
                    "wrist_image_left": wrist_image_left,
                    "joint_position": joint_position,
                    "gripper_position": gripper_position,
                    "mic1_log_mel": mic1_log_mel,
                    "mic2_log_mel": mic2_log_mel,
                    "actions": actions,
                    "task": prompt,
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
