"""
Pointcloud-aware conversion script for IsaacLab datasets to LeRobot format.

This keeps the behavior of ``convert_isaaclab_data_to_lerobot.py`` and additionally
stores point-cloud positions and colors. If the source cloud contains more than
2048 points, it is downsampled with farthest-point sampling while keeping colors
aligned with the sampled positions.
"""

import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import tyro


MAX_POINT_COUNT = 8192


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def farthest_point_sample_indices(points: np.ndarray, target_count: int) -> np.ndarray:
    """Return farthest-point-sampling indices for an ``(N, 3)`` point set."""
    num_points = points.shape[0]
    if target_count >= num_points:
        return np.arange(num_points, dtype=np.int64)

    points_tensor = torch.as_tensor(points, dtype=torch.float32)
    selected_indices = torch.empty(target_count, dtype=torch.long)
    min_distances = torch.full((num_points,), float("inf"), dtype=torch.float32)

    centroid = points_tensor.mean(dim=0, keepdim=True)
    farthest_index = torch.argmax(torch.sum((points_tensor - centroid) ** 2, dim=1))

    for sample_idx in range(target_count):
        selected_indices[sample_idx] = farthest_index
        current_point = points_tensor[farthest_index]
        current_distances = torch.sum((points_tensor - current_point) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, current_distances)
        farthest_index = torch.argmax(min_distances)

    return selected_indices.cpu().numpy()


def process_point_cloud(
    point_positions: np.ndarray,
    point_color: np.ndarray,
    max_point_count: int = MAX_POINT_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep point positions and colors aligned while FPS-downsampling oversized clouds."""
    point_positions = np.asarray(point_positions, dtype=np.float32)
    point_color = np.asarray(point_color, dtype=np.float32)

    if point_positions.shape != point_color.shape:
        raise ValueError(
            f"Point position/color shape mismatch: {point_positions.shape} vs {point_color.shape}"
        )

    if point_positions.ndim != 2 or point_positions.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {point_positions.shape}")

    valid_mask = np.isfinite(point_positions).all(axis=1)
    if valid_mask.any():
        point_positions = point_positions[valid_mask]
        point_color = point_color[valid_mask]
    else:
        point_positions = np.zeros((1, 3), dtype=np.float32)
        point_color = np.zeros((1, 3), dtype=np.float32)

    point_color = np.nan_to_num(point_color, nan=0.0, posinf=255.0, neginf=0.0)

    if point_positions.shape[0] > max_point_count:
        sampled_indices = farthest_point_sample_indices(point_positions, max_point_count)
        point_positions = point_positions[sampled_indices]
        point_color = point_color[sampled_indices]

    return point_positions, point_color


def main(
    data_file: str,
    repo_name: str,
    prompt: str,
    *,
    push_to_hub: bool = False,
    droid_action: bool = False,
    max_point_count: int = MAX_POINT_COUNT,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    file = h5py.File(data_file, "r")
    demos = file["data"]
    demo_name_list = sorted(demos.keys())
    if not demo_name_list:
        raise ValueError(f"No demos found in {data_file}")

    first_trajectory = demos[demo_name_list[0]]
    if "obs/point_positions" not in first_trajectory or "obs/point_color" not in first_trajectory:
        raise ValueError(
            "This converter expects a pointcloud dataset with 'obs/point_positions' and 'obs/point_color'."
        )

    point_count = first_trajectory["obs/point_positions"].shape[1]
    point_feature_shape = (min(point_count, max_point_count), 3)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
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
            "point_position": {
                "dtype": "float32",
                "shape": point_feature_shape,
                "names": ["point", "xyz"],
            },
            "point_color": {
                "dtype": "float32",
                "shape": point_feature_shape,
                "names": ["point", "rgb"],
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

    for demo_name in tqdm(demo_name_list, desc="Converting episodes"):
        trajectory = demos[demo_name]
        language_instruction = prompt

        trajectory_length = len(trajectory["obs/table_cam"])
        assert trajectory_length == len(trajectory["obs/wrist_cam"])
        assert trajectory_length == len(trajectory["obs/joint_pos"])
        assert trajectory_length == len(trajectory["obs/gripper_pos"])
        assert trajectory_length == len(trajectory["obs/joint_actions"])
        assert trajectory_length == len(trajectory["obs/point_positions"])
        assert trajectory_length == len(trajectory["obs/point_color"])

        for step_idx in range(trajectory_length):
            exterior_image_1_left = trajectory["obs/table_cam"][step_idx]
            wrist_image_left = trajectory["obs/wrist_cam"][step_idx]
            exterior_image_1_left = resize_image(exterior_image_1_left, (320, 180))
            wrist_image_left = resize_image(wrist_image_left, (320, 180))

            joint_position = np.asarray(trajectory["obs/joint_pos"][step_idx][:7], dtype=np.float32)
            gripper_position = np.asarray(trajectory["obs/gripper_pos"][step_idx][:1], dtype=np.float32)

            if droid_action:
                actions = np.asarray(trajectory["actions"][step_idx], dtype=np.float32)
            else:
                actions = np.asarray(
                    trajectory["obs/joint_actions"][
                        min(step_idx + 1, len(trajectory["obs/joint_actions"]) - 1)
                    ],
                    dtype=np.float32,
                )

            point_position, point_color = process_point_cloud(
                trajectory["obs/point_positions"][step_idx],
                trajectory["obs/point_color"][step_idx],
                max_point_count=max_point_count,
            )

            dataset.add_frame(
                {
                    "exterior_image_1_left": exterior_image_1_left,
                    "wrist_image_left": wrist_image_left,
                    "joint_position": joint_position,
                    "gripper_position": gripper_position,
                    "point_position": point_position,
                    "point_color": point_color,
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
