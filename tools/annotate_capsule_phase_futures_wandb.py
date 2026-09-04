#!/usr/bin/env python3
"""Annotate capsule phases, render terminal future-qpos trails, and upload videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "capsule" / "generated_dataset.hdf5"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "capsule_phase_future_qpos"

PHASE_OPEN_LID = 0
PHASE_GRASP_POD = 1
PHASE_PLACE_POD = 2
PHASE_NAMES = {
    PHASE_OPEN_LID: "OPEN_LID",
    PHASE_GRASP_POD: "GRASP_POD",
    PHASE_PLACE_POD: "PLACE_POD",
}

_RENDER_WORKER_DATA: h5py.File | None = None
_RENDER_WORKER_SIDECAR: dict[str, np.ndarray] | None = None

# CameraCfg from the capsule Droid visuomotor task. The quaternion is wxyz and
# maps ROS optical-camera coordinates (+x right, +y down, +z forward) to link0.
TABLE_CAMERA_POS_LINK0 = np.asarray(
    (0.004620336834421451, -0.5388594867462788, 0.454018368138419),
    dtype=np.float64,
)
TABLE_CAMERA_QUAT_LINK0_WXYZ = np.asarray(
    (-0.5078392969, 0.7575422903, -0.3175587775, 0.2595868830),
    dtype=np.float64,
)
TABLE_CAMERA_FOCAL_LENGTH = 1.0476
TABLE_CAMERA_HORIZONTAL_APERTURE = 2.5452
TABLE_CAMERA_VERTICAL_APERTURE = 1.4721


def _natural_demo_names(data: h5py.Group) -> list[str]:
    names = [name for name in data.keys() if name.startswith("demo_")]
    return sorted(names, key=lambda name: int(name.removeprefix("demo_")))


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray | None = None, translation: tuple[float, float, float] | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _translate(x: float, y: float, z: float) -> np.ndarray:
    return _transform(translation=(x, y, z))


def _rotate_z(theta: float) -> np.ndarray:
    cosine = math.cos(float(theta))
    sine = math.sin(float(theta))
    return _transform(
        rotation=np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    )


def _quat_transform_wxyz(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    return _transform(rotation=_quat_matrix_wxyz(np.asarray(quaternion)))


def _panda_chain_and_ee_pose_link0(joint_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(joint_position, dtype=np.float64).reshape(-1)
    if qpos.size < 7:
        raise ValueError(f"expected at least 7 robot joints, got {qpos.size}")
    static = (
        _translate(0.0, 0.0, 0.333),
        _quat_transform_wxyz((0.707107, -0.707107, 0.0, 0.0)),
        _translate(0.0, -0.316, 0.0) @ _quat_transform_wxyz((0.707107, 0.707107, 0.0, 0.0)),
        _translate(0.0825, 0.0, 0.0) @ _quat_transform_wxyz((0.707107, 0.707107, 0.0, 0.0)),
        _translate(-0.0825, 0.384, 0.0) @ _quat_transform_wxyz((0.707107, -0.707107, 0.0, 0.0)),
        _quat_transform_wxyz((0.707107, 0.707107, 0.0, 0.0)),
        _translate(0.088, 0.0, 0.0) @ _quat_transform_wxyz((0.707107, 0.707107, 0.0, 0.0)),
    )
    pose = np.eye(4, dtype=np.float64)
    points = [pose[:3, 3].copy()]
    for joint_index, offset in enumerate(static):
        pose = pose @ offset
        points.append(pose[:3, 3].copy())
        pose = pose @ _rotate_z(float(qpos[joint_index]))
    link8 = pose @ _translate(0.0, 0.0, 0.107)
    end_effector = link8 @ _translate(0.0, 0.0, 0.171574)
    points.extend((link8[:3, 3].copy(), end_effector[:3, 3].copy()))
    return np.asarray(points, dtype=np.float64), end_effector


def panda_chain_points_link0(joint_position: np.ndarray) -> np.ndarray:
    """Return Panda joint/link centers plus the policy end-effector in link0."""
    return _panda_chain_and_ee_pose_link0(joint_position)[0]


def panda_gripper_wireframe_link0(
    joint_position: np.ndarray,
    gripper_command: float | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Return an FK gripper glyph, optionally driven by a binary close command."""
    _, end_effector = _panda_chain_and_ee_pose_link0(joint_position)
    half_gap = (
        0.05
        if gripper_command is None
        else 0.043 - float(np.clip(gripper_command, 0.0, 1.0)) * (0.043 - 0.006)
    )
    local_points = np.asarray(
        [
            (0.0, 0.0, -0.08),
            (0.0, 0.0, 0.00),
            (0.0, -half_gap, 0.00),
            (0.0, -half_gap, 0.08),
            (0.0, half_gap, 0.00),
            (0.0, half_gap, 0.08),
            (0.0, 0.0, 0.10),
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate((local_points, np.ones((len(local_points), 1))), axis=1)
    points_link0 = (end_effector @ homogeneous.T).T[:, :3]
    lines = ((0, 1), (2, 4), (2, 3), (4, 5), (1, 6))
    return points_link0, lines


def project_table_camera(points_link0: np.ndarray, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_link0, dtype=np.float64)
    rotation_link0_camera = _quat_matrix_wxyz(TABLE_CAMERA_QUAT_LINK0_WXYZ)
    points_camera = (rotation_link0_camera.T @ (points - TABLE_CAMERA_POS_LINK0).T).T
    depth = points_camera[:, 2]
    fx = TABLE_CAMERA_FOCAL_LENGTH * float(width) / TABLE_CAMERA_HORIZONTAL_APERTURE
    fy = TABLE_CAMERA_FOCAL_LENGTH * float(height) / TABLE_CAMERA_VERTICAL_APERTURE
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = fx * points_camera[:, 0] / np.maximum(depth, 1e-8) + 0.5 * float(width)
    pixels[:, 1] = fy * points_camera[:, 1] / np.maximum(depth, 1e-8) + 0.5 * float(height)
    valid = (
        (depth > 1e-4)
        & (pixels[:, 0] >= -0.1 * width)
        & (pixels[:, 0] <= 1.1 * width)
        & (pixels[:, 1] >= -0.1 * height)
        & (pixels[:, 1] <= 1.1 * height)
    )
    return np.rint(pixels).astype(np.int32), valid


def infer_capsule_phase(
    *,
    capsule_joint_position: np.ndarray,
    gripper_position: np.ndarray,
    end_effector_position: np.ndarray,
    can_root_pose: np.ndarray,
    lid_joint_index: int,
    lid_open_threshold: float = -0.5,
    gripper_open_atol: float = 0.01,
    grasp_distance_threshold: float = 0.1,
    gripper_closed_threshold: float = 0.005,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Reconstruct cumulative OPEN_LID -> GRASP_POD -> PLACE_POD phases."""
    capsule_qpos = np.asarray(capsule_joint_position, dtype=np.float64)
    gripper = np.asarray(gripper_position, dtype=np.float64)
    eef = np.asarray(end_effector_position, dtype=np.float64)
    can_pose = np.asarray(can_root_pose, dtype=np.float64)
    frame_count = len(capsule_qpos)
    if not (len(gripper) == len(eef) == len(can_pose) == frame_count):
        raise ValueError("capsule phase inputs do not have matching frame counts")
    if not 0 <= int(lid_joint_index) < capsule_qpos.shape[1]:
        raise IndexError(f"lid joint index {lid_joint_index} is invalid for shape {capsule_qpos.shape}")

    lid_position = capsule_qpos[:, int(lid_joint_index)]
    gripper_open = np.all(np.isclose(gripper, 0.0, atol=gripper_open_atol, rtol=0.01), axis=1)
    lid_opened = (lid_position <= float(lid_open_threshold)) & gripper_open
    open_indices = np.flatnonzero(lid_opened)
    if not len(open_indices):
        raise ValueError("episode never satisfies the capsule coffee_lid_opened predicate")
    open_end = int(open_indices[0])

    pod_distance = np.linalg.norm(can_pose[:, :3] - eef[:, :3], axis=1)
    gripper_closed = np.all(np.abs(gripper) > float(gripper_closed_threshold), axis=1)
    pod_grasped = (pod_distance < float(grasp_distance_threshold)) & gripper_closed
    grasp_indices = np.flatnonzero(pod_grasped & (np.arange(frame_count) > open_end))
    if not len(grasp_indices):
        raise ValueError("episode never satisfies the capsule grasp_pod predicate after opening the lid")
    grasp_end = int(grasp_indices[0])

    phase = np.full(frame_count, PHASE_PLACE_POD, dtype=np.uint8)
    phase[: open_end + 1] = PHASE_OPEN_LID
    phase[open_end + 1 : grasp_end + 1] = PHASE_GRASP_POD
    metadata: dict[str, int | float] = {
        "open_lid_end": open_end,
        "grasp_pod_end": grasp_end,
        "place_pod_end": frame_count - 1,
        "open_lid_frames": open_end + 1,
        "grasp_pod_frames": grasp_end - open_end,
        "place_pod_frames": frame_count - grasp_end - 1,
        "minimum_pod_eef_distance": float(np.min(pod_distance)),
        "minimum_lid_joint_position": float(np.min(lid_position)),
    }
    return phase, metadata


def phase_future_annotations(
    phase: np.ndarray,
    qpos: np.ndarray,
    *,
    tail_frames: int,
    global_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Map every phase frame to the final ``tail_frames`` qposes of its segment."""
    labels = np.asarray(phase, dtype=np.uint8)
    robot_qpos = np.asarray(qpos, dtype=np.float32)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("phase must be a non-empty one-dimensional array")
    if robot_qpos.ndim != 2 or len(robot_qpos) != len(labels):
        raise ValueError("qpos must be [frames, joints] and align with phase")
    if int(tail_frames) < 1:
        raise ValueError("tail_frames must be positive")

    frame_count, joint_count = robot_qpos.shape
    segment_start_array = np.empty(frame_count, dtype=np.int64)
    segment_end_array = np.empty(frame_count, dtype=np.int64)
    future_indices = np.full((frame_count, int(tail_frames)), -1, dtype=np.int64)
    future_mask = np.zeros((frame_count, int(tail_frames)), dtype=bool)
    future_qpos = np.full((frame_count, int(tail_frames), joint_count), np.nan, dtype=np.float32)

    segment_start = 0
    for segment_end_value in np.flatnonzero(np.r_[labels[1:] != labels[:-1], True]):
        segment_end = int(segment_end_value)
        candidate_start = max(segment_start, segment_end - int(tail_frames) + 1)
        candidates = np.arange(candidate_start, segment_end + 1, dtype=np.int64)
        count = len(candidates)
        rows = slice(segment_start, segment_end + 1)
        segment_start_array[rows] = int(global_offset) + segment_start
        segment_end_array[rows] = int(global_offset) + segment_end
        future_indices[rows, :count] = int(global_offset) + candidates
        future_mask[rows, :count] = True
        future_qpos[rows, :count] = robot_qpos[candidates][None, :, :]
        segment_start = segment_end + 1

    return {
        "phase_segment_start": segment_start_array,
        "phase_segment_end": segment_end_array,
        "future_qpos_indices": future_indices,
        "future_qpos_mask": future_mask,
        "future_qpos": future_qpos,
    }


def _auto_lid_joint_index(data: h5py.Group, demo_names: list[str]) -> int:
    minima = []
    for name in demo_names[: min(5, len(demo_names))]:
        minima.append(np.min(np.asarray(data[name]["states/articulation/capsule/joint_position"]), axis=0))
    minimum = np.min(np.stack(minima), axis=0)
    index = int(np.argmin(minimum))
    if float(minimum[index]) > -0.5:
        raise ValueError(f"could not identify the lid joint from minima {minimum.tolist()}")
    return index


def build_sidecar(
    data_file: Path,
    output: Path,
    *,
    tail_frames: int,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"sidecar exists: {output}; pass --overwrite")
    with h5py.File(data_file, "r") as h5:
        data = h5["data"]
        demo_names = _natural_demo_names(data)
        if not demo_names:
            raise ValueError(f"no demo_N groups found in {data_file}")
        lid_joint_index = _auto_lid_joint_index(data, demo_names)
        episode_lengths = np.asarray([len(data[name]["obs/joint_pos"]) for name in demo_names], dtype=np.int64)
        episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
        total_frames = int(episode_ends[-1])
        joint_count = int(data[demo_names[0]]["obs/joint_pos"].shape[1])

        phase_all = np.empty(total_frames, dtype=np.uint8)
        qpos_all = np.empty((total_frames, joint_count), dtype=np.float32)
        policy_qpos_all = np.empty((total_frames, 8), dtype=np.float32)
        segment_start_all = np.empty(total_frames, dtype=np.int64)
        segment_end_all = np.empty(total_frames, dtype=np.int64)
        future_indices_all = np.full((total_frames, tail_frames), -1, dtype=np.int64)
        future_mask_all = np.zeros((total_frames, tail_frames), dtype=bool)
        future_qpos_all = np.full((total_frames, tail_frames, joint_count), np.nan, dtype=np.float32)
        future_policy_qpos_all = np.full((total_frames, tail_frames, 8), np.nan, dtype=np.float32)
        episode_records: list[dict[str, Any]] = []

        global_start = 0
        for demo_name, episode_length in zip(demo_names, episode_lengths, strict=True):
            demo = data[demo_name]
            global_end = global_start + int(episode_length)
            qpos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)
            gripper = np.asarray(demo["obs/gripper_pos"], dtype=np.float32)
            policy_qpos = np.concatenate((qpos[:, :7], gripper[:, :1]), axis=1)
            phase, phase_metadata = infer_capsule_phase(
                capsule_joint_position=np.asarray(
                    demo["states/articulation/capsule/joint_position"], dtype=np.float32
                ),
                gripper_position=gripper,
                end_effector_position=np.asarray(demo["obs/eef_pos"], dtype=np.float32),
                can_root_pose=np.asarray(demo["states/rigid_object/can/root_pose"], dtype=np.float32),
                lid_joint_index=lid_joint_index,
            )
            annotations = phase_future_annotations(
                phase,
                qpos,
                tail_frames=tail_frames,
                global_offset=global_start,
            )
            policy_annotations = phase_future_annotations(
                phase,
                policy_qpos,
                tail_frames=tail_frames,
                global_offset=global_start,
            )
            rows = slice(global_start, global_end)
            phase_all[rows] = phase
            qpos_all[rows] = qpos
            policy_qpos_all[rows] = policy_qpos
            segment_start_all[rows] = annotations["phase_segment_start"]
            segment_end_all[rows] = annotations["phase_segment_end"]
            future_indices_all[rows] = annotations["future_qpos_indices"]
            future_mask_all[rows] = annotations["future_qpos_mask"]
            future_qpos_all[rows] = annotations["future_qpos"]
            future_policy_qpos_all[rows] = policy_annotations["future_qpos"]
            episode_records.append(
                {
                    "episode": demo_name,
                    "frames": int(episode_length),
                    **phase_metadata,
                }
            )
            global_start = global_end

        metadata = {
            "schema": "capsule-phase-future-qpos-v1",
            "source_hdf5": str(data_file),
            "source_env_name": json.loads(str(data.attrs["env_args"]))["env_name"],
            "source_hz": 15.0,
            "num_episodes": len(demo_names),
            "num_frames": total_frames,
            "tail_frames": int(tail_frames),
            "robot_qpos_dim": joint_count,
            "policy_qpos_dim": 8,
            "lid_joint_index": lid_joint_index,
            "phase_names": PHASE_NAMES,
            "phase_semantics": "cumulative first-achievement boundaries",
            "open_lid_predicate": "lid_joint <= -0.5 and both gripper observations isclose(0, atol=0.01, rtol=0.01)",
            "grasp_pod_predicate": "norm(can_xyz - eef_xyz) < 0.1 and both abs(gripper observations) > 0.005",
            "future_semantics": "all frames in a phase map to up to the final tail_frames qposes of that phase",
            "episodes": episode_records,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        phase=phase_all,
        robot_qpos=qpos_all,
        policy_qpos=policy_qpos_all,
        episode_ends=episode_ends,
        episode_names=np.asarray(demo_names),
        phase_segment_start=segment_start_all,
        phase_segment_end=segment_end_all,
        future_qpos_indices=future_indices_all,
        future_qpos_mask=future_mask_all,
        future_qpos=future_qpos_all,
        future_policy_qpos=future_policy_qpos_all,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary.replace(output)
    phase_values, phase_counts = np.unique(phase_all, return_counts=True)
    summary = {
        "sidecar": str(output),
        "phase_counts": {
            PHASE_NAMES[int(value)]: int(count)
            for value, count in zip(phase_values, phase_counts, strict=True)
        },
        **metadata,
    }
    return output, summary


def _draw_future_pose(
    frame: np.ndarray,
    joint_position: np.ndarray,
    *,
    alpha: float,
    rank: int,
    count: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    gripper_points, gripper_lines = panda_gripper_wireframe_link0(joint_position)
    points, valid = project_table_camera(
        gripper_points,
        width=width,
        height=height,
    )
    overlay = frame.copy()
    color = (0, 255, 255)  # RGB cyan; frames are kept in RGB for imageio.
    brightness = 0.55 + 0.45 * (rank + 1) / max(count, 1)
    draw_color = tuple(int(channel * brightness) for channel in color)
    for start, end in gripper_lines:
        if valid[start] and valid[end]:
            cv2.line(
                overlay,
                tuple(points[start]),
                tuple(points[end]),
                draw_color,
                4,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if valid[index]:
            radius = 6 if index in (1, 6) else 4
            cv2.circle(overlay, tuple(point), radius, draw_color, -1, cv2.LINE_AA)
    pose_alpha = float(alpha) * (0.45 + 0.55 * (rank + 1) / max(count, 1)) / math.sqrt(max(count, 1))
    return cv2.addWeighted(overlay, pose_alpha, frame, 1.0 - pose_alpha, 0.0)


def _annotate_frame(
    frame: np.ndarray,
    *,
    episode_name: str,
    phase: int,
    local_frame: int,
    candidate_indices: np.ndarray,
    fps: float,
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 790), 76), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"{episode_name} | {PHASE_NAMES[int(phase)]} | frame {local_frame}",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "CYAN FUTURE QPOS: final %d phase frames [%d:%d] at %g Hz"
        % (
            len(candidate_indices),
            int(candidate_indices[0]),
            int(candidate_indices[-1]),
            float(fps),
        ),
        (16, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def render_episode(
    data: h5py.Group,
    sidecar: dict[str, np.ndarray],
    *,
    episode_index: int,
    out_dir: Path,
    fps: float,
    alpha: float,
    output_width: int,
) -> dict[str, Any]:
    demo_names = [str(value) for value in sidecar["episode_names"]]
    episode_ends = np.asarray(sidecar["episode_ends"], dtype=np.int64)
    demo_name = demo_names[episode_index]
    global_end = int(episode_ends[episode_index])
    global_start = int(episode_ends[episode_index - 1]) if episode_index else 0
    phase = np.asarray(sidecar["phase"][global_start:global_end], dtype=np.uint8)
    qpos = np.asarray(sidecar["robot_qpos"][global_start:global_end], dtype=np.float32)
    future_indices = np.asarray(sidecar["future_qpos_indices"][global_start:global_end], dtype=np.int64)
    future_mask = np.asarray(sidecar["future_qpos_mask"][global_start:global_end], dtype=bool)
    images = data[demo_name]["obs/table_cam"]
    input_height, input_width = int(images.shape[1]), int(images.shape[2])
    if output_width > 0 and output_width != input_width:
        output_height = int(round(input_height * output_width / input_width))
    else:
        output_width = input_width
        output_height = input_height

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{demo_name}_capsule_phase_future_qpos_tail{future_indices.shape[1]}.mp4"
    )
    writer = imageio.get_writer(
        str(out_path),
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        output_params=[
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    cached: dict[tuple[int, ...], list[np.ndarray]] = {}
    try:
        for local_frame in range(global_end - global_start):
            valid_indices = future_indices[local_frame][future_mask[local_frame]]
            local_candidates = valid_indices - global_start
            cache_key = tuple(int(value) for value in local_candidates)
            if cache_key not in cached:
                cached[cache_key] = [qpos[int(value)] for value in local_candidates]
            frame = np.asarray(images[local_frame], dtype=np.uint8)
            if (output_width, output_height) != (input_width, input_height):
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            for rank, candidate_qpos in enumerate(cached[cache_key]):
                frame = _draw_future_pose(
                    frame,
                    candidate_qpos,
                    alpha=alpha,
                    rank=rank,
                    count=len(cache_key),
                )
            frame = _annotate_frame(
                frame,
                episode_name=demo_name,
                phase=int(phase[local_frame]),
                local_frame=local_frame,
                candidate_indices=local_candidates,
                fps=fps,
            )
            writer.append_data(frame)
    finally:
        writer.close()
    phase_counts = {
        PHASE_NAMES[int(value)]: int(count)
        for value, count in zip(*np.unique(phase, return_counts=True), strict=True)
    }
    record = {
        "episode_index": int(episode_index),
        "episode_name": demo_name,
        "frames": int(global_end - global_start),
        "phase_counts": phase_counts,
        "video": str(out_path),
        "size_bytes": int(out_path.stat().st_size),
    }
    print(
        f"rendered episode={episode_index} name={demo_name} frames={record['frames']} "
        f"phases={phase_counts} video={out_path}",
        flush=True,
    )
    return record


def _initialize_render_worker(data_file: str, sidecar_path: str) -> None:
    global _RENDER_WORKER_DATA, _RENDER_WORKER_SIDECAR
    _RENDER_WORKER_DATA = h5py.File(data_file, "r")
    with np.load(sidecar_path, allow_pickle=False) as loaded:
        _RENDER_WORKER_SIDECAR = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "metadata_json"
        }


def _render_episode_worker(arguments: tuple[int, str, float, float, int]) -> dict[str, Any]:
    if _RENDER_WORKER_DATA is None or _RENDER_WORKER_SIDECAR is None:
        raise RuntimeError("render worker was not initialized")
    episode_index, out_dir, fps, alpha, output_width = arguments
    return render_episode(
        _RENDER_WORKER_DATA["data"],
        _RENDER_WORKER_SIDECAR,
        episode_index=episode_index,
        out_dir=Path(out_dir),
        fps=fps,
        alpha=alpha,
        output_width=output_width,
    )


def _parse_episode_indices(text: str | None, *, episode_count: int, max_videos: int) -> list[int]:
    if text:
        indices = []
        for raw_value in text.split(","):
            index = int(raw_value.strip())
            if index < 0:
                index += episode_count
            if not 0 <= index < episode_count:
                raise IndexError(f"episode index {index} is outside [0, {episode_count})")
            indices.append(index)
        return indices
    if max_videos <= 0 or max_videos >= episode_count:
        return list(range(episode_count))
    return np.linspace(0, episode_count - 1, max_videos, dtype=np.int64).tolist()


def upload_to_wandb(
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    sidecar_path: Path,
    out_dir: Path,
    project: str,
    entity: str | None,
    group: str,
    name: str,
    mode: str,
    fps: float,
) -> str | None:
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        mode=mode,
        config={
            key: value
            for key, value in summary.items()
            if key not in {"episodes"}
        },
    )
    try:
        table = wandb.Table(
            columns=["episode_index", "episode_name", "frames", "open_lid_frames", "grasp_pod_frames", "place_pod_frames", "video"]
        )
        episode_metadata = {
            str(record["episode"]): record
            for record in summary["episodes"]
        }
        payload: dict[str, Any] = {}
        for record in records:
            video = wandb.Video(str(record["video"]), fps=float(fps), format="mp4")
            phases = episode_metadata[record["episode_name"]]
            table.add_data(
                record["episode_index"],
                record["episode_name"],
                record["frames"],
                phases["open_lid_frames"],
                phases["grasp_pod_frames"],
                phases["place_pod_frames"],
                video,
            )
            payload[f"capsule_phase_futures/{record['episode_index']:02d}_{record['episode_name']}"] = video
        payload["capsule_phase_future_qpos_videos"] = table
        payload["num_videos"] = len(records)
        run.log(payload)
        artifact = wandb.Artifact(
            name="capsule-phase-future-qpos-tail10",
            type="dataset-annotation",
            metadata={
                "schema": summary["schema"],
                "num_episodes": summary["num_episodes"],
                "num_frames": summary["num_frames"],
                "tail_frames": summary["tail_frames"],
                "source_hz": summary["source_hz"],
            },
        )
        artifact.add_file(str(sidecar_path), name=sidecar_path.name)
        run.log_artifact(artifact)
        run_url = run.url
        manifest = {
            "wandb_url": run_url,
            "sidecar": str(sidecar_path),
            "records": records,
            "summary": summary,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wandb_url={run_url}", flush=True)
        return run_url
    finally:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--tail-frames", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--max-videos", type=int, default=0, help="0 renders every episode")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.62)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-project", default="openpi")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="capsule-phase-future-qpos")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if args.tail_frames < 1:
        parser.error("--tail-frames must be positive")
    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must be in (0, 1]")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.output_width < 0:
        parser.error("--output-width must be nonnegative")
    if args.render_workers < 1:
        parser.error("--render-workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_file = args.data_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    sidecar_path = (
        args.sidecar.expanduser().resolve()
        if args.sidecar is not None
        else out_dir / f"capsule_phase_future_qpos_tail{args.tail_frames}.npz"
    )
    sidecar_path, summary = build_sidecar(
        data_file,
        sidecar_path,
        tail_frames=int(args.tail_frames),
        overwrite=bool(args.overwrite),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2), flush=True)
    if args.annotation_only:
        return

    with np.load(sidecar_path, allow_pickle=False) as loaded:
        sidecar = {key: np.asarray(loaded[key]) for key in loaded.files if key != "metadata_json"}
    episode_indices = _parse_episode_indices(
        args.episode_indices,
        episode_count=len(sidecar["episode_names"]),
        max_videos=int(args.max_videos),
    )
    worker_arguments = [
        (
            int(episode_index),
            str(out_dir / "videos"),
            float(args.fps),
            float(args.alpha),
            int(args.output_width),
        )
        for episode_index in episode_indices
    ]
    if int(args.render_workers) == 1:
        records = []
        with h5py.File(data_file, "r") as h5:
            for arguments in worker_arguments:
                records.append(
                    render_episode(
                        h5["data"],
                        sidecar,
                        episode_index=arguments[0],
                        out_dir=Path(arguments[1]),
                        fps=arguments[2],
                        alpha=arguments[3],
                        output_width=arguments[4],
                    )
                )
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.render_workers),
            initializer=_initialize_render_worker,
            initargs=(str(data_file), str(sidecar_path)),
        ) as executor:
            records = list(executor.map(_render_episode_worker, worker_arguments))
    (out_dir / "render_records.json").write_text(json.dumps(records, indent=2) + "\n")
    if args.skip_upload:
        return

    run_name = args.wandb_name or (
        "capsule-phase-future-qpos-tail%d-%s"
        % (int(args.tail_frames), time.strftime("%Y%m%d-%H%M%S"))
    )
    upload_to_wandb(
        records=records,
        summary=summary,
        sidecar_path=sidecar_path,
        out_dir=out_dir,
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        group=str(args.wandb_group),
        name=run_name,
        mode=str(args.wandb_mode),
        fps=float(args.fps),
    )


if __name__ == "__main__":
    main()
