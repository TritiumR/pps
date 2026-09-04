#!/usr/bin/env python3
"""Render weight demos with one translucent cyan terminal keypose per stage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import wandb

from tools import annotate_capsule_phase_futures_wandb as render_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "weight" / "generated_dataset.hdf5"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "weight_stage_keyposes"

STAGE_REACH_PEAR = 0
STAGE_GRASP_PEAR = 1
STAGE_PLACE_PEAR = 2
STAGE_OPEN_GRIPPER_PEAR = 3
STAGE_REACH_APPLE = 4
STAGE_GRASP_APPLE = 5
STAGE_PLACE_APPLE = 6
STAGE_OPEN_GRIPPER_APPLE = 7
STAGE_NAMES = {
    STAGE_REACH_PEAR: "REACH_PEAR",
    STAGE_GRASP_PEAR: "GRASP_PEAR",
    STAGE_PLACE_PEAR: "PLACE_PEAR",
    STAGE_OPEN_GRIPPER_PEAR: "OPEN_GRIPPER_PEAR",
    STAGE_REACH_APPLE: "REACH_APPLE",
    STAGE_GRASP_APPLE: "GRASP_APPLE",
    STAGE_PLACE_APPLE: "PLACE_APPLE",
    STAGE_OPEN_GRIPPER_APPLE: "OPEN_GRIPPER_APPLE",
}

GRASP_DISTANCE_THRESHOLD = 0.08
GRIPPER_CLOSED_THRESHOLD = 0.005
SCALE_XY_THRESHOLD = 0.12
SCALE_Y_OFFSET = -0.05
LIFT_START_HEIGHT = 0.015
ROBOTIQ_OPEN_HALF_APERTURE = 0.043
ROBOTIQ_MIN_HALF_APERTURE = 0.006

_WORKER_DATA: h5py.File | None = None


def _natural_demo_names(data: h5py.Group) -> list[str]:
    names = [name for name in data.keys() if name.startswith("demo_")]
    return sorted(names, key=lambda name: int(name.removeprefix("demo_")))


def _first_true_after(mask: np.ndarray, start: int, *, name: str) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool) & (np.arange(len(mask)) >= start))
    if not len(indices):
        raise ValueError(f"episode never satisfies {name} at or after frame {start}")
    return int(indices[0])


def infer_weight_stages(
    *,
    end_effector_position: np.ndarray,
    gripper_position: np.ndarray,
    joint_actions: np.ndarray,
    pear_root_pose: np.ndarray,
    apple_root_pose: np.ndarray,
    scale_root_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Split each Mimic grasp segment into reach, grasp-close, and place."""
    eef = np.asarray(end_effector_position, dtype=np.float64)
    gripper = np.asarray(gripper_position, dtype=np.float64)
    action = np.asarray(joint_actions, dtype=np.float64)
    pear = np.asarray(pear_root_pose, dtype=np.float64)
    apple = np.asarray(apple_root_pose, dtype=np.float64)
    scale = np.asarray(scale_root_pose, dtype=np.float64)
    frame_count = len(eef)
    if frame_count < 4 or not all(
        len(value) == frame_count for value in (gripper, action, pear, apple, scale)
    ):
        raise ValueError("weight stage inputs must have matching nontrivial frame counts")
    if action.ndim != 2 or action.shape[1] < 8:
        raise ValueError(f"expected joint_actions with a gripper column, got {action.shape}")

    gripper_closed = np.all(np.abs(gripper) > GRIPPER_CLOSED_THRESHOLD, axis=1)
    pear_grasped = (
        np.linalg.norm(pear[:, :3] - eef[:, :3], axis=1) < GRASP_DISTANCE_THRESHOLD
    ) & gripper_closed
    apple_grasped = (
        np.linalg.norm(apple[:, :3] - eef[:, :3], axis=1) < GRASP_DISTANCE_THRESHOLD
    ) & gripper_closed
    scale_target_xy = scale[:, :2] + np.asarray([0.0, SCALE_Y_OFFSET])
    pear_on_scale = (
        np.linalg.norm(pear[:, :2] - scale_target_xy, axis=1) <= SCALE_XY_THRESHOLD
    )
    apple_on_scale = (
        np.linalg.norm(apple[:, :2] - scale_target_xy, axis=1) <= SCALE_XY_THRESHOLD
    )

    grasp_pear_end = _first_true_after(pear_grasped, 0, name="grasp_pear")
    pear_on_scale_frame = _first_true_after(
        pear_on_scale, grasp_pear_end + 1, name="pear_on_scale"
    )
    grasp_apple_end = _first_true_after(
        apple_grasped, pear_on_scale_frame + 1, name="grasp_apple"
    )
    apple_on_scale_frame = _first_true_after(
        apple_on_scale, grasp_apple_end + 1, name="apple_on_scale"
    )
    gripper_close_command = action[:, 7] >= 0.5
    reach_pear_candidates = np.flatnonzero(
        ~gripper_close_command & (np.arange(frame_count) <= grasp_pear_end)
    )
    reach_apple_candidates = np.flatnonzero(
        ~gripper_close_command
        & (np.arange(frame_count) > pear_on_scale_frame)
        & (np.arange(frame_count) <= grasp_apple_end)
    )
    if not len(reach_pear_candidates) or not len(reach_apple_candidates):
        raise ValueError("could not find an open-gripper reach endpoint")
    reach_pear_end = int(reach_pear_candidates[-1])
    reach_apple_end = int(reach_apple_candidates[-1])

    pear_lifted = pear[:, 2] >= pear[reach_pear_end, 2] + LIFT_START_HEIGHT
    apple_lifted = apple[:, 2] >= apple[reach_apple_end, 2] + LIFT_START_HEIGHT
    pear_lift_start = _first_true_after(
        pear_lifted, reach_pear_end + 1, name="pear lift start"
    )
    apple_lift_start = _first_true_after(
        apple_lifted, reach_apple_end + 1, name="apple lift start"
    )
    grasp_pear_close_end = min(pear_lift_start - 1, pear_on_scale_frame - 1)
    grasp_apple_close_end = min(apple_lift_start - 1, frame_count - 2)
    grasp_pear_close_end = max(grasp_pear_close_end, reach_pear_end + 1)
    grasp_apple_close_end = max(grasp_apple_close_end, reach_apple_end + 1)

    gripper_open = np.all(np.isclose(gripper, 0.0, atol=0.01, rtol=0.01), axis=1)
    pear_release_start = _first_true_after(
        ~gripper_close_command, pear_on_scale_frame + 1, name="pear release command"
    )
    apple_release_start = _first_true_after(
        ~gripper_close_command, apple_on_scale_frame + 1, name="apple release command"
    )
    open_gripper_pear_end = _first_true_after(
        gripper_open, pear_release_start, name="pear gripper open"
    )

    stage = np.full(frame_count, STAGE_OPEN_GRIPPER_APPLE, dtype=np.uint8)
    stage[: reach_pear_end + 1] = STAGE_REACH_PEAR
    stage[reach_pear_end + 1 : grasp_pear_close_end + 1] = STAGE_GRASP_PEAR
    stage[grasp_pear_close_end + 1 : pear_release_start] = STAGE_PLACE_PEAR
    stage[pear_release_start : open_gripper_pear_end + 1] = STAGE_OPEN_GRIPPER_PEAR
    stage[open_gripper_pear_end + 1 : reach_apple_end + 1] = STAGE_REACH_APPLE
    stage[reach_apple_end + 1 : grasp_apple_close_end + 1] = STAGE_GRASP_APPLE
    stage[grasp_apple_close_end + 1 : apple_release_start] = STAGE_PLACE_APPLE
    keypose_frames = {
        "reach_pear": reach_pear_end,
        "grasp_pear": grasp_pear_close_end,
        "place_pear": pear_release_start - 1,
        "open_gripper_pear": open_gripper_pear_end,
        "reach_apple": reach_apple_end,
        "grasp_apple": grasp_apple_close_end,
        "place_apple": apple_release_start - 1,
        "open_gripper_apple": frame_count - 1,
    }
    return stage, keypose_frames


def _episode_annotation(demo: h5py.Group) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    stage, keypose_frames = infer_weight_stages(
        end_effector_position=demo["obs/eef_pos"],
        gripper_position=demo["obs/gripper_pos"],
        joint_actions=demo["obs/joint_actions"],
        pear_root_pose=demo["states/rigid_object/pear/root_pose"],
        apple_root_pose=demo["states/rigid_object/apple/root_pose"],
        scale_root_pose=demo["states/rigid_object/scale/root_pose"],
    )
    qpos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)[:, :7]
    gripper_command = np.asarray(demo["obs/joint_actions"], dtype=np.float32)[:, 7]
    stage_keyposes = np.stack(
        [
            np.concatenate(
                (
                    qpos[keypose_frames[name]],
                    np.asarray([gripper_command[keypose_frames[name]]], dtype=np.float32),
                )
            )
            for name in (
                "reach_pear",
                "grasp_pear",
                "place_pear",
                "open_gripper_pear",
                "reach_apple",
                "grasp_apple",
                "place_apple",
                "open_gripper_apple",
            )
        ],
        axis=0,
    )
    return stage, stage_keyposes, keypose_frames


def build_training_sidecar(
    data_file: Path,
    output: Path,
    *,
    tail_frames: int = 12,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build random-sampling candidates for ``SubphaseKeyposeDataset``."""
    data_file = data_file.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"sidecar exists: {output}; pass overwrite=True")
    if tail_frames < 1:
        raise ValueError("tail_frames must be positive")

    with h5py.File(data_file, "r") as h5:
        data = h5["data"]
        demo_names = _natural_demo_names(data)
        episode_lengths = np.asarray(
            [len(data[name]["obs/joint_pos"]) for name in demo_names],
            dtype=np.int64,
        )
        episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
        total_frames = int(episode_ends[-1])

        phase_all = np.empty(total_frames, dtype=np.uint8)
        policy_qpos_all = np.empty((total_frames, 8), dtype=np.float32)
        segment_start_all = np.empty(total_frames, dtype=np.int64)
        segment_end_all = np.empty(total_frames, dtype=np.int64)
        future_indices_all = np.full(
            (total_frames, tail_frames), -1, dtype=np.int64
        )
        future_mask_all = np.zeros((total_frames, tail_frames), dtype=bool)
        future_policy_qpos_all = np.full(
            (total_frames, tail_frames, 8), np.nan, dtype=np.float32
        )
        episode_records: list[dict[str, Any]] = []

        global_start = 0
        for demo_name, episode_length in zip(
            demo_names, episode_lengths, strict=True
        ):
            demo = data[demo_name]
            global_end = global_start + int(episode_length)
            phase, _, keypose_frames = _episode_annotation(demo)
            arm_qpos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)[:, :7]
            gripper_command = np.asarray(
                demo["obs/joint_actions"], dtype=np.float32
            )[:, 7:8]
            policy_qpos = np.concatenate((arm_qpos, gripper_command), axis=1)
            annotations = render_utils.phase_future_annotations(
                phase,
                policy_qpos,
                tail_frames=tail_frames,
                global_offset=global_start,
            )
            rows = slice(global_start, global_end)
            phase_all[rows] = phase
            policy_qpos_all[rows] = policy_qpos
            segment_start_all[rows] = annotations["phase_segment_start"]
            segment_end_all[rows] = annotations["phase_segment_end"]
            future_indices_all[rows] = annotations["future_qpos_indices"]
            future_mask_all[rows] = annotations["future_qpos_mask"]
            future_policy_qpos_all[rows] = annotations["future_qpos"]
            phase_values, phase_counts = np.unique(phase, return_counts=True)
            episode_records.append(
                {
                    "episode": demo_name,
                    "frames": int(episode_length),
                    "phase_counts": {
                        STAGE_NAMES[int(value)]: int(count)
                        for value, count in zip(
                            phase_values, phase_counts, strict=True
                        )
                    },
                    "keypose_frames": keypose_frames,
                }
            )
            global_start = global_end

    metadata = {
        "schema": "weight-eight-phase-future-policy-qpos-v1",
        "source_hdf5": str(data_file),
        "source_hz": 15.0,
        "num_episodes": len(demo_names),
        "num_frames": total_frames,
        "tail_frames": int(tail_frames),
        "tail_seconds": float(tail_frames) / 15.0,
        "policy_qpos_dim": 8,
        "phase_names": STAGE_NAMES,
        "gripper_keypose_source": "obs/joint_actions[:, 7]",
        "gripper_command_semantics": {"0": "open", "1": "closed"},
        "future_semantics": (
            "each row maps to its phase's final tail_frames policy qposes; "
            "the dataset wrapper rejects candidates at or before the current row"
        ),
        "episodes": episode_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        phase=phase_all,
        phase_names=np.asarray(
            [STAGE_NAMES[index] for index in range(len(STAGE_NAMES))]
        ),
        policy_qpos=policy_qpos_all,
        episode_ends=episode_ends,
        episode_names=np.asarray(demo_names),
        phase_segment_start=segment_start_all,
        phase_segment_end=segment_end_all,
        future_qpos_indices=future_indices_all,
        future_qpos_mask=future_mask_all,
        future_policy_qpos=future_policy_qpos_all,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary.replace(output)
    summary = {
        key: value for key, value in metadata.items() if key != "episodes"
    }
    summary["sidecar"] = str(output)
    return output, summary


def _gripper_half_aperture(gripper_command: float | np.ndarray) -> float:
    closure = np.clip(float(np.asarray(gripper_command).reshape(-1)[0]), 0.0, 1.0)
    return ROBOTIQ_OPEN_HALF_APERTURE - closure * (
        ROBOTIQ_OPEN_HALF_APERTURE - ROBOTIQ_MIN_HALF_APERTURE
    )


def _draw_keypose(
    frame: np.ndarray,
    keypose: np.ndarray,
    *,
    alpha: float,
    rank: int = 0,
    count: int = 1,
) -> np.ndarray:
    """Draw arm FK with finger gap driven by the binary gripper action command."""
    keypose = np.asarray(keypose, dtype=np.float64).reshape(-1)
    if keypose.size < 8:
        raise ValueError(f"expected an 8D arm+gripper-command keypose, got {keypose.shape}")
    _, end_effector = render_utils._panda_chain_and_ee_pose_link0(keypose[:7])
    half_gap = _gripper_half_aperture(keypose[7])
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
    points, valid = render_utils.project_table_camera(
        points_link0,
        width=frame.shape[1],
        height=frame.shape[0],
    )
    overlay = frame.copy()
    brightness = 0.55 + 0.45 * (rank + 1) / max(count, 1)
    cyan = (0, int(255 * brightness), int(255 * brightness))
    for start, end in ((0, 1), (2, 4), (2, 3), (4, 5), (1, 6)):
        if valid[start] and valid[end]:
            cv2.line(overlay, tuple(points[start]), tuple(points[end]), cyan, 5, cv2.LINE_AA)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(overlay, tuple(point), 5, cyan, -1, cv2.LINE_AA)
    pose_alpha = (
        float(alpha)
        * (0.45 + 0.55 * (rank + 1) / max(count, 1))
        / np.sqrt(max(count, 1))
    )
    return cv2.addWeighted(overlay, pose_alpha, frame, 1.0 - pose_alpha, 0.0)


def _annotate_frame(
    frame: np.ndarray,
    *,
    demo_name: str,
    stage: int,
    frame_index: int,
    candidate_indices: np.ndarray,
    candidate_keyposes: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 850), 76), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"{demo_name} | {STAGE_NAMES[stage]} | frame {frame_index}",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if len(candidate_indices):
        commands = sorted({int(round(float(keypose[7]))) for keypose in candidate_keyposes})
        detail = "CYAN FUTURES: %d strictly-future tail keyposes [%d:%d] | commands %s" % (
            len(candidate_indices),
            int(candidate_indices[0]),
            int(candidate_indices[-1]),
            commands,
        )
    else:
        detail = "CYAN FUTURES: none remain after the current frame"
    cv2.putText(
        output,
        detail,
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
    *,
    episode_index: int,
    out_dir: Path,
    fps: float,
    alpha: float,
    output_width: int,
    future_tail_frames: int = 0,
) -> dict[str, Any]:
    demo_name = _natural_demo_names(data)[episode_index]
    demo = data[demo_name]
    stage, stage_keyposes, keypose_frames = _episode_annotation(demo)
    keypose_frame_by_stage = np.asarray(
        [
            keypose_frames["reach_pear"],
            keypose_frames["grasp_pear"],
            keypose_frames["place_pear"],
            keypose_frames["open_gripper_pear"],
            keypose_frames["reach_apple"],
            keypose_frames["grasp_apple"],
            keypose_frames["place_apple"],
            keypose_frames["open_gripper_apple"],
        ],
        dtype=np.int64,
    )
    frame_qpos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)[:, :7]
    frame_gripper_command = np.asarray(
        demo["obs/joint_actions"], dtype=np.float32
    )[:, 7:8]
    frame_keyposes = np.concatenate((frame_qpos, frame_gripper_command), axis=1)
    tail_indices_by_stage = {
        stage_id: np.flatnonzero(stage == stage_id)[-future_tail_frames:]
        for stage_id in STAGE_NAMES
    } if future_tail_frames > 0 else {}
    images = demo["obs/table_cam"]
    input_height, input_width = int(images.shape[1]), int(images.shape[2])
    if output_width > 0 and output_width != input_width:
        output_height = int(round(input_height * output_width / input_width))
    else:
        output_width = input_width
        output_height = input_height

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"weight_phase_future_tail{future_tail_frames}"
        if future_tail_frames > 0
        else "weight_reach_grasp_keyposes"
    )
    out_path = out_dir / f"{demo_name}_{suffix}.mp4"
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
    try:
        for frame_index in range(len(stage)):
            stage_id = int(stage[frame_index])
            frame = np.asarray(images[frame_index], dtype=np.uint8)
            if (output_width, output_height) != (input_width, input_height):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            if future_tail_frames > 0:
                candidate_indices = tail_indices_by_stage[stage_id]
                candidate_indices = candidate_indices[candidate_indices > frame_index]
                candidate_keyposes = frame_keyposes[candidate_indices]
            else:
                candidate_indices = np.asarray(
                    [keypose_frame_by_stage[stage_id]], dtype=np.int64
                )
                candidate_keyposes = stage_keyposes[stage_id : stage_id + 1]
            for rank, candidate_keypose in enumerate(candidate_keyposes):
                frame = _draw_keypose(
                    frame,
                    candidate_keypose,
                    alpha=alpha,
                    rank=rank,
                    count=len(candidate_keyposes),
                )
            frame = _annotate_frame(
                frame,
                demo_name=demo_name,
                stage=stage_id,
                frame_index=frame_index,
                candidate_indices=candidate_indices,
                candidate_keyposes=candidate_keyposes,
            )
            writer.append_data(frame)
    finally:
        writer.close()

    stage_counts = {
        STAGE_NAMES[int(value)]: int(count)
        for value, count in zip(*np.unique(stage, return_counts=True), strict=True)
    }
    record = {
        "episode_index": episode_index,
        "episode_name": demo_name,
        "frames": len(stage),
        "stage_counts": stage_counts,
        "keypose_frames": keypose_frames,
        "future_tail_frames": int(future_tail_frames),
        "strictly_future_only": bool(future_tail_frames > 0),
        "video": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    print(
        f"rendered episode={episode_index} name={demo_name} "
        f"keyposes={keypose_frames} video={out_path}",
        flush=True,
    )
    return record


def _initialize_worker(data_file: str) -> None:
    global _WORKER_DATA
    _WORKER_DATA = h5py.File(data_file, "r")


def _render_worker(arguments: tuple[int, str, float, float, int, int]) -> dict[str, Any]:
    if _WORKER_DATA is None:
        raise RuntimeError("render worker was not initialized")
    episode_index, out_dir, fps, alpha, output_width, future_tail_frames = arguments
    return render_episode(
        _WORKER_DATA["data"],
        episode_index=episode_index,
        out_dir=Path(out_dir),
        fps=fps,
        alpha=alpha,
        output_width=output_width,
        future_tail_frames=future_tail_frames,
    )


def upload_to_wandb(
    *,
    records: list[dict[str, Any]],
    data_file: Path,
    out_dir: Path,
    fps: float,
    alpha: float,
    project: str,
    entity: str | None,
    group: str,
    name: str,
) -> str:
    future_tail_frames = int(records[0].get("future_tail_frames", 0)) if records else 0
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config={
            "source_dataset": str(data_file),
            "episodes": len(records),
            "fps": fps,
            "overlay": (
                "strictly-future terminal phase-tail arm qposes with binary gripper commands"
                if future_tail_frames > 0
                else "terminal stage arm qpos with binary gripper action command"
            ),
            "overlay_color": "cyan",
            "overlay_alpha": alpha,
            "gripper_keypose_source": "obs/joint_actions[:, 7]",
            "gripper_command_semantics": {"0": "open", "1": "closed"},
            "future_tail_frames": future_tail_frames,
            "future_tail_seconds": future_tail_frames / fps,
            "strictly_future_only": future_tail_frames > 0,
            "stages": list(STAGE_NAMES.values()),
            "grasp_distance_threshold": GRASP_DISTANCE_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "scale_xy_threshold": SCALE_XY_THRESHOLD,
            "scale_y_offset": SCALE_Y_OFFSET,
            "lift_start_height": LIFT_START_HEIGHT,
        },
    )
    try:
        table = wandb.Table(
            columns=[
                "episode_index",
                "episode_name",
                "frames",
                "reach_pear_keypose",
                "grasp_pear_keypose",
                "place_pear_keypose",
                "open_gripper_pear_keypose",
                "reach_apple_keypose",
                "grasp_apple_keypose",
                "place_apple_keypose",
                "open_gripper_apple_keypose",
                "video",
            ]
        )
        payload: dict[str, Any] = {}
        for record in records:
            video = wandb.Video(record["video"], fps=fps, format="mp4")
            keyposes = record["keypose_frames"]
            table.add_data(
                record["episode_index"],
                record["episode_name"],
                record["frames"],
                keyposes["reach_pear"],
                keyposes["grasp_pear"],
                keyposes["place_pear"],
                keyposes["open_gripper_pear"],
                keyposes["reach_apple"],
                keyposes["grasp_apple"],
                keyposes["place_apple"],
                keyposes["open_gripper_apple"],
                video,
            )
            payload[
                f"weight_reach_grasp_keyposes/{record['episode_index']:02d}_{record['episode_name']}"
            ] = video
        payload["weight_reach_grasp_keypose_videos"] = table
        payload["num_videos"] = len(records)
        run.log(payload)
        run_url = run.url
        manifest = {
            "wandb_url": run_url,
            "data_file": str(data_file),
            "records": records,
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--future-tail-seconds", type=float, default=0.0)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--wandb-project", default="openpi")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="weight-stage-keyposes")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")
    if args.output_width < 0:
        parser.error("--output-width must be nonnegative")
    if args.render_workers < 1:
        parser.error("--render-workers must be positive")
    if args.future_tail_seconds < 0:
        parser.error("--future-tail-seconds must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    data_file = args.data_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_file, "r") as h5:
        episode_count = len(_natural_demo_names(h5["data"]))
    requested_count = min(args.episodes, episode_count)
    future_tail_frames = int(round(args.future_tail_seconds * args.fps))
    if args.future_tail_seconds > 0 and future_tail_frames < 1:
        raise ValueError("--future-tail-seconds is shorter than one frame")
    arguments = [
        (
            episode_index,
            str(out_dir / "videos"),
            float(args.fps),
            float(args.alpha),
            int(args.output_width),
            future_tail_frames,
        )
        for episode_index in range(requested_count)
    ]
    if args.render_workers == 1:
        with h5py.File(data_file, "r") as h5:
            records = [
                render_episode(
                    h5["data"],
                    episode_index=item[0],
                    out_dir=Path(item[1]),
                    fps=item[2],
                    alpha=item[3],
                    output_width=item[4],
                    future_tail_frames=item[5],
                )
                for item in arguments
            ]
    else:
        with ProcessPoolExecutor(
            max_workers=args.render_workers,
            initializer=_initialize_worker,
            initargs=(str(data_file),),
        ) as executor:
            records = list(executor.map(_render_worker, arguments))
    (out_dir / "render_records.json").write_text(json.dumps(records, indent=2) + "\n")
    if args.skip_upload:
        return

    run_name = args.wandb_name or f"weight-stage-keyposes-{time.strftime('%Y%m%d-%H%M%S')}"
    upload_to_wandb(
        records=records,
        data_file=data_file,
        out_dir=out_dir,
        fps=float(args.fps),
        alpha=float(args.alpha),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
    )


if __name__ == "__main__":
    main()
