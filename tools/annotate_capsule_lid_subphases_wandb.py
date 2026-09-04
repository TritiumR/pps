#!/usr/bin/env python3
"""Render capsule demos with separate lid approach, pull, and release subphases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import annotate_capsule_phase_futures_wandb as base  # noqa: E402


DEFAULT_DATA_FILE = REPO_ROOT / "data" / "capsule" / "generated_dataset.hdf5"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "capsule_lid_subphase_future_qpos"

LID_APPROACH = 0
LID_PULL = 1
LID_RELEASE = 2
GRASP_POD = 3
PLACE_POD = 4
PHASE_NAMES = {
    LID_APPROACH: "LID_APPROACH",
    LID_PULL: "LID_PULL",
    LID_RELEASE: "LID_RELEASE",
    GRASP_POD: "GRASP_POD",
    PLACE_POD: "PLACE_POD",
}


def infer_capsule_subphases(
    *,
    capsule_joint_position: np.ndarray,
    gripper_position: np.ndarray,
    end_effector_position: np.ndarray,
    can_root_pose: np.ndarray,
    lid_joint_index: int,
    lid_motion_threshold: float = 0.02,
    lid_open_threshold: float = -0.5,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Split OPEN_LID using lid-motion onset, angle threshold, and release completion."""
    coarse_phase, coarse_metadata = base.infer_capsule_phase(
        capsule_joint_position=capsule_joint_position,
        gripper_position=gripper_position,
        end_effector_position=end_effector_position,
        can_root_pose=can_root_pose,
        lid_joint_index=lid_joint_index,
        lid_open_threshold=lid_open_threshold,
    )
    del coarse_phase
    lid_position = np.asarray(capsule_joint_position, dtype=np.float64)[:, lid_joint_index]
    open_lid_end = int(coarse_metadata["open_lid_end"])
    grasp_pod_end = int(coarse_metadata["grasp_pod_end"])

    motion_indices = np.flatnonzero(
        lid_position <= lid_position[0] - float(lid_motion_threshold)
    )
    if not len(motion_indices):
        raise ValueError("episode never shows sustained lid motion")
    pull_start = int(motion_indices[0])
    release_indices = np.flatnonzero(lid_position <= float(lid_open_threshold))
    if not len(release_indices):
        raise ValueError("episode never reaches the lid-open angle")
    release_start = int(release_indices[0])
    if not (0 < pull_start < release_start <= open_lid_end < grasp_pod_end):
        raise ValueError(
            "invalid capsule subphase ordering: "
            f"pull_start={pull_start}, release_start={release_start}, "
            f"open_lid_end={open_lid_end}, grasp_pod_end={grasp_pod_end}"
        )

    frame_count = len(lid_position)
    phase = np.full(frame_count, PLACE_POD, dtype=np.uint8)
    phase[:pull_start] = LID_APPROACH
    phase[pull_start:release_start] = LID_PULL
    phase[release_start : open_lid_end + 1] = LID_RELEASE
    phase[open_lid_end + 1 : grasp_pod_end + 1] = GRASP_POD
    metadata: dict[str, int | float] = {
        **coarse_metadata,
        "lid_approach_end": pull_start - 1,
        "lid_pull_start": pull_start,
        "lid_pull_end": release_start - 1,
        "lid_release_start": release_start,
        "lid_release_end": open_lid_end,
        "lid_approach_frames": pull_start,
        "lid_pull_frames": release_start - pull_start,
        "lid_release_frames": open_lid_end - release_start + 1,
        "lid_motion_threshold": float(lid_motion_threshold),
    }
    return phase, metadata


def build_sidecar(
    data_file: Path,
    output: Path,
    *,
    tail_frames: int,
    lid_motion_threshold: float,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"sidecar exists: {output}; pass --overwrite")
    with h5py.File(data_file, "r") as h5:
        data = h5["data"]
        demo_names = base._natural_demo_names(data)
        lid_joint_index = base._auto_lid_joint_index(data, demo_names)
        episode_lengths = np.asarray(
            [len(data[name]["obs/joint_pos"]) for name in demo_names],
            dtype=np.int64,
        )
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
        future_qpos_all = np.full(
            (total_frames, tail_frames, joint_count),
            np.nan,
            dtype=np.float32,
        )
        future_policy_qpos_all = np.full(
            (total_frames, tail_frames, 8),
            np.nan,
            dtype=np.float32,
        )
        episode_records = []

        global_start = 0
        for demo_name, episode_length in zip(
            demo_names,
            episode_lengths,
            strict=True,
        ):
            demo = data[demo_name]
            global_end = global_start + int(episode_length)
            qpos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)
            gripper = np.asarray(demo["obs/gripper_pos"], dtype=np.float32)
            policy_qpos = np.concatenate((qpos[:, :7], gripper[:, :1]), axis=1)
            phase, phase_metadata = infer_capsule_subphases(
                capsule_joint_position=np.asarray(
                    demo["states/articulation/capsule/joint_position"],
                    dtype=np.float32,
                ),
                gripper_position=gripper,
                end_effector_position=np.asarray(
                    demo["obs/eef_pos"],
                    dtype=np.float32,
                ),
                can_root_pose=np.asarray(
                    demo["states/rigid_object/can/root_pose"],
                    dtype=np.float32,
                ),
                lid_joint_index=lid_joint_index,
                lid_motion_threshold=lid_motion_threshold,
            )
            annotations = base.phase_future_annotations(
                phase,
                qpos,
                tail_frames=tail_frames,
                global_offset=global_start,
            )
            policy_annotations = base.phase_future_annotations(
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
            "schema": "capsule-lid-subphase-future-qpos-v1",
            "source_hdf5": str(data_file),
            "source_hz": 15.0,
            "num_episodes": len(demo_names),
            "num_frames": total_frames,
            "tail_frames": int(tail_frames),
            "robot_qpos_dim": joint_count,
            "policy_qpos_dim": 8,
            "lid_joint_index": lid_joint_index,
            "lid_motion_threshold": float(lid_motion_threshold),
            "phase_names": PHASE_NAMES,
            "phase_semantics": (
                "LID_APPROACH until sustained 0.02-rad motion; LID_PULL until "
                "lid <= -0.5; LID_RELEASE until gripper-open completion; then "
                "GRASP_POD and PLACE_POD"
            ),
            "future_semantics": (
                "all frames in each subphase map to up to its final tail_frames qposes"
            ),
            "episodes": episode_records,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        phase=phase_all,
        phase_names=np.asarray([PHASE_NAMES[index] for index in range(len(PHASE_NAMES))]),
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


def _initialize_worker(data_file: str, sidecar_path: str) -> None:
    base.PHASE_NAMES = PHASE_NAMES
    base._initialize_render_worker(data_file, sidecar_path)


def _render_worker(arguments: tuple[int, str, float, float, int]) -> dict[str, Any]:
    base.PHASE_NAMES = PHASE_NAMES
    return base._render_episode_worker(arguments)


def render_videos(
    *,
    data_file: Path,
    sidecar_path: Path,
    episode_indices: list[int],
    video_dir: Path,
    render_workers: int,
    fps: float,
    alpha: float,
    output_width: int,
) -> list[dict[str, Any]]:
    arguments = [
        (index, str(video_dir), fps, alpha, output_width)
        for index in episode_indices
    ]
    with ProcessPoolExecutor(
        max_workers=render_workers,
        initializer=_initialize_worker,
        initargs=(str(data_file), str(sidecar_path)),
    ) as executor:
        return list(executor.map(_render_worker, arguments))


def upload(
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    sidecar_path: Path,
    out_dir: Path,
    project: str,
    entity: str | None,
    group: str,
    name: str,
    fps: float,
) -> str | None:
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config={key: value for key, value in summary.items() if key != "episodes"},
    )
    try:
        table = wandb.Table(
            columns=[
                "episode_index",
                "episode_name",
                "frames",
                "lid_approach_frames",
                "lid_pull_frames",
                "lid_release_frames",
                "grasp_pod_frames",
                "place_pod_frames",
                "video",
            ]
        )
        episode_metadata = {
            str(record["episode"]): record
            for record in summary["episodes"]
        }
        payload: dict[str, Any] = {}
        for record in records:
            metadata = episode_metadata[record["episode_name"]]
            video = wandb.Video(record["video"], fps=fps, format="mp4")
            table.add_data(
                record["episode_index"],
                record["episode_name"],
                record["frames"],
                metadata["lid_approach_frames"],
                metadata["lid_pull_frames"],
                metadata["lid_release_frames"],
                metadata["grasp_pod_frames"],
                metadata["place_pod_frames"],
                video,
            )
            payload[
                f"capsule_lid_subphase_futures/{record['episode_index']:02d}"
            ] = video
        payload["capsule_lid_subphase_future_qpos_videos"] = table
        payload["num_videos"] = len(records)
        run.log(payload)
        artifact = wandb.Artifact(
            name="capsule-lid-subphase-future-qpos-tail10",
            type="dataset-annotation",
            metadata={
                "schema": summary["schema"],
                "num_episodes": summary["num_episodes"],
                "num_frames": summary["num_frames"],
                "tail_frames": summary["tail_frames"],
            },
        )
        artifact.add_file(str(sidecar_path), name=sidecar_path.name)
        run.log_artifact(artifact)
        manifest = {
            "wandb_url": run.url,
            "sidecar": str(sidecar_path),
            "records": records,
            "summary": summary,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wandb_url={run.url}", flush=True)
        return run.url
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tail-frames", type=int, default=10)
    parser.add_argument("--lid-motion-threshold", type=float, default=0.02)
    parser.add_argument("--episode-indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.62)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-project", default="openpi")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-group",
        default="capsule-lid-subphase-future-qpos",
    )
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    data_file = args.data_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    sidecar_path = out_dir / (
        f"capsule_lid_subphase_future_qpos_tail{args.tail_frames}.npz"
    )
    sidecar_path, summary = build_sidecar(
        data_file,
        sidecar_path,
        tail_frames=args.tail_frames,
        lid_motion_threshold=args.lid_motion_threshold,
        overwrite=args.overwrite,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    episode_indices = [
        int(value.strip())
        for value in args.episode_indices.split(",")
        if value.strip()
    ]
    base.PHASE_NAMES = PHASE_NAMES
    records = render_videos(
        data_file=data_file,
        sidecar_path=sidecar_path,
        episode_indices=episode_indices,
        video_dir=out_dir / "videos",
        render_workers=args.render_workers,
        fps=args.fps,
        alpha=args.alpha,
        output_width=args.output_width,
    )
    (out_dir / "render_records.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    if args.skip_upload:
        return
    run_name = args.wandb_name or (
        "capsule-lid-subphase-future-qpos-tail10-"
        + time.strftime("%Y%m%d-%H%M%S")
    )
    upload(
        records=records,
        summary=summary,
        sidecar_path=sidecar_path,
        out_dir=out_dir,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
