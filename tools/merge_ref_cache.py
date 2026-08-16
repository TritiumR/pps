#!/usr/bin/env python3
"""Merge disjoint v5 ref-cache shards and reindex their trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np


ARRAY_KEYS = (
    "demo_name",
    "step_index",
    "trajectory_id",
    "iteration",
    "time",
    "x_t",
    "epsilon",
    "cost_min",
    "score_norm",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--allow-teacher-code-mismatch",
        action="store_true",
        help=(
            "Allow shards whose only invariant mismatch is teacher_code_sha256. "
            "The merged cache records all source hashes and marks the teacher hash as mixed."
        ),
    )
    return parser.parse_args()


OBSERVATION_ARRAY_FILES = ("images.npy", "image_masks.npy", "states.npy")
OBSERVATION_SHARED_FILES = ("tokenized_prompt.npy", "tokenized_prompt_mask.npy")


def _merge_observation_sidecars(
    inputs: list[pathlib.Path], output: pathlib.Path, expected_observations: int
) -> pathlib.Path:
    sources = [pathlib.Path(f"{path}.observations") for path in inputs]
    target = pathlib.Path(f"{output}.observations")
    metadatas = []
    for source in sources:
        metadata_path = source / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing runtime observation sidecar: {source}")
        metadatas.append(json.loads(metadata_path.read_text()))
    reference = metadatas[0]
    invariant = {key: value for key, value in reference.items() if key != "num_observations"}
    for source, metadata in zip(sources[1:], metadatas[1:], strict=True):
        current = {key: value for key, value in metadata.items() if key != "num_observations"}
        if current != invariant:
            raise ValueError(f"Observation sidecar metadata differs: {source}")
    total = sum(int(metadata["num_observations"]) for metadata in metadatas)
    if total != expected_observations:
        raise ValueError(f"Observation sidecars contain {total}, expected {expected_observations}.")

    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for filename in OBSERVATION_ARRAY_FILES:
        arrays = [np.load(source / filename, mmap_mode="r") for source in sources]
        merged = np.lib.format.open_memmap(
            temporary / filename,
            mode="w+",
            dtype=arrays[0].dtype,
            shape=(total, *arrays[0].shape[1:]),
        )
        offset = 0
        for array in arrays:
            if array.shape[1:] != arrays[0].shape[1:] or array.dtype != arrays[0].dtype:
                raise ValueError(f"Incompatible {filename} in runtime observation sidecars.")
            merged[offset : offset + len(array)] = array
            offset += len(array)
        merged.flush()
    for filename in OBSERVATION_SHARED_FILES:
        value = np.load(sources[0] / filename)
        for source in sources[1:]:
            if not np.array_equal(value, np.load(source / filename)):
                raise ValueError(f"Shared observation value differs for {filename}: {source}")
        np.save(temporary / filename, value)
    merged_metadata = dict(reference)
    merged_metadata["num_observations"] = total
    (temporary / "metadata.json").write_text(
        json.dumps(merged_metadata, indent=2, sort_keys=True)
    )
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return target


def main() -> None:
    args = _args()
    if len(args.inputs) < 1:
        raise ValueError("At least one shard is required.")
    shards = []
    metadatas = []
    for path in args.inputs:
        shard = np.load(path, allow_pickle=False)
        missing = [key for key in (*ARRAY_KEYS, "metadata_json") if key not in shard.files]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        metadata = json.loads(str(shard["metadata_json"].item()))
        if int(metadata.get("cache_format_version", -1)) != 5:
            raise ValueError(f"{path} is not a v5 epsilon ref cache.")
        shards.append(shard)
        metadatas.append(metadata)

    invariant_keys = (
        "label_type",
        "state_source",
        "initial_state_distribution",
        "trajectory_update",
        "trajectories_per_observation",
        "labels_per_trajectory",
        "config",
        "base_config",
        "base_checkpoint_dir",
        "base_action_stats",
        "base_action_stats_sha256",
        "hdf5_path",
        "task",
        "vlm_cost_config",
        "teacher_code_sha256",
        "prompt",
        "num_steps",
        "num_iterations",
        "prediction_type",
        "target_transform",
        "ddim_num_train_timesteps",
        "score_model_action_dim",
        "score_model_action_horizon",
        "stored_action_dim",
        "norm_stats_fingerprint",
        "use_quantile_norm",
        "subtask_mode",
        "mpc",
    )
    reference = metadatas[0]
    for path, metadata in zip(args.inputs[1:], metadatas[1:], strict=True):
        mismatches = [key for key in invariant_keys if metadata.get(key) != reference.get(key)]
        if args.allow_teacher_code_mismatch:
            mismatches = [key for key in mismatches if key != "teacher_code_sha256"]
        if mismatches:
            raise ValueError(f"{path} differs from the first shard in: {mismatches}")

    labels_per_trajectory = int(reference["labels_per_trajectory"])
    merged = {
        key: np.concatenate([shard[key] for shard in shards], axis=0)
        for key in ARRAY_KEYS
        if key != "trajectory_id"
    }
    total_labels = len(merged["iteration"])
    if total_labels % labels_per_trajectory:
        raise ValueError("Merged label count is not divisible by labels_per_trajectory.")
    total_trajectories = total_labels // labels_per_trajectory
    merged["trajectory_id"] = np.repeat(
        np.arange(total_trajectories, dtype=np.int64), labels_per_trajectory
    )

    metadata = dict(reference)
    teacher_code_hashes = sorted(
        {str(item["teacher_code_sha256"]) for item in metadatas}
    )
    if len(teacher_code_hashes) > 1:
        metadata["teacher_code_sha256"] = "mixed"
        metadata["teacher_code_sha256_values"] = teacher_code_hashes
    metadata["num_observations"] = int(sum(item["num_observations"] for item in metadatas))
    metadata["num_trajectories"] = int(total_trajectories)
    metadata["merged_shards"] = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "num_observations": int(item["num_observations"]),
            "num_trajectories": int(item["num_trajectories"]),
            "obs_shard": item.get("obs_shard"),
            "obs_num_shards": item.get("obs_num_shards"),
            "label_seed": item.get("label_seed"),
            "teacher_code_sha256": item.get("teacher_code_sha256"),
        }
        for path, item in zip(args.inputs, metadatas, strict=True)
    ]
    expected_trajectories = (
        int(metadata["num_observations"])
        * int(metadata["trajectories_per_observation"])
    )
    if total_trajectories != expected_trajectories:
        raise ValueError(
            f"Merged trajectories={total_trajectories}, expected {expected_trajectories}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **merged,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    observation_output = None
    if reference.get("observation_source") == "eval_steering_live_model_inputs":
        observation_output = _merge_observation_sidecars(
            args.inputs, args.output, int(metadata["num_observations"])
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_shards": len(shards),
                "num_observations": metadata["num_observations"],
                "num_trajectories": total_trajectories,
                "labels": total_labels,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
