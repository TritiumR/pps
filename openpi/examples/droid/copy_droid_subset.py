"""
Copies a random subset of DROID RLDS shards (~100GB) to a new directory,
creating a fully self-contained physical dataset ready for training.

Unlike create_droid_subset.py (which uses symlinks), this script performs
actual file copies so the output directory is independent of the original
dataset location.

Usage:
    uv run --group rlds examples/droid/copy_droid_subset.py \\
        --droid_dir /home/chuanruo/openpi/droid \\
        --output_dir /path/to/droid_subset \\
        --filter_dict_path /path/to/droid_subset_filter_dict.json \\
        --target_gb 100

After running, update src/openpi/training/config.py:
    rlds_data_dir = "{output_dir}"
    filter_dict_path = "{filter_dict_path}"

Note: copying ~100GB takes time. Progress is shown per shard.
Interrupted copies can be resumed safely -- already-copied shards are skipped.
"""

import json
import os
import random
import shutil
from pathlib import Path

import dataclasses
import numpy as np
from tqdm import tqdm
import tyro


@dataclasses.dataclass
class Args:
    # Path to the original droid dataset directory (containing 1.0.1/1.0.1/).
    droid_dir: str

    # Destination directory. Will create {output_dir}/droid/1.0.1/1.0.1/.
    output_dir: str

    # Where to save the filter_dict JSON.
    filter_dict_path: str

    # Target subset size in GB.
    target_gb: float = 100.0

    # Random seed for shard selection.
    seed: int = 42

    # Idle filtering parameters (same defaults as compute_droid_nonidle_ranges.py).
    min_idle_len: int = 7
    min_non_idle_len: int = 16
    filter_last_n_in_ranges: int = 10

    # Skip the filter dict computation (only copy shards).
    skip_filter_computation: bool = False


def copy_subset_dir(args: Args, src_tfds_dir: Path, dst_tfds_dir: Path) -> list[int]:
    """
    Selects a random subset of shards, physically copies them to dst_tfds_dir
    with new contiguous indices, and writes an updated dataset_info.json.

    Already-copied shards are skipped (safe to resume after interruption).
    Returns the sorted list of selected original shard indices.
    """
    with (src_tfds_dir / "dataset_info.json").open() as f:
        dataset_info = json.load(f)

    split_info = dataset_info["splits"][0]
    shard_lengths = [int(x) for x in split_info["shardLengths"]]
    total_bytes = int(split_info["numBytes"])
    n_total_shards = len(shard_lengths)

    bytes_per_episode = total_bytes / sum(shard_lengths)

    # Randomly select shards until we reach target_gb.
    random.seed(args.seed)
    shuffled_indices = list(range(n_total_shards))
    random.shuffle(shuffled_indices)

    selected: list[int] = []
    accumulated_bytes = 0.0
    target_bytes = args.target_gb * 1e9
    for idx in shuffled_indices:
        selected.append(idx)
        accumulated_bytes += shard_lengths[idx] * bytes_per_episode
        if accumulated_bytes >= target_bytes:
            break

    selected_sorted = sorted(selected)
    selected_lengths = [shard_lengths[i] for i in selected_sorted]
    n_new_shards = len(selected_sorted)
    subset_gb = accumulated_bytes / 1e9
    subset_episodes = sum(selected_lengths)

    print(f"\n--- Subset selection ---")
    print(f"  Original : {n_total_shards} shards, {sum(shard_lengths):,} episodes, {total_bytes/1e9:.1f} GB")
    print(f"  Selected : {n_new_shards} shards, {subset_episodes:,} episodes, ~{subset_gb:.1f} GB")
    print(f"  Output   : {dst_tfds_dir}")

    dst_tfds_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_info["name"]       # "droid_101"
    file_format = dataset_info["fileFormat"]  # "tfrecord"

    def shard_filename(dataset: str, fmt: str, shard_x: int, shard_y: int) -> str:
        return f"{dataset}-train.{fmt}-{shard_x:05d}-of-{shard_y:05d}"

    # Copy shards with new contiguous indices.
    print(f"\n--- Copying {n_new_shards} shards ---")
    for new_idx, orig_idx in enumerate(tqdm(selected_sorted, desc="Copying shards", unit="shard")):
        src_name = shard_filename(dataset_name, file_format, orig_idx, n_total_shards)
        dst_name = shard_filename(dataset_name, file_format, new_idx, n_new_shards)
        src_path = src_tfds_dir / src_name
        dst_path = dst_tfds_dir / dst_name

        if not src_path.exists():
            raise FileNotFoundError(f"Source shard not found: {src_path}")

        if dst_path.exists():
            # Skip already-copied shards (allows resuming interrupted runs).
            continue

        shutil.copy2(src_path, dst_path)

    print(f"  Done copying.")

    # Remove stale files from any previous run with a different shard count.
    for p in dst_tfds_dir.glob(f"{dataset_name}-train.{file_format}-*"):
        parts = p.name.rsplit("-of-", 1)
        if len(parts) == 2 and int(parts[1]) != n_new_shards:
            print(f"  Removing stale shard: {p.name}")
            p.unlink()

    # Write updated dataset_info.json.
    subset_info = json.loads(json.dumps(dataset_info))
    subset_info["splits"][0]["shardLengths"] = [str(x) for x in selected_lengths]
    subset_info["splits"][0]["numBytes"] = str(int(bytes_per_episode * subset_episodes))

    with (dst_tfds_dir / "dataset_info.json").open("w") as f:
        json.dump(subset_info, f, indent=2)
    print("  Wrote dataset_info.json.")

    # Copy features.json (required by tfds.builder_from_directory to deserialize TFRecords).
    features_src = src_tfds_dir / "features.json"
    features_dst = dst_tfds_dir / "features.json"
    if features_src.exists():
        shutil.copy2(features_src, features_dst)
        print("  Copied features.json.")
    else:
        print("  WARNING: features.json not found in source directory; tfds may fail to read data.")

    return selected_sorted


def compute_filter_dict(args: Args, tfds_dir: Path) -> None:
    """
    Iterates the TFDS dataset at tfds_dir and computes non-idle timestep ranges.
    Saves the result to args.filter_dict_path. Supports resuming.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tf.config.set_visible_devices([], "GPU")

    filter_dict_path = Path(args.filter_dict_path)

    keep_ranges_map: dict = {}
    if filter_dict_path.exists():
        with filter_dict_path.open() as f:
            keep_ranges_map = json.load(f)
        print(f"\nResuming: {len(keep_ranges_map)} episodes already processed.")

    print(f"\n--- Filter dict computation ---")
    print(f"  TFDS dir : {tfds_dir}")
    print(f"  Output   : {filter_dict_path}")

    builder = tfds.builder_from_directory(builder_dir=str(tfds_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    n_skipped = 0
    n_processed = 0

    for ep_idx, ep in enumerate(tqdm(ds, desc="Computing non-idle ranges")):
        recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
        file_path = ep["episode_metadata"]["file_path"].numpy().decode()
        key = f"{recording_folderpath}--{file_path}"

        if key in keep_ranges_map:
            n_skipped += 1
            continue

        joint_velocities = [step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]]
        joint_velocities = np.array(joint_velocities)
        T = len(joint_velocities)

        is_idle = np.hstack(
            [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
        )

        padded = np.concatenate([[False], is_idle, [False]])
        diff = np.diff(padded.astype(int))
        idle_starts = np.where(diff == 1)[0]
        idle_ends = np.where(diff == -1)[0]

        long_idle = (idle_ends - idle_starts) >= args.min_idle_len
        idle_starts = idle_starts[long_idle]
        idle_ends = idle_ends[long_idle]

        keep_mask = np.ones(T, dtype=bool)
        for start, end in zip(idle_starts, idle_ends, strict=True):
            keep_mask[start:end] = False

        keep_padded = np.concatenate([[False], keep_mask, [False]])
        keep_diff = np.diff(keep_padded.astype(int))
        keep_starts = np.where(keep_diff == 1)[0]
        keep_ends = np.where(keep_diff == -1)[0]

        long_enough = (keep_ends - keep_starts) >= args.min_non_idle_len
        keep_starts = keep_starts[long_enough]
        keep_ends = keep_ends[long_enough]

        ranges = []
        for start, end in zip(keep_starts, keep_ends, strict=True):
            trimmed_end = int(end) - args.filter_last_n_in_ranges
            if trimmed_end > int(start):
                ranges.append([int(start), trimmed_end])

        keep_ranges_map[key] = ranges
        n_processed += 1

        if ep_idx % 500 == 0 and ep_idx > 0:
            with filter_dict_path.open("w") as f:
                json.dump(keep_ranges_map, f)

    with filter_dict_path.open("w") as f:
        json.dump(keep_ranges_map, f)

    total = n_skipped + n_processed
    total_frames = sum(e - s for ranges in keep_ranges_map.values() for s, e in ranges)
    eps_with_content = sum(1 for v in keep_ranges_map.values() if v)
    print(f"\nDone. {n_processed} new episodes processed, {n_skipped} skipped (cached).")
    print(f"Episodes with non-idle ranges: {eps_with_content}/{total}")
    print(f"Total non-idle frames: {total_frames:,}")
    print(f"Filter dict saved to: {filter_dict_path}")


def main(args: Args) -> None:
    droid_dir = Path(args.droid_dir)
    output_dir = Path(args.output_dir)

    # The TFDS data lives one level in: droid_dir/1.0.1/
    src_tfds_dir = droid_dir / "1.0.1"
    if not src_tfds_dir.exists():
        raise FileNotFoundError(
            f"Expected TFDS data at {src_tfds_dir}. "
            "Make sure --droid_dir points to the directory containing 1.0.1/."
        )

    dst_tfds_dir = output_dir / "droid" / "1.0.1"

    copy_subset_dir(args, src_tfds_dir, dst_tfds_dir)

    if not args.skip_filter_computation:
        compute_filter_dict(args, dst_tfds_dir)
    else:
        print("\nSkipping filter dict computation.")
        print("You can use the original GCS filter dict (valid for droid/1.0.1):")
        print("  filter_dict_path=\"gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json\"")

    print("\n=== Next steps ===")
    print("Edit src/openpi/training/config.py, in the pi05_full_droid_finetune TrainConfig:")
    print(f'    rlds_data_dir="{output_dir}",')
    print(f'    filter_dict_path="{args.filter_dict_path}",')


if __name__ == "__main__":
    main(tyro.cli(Args))
