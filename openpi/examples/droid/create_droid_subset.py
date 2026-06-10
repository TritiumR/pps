"""
Creates a physical subset of the DROID RLDS dataset (~100GB) and optionally computes
non-idle timestep ranges (filter_dict) for use during training.

Overview
--------
The full DROID dataset is ~1.7TB spread across 2048 TFDS shards.  Rather than
reading all shards during every training run (slow), this script creates a small
TFDS dataset that contains only the shards you actually want to train on.

What the script does
--------------------
  1. Selects a random set of TFDS shards totalling ~target_gb from the original
     dataset.  Creates a new TFDS directory that symlinks those shards with new
     contiguous indices, and writes an updated dataset_info.json.  Training will
     only read this subset's data (~100GB instead of 1.7TB).

  2. (Optional) Iterates the subset and runs the same idle-filtering logic as
     compute_droid_nonidle_ranges.py to produce a filter_dict JSON.

Can I reuse the original GCS filter_dict instead of computing a new one?
------------------------------------------------------------------------
YES.  The original filter_dict at
  gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json
contains non-idle ranges for *all* episodes in DROID 1.0.1.  Since the physical
subset is a strict subset of those episodes, the original filter_dict is fully
valid: when training reads a subset episode, its key IS present in the original
filter_dict and the correct idle ranges are applied.

Advantages of computing a custom filter_dict for the subset:
  - Smaller file (~5k entries vs ~76k), loads faster during training startup.
  - No network download from GCS required.

If you are happy downloading from GCS at training startup, pass
  --skip_filter_computation
and use the default filter_dict_path in the training config.

Directory structure created
---------------------------
    {output_dir}/
      droid/
        1.0.1/
          1.0.1/
            dataset_info.json   <- updated shard count & lengths
            droid_101-train.tfrecord-00000-of-NNNNN  -> symlink to original shard
            droid_101-train.tfrecord-00001-of-NNNNN  -> symlink to original shard
            ...

Usage
-----
Full run (create subset + compute filter dict):
    uv run --group rlds examples/droid/create_droid_subset.py \\
        --droid_dir /home/chuanruo/openpi/droid \\
        --output_dir /home/chuanruo/openpi/droid_subset \\
        --filter_dict_path /home/chuanruo/openpi/droid_subset_filter_dict.json \\
        --target_gb 100

Subset only (reuse original GCS filter_dict during training):
    uv run --group rlds examples/droid/create_droid_subset.py \\
        --droid_dir /home/chuanruo/openpi/droid \\
        --output_dir /home/chuanruo/openpi/droid_subset \\
        --filter_dict_path /dev/null \\
        --skip_filter_computation

Training config changes (src/openpi/training/config.py)
-------------------------------------------------------
In the pi05_full_droid_finetune TrainConfig, update RLDSDroidDataConfig:

    data=RLDSDroidDataConfig(
        rlds_data_dir="{output_dir}",   # <-- parent of the new droid/ directory
        ...
        datasets=(
            droid_rlds_dataset.RLDSDataset(
                name="droid",
                version="1.0.1",
                weight=1.0,
                # Use your custom filter dict:
                filter_dict_path="/path/to/droid_subset_filter_dict.json",
                # OR keep using the original GCS filter dict:
                # filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
            ),
        ),
    ),
"""

import json
import os
import random
from pathlib import Path

import dataclasses
import numpy as np
from tqdm import tqdm
import tyro


@dataclasses.dataclass
class Args:
    # Path to the original droid dataset directory (the one containing 1.0.1/1.0.1/).
    # E.g. /home/chuanruo/openpi/droid
    droid_dir: str

    # Where to create the subset TFDS directory.
    # E.g. /home/chuanruo/openpi/droid_subset
    # Will create {output_dir}/droid/1.0.1/1.0.1/ mirroring the original structure.
    output_dir: str

    # Path to save the filter_dict JSON.
    filter_dict_path: str

    # Target subset size in GB.
    target_gb: float = 100.0

    # Random seed for shard selection.
    seed: int = 42

    # Idle filtering parameters (same defaults as compute_droid_nonidle_ranges.py).
    min_idle_len: int = 7
    min_non_idle_len: int = 16
    filter_last_n_in_ranges: int = 10

    # Skip creating the subset directory (if it already exists and you only want
    # to (re)compute the filter dict for an existing subset).
    skip_subset_creation: bool = False

    # Skip the filter dict computation (only create the subset directory).
    skip_filter_computation: bool = False


def create_subset_dir(args: Args, src_tfds_dir: Path, dst_tfds_dir: Path) -> list[int]:
    """
    Selects a random subset of shards from src_tfds_dir, creates symlinks in
    dst_tfds_dir with contiguous shard indices, and writes a new dataset_info.json.

    Returns the list of selected original shard indices.
    """
    with (src_tfds_dir / "dataset_info.json").open() as f:
        dataset_info = json.load(f)

    split_info = dataset_info["splits"][0]
    shard_lengths = [int(x) for x in split_info["shardLengths"]]
    total_bytes = int(split_info["numBytes"])
    n_total_shards = len(shard_lengths)

    # Compute per-shard byte estimate from the metadata's total.
    bytes_per_episode = total_bytes / sum(shard_lengths)

    # Greedily select shards until we exceed target_gb.
    # We sample randomly (without replacement) for diversity, then sort for
    # reproducible symlink naming.
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
    print(f"  Original dataset : {n_total_shards} shards, {sum(shard_lengths):,} episodes, {total_bytes/1e9:.1f} GB")
    print(f"  Selected         : {n_new_shards} shards, {subset_episodes:,} episodes, ~{subset_gb:.1f} GB")
    print(f"  Output dir       : {dst_tfds_dir}")

    dst_tfds_dir.mkdir(parents=True, exist_ok=True)

    # Determine the dataset/split/format names from the template and dataset_info.
    dataset_name = dataset_info["name"]       # "droid_101"
    file_format = dataset_info["fileFormat"]  # "tfrecord"

    def shard_filename(dataset: str, fmt: str, shard_x: int, shard_y: int) -> str:
        return f"{dataset}-train.{fmt}-{shard_x:05d}-of-{shard_y:05d}"

    # Create symlinks with new contiguous indices.
    for new_idx, orig_idx in enumerate(selected_sorted):
        src_name = shard_filename(dataset_name, file_format, orig_idx, n_total_shards)
        dst_name = shard_filename(dataset_name, file_format, new_idx, n_new_shards)
        src_path = src_tfds_dir / src_name
        dst_path = dst_tfds_dir / dst_name

        if not src_path.exists():
            raise FileNotFoundError(f"Source shard not found: {src_path}")

        if dst_path.is_symlink() or dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_path.resolve())

    print(f"  Created {n_new_shards} symlinks.")

    # Remove any stale symlinks from a previous run with a different shard count.
    for p in dst_tfds_dir.glob(f"{dataset_name}-train.{file_format}-*"):
        if p.is_symlink():
            # Check if it follows the new naming convention; remove if stale.
            parts = p.name.rsplit("-of-", 1)
            if len(parts) == 2 and int(parts[1]) != n_new_shards:
                p.unlink()

    # Write updated dataset_info.json.
    subset_info = json.loads(json.dumps(dataset_info))  # deep copy
    subset_bytes = int(bytes_per_episode * subset_episodes)
    subset_info["splits"][0]["shardLengths"] = [str(x) for x in selected_lengths]
    subset_info["splits"][0]["numBytes"] = str(subset_bytes)

    with (dst_tfds_dir / "dataset_info.json").open("w") as f:
        json.dump(subset_info, f, indent=2)
    print("  Wrote dataset_info.json.")

    # Symlink features.json (required by tfds.builder_from_directory to deserialize TFRecords).
    features_src = src_tfds_dir / "features.json"
    features_dst = dst_tfds_dir / "features.json"
    if features_src.exists():
        if features_dst.is_symlink() or features_dst.exists():
            features_dst.unlink()
        features_dst.symlink_to(features_src.resolve())
        print("  Symlinked features.json.")
    else:
        print("  WARNING: features.json not found in source directory; tfds may fail to read data.")

    return selected_sorted


def compute_filter_dict(args: Args, tfds_dir: Path) -> None:
    """
    Iterates the TFDS dataset at tfds_dir and computes non-idle timestep ranges
    for each episode. Saves the result to args.filter_dict_path.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tf.config.set_visible_devices([], "GPU")

    filter_dict_path = Path(args.filter_dict_path)

    # Support resuming from a partial run.
    keep_ranges_map: dict = {}
    if filter_dict_path.exists():
        with filter_dict_path.open() as f:
            keep_ranges_map = json.load(f)
        print(f"\nResuming filter computation: {len(keep_ranges_map)} episodes already processed.")

    print(f"\n--- Filter dict computation ---")
    print(f"  TFDS dir : {tfds_dir}")
    print(f"  Output   : {filter_dict_path}")
    print(f"  Params   : min_idle_len={args.min_idle_len}, min_non_idle_len={args.min_non_idle_len}, "
          f"filter_last_n={args.filter_last_n_in_ranges}")

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

        # Collect joint velocities for all steps in this episode.
        joint_velocities = [step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]]
        joint_velocities = np.array(joint_velocities)
        T = len(joint_velocities)

        # --- Idle detection (identical to compute_droid_nonidle_ranges.py) ---

        # A timestep is idle if its joint velocity hasn't changed from the previous step.
        is_idle = np.hstack(
            [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
        )

        # Find starts/ends of idle segments.
        padded = np.concatenate([[False], is_idle, [False]])
        diff = np.diff(padded.astype(int))
        idle_starts = np.where(diff == 1)[0]
        idle_ends = np.where(diff == -1)[0]

        # Only mask out idle segments long enough to matter.
        long_idle = (idle_ends - idle_starts) >= args.min_idle_len
        idle_starts = idle_starts[long_idle]
        idle_ends = idle_ends[long_idle]

        keep_mask = np.ones(T, dtype=bool)
        for start, end in zip(idle_starts, idle_ends, strict=True):
            keep_mask[start:end] = False

        # Find contiguous non-idle segments.
        keep_padded = np.concatenate([[False], keep_mask, [False]])
        keep_diff = np.diff(keep_padded.astype(int))
        keep_starts = np.where(keep_diff == 1)[0]
        keep_ends = np.where(keep_diff == -1)[0]

        # Only keep segments long enough to form useful action chunks.
        long_enough = (keep_ends - keep_starts) >= args.min_non_idle_len
        keep_starts = keep_starts[long_enough]
        keep_ends = keep_ends[long_enough]

        # Trim the tail of each range (those timesteps would produce action chunks
        # that are mostly idle, since the robot is winding down).
        ranges = []
        for start, end in zip(keep_starts, keep_ends, strict=True):
            trimmed_end = int(end) - args.filter_last_n_in_ranges
            if trimmed_end > int(start):
                ranges.append([int(start), trimmed_end])

        keep_ranges_map[key] = ranges
        n_processed += 1

        # Checkpoint periodically.
        if ep_idx % 500 == 0 and ep_idx > 0:
            with filter_dict_path.open("w") as f:
                json.dump(keep_ranges_map, f)

    # Final save.
    with filter_dict_path.open("w") as f:
        json.dump(keep_ranges_map, f)

    total = n_skipped + n_processed
    print(f"\nDone. Processed {n_processed} new episodes, {n_skipped} already cached.")
    print(f"Total episodes in filter dict: {total}")

    # Summary statistics.
    total_frames = sum(e - s for ranges in keep_ranges_map.values() for s, e in ranges)
    eps_with_content = sum(1 for v in keep_ranges_map.values() if v)
    print(f"Episodes with at least one non-idle range: {eps_with_content}/{total}")
    print(f"Total non-idle frames across all episodes: {total_frames:,}")
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

    # Mirror the same directory structure in the output.
    # Training config's rlds_data_dir should be set to output_dir.
    dst_tfds_dir = output_dir / "droid" / "1.0.1"

    if not args.skip_subset_creation:
        create_subset_dir(args, src_tfds_dir, dst_tfds_dir)
    else:
        print(f"Skipping subset creation, using existing: {dst_tfds_dir}")
        if not dst_tfds_dir.exists():
            raise FileNotFoundError(f"Subset dir not found: {dst_tfds_dir}")

    if not args.skip_filter_computation:
        compute_filter_dict(args, dst_tfds_dir)
    else:
        print("Skipping filter dict computation.")

    print("\n=== Next steps ===")
    print("Edit src/openpi/training/config.py, in the pi05_full_droid_finetune TrainConfig:")
    print()
    print("    data=RLDSDroidDataConfig(")
    print(f'        rlds_data_dir="{output_dir}",')
    print("        action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,")
    print("        ...,")
    print("        datasets=(")
    print("            droid_rlds_dataset.RLDSDataset(")
    print('                name="droid",')
    print('                version="1.0.1",')
    print("                weight=1.0,")
    if not args.skip_filter_computation:
        print(f'                filter_dict_path="{args.filter_dict_path}",')
    else:
        print('                # Option A: use your computed filter dict:')
        print(f'                # filter_dict_path="{args.filter_dict_path}",')
        print('                # Option B: reuse the original GCS filter dict (valid for droid/1.0.1):')
        print('                filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",')
    print("            ),")
    print("        ),")
    print("    ),")
    print()
    print("NOTE: The original GCS filter_dict is valid for this subset because it")
    print("  contains non-idle ranges for all DROID 1.0.1 episodes, including those")
    print("  in your subset. A custom filter_dict only has entries for the subset")
    print("  episodes (faster to load, no GCS download needed).")


if __name__ == "__main__":
    main(tyro.cli(Args))
