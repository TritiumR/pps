"""Deterministic observation-level train/validation splits for ref A/B pilots."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np


def split_observations(
    demo_names,
    step_indices,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    demo_names = np.asarray(demo_names).astype(str)
    step_indices = np.asarray(step_indices, dtype=np.int64)
    if len(demo_names) != len(step_indices):
        raise ValueError("demo_names and step_indices have different lengths.")
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    num_observations = len(demo_names)
    if num_observations == 0:
        raise ValueError("Cannot split an empty observation set.")

    all_indices = np.arange(num_observations, dtype=np.int64)
    if val_fraction == 0.0:
        val_indices = np.empty(0, dtype=np.int64)
    else:
        num_val = max(1, int(round(num_observations * float(val_fraction))))
        if num_val >= num_observations:
            raise ValueError("Validation split leaves no training observations.")
        rng = np.random.default_rng(int(seed))
        val_indices = np.sort(rng.choice(all_indices, size=num_val, replace=False))
    train_indices = np.setdiff1d(all_indices, val_indices, assume_unique=True)

    digest = hashlib.sha256()
    for demo_name, step_index in zip(demo_names, step_indices, strict=True):
        digest.update(demo_name.encode("utf-8"))
        digest.update(np.asarray(step_index, dtype=np.int64).tobytes())
    manifest = {
        "split_unit": "observation",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "observation_fingerprint": digest.hexdigest(),
        "num_observations": int(num_observations),
        "num_train_observations": int(len(train_indices)),
        "num_val_observations": int(len(val_indices)),
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "val_observations": [
            {"demo_name": demo_names[idx], "step_index": int(step_indices[idx])}
            for idx in val_indices
        ],
    }
    return train_indices, val_indices, manifest


def write_split_manifest(path: pathlib.Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
