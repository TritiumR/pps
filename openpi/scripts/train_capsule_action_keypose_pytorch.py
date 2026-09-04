#!/usr/bin/env python3
"""Train capsule actions and a terminal subphase keypose with one Gemma expert.

This reuses the standard PyTorch proxy trainer and model unchanged. The only
data-path change is that the model horizon's final token is replaced with a
randomly sampled, strictly future keypose from the current subphase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import openpi.training.data_loader as data_loader
from openpi.training.subphase_keypose_dataset import (
    SubphaseKeyposeDataset,
    executable_action_horizon,
)
import train_pytorch


_create_torch_dataset = data_loader.create_torch_dataset


def _create_capsule_action_keypose_dataset(
    data_config,
    action_horizon,
    model_config,
):
    sidecar = os.environ.get("PPS_CAPSULE_KEYPOSE_SIDECAR")
    if not sidecar:
        raise ValueError("PPS_CAPSULE_KEYPOSE_SIDECAR is required")
    sidecar_path = Path(sidecar).expanduser().resolve()
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Capsule keypose sidecar not found: {sidecar_path}")

    action_prefix_horizon = executable_action_horizon(action_horizon)
    dataset = _create_torch_dataset(
        data_config,
        action_prefix_horizon,
        model_config,
    )
    wrapped = SubphaseKeyposeDataset(
        dataset,
        sidecar_path,
        action_keys=data_config.action_sequence_keys,
    )
    logging.info(
        "Capsule action/keypose dataset: %d executable actions + 1 keypose; "
        "%d/%d rows have a strictly future subphase keypose.",
        action_prefix_horizon,
        len(wrapped),
        len(dataset),
    )
    return wrapped


def main() -> None:
    data_loader.create_torch_dataset = _create_capsule_action_keypose_dataset
    train_pytorch.main()


if __name__ == "__main__":
    main()
