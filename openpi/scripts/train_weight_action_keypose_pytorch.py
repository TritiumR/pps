#!/usr/bin/env python3
"""Train weight actions plus one random future phase-tail keypose.

The model output layout is ``[15 executable actions, 1 keypose]`` by default.
The final token is sampled uniformly from the current phase's valid final
0.8-second tail, restricted to keyposes strictly after the observation frame.
All action/keypose diffusion tokens use the bidirectional two-block attention
path (``attention_mode=two_block_diffusion``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


PPS_ROOT = Path(__file__).resolve().parents[2]
if str(PPS_ROOT) not in sys.path:
    sys.path.insert(0, str(PPS_ROOT))

import openpi.training.data_loader as data_loader  # noqa: E402
from openpi.training.subphase_keypose_dataset import (  # noqa: E402
    SubphaseKeyposeDataset,
    executable_action_horizon,
)
import train_pytorch  # noqa: E402


DEFAULT_SIDECAR = (
    PPS_ROOT
    / "artifacts"
    / "weight_phase_future_tail12_action_gripper"
    / "weight_phase_future_policy_qpos_tail12.npz"
)
_create_torch_dataset = data_loader.create_torch_dataset


def _sidecar_path() -> Path:
    configured = os.environ.get("PPS_WEIGHT_KEYPOSE_SIDECAR")
    path = Path(configured).expanduser().resolve() if configured else DEFAULT_SIDECAR
    if not path.is_file():
        raise FileNotFoundError(
            f"Weight keypose sidecar not found: {path}. "
            "Build it from data/weight/generated_dataset.hdf5 first."
        )
    return path


def _create_weight_action_keypose_dataset(
    data_config,
    action_horizon,
    model_config,
):
    attention_mode = getattr(model_config, "attention_mode", None)
    if attention_mode != "two_block_diffusion":
        raise ValueError(
            "Weight action/keypose training requires "
            "--model.attention-mode two_block_diffusion; "
            f"got {attention_mode!r}."
        )

    action_prefix_horizon = executable_action_horizon(action_horizon)
    dataset = _create_torch_dataset(
        data_config,
        action_prefix_horizon,
        model_config,
    )
    wrapped = SubphaseKeyposeDataset(
        dataset,
        _sidecar_path(),
        action_keys=data_config.action_sequence_keys,
    )
    logging.info(
        "Weight action/keypose dataset: %d executable actions + 1 uniformly "
        "sampled strictly-future phase-tail keypose; %d/%d rows eligible; "
        "attention_mode=%s.",
        action_prefix_horizon,
        len(wrapped),
        len(dataset),
        attention_mode,
    )
    return wrapped


def _set_model_override(
    flag: str,
    default: str,
    *,
    required: str | None = None,
) -> None:
    """Set a default CLI override and reject an incompatible required value."""
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"Missing value for {flag}")
            value = sys.argv[index + 1]
            if required is not None and value != required:
                raise ValueError(f"{flag} must be {required!r}, got {value!r}")
            return
        prefix = flag + "="
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
            if required is not None and value != required:
                raise ValueError(f"{flag} must be {required!r}, got {value!r}")
            sys.argv[index : index + 1] = [flag, value]
            return
    sys.argv.extend([flag, required if required is not None else default])


def main() -> None:
    # Preserve the base weight policy's 15 executable actions while reserving
    # the final output token for the sampled keypose.
    _set_model_override("--model.action-horizon", "16")
    _set_model_override(
        "--model.attention-mode",
        "two_block_diffusion",
        required="two_block_diffusion",
    )
    data_loader.create_torch_dataset = _create_weight_action_keypose_dataset
    train_pytorch.main()


if __name__ == "__main__":
    main()
