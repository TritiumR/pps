"""Decode-only policy surface: transforms and norm stats without model weights.

Modes that drive the action chunk from an external planner never forward-pass the checkpoint, but
still need its decode surface -- transforms, norm stats and action shape -- so this supplies those
without loading ~2.3B parameters.

sample_actions raises rather than returning something plausible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from typing_extensions import override

import openpi.policies.policy as _policy


class DecodeOnlyModel:
    """Stand-in for a loaded model exposing only what action decoding reads.

    Mirrors the loaded model's `config`, `sample_noise`, `to` and `eval` surface. `sample_noise`
    is a byte-for-byte copy of `PI0Pytorch.sample_noise` so the noise draw is identical for an
    identical RNG state.
    """

    def __init__(self, config: Any):
        self.config = config

    def to(self, device: Any) -> "DecodeOnlyModel":
        del device
        return self

    def eval(self) -> "DecodeOnlyModel":
        return self

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_actions(self, *args, **kwargs):
        raise RuntimeError(
            "This policy was built decode-only (no weights loaded); it cannot sample actions. "
            "Drop the decode-only flag for any mode that forward-passes the base network."
        )


# Placeholder for a camera-free rollout: never read, only shaped so the image transforms run.
_PLACEHOLDER_IMAGE_HW = 224


class DecodeOnlyPolicy(_policy.Policy):
    """Policy whose model is a `DecodeOnlyModel`.

    Adds one behavior over `Policy`: missing camera observations are filled with zero images so
    the checkpoint's own input pipeline still runs unchanged. The state path -- the only part the
    decode uses -- is untouched by the placeholder.
    """

    _IMAGE_KEYS = (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    )

    @override
    def obs_to_input(self, obs: dict):
        if any(key not in obs for key in self._IMAGE_KEYS):
            obs = dict(obs)
            placeholder = np.zeros(
                (_PLACEHOLDER_IMAGE_HW, _PLACEHOLDER_IMAGE_HW, 3), dtype=np.uint8
            )
            for key in self._IMAGE_KEYS:
                obs.setdefault(key, placeholder)
        return super().obs_to_input(obs)
