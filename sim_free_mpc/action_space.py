from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DecodedActionChunk:
    """Decoded action chunk used by sim-free cost evaluation."""

    real_actions: torch.Tensor
    model_actions: torch.Tensor


def _state_for_sample(policy_inputs: dict[str, Any]) -> torch.Tensor:
    state = policy_inputs["state"]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    if state.ndim == 2:
        return state[0]
    return state


def decode_model_action_chunks(
    policy: Any,
    policy_inputs: dict[str, Any],
    model_chunks: torch.Tensor,
) -> DecodedActionChunk:
    """Decode model-space chunks to executable Droid joint-position actions.

    OpenPI output transforms are numpy-oriented and include unnormalization,
    absolute-action reconstruction, and `DroidOutputs` truncation. For the MVP we
    intentionally reuse the policy wrapper's output path per sample so the MPC
    sees the same real action space as `env.step`.
    """
    if model_chunks.ndim == 2:
        model_chunks = model_chunks.unsqueeze(0)
    if model_chunks.ndim != 3:
        raise ValueError(f"Expected model_chunks [N,H,D], got {tuple(model_chunks.shape)}")

    device = model_chunks.device
    dtype = model_chunks.dtype
    state = _state_for_sample(policy_inputs)
    state = state.to(device=device, dtype=dtype)

    real_chunks = []
    with torch.no_grad():
        for chunk in model_chunks:
            # `output_to_actions` expects a batch dimension and returns [H, 8].
            decoded = policy.output_to_actions(
                {"state": state.unsqueeze(0)},
                chunk.unsqueeze(0),
            )
            real_chunks.append(torch.as_tensor(decoded, device=device, dtype=dtype))

    return DecodedActionChunk(
        real_actions=torch.stack(real_chunks, dim=0),
        model_actions=model_chunks,
    )


def decode_numpy_action_chunk(
    policy: Any,
    policy_inputs: dict[str, Any],
    model_chunk: torch.Tensor,
) -> np.ndarray:
    """Convenience helper for debug paths that need a numpy executable chunk."""
    decoded = decode_model_action_chunks(policy, policy_inputs, model_chunk)
    return np.asarray(decoded.real_actions[0].detach().cpu())
