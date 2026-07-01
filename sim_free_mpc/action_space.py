from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .fk import PANDA_JOINT_LIMITS


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


def _stats_tensor(stats: Any, name: str, device, dtype) -> torch.Tensor | None:
    value = getattr(stats, name, None)
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=dtype)


def _pad_to_dim_torch(
    value: torch.Tensor,
    target_dim: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    current_dim = value.shape[-1]
    if current_dim == target_dim:
        return value
    if current_dim > target_dim:
        return value[..., :target_dim]
    pad_shape = (*value.shape[:-1], target_dim - current_dim)
    pad = torch.full(pad_shape, fill_value, device=value.device, dtype=value.dtype)
    return torch.cat((value, pad), dim=-1)


def _unnormalize_torch(
    values: torch.Tensor,
    stats: Any,
    *,
    use_quantile_norm: bool,
) -> torch.Tensor | None:
    device = values.device
    dtype = values.dtype
    if use_quantile_norm:
        q01 = _stats_tensor(stats, "q01", device, dtype)
        q99 = _stats_tensor(stats, "q99", device, dtype)
        if q01 is None or q99 is None:
            return None
        stats_dim = q01.shape[-1]
        data_dim = values.shape[-1]
        if stats_dim < data_dim:
            head = (values[..., :stats_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return torch.cat((head, values[..., stats_dim:]), dim=-1)
        else:
            q01 = q01[..., :data_dim]
            q99 = q99[..., :data_dim]
            return (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        mean = _stats_tensor(stats, "mean", device, dtype)
        std = _stats_tensor(stats, "std", device, dtype)
        if mean is None or std is None:
            return None
        mean = _pad_to_dim_torch(mean, values.shape[-1], fill_value=0.0)
        std = _pad_to_dim_torch(std, values.shape[-1], fill_value=1.0)
        return values * (std + 1e-6) + mean


def _torch_output_to_actions(
    policy: Any,
    policy_inputs: dict[str, Any],
    model_chunks: torch.Tensor,
) -> torch.Tensor | None:
    metadata = getattr(policy, "_metadata", {}) or {}
    output_norm_stats = metadata.get("output_norm_stats")
    if (
        not output_norm_stats
        or "actions" not in output_norm_stats
        or "state" not in output_norm_stats
    ):
        return None

    device = model_chunks.device
    dtype = model_chunks.dtype
    use_quantile_norm = bool(metadata.get("use_quantile_norm", False))

    actions = _unnormalize_torch(
        model_chunks,
        output_norm_stats["actions"],
        use_quantile_norm=use_quantile_norm,
    )
    if actions is None:
        return None

    state = _state_for_sample(policy_inputs).to(device=device, dtype=dtype)
    state = _unnormalize_torch(
        state,
        output_norm_stats["state"],
        use_quantile_norm=use_quantile_norm,
    )
    if state is None:
        return None
    if state.ndim != 1:
        return None
    delta_dims = min(7, actions.shape[-1], state.shape[-1])
    actions = actions.clone()
    actions[..., :delta_dims] = actions[..., :delta_dims] + state[:delta_dims]
    return actions[..., :8]


def _joint_limit_tensors(device, dtype, margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    limits = torch.as_tensor(PANDA_JOINT_LIMITS, device=device, dtype=dtype)
    lower = limits[:, 0] + margin
    upper = limits[:, 1] - margin
    return lower, upper


def clamp_real_action_chunk(
    real_actions: torch.Tensor,
    *,
    current_joint_pos: torch.Tensor | np.ndarray | None = None,
    max_joint_delta: float | None = None,
    joint_limit_margin: float = 0.0,
) -> torch.Tensor:
    """Clamp executable Droid actions to Franka joint limits and optional per-step motion.

    `real_actions` are decoded environment actions: first 7 dims are absolute
    joint-position targets, dim 7 is the gripper command.
    """
    if real_actions.shape[-1] < 7:
        return real_actions

    out = real_actions.clone()
    lower, upper = _joint_limit_tensors(out.device, out.dtype, margin=joint_limit_margin)
    out[..., :7] = torch.clamp(out[..., :7], lower, upper)

    if max_joint_delta is not None and max_joint_delta > 0.0 and current_joint_pos is not None:
        current = torch.as_tensor(current_joint_pos, device=out.device, dtype=out.dtype)[..., :7]
        if current.ndim > 1:
            current = current.reshape(-1, current.shape[-1])[0]

        arm = out[..., :7]
        if arm.ndim == 1:
            delta = torch.clamp(arm - current, -max_joint_delta, max_joint_delta)
            out[..., :7] = torch.clamp(current + delta, lower, upper)
        else:
            original_shape = arm.shape
            horizon = original_shape[-2]
            flat = arm.reshape(-1, horizon, 7)
            prev = current.view(1, 7).expand(flat.shape[0], 7)
            for step in range(horizon):
                delta = torch.clamp(flat[:, step, :] - prev, -max_joint_delta, max_joint_delta)
                next_joint = torch.clamp(prev + delta, lower, upper)
                flat[:, step, :] = next_joint
                prev = next_joint
            out[..., :7] = flat.reshape(original_shape)

    if out.shape[-1] > 7:
        out[..., 7] = torch.clamp(out[..., 7], 0.0, 1.0)
    return out


def decode_model_action_chunks(
    policy: Any,
    policy_inputs: dict[str, Any],
    model_chunks: torch.Tensor,
    *,
    current_joint_pos: torch.Tensor | np.ndarray | None = None,
    max_joint_delta: float | None = None,
    joint_limit_margin: float = 0.0,
) -> DecodedActionChunk:
    """Decode model-space chunks to executable Droid joint-position actions.

    The fast path mirrors OpenPI's output transforms in torch: unnormalize,
    reconstruct absolute arm actions from deltas, and apply `DroidOutputs`
    truncation. If the policy does not expose the needed metadata, fall back to
    the original numpy-oriented output path.
    """
    if model_chunks.ndim == 2:
        model_chunks = model_chunks.unsqueeze(0)
    if model_chunks.ndim != 3:
        raise ValueError(f"Expected model_chunks [N,H,D], got {tuple(model_chunks.shape)}")

    fast_actions = _torch_output_to_actions(policy, policy_inputs, model_chunks)
    if fast_actions is not None:
        fast_actions = clamp_real_action_chunk(
            fast_actions,
            current_joint_pos=current_joint_pos,
            max_joint_delta=max_joint_delta,
            joint_limit_margin=joint_limit_margin,
        )
        return DecodedActionChunk(real_actions=fast_actions, model_actions=model_chunks)

    real_chunks = []
    with torch.no_grad():
        for chunk in model_chunks:
            # `output_to_actions` expects a batch dimension and returns [H, 8].
            decoded = policy.output_to_actions(
                {"state": _state_for_sample(policy_inputs).unsqueeze(0)},
                chunk.unsqueeze(0),
            )
            real_chunks.append(
                torch.as_tensor(decoded, device=model_chunks.device, dtype=model_chunks.dtype)
            )

    return DecodedActionChunk(
        real_actions=clamp_real_action_chunk(
            torch.stack(real_chunks, dim=0),
            current_joint_pos=current_joint_pos,
            max_joint_delta=max_joint_delta,
            joint_limit_margin=joint_limit_margin,
        ),
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
