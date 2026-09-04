from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .fk import PANDA_JOINT_LIMITS


_PRINTED_TORCH_OUTPUT_NORM_DEBUG_IDS: set[int] = set()
_PRINTED_TORCH_OUTPUT_NORM_UNAVAILABLE_IDS: set[int] = set()


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


def _normalize_torch(
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
            head = 2.0 * (values[..., :stats_dim] - q01) / (q99 - q01 + 1e-6) - 1.0
            return torch.cat((head, values[..., stats_dim:]), dim=-1)
        q01 = q01[..., :data_dim]
        q99 = q99[..., :data_dim]
        return 2.0 * (values - q01) / (q99 - q01 + 1e-6) - 1.0

    mean = _stats_tensor(stats, "mean", device, dtype)
    std = _stats_tensor(stats, "std", device, dtype)
    if mean is None or std is None:
        return None
    mean = _pad_to_dim_torch(mean, values.shape[-1], fill_value=0.0)
    std = _pad_to_dim_torch(std, values.shape[-1], fill_value=1.0)
    return (values - mean) / (std + 1e-6)


def rebase_model_action_chunk(
    policy: Any,
    model_chunk: torch.Tensor,
    *,
    previous_state: torch.Tensor | np.ndarray,
    current_state: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Preserve absolute arm targets when moving a delta-action chunk to a new state."""
    metadata = getattr(policy, "_metadata", {}) or {}
    output_norm_stats = metadata.get("output_norm_stats")
    if (
        not output_norm_stats
        or "actions" not in output_norm_stats
        or "state" not in output_norm_stats
    ):
        raise ValueError("Action warm rebasing requires action and state normalization stats.")

    device = model_chunk.device
    dtype = model_chunk.dtype
    use_quantile_norm = bool(metadata.get("use_quantile_norm", False))
    actions = _unnormalize_torch(
        model_chunk,
        output_norm_stats["actions"],
        use_quantile_norm=use_quantile_norm,
    )
    previous = _unnormalize_torch(
        _state_for_sample({"state": previous_state}).to(device=device, dtype=dtype),
        output_norm_stats["state"],
        use_quantile_norm=use_quantile_norm,
    )
    current = _unnormalize_torch(
        _state_for_sample({"state": current_state}).to(device=device, dtype=dtype),
        output_norm_stats["state"],
        use_quantile_norm=use_quantile_norm,
    )
    if actions is None or previous is None or current is None:
        raise ValueError("Action warm rebasing could not apply the configured normalization.")

    delta_dims = min(7, actions.shape[-1], previous.shape[-1], current.shape[-1])
    actions = actions.clone()
    actions[..., :delta_dims] += previous[:delta_dims] - current[:delta_dims]
    rebased = _normalize_torch(
        actions,
        output_norm_stats["actions"],
        use_quantile_norm=use_quantile_norm,
    )
    if rebased is None:
        raise ValueError("Action warm rebasing could not restore normalized model actions.")
    return rebased


def _norm_stat_debug_head(stats: Any, field: str, *, count: int = 8) -> str:
    value = getattr(stats, field, None)
    if value is None:
        return "None"
    arr = np.asarray(value).reshape(-1)
    head = ", ".join(f"{float(v):.6g}" for v in arr[:count])
    return f"len={arr.size} first{min(count, arr.size)}=[{head}]"


def _maybe_print_torch_output_norm_debug(
    policy: Any,
    output_norm_stats: dict[str, Any],
    *,
    use_quantile_norm: bool,
    model_chunks: torch.Tensor,
):
    metadata = getattr(policy, "_metadata", {}) or {}
    if not metadata.get("debug_torch_output_to_actions_norm_stats", False):
        return

    policy_id = id(policy)
    if policy_id in _PRINTED_TORCH_OUTPUT_NORM_DEBUG_IDS:
        return
    _PRINTED_TORCH_OUTPUT_NORM_DEBUG_IDS.add(policy_id)

    print(
        "torch_output_to_actions_norm_debug: "
        f"source={metadata.get('output_norm_stats_source', '<unknown>')} "
        f"use_quantile_norm={use_quantile_norm} "
        f"model_chunks_shape={tuple(model_chunks.shape)} "
        f"output_keys={sorted(output_norm_stats.keys())}",
        flush=True,
    )
    for key in ("state", "actions"):
        stats = output_norm_stats.get(key)
        if stats is None:
            print(f"torch_output_to_actions_norm_debug {key}: MISSING", flush=True)
            continue
        for field in ("mean", "std", "q01", "q99"):
            print(
                f"torch_output_to_actions_norm_debug {key}.{field}: "
                f"{_norm_stat_debug_head(stats, field)}",
                flush=True,
            )


def _maybe_print_torch_output_norm_unavailable(
    policy: Any,
    output_norm_stats: Any,
):
    metadata = getattr(policy, "_metadata", {}) or {}
    if not metadata.get("debug_torch_output_to_actions_norm_stats", False):
        return

    policy_id = id(policy)
    if policy_id in _PRINTED_TORCH_OUTPUT_NORM_UNAVAILABLE_IDS:
        return
    _PRINTED_TORCH_OUTPUT_NORM_UNAVAILABLE_IDS.add(policy_id)

    keys = sorted(output_norm_stats.keys()) if isinstance(output_norm_stats, dict) else None
    print(
        "torch_output_to_actions_norm_debug: fast_path_disabled "
        f"source={metadata.get('output_norm_stats_source', '<unknown>')} "
        f"output_keys={keys} required_keys=['actions', 'state']",
        flush=True,
    )


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
        _maybe_print_torch_output_norm_unavailable(policy, output_norm_stats)
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

    state = policy_inputs["state"]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    state = state.to(device=device, dtype=dtype)
    state = _unnormalize_torch(
        state,
        output_norm_stats["state"],
        use_quantile_norm=use_quantile_norm,
    )
    if state is None:
        return None
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[0] not in (1, model_chunks.shape[0]):
        return None
    _maybe_print_torch_output_norm_debug(
        policy,
        output_norm_stats,
        use_quantile_norm=use_quantile_norm,
        model_chunks=model_chunks,
    )
    delta_dims = min(7, actions.shape[-1], state.shape[-1])
    actions = actions.clone()
    actions[..., :delta_dims] = (
        actions[..., :delta_dims] + state[:, None, :delta_dims]
    )
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
        arm = out[..., :7]
        if arm.ndim == 1:
            if current.ndim > 1:
                current = current.reshape(-1, current.shape[-1])[0]
            delta = torch.clamp(arm - current, -max_joint_delta, max_joint_delta)
            out[..., :7] = torch.clamp(current + delta, lower, upper)
        else:
            original_shape = arm.shape
            horizon = original_shape[-2]
            flat = arm.reshape(-1, horizon, 7)
            if current.ndim == 1:
                prev = current.view(1, 7).expand(flat.shape[0], 7)
            else:
                current = current.reshape(-1, current.shape[-1])
                if current.shape[0] not in (1, flat.shape[0]):
                    raise ValueError(
                        "current_joint_pos batch does not match action batch: "
                        f"current={tuple(current.shape)}, actions={tuple(original_shape)}"
                    )
                prev = current[:, :7].expand(flat.shape[0], 7)
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
    apply_clamp: bool = True,
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
        if not apply_clamp:
            return DecodedActionChunk(real_actions=fast_actions, model_actions=model_chunks)
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

    real_actions = torch.stack(real_chunks, dim=0)
    if apply_clamp:
        real_actions = clamp_real_action_chunk(
            real_actions,
            current_joint_pos=current_joint_pos,
            max_joint_delta=max_joint_delta,
            joint_limit_margin=joint_limit_margin,
        )
    return DecodedActionChunk(real_actions=real_actions, model_actions=model_chunks)


def decode_numpy_action_chunk(
    policy: Any,
    policy_inputs: dict[str, Any],
    model_chunk: torch.Tensor,
) -> np.ndarray:
    """Convenience helper for debug paths that need a numpy executable chunk."""
    decoded = decode_model_action_chunks(policy, policy_inputs, model_chunk)
    return np.asarray(decoded.real_actions[0].detach().cpu())


# ------------------------------------------------------------------ demo-delta decode surface
# Everything below is opt-in: nothing above constructs it, so a caller that never builds a
# DemoDeltaDecodePolicy decodes exactly as before.


@dataclass(frozen=True)
class AffineNormStats:
    """Mean/std pair in the shape `_unnormalize_torch` reads."""

    mean: np.ndarray
    std: np.ndarray


def affine_stats_from_quantiles(stats: Any) -> AffineNormStats:
    """Rewrite a q01/q99 band as the mean/std that unnormalizes identically.

    Both branches of `_unnormalize_torch` are affine, so a quantile band has an exact mean/std
    twin. That lets one stats dict hold a quantile-normalized state next to mean/std actions.
    """
    q01 = getattr(stats, "q01", None)
    q99 = getattr(stats, "q99", None)
    if q01 is None or q99 is None:
        raise ValueError("Quantile-normalized stats must carry q01 and q99.")
    half = (np.asarray(q99, dtype=np.float64) - np.asarray(q01, dtype=np.float64) + 1e-6) / 2.0
    return AffineNormStats(
        mean=(np.asarray(q01, dtype=np.float64) + half).astype(np.float32),
        std=(half - 1e-6).astype(np.float32),
    )


def load_action_norm_stats_json(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an action_norm_stats-style JSON into (mean, std) float32 arrays."""
    raw = json.loads(pathlib.Path(path).read_text())
    for key in ("mean", "std"):
        if key not in raw:
            raise ValueError(f"{path}: action norm stats need a '{key}' array.")
    return (
        np.asarray(raw["mean"], dtype=np.float32),
        np.asarray(raw["std"], dtype=np.float32),
    )


class DemoDeltaDecodePolicy:
    """Decode surface that reads a model chunk as demonstration joint deltas.

    Mirrors the ChunkDecodePolicy of mujoco_eval/robolab_eval: a chunk row decodes to
    `q_now + mean + std * x`, so `std` is the model-space scale the planner explores in. The
    state half is inherited from `policy`, so callers keep feeding that policy's own normalized
    state and only the ACTION representation changes.
    """

    def __init__(
        self,
        policy: Any,
        action_mean,
        action_std,
        *,
        source: str = "demo_delta_stats",
    ):
        metadata = getattr(policy, "_metadata", {}) or {}
        output_norm_stats = metadata.get("output_norm_stats") or {}
        if "state" not in output_norm_stats:
            raise ValueError("Demo-delta decoding needs the policy's state normalization stats.")
        state_stats = output_norm_stats["state"]
        if bool(metadata.get("use_quantile_norm", False)):
            state_stats = affine_stats_from_quantiles(state_stats)
        self._metadata = {
            "output_norm_stats": {
                "actions": AffineNormStats(
                    mean=np.asarray(action_mean, dtype=np.float32),
                    std=np.asarray(action_std, dtype=np.float32),
                ),
                "state": state_stats,
            },
            "use_quantile_norm": False,
            "output_norm_stats_source": source,
            "debug_torch_output_to_actions_norm_stats": bool(
                metadata.get("debug_torch_output_to_actions_norm_stats", False)
            ),
        }
