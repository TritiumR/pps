from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .action_space import _state_for_sample, _unnormalize_torch
from .fk import PANDA_JOINT_LIMITS


def _expand_scale(scale: torch.Tensor | float, mean: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(scale, device=mean.device, dtype=mean.dtype)
    if value.ndim == 0:
        return value.expand_as(mean)
    if value.ndim == 1 and value.shape[0] == mean.shape[0]:
        return value[:, None].expand_as(mean)
    if value.shape == mean.shape:
        return value
    raise ValueError(
        "noise scale must be scalar, [H], or [H,D], got "
        f"{tuple(value.shape)} for mean {tuple(mean.shape)}"
    )


def _sample_truncated_normal(
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample independent truncated normals using inverse-CDF sampling."""
    if torch.any(lower > upper):
        raise ValueError("Truncated-normal lower bounds must not exceed upper bounds.")
    if torch.any(std < 0.0):
        raise ValueError("Truncated-normal standard deviations must be non-negative.")

    # Float64 preserves tail probability that would vanish in float32. When a
    # conditional interval is still numerically degenerate, use the closest
    # feasible point; this is also the correct safe behavior for zero variance.
    work_mean = mean.to(torch.float64)
    work_std = std.to(torch.float64)
    work_lower = lower.to(torch.float64)
    work_upper = upper.to(torch.float64)
    safe_std = torch.clamp(work_std, min=torch.finfo(torch.float64).tiny)
    inv_sqrt_two = 1.0 / np.sqrt(2.0)
    cdf_lower = 0.5 * (
        1.0 + torch.erf((work_lower - work_mean) / safe_std * inv_sqrt_two)
    )
    cdf_upper = 0.5 * (
        1.0 + torch.erf((work_upper - work_mean) / safe_std * inv_sqrt_two)
    )
    cdf_span = cdf_upper - cdf_lower

    uniform = torch.rand(
        mean.shape,
        device=mean.device,
        dtype=torch.float64,
        generator=generator,
    )
    probability = cdf_lower + uniform * cdf_span
    probability = torch.clamp(
        probability,
        min=torch.finfo(torch.float64).eps,
        max=1.0 - torch.finfo(torch.float64).eps,
    )
    sampled = (
        work_mean
        + safe_std * np.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    )
    fallback = torch.clamp(work_mean, min=work_lower, max=work_upper)
    usable = (work_std > 0.0) & (cdf_span > torch.finfo(torch.float64).eps)
    sampled = torch.where(usable, sampled, fallback)
    return torch.clamp(sampled, min=work_lower, max=work_upper).to(mean.dtype)


def sample_truncated_model_action_chunks(
    policy: Any,
    policy_inputs: dict[str, Any],
    mean: torch.Tensor,
    noise_scale: torch.Tensor | float,
    num_samples: int,
    *,
    current_joint_pos: torch.Tensor | np.ndarray,
    max_joint_delta: float,
    joint_limit_margin: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively sample model actions with valid decoded joint targets."""
    if mean.ndim != 2:
        raise ValueError(f"Expected proposal mean [H,D], got {tuple(mean.shape)}")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_joint_delta <= 0.0:
        raise ValueError(
            "Truncated action sampling requires a positive max_joint_delta."
        )

    metadata = getattr(policy, "_metadata", {}) or {}
    output_norm_stats = metadata.get("output_norm_stats")
    if (
        not output_norm_stats
        or "actions" not in output_norm_stats
        or "state" not in output_norm_stats
    ):
        raise ValueError(
            "Truncated action sampling requires action and state normalization metadata."
        )

    use_quantile_norm = bool(metadata.get("use_quantile_norm", False))
    horizon, action_dims = mean.shape
    scale = _expand_scale(noise_scale, mean)
    if torch.any(scale < 0.0):
        raise ValueError("noise scale must be non-negative")

    expanded_mean = mean.unsqueeze(0).expand(num_samples, -1, -1)
    expanded_scale = scale.unsqueeze(0).expand_as(expanded_mean)
    samples = expanded_mean + torch.randn(
        expanded_mean.shape,
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    ) * expanded_scale

    model_zero = torch.zeros(action_dims, device=mean.device, dtype=mean.dtype)
    model_one = torch.ones_like(model_zero)
    action_zero = _unnormalize_torch(
        model_zero,
        output_norm_stats["actions"],
        use_quantile_norm=use_quantile_norm,
    )
    action_one = _unnormalize_torch(
        model_one,
        output_norm_stats["actions"],
        use_quantile_norm=use_quantile_norm,
    )
    state = _unnormalize_torch(
        _state_for_sample(policy_inputs).to(device=mean.device, dtype=mean.dtype),
        output_norm_stats["state"],
        use_quantile_norm=use_quantile_norm,
    )
    if action_zero is None or action_one is None or state is None:
        raise ValueError(
            "Truncated action sampling could not apply policy normalization."
        )

    affine_scale = action_one - action_zero
    joint_dims = min(7, action_dims, state.shape[-1])
    if joint_dims > 0 and torch.any(
        affine_scale[:joint_dims].abs() <= torch.finfo(mean.dtype).eps
    ):
        raise ValueError(
            "Truncated action sampling requires non-degenerate action scales."
        )

    current = torch.as_tensor(
        current_joint_pos, device=mean.device, dtype=mean.dtype
    )
    if current.ndim > 1:
        current = current.reshape(-1, current.shape[-1])[0]
    current = current.flatten()[:joint_dims]
    if current.shape[0] != joint_dims:
        raise ValueError(
            f"Expected at least {joint_dims} current joints, got {current.shape[0]}."
        )

    previous = current.unsqueeze(0).expand(num_samples, -1).clone()
    limits = torch.as_tensor(
        PANDA_JOINT_LIMITS, device=mean.device, dtype=mean.dtype
    )
    joint_lower = limits[:joint_dims, 0] + float(joint_limit_margin)
    joint_upper = limits[:joint_dims, 1] - float(joint_limit_margin)
    decoded_offset = action_zero[:joint_dims] + state[:joint_dims]
    decoded_scale = affine_scale[:joint_dims]

    for step in range(horizon):
        real_lower = torch.maximum(
            joint_lower, previous - float(max_joint_delta)
        )
        real_upper = torch.minimum(
            joint_upper, previous + float(max_joint_delta)
        )
        model_a = (real_lower - decoded_offset) / decoded_scale
        model_b = (real_upper - decoded_offset) / decoded_scale
        model_lower = torch.minimum(model_a, model_b)
        model_upper = torch.maximum(model_a, model_b)
        step_sample = _sample_truncated_normal(
            expanded_mean[:, step, :joint_dims],
            expanded_scale[:, step, :joint_dims],
            model_lower,
            model_upper,
            generator=generator,
        )
        samples[:, step, :joint_dims] = step_sample
        previous = decoded_offset + decoded_scale * step_sample

    if action_dims > 7:
        gripper_scale = affine_scale[7]
        if gripper_scale.abs() <= torch.finfo(mean.dtype).eps:
            raise ValueError(
                "Truncated action sampling requires a non-degenerate gripper scale."
            )
        gripper_a = -action_zero[7] / gripper_scale
        gripper_b = (1.0 - action_zero[7]) / gripper_scale
        gripper_lower = torch.minimum(gripper_a, gripper_b)
        gripper_upper = torch.maximum(gripper_a, gripper_b)
        samples[:, :, 7] = _sample_truncated_normal(
            expanded_mean[:, :, 7],
            expanded_scale[:, :, 7],
            gripper_lower.expand(num_samples, horizon),
            gripper_upper.expand(num_samples, horizon),
            generator=generator,
        )

    return samples
