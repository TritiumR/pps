"""Core Iterative Denoising Energy Matching utilities.

This module deliberately contains no policy or simulator dependencies. The
training script supplies energy values and this module implements a Tweedie
Monte-Carlo denoiser plus the rectified-flow change of variables used by PPS.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch


Tensor = torch.Tensor
ProposalFn = Callable[[Tensor, Tensor, int, torch.Generator | None], Tensor]


@dataclass(frozen=True)
class ProcedureBlocks:
    """Token slices for a 16-action, 5-waypoint, 1-keypose procedure."""

    actions: slice = field(default_factory=lambda: slice(0, 16))
    waypoints: slice = field(default_factory=lambda: slice(16, 21))
    keypose: slice = field(default_factory=lambda: slice(21, 22))


def waypoint_smoothness_energy(
    procedure: Tensor,
    blocks: ProcedureBlocks = ProcedureBlocks(),
) -> Tensor:
    """Waypoint path energy with a stop-gradient terminal keypose boundary.

    The detach is intentional: the terminal keypose is optimized only by its
    semantic energy, never by waypoint smoothness.
    """
    waypoints = procedure[..., blocks.waypoints, :]
    keypose_boundary = procedure[..., blocks.keypose, :].detach()
    path = torch.cat((waypoints, keypose_boundary), dim=-2)
    velocities = path[..., 1:, :] - path[..., :-1, :]
    if velocities.shape[-2] < 2:
        return torch.zeros(path.shape[:-2], device=path.device, dtype=path.dtype)
    acceleration = velocities[..., 1:, :] - velocities[..., :-1, :]
    return acceleration.square().mean(dim=(-2, -1))


def procedure_reference_energies(
    procedure: Tensor,
    reference: Tensor,
    *,
    blocks: ProcedureBlocks = ProcedureBlocks(),
    action_smoothness_weight: float = 0.05,
    waypoint_smoothness_weight: float = 0.10,
) -> dict[str, Tensor]:
    """Differentiable block energies for VLM-segmented procedure targets.

    Each returned tensor has the proposal-leading shape.  Crucially, the
    keypose term is terminal reference error only.
    """
    if procedure.shape != reference.shape:
        reference = torch.broadcast_to(reference, procedure.shape)
    action = procedure[..., blocks.actions, :]
    action_ref = reference[..., blocks.actions, :]
    action_energy = (action - action_ref).square().mean(dim=(-2, -1))
    if action.shape[-2] > 1:
        action_energy = action_energy + float(action_smoothness_weight) * (
            action[..., 1:, :] - action[..., :-1, :]
        ).square().mean(dim=(-2, -1))

    waypoint = procedure[..., blocks.waypoints, :]
    waypoint_ref = reference[..., blocks.waypoints, :]
    waypoint_energy = (waypoint - waypoint_ref).square().mean(dim=(-2, -1))
    waypoint_energy = waypoint_energy + float(waypoint_smoothness_weight) * (
        waypoint_smoothness_energy(procedure, blocks)
    )

    keypose = procedure[..., blocks.keypose, :]
    keypose_ref = reference[..., blocks.keypose, :]
    keypose_energy = (keypose - keypose_ref).square().mean(dim=(-2, -1))
    return {
        "actions": action_energy,
        "waypoints": waypoint_energy,
        "keypose": keypose_energy,
    }


def _sample_truncated_normal(
    center: Tensor,
    scale: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample independent truncated normals with a stable projected fallback."""
    if torch.any(lower > upper):
        raise ValueError("truncated-normal lower bound exceeds upper bound")
    work_center = center.to(torch.float64)
    work_scale = scale.to(torch.float64)
    work_lower = lower.to(torch.float64)
    work_upper = upper.to(torch.float64)
    safe_scale = work_scale.clamp_min(torch.finfo(torch.float64).tiny)
    inv_sqrt_two = 2.0 ** -0.5
    lower_cdf = 0.5 * (
        1.0 + torch.erf((work_lower - work_center) / safe_scale * inv_sqrt_two)
    )
    upper_cdf = 0.5 * (
        1.0 + torch.erf((work_upper - work_center) / safe_scale * inv_sqrt_two)
    )
    span = upper_cdf - lower_cdf
    uniform = torch.rand(
        center.shape,
        device=center.device,
        dtype=torch.float64,
        generator=generator,
    )
    probability = lower_cdf + uniform * span
    epsilon = torch.finfo(torch.float64).eps
    probability = probability.clamp(epsilon, 1.0 - epsilon)
    sampled = work_center + safe_scale * (2.0 ** 0.5) * torch.erfinv(
        2.0 * probability - 1.0
    )
    projected = work_center.clamp(min=work_lower, max=work_upper)
    usable = (work_scale > 0.0) & (span > epsilon) & torch.isfinite(sampled)
    sampled = torch.where(usable, sampled, projected)
    return sampled.clamp(min=work_lower, max=work_upper).to(center.dtype)


def sample_autoregressive_truncated_action_cloud(
    center: Tensor,
    scale: Tensor,
    proposals: int,
    generator: torch.Generator | None,
    *,
    action_mean: Tensor,
    action_std: Tensor,
    action_lower: Tensor,
    action_upper: Tensor,
    initial_reference: Tensor,
    first_delta_limit: float,
    step_delta_limit: float,
) -> Tensor:
    """Sample normalized action chunks under physical control-delta limits.

    The first action is constrained around the current physical robot state;
    every later action is constrained around the preceding sampled control.
    All intervals are also intersected with the demonstrated control bounds.
    """
    if center.ndim != 3:
        raise ValueError("action proposal center must have shape [B,H,D]")
    if proposals < 1:
        raise ValueError("proposals must be positive")
    if first_delta_limit <= 0.0 or step_delta_limit <= 0.0:
        raise ValueError("control-delta limits must be positive")
    batch, horizon, action_dim = center.shape
    vectors = {
        "action_mean": action_mean,
        "action_std": action_std,
        "action_lower": action_lower,
        "action_upper": action_upper,
    }
    for name, value in vectors.items():
        if value.shape != (action_dim,):
            raise ValueError(f"{name} must have shape {(action_dim,)}")
    if initial_reference.shape != (batch, action_dim):
        raise ValueError(
            "initial_reference must have shape "
            f"{(batch, action_dim)}, got {tuple(initial_reference.shape)}"
        )
    if torch.any(action_std <= 0.0):
        raise ValueError("action_std must be positive")
    if torch.any(action_lower > action_upper):
        raise ValueError("action lower bound exceeds upper bound")

    proposal_scale = torch.as_tensor(
        scale, device=center.device, dtype=center.dtype
    )
    if proposal_scale.ndim == 0:
        proposal_scale = proposal_scale.expand(batch, 1, 1)
    elif proposal_scale.shape == (batch, 1, 1):
        pass
    else:
        raise ValueError(
            "action proposal scale must be scalar or [B,1,1], got "
            f"{tuple(proposal_scale.shape)}"
        )
    proposal_scale = proposal_scale[:, None].expand(
        batch, proposals, 1, action_dim
    )
    samples = torch.empty(
        (batch, proposals, horizon, action_dim),
        device=center.device,
        dtype=center.dtype,
    )
    previous = initial_reference[:, None].expand(-1, proposals, -1).clone()
    mean = action_mean[None, None]
    std = action_std[None, None]
    physical_lower = action_lower[None, None]
    physical_upper = action_upper[None, None]
    for step in range(horizon):
        delta_limit = first_delta_limit if step == 0 else step_delta_limit
        lower = torch.maximum(physical_lower, previous - float(delta_limit))
        upper = torch.minimum(physical_upper, previous + float(delta_limit))
        infeasible = lower > upper
        if torch.any(infeasible):
            nearest = previous.clamp(min=physical_lower, max=physical_upper)
            lower = torch.where(infeasible, nearest, lower)
            upper = torch.where(infeasible, nearest, upper)
        normalized_lower = (lower - mean) / std
        normalized_upper = (upper - mean) / std
        step_center = center[:, None, step, :].expand(-1, proposals, -1)
        step_sample = _sample_truncated_normal(
            step_center,
            proposal_scale[:, :, 0, :],
            normalized_lower,
            normalized_upper,
            generator=generator,
        )
        samples[:, :, step, :] = step_sample
        previous = step_sample * std + mean
    return samples


def local_tweedie_score(
    x_t: Tensor,
    time: Tensor,
    energy_fn: Callable[[Tensor], Tensor],
    *,
    proposals: int,
    generator: torch.Generator | None = None,
    proposal_fn: ProposalFn | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate the rectified marginal score using Tweedie local proposals.

    For ``x_t=(1-t)x_0+t*eps``, set ``y=x_t/(1-t)`` and
    ``sigma=t/(1-t)`` and sample ``x_0^i ~ N(y,sigma^2 I)``. Importance
    weights ``softmax(-E_i)`` yield a clean denoiser, from which Tweedie's
    identity gives the score. No energy gradients are evaluated.
    """
    if proposals < 1:
        raise ValueError("proposals must be positive")
    if time.ndim != 1 or time.shape[0] != x_t.shape[0]:
        raise ValueError("time must have shape [batch]")
    expand = (slice(None),) + (None,) * (x_t.ndim - 1)
    t = time[expand]
    one_minus_t = 1.0 - t
    center = x_t / one_minus_t
    sigma = t / one_minus_t
    if proposal_fn is None:
        noise = torch.randn(
            (x_t.shape[0], proposals, *x_t.shape[1:]),
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        samples = center[:, None] + sigma[:, None] * noise
    else:
        samples = proposal_fn(center, sigma, proposals, generator)
        expected = (x_t.shape[0], proposals, *x_t.shape[1:])
        if samples.shape != expected:
            raise ValueError(
                f"proposal_fn must return {expected}, got {tuple(samples.shape)}"
            )
    energies = energy_fn(samples)
    if energies.shape != (x_t.shape[0], proposals):
        raise ValueError(
            f"energy_fn must return [batch, proposals], got {tuple(energies.shape)}"
        )
    weights = torch.softmax(-energies.detach(), dim=1)
    weight_shape = (*weights.shape, *((1,) * (x_t.ndim - 1)))
    denoised = (weights.reshape(weight_shape) * samples).sum(dim=1)
    score_x = (one_minus_t * denoised - x_t) / t.square()
    return score_x.detach(), energies.detach()


def score_to_rectified_velocity(x_t: Tensor, score_x: Tensor, time: Tensor) -> Tensor:
    """Convert a rectified-path marginal score into policy velocity."""
    expand = (slice(None),) + (None,) * (x_t.ndim - 1)
    t = time[expand]
    return -(x_t + t * score_x) / (1.0 - t)


def blockwise_idem_velocity_target(
    x_t: Tensor,
    time: Tensor,
    energy_fns: Mapping[str, Callable[[Tensor], Tensor]],
    *,
    proposals: int,
    blocks: ProcedureBlocks = ProcedureBlocks(),
    generator: torch.Generator | None = None,
    proposal_fns: Mapping[str, ProposalFn] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build independent iDEM targets for action, waypoint and keypose blocks."""
    target_score = torch.zeros_like(x_t)
    diagnostics: dict[str, Tensor] = {}
    slices = {
        "actions": blocks.actions,
        "waypoints": blocks.waypoints,
        "keypose": blocks.keypose,
    }
    for name, token_slice in slices.items():
        base = x_t.detach()

        def block_energy(block_samples: Tensor, *, _slice=token_slice, _name=name):
            expanded = base[:, None].expand(-1, proposals, -1, -1).clone()
            expanded[..., _slice, :] = block_samples
            return energy_fns[_name](expanded)

        score, energy = local_tweedie_score(
            x_t[..., token_slice, :],
            time,
            block_energy,
            proposals=proposals,
            generator=generator,
            proposal_fn=(
                None if proposal_fns is None else proposal_fns.get(name)
            ),
        )
        target_score[..., token_slice, :] = score
        diagnostics[name] = energy.mean()
    return score_to_rectified_velocity(x_t, target_score, time).detach(), diagnostics
