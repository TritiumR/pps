"""Generic rectified-flow MBD guidance with externally supplied costs.

This module deliberately knows nothing about a robot, simulator, or task.  A
caller supplies normalized token bounds and a batched clean-trajectory cost
callback.  That keeps policy/MBD mechanics in PPS while allowing downstream
applications to own FK, collision, phase, and scene-state semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import torch

from .tempered_smc import TemperedSMCConfig, adaptive_tempered_smc


@dataclass(frozen=True)
class ProposalCostResult:
    """Proposal costs plus an optional hard eligibility mask."""

    costs: np.ndarray | torch.Tensor
    eligible: np.ndarray | torch.Tensor | None = None


CleanTrajectoryCost = Callable[
    [torch.Tensor], np.ndarray | torch.Tensor | ProposalCostResult
]


@dataclass(frozen=True)
class TokenBlockLayout:
    """Disjoint action, waypoint, and terminal-keypose token blocks."""

    trajectory_start: int
    keypose_index: int

    def validate(self, horizon: int) -> None:
        if not 0 <= int(self.trajectory_start) <= int(self.keypose_index) < int(horizon):
            raise ValueError(
                "token blocks require 0 <= trajectory_start <= keypose_index < horizon"
            )

    @property
    def action(self) -> slice:
        return slice(0, int(self.trajectory_start))

    @property
    def trajectory(self) -> slice:
        return slice(int(self.trajectory_start), int(self.keypose_index))

    @property
    def keypose(self) -> slice:
        return slice(int(self.keypose_index), int(self.keypose_index) + 1)

    @property
    def trajectory_with_keypose(self) -> slice:
        return slice(int(self.trajectory_start), int(self.keypose_index) + 1)


@dataclass(frozen=True)
class FlowBlendCoefficients:
    action: float
    trajectory: float
    keypose: float | None = None

    def resolved_keypose(self) -> float:
        return float(self.trajectory if self.keypose is None else self.keypose)


@dataclass(frozen=True)
class RectifiedFlowMBDConfig:
    proposals_per_particle: int = 256
    temperature: float = 0.01
    proposal_std: float = 0.40
    proposal_sampler: str = "clipped_gaussian"
    inference_sampler: str = "direct"
    smc_final_beta: float | None = None
    smc_target_ess_fraction: float = 0.5
    smc_resample_ess_fraction: float = 0.5
    smc_resampling_method: str = "systematic"
    smc_mala_steps: int = 0
    smc_mala_step_size: float = 0.001
    smc_mala_schedule: str = "every_resample"
    smc_beta_tolerance: float = 1e-4
    smc_max_tempering_stages: int = 64

    def validate(self) -> None:
        if int(self.proposals_per_particle) < 1:
            raise ValueError("proposals_per_particle must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(self.proposal_std) <= 0.0:
            raise ValueError("proposal_std must be positive")
        if self.proposal_sampler not in ("clipped_gaussian", "truncated_gaussian"):
            raise ValueError(
                "proposal_sampler must be 'clipped_gaussian' or 'truncated_gaussian'"
            )
        if self.inference_sampler not in ("direct", "adaptive_smc"):
            raise ValueError("inference_sampler must be direct or adaptive_smc")
        final_beta = (
            1.0 / float(self.temperature)
            if self.smc_final_beta is None
            else float(self.smc_final_beta)
        )
        TemperedSMCConfig(
            final_beta=final_beta,
            target_ess_fraction=float(self.smc_target_ess_fraction),
            resample_ess_fraction=float(self.smc_resample_ess_fraction),
            resampling_method=str(self.smc_resampling_method),
            mala_steps=int(self.smc_mala_steps),
            mala_step_size=float(self.smc_mala_step_size),
            mala_schedule=str(self.smc_mala_schedule),
            beta_tolerance=float(self.smc_beta_tolerance),
            max_tempering_stages=int(self.smc_max_tempering_stages),
        ).validate()
        if (
            self.inference_sampler == "adaptive_smc"
            and self.smc_mala_steps > 0
            and self.proposal_sampler != "truncated_gaussian"
        ):
            raise ValueError(
                "MALA requires continuous truncated_gaussian proposals; "
                "clipped_gaussian has boundary point masses"
            )


@dataclass(frozen=True)
class ProposalResult:
    candidates: torch.Tensor
    costs: np.ndarray
    weights: torch.Tensor
    reward_logits: np.ndarray
    mean: torch.Tensor
    effective_sample_size: np.ndarray
    proposal_scale: float
    diagnostics: dict[str, Any] | None = None

    @property
    def log_mean_reward(self) -> np.ndarray:
        return logmeanexp(self.reward_logits, axis=1)


@dataclass(frozen=True)
class ScoreGuidanceResult:
    guided_flow: torch.Tensor
    policy_score: torch.Tensor
    mbd_score: torch.Tensor
    guided_score: torch.Tensor
    proposals: ProposalResult


@dataclass(frozen=True)
class SequentialScoreGuidanceResult:
    """Keypose guidance followed by waypoint guidance conditioned on it."""

    guided_flow: torch.Tensor
    fixed_keypose: torch.Tensor
    waypoint_conditioning_keypose: torch.Tensor
    keypose: ScoreGuidanceResult
    waypoints: ScoreGuidanceResult | None


@dataclass(frozen=True)
class QuadraticSmoothnessGaussianProposal:
    """Conjugate Gaussian proposal for anchored Laplacian smoothness.

    The base clean-space Gaussian is multiplied exactly by
    ``exp(-weight * mean(||D2 path||^2) / temperature)``. ``initial`` and
    ``terminal`` are fixed endpoints in normalized action coordinates, while
    ``physical_scale`` maps normalized coordinates to physical joints.
    """

    initial: torch.Tensor
    terminal: torch.Tensor
    physical_scale: torch.Tensor
    weight: float
    residual_bounds: torch.Tensor | None = None
    constrained_dims: int | None = None


def logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.mean(np.exp(values - maximum), axis=axis)
    )


def normalized_weights(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(logits), 1.0 / len(logits), dtype=np.float64)
    return weights / total


def bounded_gaussian_log_prob(
    samples: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    sampler: str,
) -> torch.Tensor:
    """Log density/mass of the bounded proposal, reduced over trajectory axes."""
    if samples.ndim != 4 or center.ndim != 3:
        raise ValueError("samples/center must have shapes [P,N,W,D]/[P,W,D]")
    if samples.shape[0] != center.shape[0] or samples.shape[2:] != center.shape[1:]:
        raise ValueError("samples and center have incompatible trajectory shapes")
    if float(scale) <= 0.0:
        raise ValueError("bounded Gaussian scale must be positive")

    work_dtype = torch.float64
    value = samples.to(dtype=work_dtype)
    mean = center.to(device=samples.device, dtype=work_dtype)[:, None, :, :]
    lo = lower.to(device=samples.device, dtype=work_dtype)[None, None, None, :]
    hi = upper.to(device=samples.device, dtype=work_dtype)[None, None, None, :]
    sigma = torch.as_tensor(float(scale), device=samples.device, dtype=work_dtype)
    degenerate = (hi <= lo).expand_as(value)
    standardized = (value - mean) / sigma
    log_pdf = (
        -0.5 * standardized.square()
        - torch.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
    )

    if sampler == "truncated_gaussian":
        z_lower = (lo - mean) / sigma
        z_upper = (hi - mean) / sigma
        log_cdf_lower = torch.special.log_ndtr(z_lower)
        log_cdf_upper = torch.special.log_ndtr(z_upper)
        log_ratio = torch.clamp(
            log_cdf_lower - log_cdf_upper,
            max=-torch.finfo(work_dtype).eps,
        )
        log_normalizer = log_cdf_upper + torch.log1p(-torch.exp(log_ratio))
        coordinate_log_prob = log_pdf - log_normalizer
    elif sampler == "clipped_gaussian":
        z_lower = (lo - mean) / sigma
        z_upper = (hi - mean) / sigma
        lower_log_mass = torch.special.log_ndtr(z_lower)
        upper_log_mass = torch.special.log_ndtr(-z_upper)
        at_lower = value == lo
        at_upper = value == hi
        coordinate_log_prob = torch.where(
            at_lower,
            lower_log_mass,
            torch.where(at_upper, upper_log_mass, log_pdf),
        )
    else:
        raise ValueError(f"Unknown proposal sampler {sampler!r}")

    coordinate_log_prob = torch.where(
        degenerate, torch.zeros_like(coordinate_log_prob), coordinate_log_prob
    )
    return coordinate_log_prob.sum(dim=(-1, -2))


def rectified_flow_score(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    *,
    time_value: float,
) -> torch.Tensor:
    """Convert velocity to score for ``x_t=(1-t)x_0+t*epsilon``."""
    time = float(time_value)
    if not 0.0 < time <= 1.0:
        raise ValueError(f"rectified-flow score requires t in (0, 1], got {time}")
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have identical shapes")
    return -(x_t + (1.0 - time) * velocity) / time


def rectified_flow_from_score(
    x_t: torch.Tensor,
    score: torch.Tensor,
    *,
    time_value: float,
) -> torch.Tensor:
    """Convert score to velocity under the linear flow-matching interpolant."""
    time = float(time_value)
    if not 0.0 < time < 1.0:
        raise ValueError(f"score-to-flow conversion requires t in (0, 1), got {time}")
    if x_t.shape != score.shape:
        raise ValueError("x_t and score must have identical shapes")
    return -(x_t + time * score) / (1.0 - time)


def flow_matching_clean_proposal_scale(
    *,
    time_value: float,
    noise_multiplier: float,
) -> float:
    """Return clean-space proposal noise matching the rectified-flow path."""
    time = float(time_value)
    if not 0.0 < time < 1.0:
        raise ValueError(
            f"matched clean proposal noise requires t in (0, 1), got {time}"
        )
    if float(noise_multiplier) <= 0.0:
        raise ValueError("noise_multiplier must be positive")
    return float(noise_multiplier) * time / (1.0 - time)


def flow_matching_clean_likelihood_center(
    x_t: torch.Tensor,
    *,
    time_value: float,
) -> torch.Tensor:
    """Return the clean-space FM likelihood center for a fixed ``x_t``.

    Under this repository's interpolation convention
    ``x_t=(1-t)y+t*epsilon`` with ``epsilon ~ N(0, I)``, the likelihood as a
    function of the clean endpoint is
    ``q_t(y|x_t) = N(x_t/(1-t), (t/(1-t))^2 I)`` up to normalization.
    This is a target-distribution quantity, not a learned-policy prediction.
    """
    time = float(time_value)
    if not 0.0 < time < 1.0:
        raise ValueError(
            f"FM clean likelihood center requires t in (0, 1), got {time}"
        )
    return x_t / (1.0 - time)


def sample_independent_truncated_gaussian(
    center: torch.Tensor,
    *,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample an exact axis-aligned truncated Gaussian with inverse CDF."""
    if center.ndim != 2:
        raise ValueError(f"expected [particles, action_dim], got {center.shape}")
    if float(scale) <= 0.0:
        raise ValueError("truncated Gaussian scale must be positive")
    work_dtype = torch.float64
    center64 = center.to(dtype=work_dtype)
    lower64 = lower.to(device=center.device, dtype=work_dtype)
    upper64 = upper.to(device=center.device, dtype=work_dtype)
    scale64 = torch.as_tensor(float(scale), device=center.device, dtype=work_dtype)
    sqrt_two = math.sqrt(2.0)
    standardized_lower = (lower64[None, :] - center64) / scale64
    standardized_upper = (upper64[None, :] - center64) / scale64
    lower_cdf = 0.5 * (1.0 + torch.erf(standardized_lower / sqrt_two))
    upper_cdf = 0.5 * (1.0 + torch.erf(standardized_upper / sqrt_two))
    if torch.any(upper_cdf <= lower_cdf):
        raise RuntimeError("truncated Gaussian has an empty CDF interval")
    uniform = torch.rand(
        (int(center.shape[0]), int(num_samples), int(center.shape[1])),
        device=center.device,
        dtype=work_dtype,
        generator=generator,
    )
    epsilon = torch.finfo(work_dtype).eps
    uniform = epsilon + (1.0 - 2.0 * epsilon) * uniform
    quantiles = lower_cdf[:, None, :] + uniform * (
        upper_cdf - lower_cdf
    )[:, None, :]
    quantiles = torch.clamp(quantiles, epsilon, 1.0 - epsilon)
    standard_normal = sqrt_two * torch.erfinv(2.0 * quantiles - 1.0)
    samples = center64[:, None, :] + scale64 * standard_normal
    if not bool(torch.all(torch.isfinite(samples)).item()):
        raise RuntimeError("truncated Gaussian produced non-finite samples")
    samples = torch.maximum(
        torch.minimum(samples, upper64[None, None, :]),
        lower64[None, None, :],
    )
    return samples.to(dtype=center.dtype)


def sample_trajectory_proposals(
    center: torch.Tensor,
    *,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    sampler: str,
    generator: torch.Generator,
    include_center: bool = True,
) -> torch.Tensor:
    """Sample ``[particle, proposal, pose, joint]`` clean trajectories."""
    if center.ndim != 3:
        raise ValueError("trajectory center must have shape [P,W,D]")
    particles, poses, action_dim = center.shape
    if sampler == "truncated_gaussian":
        flattened = sample_independent_truncated_gaussian(
            center.reshape(particles * poses, action_dim),
            scale=scale,
            lower=lower,
            upper=upper,
            num_samples=num_samples,
            generator=generator,
        )
        candidates = flattened.reshape(
            particles, poses, num_samples, action_dim
        ).permute(0, 2, 1, 3)
    elif sampler == "clipped_gaussian":
        candidates = center[:, None, :, :] + float(scale) * torch.randn(
            (particles, num_samples, poses, action_dim),
            device=center.device,
            dtype=center.dtype,
            generator=generator,
        )
        candidates = torch.maximum(
            torch.minimum(candidates, upper[None, None, None, :]),
            lower[None, None, None, :],
        )
    else:
        raise ValueError(f"Unknown proposal sampler {sampler!r}")
    if include_center:
        candidates[:, 0, :, :] = center
    return candidates


def _sample_dynamic_truncated_gaussian(
    center: torch.Tensor,
    *,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a Gaussian with per-candidate axis-aligned truncation bounds."""
    if not (center.shape == lower.shape == upper.shape):
        raise ValueError("dynamic truncated Gaussian tensors must match")
    if float(scale) <= 0.0 or bool(torch.any(upper <= lower).item()):
        raise ValueError("dynamic truncated Gaussian has invalid scale/bounds")
    dtype = torch.float64
    mean = center.to(dtype=dtype)
    lo = lower.to(device=center.device, dtype=dtype)
    hi = upper.to(device=center.device, dtype=dtype)
    sigma = torch.as_tensor(float(scale), device=center.device, dtype=dtype)
    sqrt_two = math.sqrt(2.0)
    cdf_lo = 0.5 * (1.0 + torch.erf((lo - mean) / sigma / sqrt_two))
    cdf_hi = 0.5 * (1.0 + torch.erf((hi - mean) / sigma / sqrt_two))
    if bool(torch.any(cdf_hi <= cdf_lo).item()):
        raise RuntimeError("dynamic truncated Gaussian has an empty CDF interval")
    eps = torch.finfo(dtype).eps
    uniform = torch.rand(
        center.shape,
        device=center.device,
        dtype=dtype,
        generator=generator,
    ).clamp(eps, 1.0 - eps)
    quantile = (cdf_lo + uniform * (cdf_hi - cdf_lo)).clamp(
        eps, 1.0 - eps
    )
    result = mean + sigma * sqrt_two * torch.erfinv(2.0 * quantile - 1.0)
    result = torch.maximum(torch.minimum(result, hi), lo)
    if not bool(torch.all(torch.isfinite(result)).item()):
        raise RuntimeError("dynamic truncated Gaussian produced non-finite values")
    return result.to(dtype=center.dtype)


def sample_autoregressive_waypoint_proposals(
    policy_center: torch.Tensor,
    *,
    initial: torch.Tensor,
    terminal: torch.Tensor,
    residual_bounds: torch.Tensor,
    autoregressive_dims: int,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample arm waypoints sequentially inside a fixed-keypose p99 tube.

    The first autoregressive_dims coordinates use the conditional mean

      previous + (terminal - previous) / remaining_segments.

    Remaining coordinates retain the policy-centered independent truncated
    Gaussian because the measured p99 bounds cover only the arm joints.
    """
    if policy_center.ndim != 3:
        raise ValueError("policy_center must have shape [P,W,D]")
    particles, waypoints, action_dim = policy_center.shape
    dims = int(autoregressive_dims)
    if not 1 <= dims <= action_dim:
        raise ValueError("autoregressive_dims is outside the action dimension")
    if initial.shape != (particles, action_dim):
        raise ValueError("initial must have shape [P,D]")
    if terminal.shape == (particles, 1, action_dim):
        terminal = terminal[:, 0, :]
    if terminal.shape != (particles, action_dim):
        raise ValueError("terminal must have shape [P,D] or [P,1,D]")
    bounds = residual_bounds.to(
        device=policy_center.device, dtype=policy_center.dtype
    ).reshape(-1)
    if bounds.shape != (dims,) or bool(torch.any(bounds <= 0.0).item()):
        raise ValueError("residual_bounds must be positive and match AR dims")

    candidates = sample_trajectory_proposals(
        policy_center,
        scale=float(scale),
        lower=lower,
        upper=upper,
        num_samples=int(num_samples),
        sampler="truncated_gaussian",
        generator=generator,
        include_center=False,
    )
    previous = initial[:, None, :dims].expand(-1, int(num_samples), -1)
    goal = terminal[:, None, :dims].expand(-1, int(num_samples), -1)
    global_lower = lower[:dims].reshape(1, 1, dims)
    global_upper = upper[:dims].reshape(1, 1, dims)
    radius = bounds.reshape(1, 1, dims)
    for waypoint_index in range(waypoints):
        remaining = float(waypoints + 1 - waypoint_index)
        conditional_mean = previous + (goal - previous) / remaining
        local_lower = torch.maximum(
            global_lower, conditional_mean - radius
        )
        local_upper = torch.minimum(
            global_upper, conditional_mean + radius
        )
        sampled = _sample_dynamic_truncated_gaussian(
            conditional_mean,
            scale=float(scale),
            lower=local_lower,
            upper=local_upper,
            generator=generator,
        )
        candidates[:, :, waypoint_index, :dims] = sampled
        previous = sampled
    return candidates


def sample_quadratic_smoothness_gaussian_proposals(
    center: torch.Tensor,
    *,
    initial: torch.Tensor,
    terminal: torch.Tensor,
    physical_scale: torch.Tensor,
    smoothness_weight: float,
    temperature: float,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    residual_bounds: torch.Tensor | None = None,
    constrained_dims: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample ``q * exp(-quadratic smoothness / tau)`` exactly.

    The correlated conjugate Gaussian is sampled through its sequential
    conditionals. Each conditional is truncated by joint limits and the p99
    tube. The returned log density ratio corrects the history-dependent
    truncation normalizers, so the subsequent Tweedie estimate remains IS for
    the hard-supported conjugate target.
    """
    if center.ndim != 3:
        raise ValueError("quadratic smoothness center must have [P,W,D]")
    particles, waypoints, action_dim = center.shape
    if waypoints < 1 or int(num_samples) < 1:
        raise ValueError("quadratic smoothness proposal dimensions are invalid")
    if float(scale) <= 0.0 or float(temperature) <= 0.0:
        raise ValueError("quadratic smoothness scale/temperature must be positive")
    if float(smoothness_weight) <= 0.0:
        raise ValueError("quadratic smoothness weight must be positive")
    initial = initial.to(device=center.device, dtype=center.dtype).reshape(
        particles, action_dim
    )
    terminal = terminal.to(device=center.device, dtype=center.dtype).reshape(
        particles, action_dim
    )
    physical_scale = physical_scale.to(
        device=center.device, dtype=center.dtype
    ).reshape(action_dim)

    # B maps free waypoints to second differences of the anchored path.
    b_matrix = torch.zeros(
        (waypoints, waypoints), device=center.device, dtype=center.dtype
    )
    index = torch.arange(waypoints, device=center.device)
    b_matrix[index, index] = -2.0
    if waypoints > 1:
        neighbor = torch.arange(waypoints - 1, device=center.device)
        b_matrix[neighbor, neighbor + 1] = 1.0
        b_matrix[neighbor + 1, neighbor] = 1.0
    anchor = torch.zeros(
        (particles, waypoints, action_dim),
        device=center.device,
        dtype=center.dtype,
    )
    anchor[:, 0, :] = initial
    anchor[:, -1, :] += terminal

    inverse_variance = 1.0 / float(scale) ** 2
    gamma = (
        float(smoothness_weight)
        * physical_scale.square()
        / float(waypoints * action_dim)
    )
    curvature = 2.0 * gamma / float(temperature)
    btb = torch.matmul(b_matrix.T, b_matrix)
    identity = torch.eye(
        waypoints, device=center.device, dtype=center.dtype
    )
    precision = (
        inverse_variance * identity[None, :, :]
        + curvature[:, None, None] * btb[None, :, :]
    )
    rhs = inverse_variance * center.permute(0, 2, 1)
    rhs -= curvature[None, :, None] * torch.einsum(
        "ij,pjd->pdi", b_matrix.T, anchor
    )
    posterior_mean = torch.linalg.solve(
        precision[None, :, :, :], rhs[..., None]
    )[..., 0]
    covariance = torch.linalg.inv(precision)
    candidates = torch.empty(
        (particles, int(num_samples), waypoints, action_dim),
        device=center.device,
        dtype=center.dtype,
    )
    log_density_ratio = torch.zeros(
        (particles, int(num_samples)),
        device=center.device,
        dtype=torch.float64,
    )
    previous = initial[:, None, :].expand(-1, int(num_samples), -1)
    goal = terminal[:, None, :].expand(-1, int(num_samples), -1)
    bounds = None
    dims = action_dim if constrained_dims is None else int(constrained_dims)
    if residual_bounds is not None:
        bounds = residual_bounds.to(
            device=center.device, dtype=center.dtype
        ).reshape(-1)
        if bounds.shape != (dims,) or bool(torch.any(bounds <= 0.0).item()):
            raise ValueError("quadratic smoothness residual bounds are invalid")
    eps = torch.finfo(torch.float64).eps
    sqrt_two = math.sqrt(2.0)
    for waypoint_index in range(waypoints):
        conditional_mean = posterior_mean[:, :, waypoint_index]
        conditional_variance = covariance[:, waypoint_index, waypoint_index]
        if waypoint_index > 0:
            prefix_covariance = covariance[
                :, :waypoint_index, :waypoint_index
            ]
            cross_covariance = covariance[
                :, waypoint_index, :waypoint_index
            ]
            gain = torch.linalg.solve(
                prefix_covariance,
                cross_covariance[..., None],
            )[..., 0]
            prefix_delta = (
                candidates[:, :, :waypoint_index, :].permute(0, 1, 3, 2)
                - posterior_mean[:, :, :waypoint_index][:, None, :, :]
            )
            conditional_mean = conditional_mean[:, None, :] + torch.sum(
                prefix_delta * gain[None, :, :], dim=-1
            )
            conditional_variance = conditional_variance - torch.sum(
                cross_covariance * gain, dim=-1
            )
        else:
            conditional_mean = conditional_mean[:, None, :].expand(
                -1, int(num_samples), -1
            )
        conditional_std = torch.sqrt(
            torch.clamp(conditional_variance, min=torch.finfo(center.dtype).eps)
        )[None, None, :]
        local_lower = lower[None, None, :].expand_as(conditional_mean)
        local_upper = upper[None, None, :].expand_as(conditional_mean)
        if bounds is not None:
            remaining = float(waypoints + 1 - waypoint_index)
            tube_center = previous + (goal - previous) / remaining
            local_lower = local_lower.clone()
            local_upper = local_upper.clone()
            local_lower[..., :dims] = torch.maximum(
                local_lower[..., :dims], tube_center[..., :dims] - bounds
            )
            local_upper[..., :dims] = torch.minimum(
                local_upper[..., :dims], tube_center[..., :dims] + bounds
            )
        mean64 = conditional_mean.to(torch.float64)
        std64 = conditional_std.to(torch.float64)
        lo64 = local_lower.to(torch.float64)
        hi64 = local_upper.to(torch.float64)
        cdf_lo = 0.5 * (1.0 + torch.erf((lo64 - mean64) / std64 / sqrt_two))
        cdf_hi = 0.5 * (1.0 + torch.erf((hi64 - mean64) / std64 / sqrt_two))
        interval = torch.clamp(cdf_hi - cdf_lo, min=eps)
        uniform = torch.rand(
            conditional_mean.shape,
            device=center.device,
            dtype=torch.float64,
            generator=generator,
        ).clamp(eps, 1.0 - eps)
        # Match the direct/keypose sampler's exact-center admission. For a
        # truncated correlated Gaussian, the valid analogue is the sequence
        # of conditional medians. With inactive truncation this is exactly the
        # analytical posterior mean; with hard support it remains central and
        # cannot violate joint limits or the autoregressive p99 tube.
        uniform[:, 0, :] = 0.5
        quantile = torch.clamp(
            cdf_lo + uniform * interval, min=eps, max=1.0 - eps
        )
        sampled = mean64 + std64 * sqrt_two * torch.erfinv(
            2.0 * quantile - 1.0
        )
        sampled = torch.maximum(torch.minimum(sampled, hi64), lo64)
        candidates[:, :, waypoint_index, :] = sampled.to(center.dtype)
        log_density_ratio += torch.sum(torch.log(interval), dim=-1)
        previous = candidates[:, :, waypoint_index, :]
    return candidates, log_density_ratio.to(center.dtype), {
        "conjugate_quadratic_smoothness": True,
        "included_central_trajectory": True,
        "analytical_posterior_mean_min": float(posterior_mean.min().item()),
        "analytical_posterior_mean_max": float(posterior_mean.max().item()),
        "base_variance": float(scale) ** 2,
        "absorbed_variance_mean": float(
            torch.diagonal(covariance, dim1=-2, dim2=-1).mean().item()
        ),
        "absorbed_variance_min": float(
            torch.diagonal(covariance, dim1=-2, dim2=-1).min().item()
        ),
        "absorbed_variance_max": float(
            torch.diagonal(covariance, dim1=-2, dim2=-1).max().item()
        ),
        "mean_absolute_cross_waypoint_covariance": float(
            (covariance - torch.diag_embed(torch.diagonal(
                covariance, dim1=-2, dim2=-1
            ))).abs().mean().item()
        ),
        "sequential_truncation_log_ratio_mean": float(
            log_density_ratio.mean().item()
        ),
        "hard_support": (
            "joint_bounds_and_autoregressive_p99"
            if residual_bounds is not None
            else "joint_bounds"
        ),
    }


def straight_line_cspace_path(
    trajectories: torch.Tensor,
    *,
    substeps: int,
) -> torch.Tensor:
    """Interpolate adjacent configuration-space waypoints without duplicate knots."""
    if trajectories.ndim < 3 or trajectories.shape[-2] < 2:
        raise ValueError("Waypoint trajectories must have at least two poses")
    if int(substeps) < 1:
        raise ValueError("Waypoint collision substeps must be positive")
    alphas = torch.linspace(
        0.0,
        1.0,
        int(substeps) + 1,
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    alpha_shape = (*((1,) * (trajectories.ndim - 1)), int(substeps) + 1, 1)
    starts = trajectories[..., :-1, None, :]
    segments = starts + alphas.reshape(alpha_shape) * (
        trajectories[..., 1:, None, :] - starts
    )
    return torch.cat(
        [segments[..., 0, :, :], segments[..., 1:, 1:, :].flatten(-3, -2)],
        dim=-2,
    )


def memoryless_sde_kl_blocks(
    flow_delta: torch.Tensor,
    *,
    dt: float,
    diffusion: float,
    layout: TokenBlockLayout,
) -> dict[str, torch.Tensor]:
    """Return discrete Girsanov KL split across disjoint token blocks."""
    if flow_delta.ndim != 3:
        raise ValueError("flow_delta must have shape [batch, horizon, action_dim]")
    layout.validate(flow_delta.shape[1])
    if float(diffusion) <= 0.0:
        raise ValueError("memoryless SDE diffusion must be positive")
    factor = abs(float(dt)) / (2.0 * float(diffusion) ** 2)
    squared = torch.square(flow_delta)

    def reduce_block(block: slice) -> torch.Tensor:
        return factor * squared[:, block, :].sum(dim=(1, 2))

    action = reduce_block(layout.action)
    trajectory = reduce_block(layout.trajectory)
    keypose = reduce_block(layout.keypose)
    return {
        "action": action,
        "trajectory": trajectory,
        "keypose": keypose,
        "total": action + trajectory + keypose,
    }


def maximum_task_weight_under_mbd_kl(task_mbd_kl: float, *, limit: float) -> float:
    """Largest task-flow weight whose KL from MBD is at most ``limit``."""
    if float(task_mbd_kl) < 0.0:
        raise ValueError("task/MBD KL must be nonnegative")
    if float(limit) <= 0.0:
        raise ValueError("steered/MBD KL limit must be positive")
    return min(1.0, math.sqrt(float(limit) / max(float(task_mbd_kl), 1e-30)))


def recenter_particle_noise(
    noise: torch.Tensor,
    *,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Shift a particle population without changing its centered residuals."""
    if noise.ndim != 3:
        raise ValueError("noise must have shape [particle, token, dimension]")
    expected_mean_shape = (1, *noise.shape[1:])
    if mean.shape != expected_mean_shape:
        raise ValueError(
            "mean must define one shared particle-population mean; expected "
            f"{expected_mean_shape}, got {tuple(mean.shape)}"
        )
    centered = noise - noise.mean(dim=0, keepdim=True)
    return centered + mean.to(device=noise.device, dtype=noise.dtype)


def blend_policy_mbd_flows(
    policy_flow: torch.Tensor,
    mbd_flow: torch.Tensor,
    *,
    layout: TokenBlockLayout,
    coefficients: FlowBlendCoefficients,
) -> torch.Tensor:
    """Blend policy and MBD flows independently over configured token blocks."""
    if policy_flow.shape != mbd_flow.shape or policy_flow.ndim != 3:
        raise ValueError("policy and MBD flows must share [batch, horizon, dim]")
    layout.validate(policy_flow.shape[1])
    result = policy_flow.clone()
    block_coefficients = (
        (layout.action, float(coefficients.action)),
        (layout.trajectory, float(coefficients.trajectory)),
        (layout.keypose, coefficients.resolved_keypose()),
    )
    for block, coefficient in block_coefficients:
        result[:, block] += coefficient * (
            mbd_flow[:, block] - policy_flow[:, block]
        )
    return result


def _evaluate_cost_on_device(
    cost_fn: CleanTrajectoryCost,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Materialize full costs and an optional hard mask on candidate device."""
    value = cost_fn(candidates)
    eligible_value = None
    if isinstance(value, ProposalCostResult):
        eligible_value = value.eligible
        value = value.costs
    costs = (
        value.detach().to(device=candidates.device, dtype=candidates.dtype)
        if torch.is_tensor(value)
        else torch.as_tensor(value, device=candidates.device, dtype=candidates.dtype)
    )
    eligible = None
    if eligible_value is not None:
        eligible = (
            eligible_value.detach().to(device=candidates.device, dtype=torch.bool)
            if torch.is_tensor(eligible_value)
            else torch.as_tensor(
                eligible_value, device=candidates.device, dtype=torch.bool
            )
        )
    expected = candidates.shape[:2]
    if costs.shape != expected:
        raise ValueError(
            f"cost_fn must return {tuple(expected)}, got {tuple(costs.shape)}"
        )
    if eligible is not None and eligible.shape != expected:
        raise ValueError(
            "eligible mask must match proposal costs; "
            f"expected {tuple(expected)}, got {tuple(eligible.shape)}"
        )
    return costs, eligible


class RectifiedFlowMBD:
    """Proposal and score-guidance engine for a rectified-flow policy."""

    def __init__(self, config: RectifiedFlowMBDConfig):
        config.validate()
        self.config = config

    def optimize_clean_trajectories(
        self,
        center: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
        cost_fn: CleanTrajectoryCost,
        generator: torch.Generator,
        proposal_scale: float,
        proposals_per_particle: int | None = None,
        gradient_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_center: torch.Tensor | None = None,
        target_scale: float | None = None,
        mixture_with_target: bool = False,
        proposal_mixture_center: torch.Tensor | None = None,
        proposal_mixture_centers: tuple[torch.Tensor, ...] | None = None,
        proposal_mixture_counts: tuple[int, ...] | None = None,
        proposal_mixture_labels: tuple[str, ...] | None = None,
        autoregressive_initial: torch.Tensor | None = None,
        autoregressive_terminal: torch.Tensor | None = None,
        autoregressive_residual_bounds: torch.Tensor | None = None,
        autoregressive_dims: int = 7,
        quadratic_smoothness_proposal: (
            QuadraticSmoothnessGaussianProposal | None
        ) = None,
    ) -> ProposalResult:
        """Optimize clean trajectories with direct IS or adaptive SMC/MALA."""
        proposal_count = (
            int(self.config.proposals_per_particle)
            if proposals_per_particle is None
            else int(proposals_per_particle)
        )
        if proposal_count < 1:
            raise ValueError("proposals_per_particle must be positive")
        if target_center is not None and target_center.shape != center.shape:
            raise ValueError("target_center must match the proposal center")
        if target_scale is not None and target_center is None:
            raise ValueError("target_scale requires target_center")
        if target_scale is not None and float(target_scale) <= 0.0:
            raise ValueError("target_scale must be positive")
        if mixture_with_target and target_center is None:
            raise ValueError("mixture_with_target requires target_center")
        if proposal_mixture_center is not None:
            if proposal_mixture_center.shape != center.shape:
                raise ValueError(
                    "proposal_mixture_center must match the proposal center"
                )
            if target_center is None:
                raise ValueError(
                    "proposal_mixture_center requires an importance target"
                )
            if mixture_with_target:
                raise ValueError(
                    "proposal_mixture_center and mixture_with_target are exclusive"
                )
        multi_center_mixture = proposal_mixture_centers is not None
        if multi_center_mixture:
            if target_center is None:
                raise ValueError(
                    "proposal_mixture_centers require an importance target"
                )
            if mixture_with_target or proposal_mixture_center is not None:
                raise ValueError(
                    "multi-center and legacy proposal mixtures are exclusive"
                )
            if proposal_mixture_counts is None:
                raise ValueError(
                    "proposal_mixture_counts are required for a multi-center mixture"
                )
            if len(proposal_mixture_centers) < 1:
                raise ValueError("proposal_mixture_centers cannot be empty")
            if len(proposal_mixture_counts) != len(proposal_mixture_centers):
                raise ValueError(
                    "proposal mixture centers and counts must have equal length"
                )
            if any(int(count) < 1 for count in proposal_mixture_counts):
                raise ValueError("proposal mixture counts must be positive")
            for mixture_center in proposal_mixture_centers:
                if mixture_center.shape != center.shape:
                    raise ValueError(
                        "every proposal mixture center must match the proposal center"
                    )
            if proposal_mixture_labels is not None:
                if len(proposal_mixture_labels) != len(proposal_mixture_centers):
                    raise ValueError(
                        "proposal mixture labels must match the mixture centers"
                    )
                if len(set(proposal_mixture_labels)) != len(
                    proposal_mixture_labels
                ):
                    raise ValueError("proposal mixture labels must be unique")
        elif (
            proposal_mixture_counts is not None
            or proposal_mixture_labels is not None
        ):
            raise ValueError(
                "proposal mixture counts/labels require proposal_mixture_centers"
            )
        autoregressive = any(
            value is not None
            for value in (
                autoregressive_initial,
                autoregressive_terminal,
                autoregressive_residual_bounds,
            )
        )
        if autoregressive and not all(
            value is not None
            for value in (
                autoregressive_initial,
                autoregressive_terminal,
                autoregressive_residual_bounds,
            )
        ):
            raise ValueError("autoregressive waypoint arguments are incomplete")
        if autoregressive and self.config.proposal_sampler != "truncated_gaussian":
            raise ValueError("autoregressive waypoints require truncated_gaussian")
        if autoregressive and (
            target_center is not None
            or mixture_with_target
            or proposal_mixture_center is not None
            or multi_center_mixture
        ):
            raise ValueError("autoregressive waypoints own their proposal center")
        if quadratic_smoothness_proposal is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError(
                    "quadratic smoothness proposals currently require direct IS"
                )
            if autoregressive:
                raise ValueError(
                    "quadratic smoothness proposal owns the autoregressive support"
                )
            if (
                target_center is not None
                or mixture_with_target
                or proposal_mixture_center is not None
                or multi_center_mixture
            ):
                raise ValueError(
                    "quadratic smoothness proposal owns its Gaussian center"
                )
        if self.config.inference_sampler == "adaptive_smc":
            if target_center is not None:
                raise ValueError(
                    "shifted-proposal importance correction is currently "
                    "implemented only for the direct sampler"
                )
            return self._optimize_clean_trajectories_smc(
                center,
                lower=lower,
                upper=upper,
                cost_fn=cost_fn,
                gradient_cost_fn=gradient_cost_fn,
                generator=generator,
                proposal_scale=float(proposal_scale),
                proposal_count=proposal_count,
            )

        started = time.perf_counter()
        proposal_diagnostics: dict[str, Any] = {}
        smoothness_log_density_ratio: torch.Tensor | None = None
        if quadratic_smoothness_proposal is not None:
            candidates, smoothness_log_density_ratio, proposal_diagnostics = (
                sample_quadratic_smoothness_gaussian_proposals(
                    center,
                    initial=quadratic_smoothness_proposal.initial,
                    terminal=quadratic_smoothness_proposal.terminal,
                    physical_scale=quadratic_smoothness_proposal.physical_scale,
                    smoothness_weight=float(
                        quadratic_smoothness_proposal.weight
                    ),
                    temperature=float(self.config.temperature),
                    scale=float(proposal_scale),
                    lower=lower,
                    upper=upper,
                    num_samples=proposal_count,
                    generator=generator,
                    residual_bounds=(
                        quadratic_smoothness_proposal.residual_bounds
                    ),
                    constrained_dims=(
                        quadratic_smoothness_proposal.constrained_dims
                    ),
                )
            )
        elif autoregressive:
            candidates = sample_autoregressive_waypoint_proposals(
                center,
                initial=autoregressive_initial,
                terminal=autoregressive_terminal,
                residual_bounds=autoregressive_residual_bounds,
                autoregressive_dims=int(autoregressive_dims),
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                generator=generator,
            )
        elif multi_center_mixture:
            candidates = torch.cat(
                tuple(
                    sample_trajectory_proposals(
                        mixture_center,
                        scale=float(proposal_scale),
                        lower=lower,
                        upper=upper,
                        num_samples=int(count),
                        sampler=self.config.proposal_sampler,
                        generator=generator,
                        include_center=False,
                    )
                    for mixture_center, count in zip(
                        proposal_mixture_centers,
                        proposal_mixture_counts,
                        strict=True,
                    )
                ),
                dim=1,
            )
        elif mixture_with_target or proposal_mixture_center is not None:
            first_center = (
                target_center
                if proposal_mixture_center is None
                else proposal_mixture_center
            )
            first_candidates = sample_trajectory_proposals(
                first_center,
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                sampler=self.config.proposal_sampler,
                generator=generator,
                include_center=False,
            )
            shifted_candidates = sample_trajectory_proposals(
                center,
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                sampler=self.config.proposal_sampler,
                generator=generator,
                include_center=False,
            )
            candidates = torch.cat(
                (first_candidates, shifted_candidates), dim=1
            )
        else:
            candidates = sample_trajectory_proposals(
                center,
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                sampler=self.config.proposal_sampler,
                generator=generator,
                include_center=target_center is None,
            )
        costs_value = cost_fn(candidates)
        eligible_value = None
        if isinstance(costs_value, ProposalCostResult):
            eligible_value = costs_value.eligible
            costs_value = costs_value.costs
        if torch.is_tensor(costs_value):
            costs = costs_value.detach().cpu().numpy()
        else:
            costs = np.asarray(costs_value)
        expected = candidates.shape[:2]
        if costs.shape != expected:
            raise ValueError(
                f"cost_fn must return {tuple(expected)}, got {tuple(costs.shape)}"
            )
        reward_logits = -costs / float(self.config.temperature)
        log_density_ratio = None
        if smoothness_log_density_ratio is not None:
            log_density_ratio = (
                smoothness_log_density_ratio.detach().cpu().numpy()
            )
            reward_logits = reward_logits + log_density_ratio
        if target_center is not None:
            resolved_target_scale = (
                float(proposal_scale)
                if target_scale is None
                else float(target_scale)
            )
            target_log_prob = bounded_gaussian_log_prob(
                candidates,
                center=target_center,
                scale=resolved_target_scale,
                lower=lower,
                upper=upper,
                sampler=self.config.proposal_sampler,
            )
            proposal_log_prob = bounded_gaussian_log_prob(
                candidates,
                center=center,
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                sampler=self.config.proposal_sampler,
            )
            if multi_center_mixture:
                total_samples = float(sum(proposal_mixture_counts))
                component_log_probs = torch.stack(
                    tuple(
                        bounded_gaussian_log_prob(
                            candidates,
                            center=mixture_center,
                            scale=float(proposal_scale),
                            lower=lower,
                            upper=upper,
                            sampler=self.config.proposal_sampler,
                        )
                        + math.log(float(count) / total_samples)
                        for mixture_center, count in zip(
                            proposal_mixture_centers,
                            proposal_mixture_counts,
                            strict=True,
                        )
                    ),
                    dim=0,
                )
                importance_log_prob = torch.logsumexp(
                    component_log_probs, dim=0
                )
            elif mixture_with_target or proposal_mixture_center is not None:
                first_center = (
                    target_center
                    if proposal_mixture_center is None
                    else proposal_mixture_center
                )
                first_log_prob = bounded_gaussian_log_prob(
                    candidates,
                    center=first_center,
                    scale=float(proposal_scale),
                    lower=lower,
                    upper=upper,
                    sampler=self.config.proposal_sampler,
                )
                importance_log_prob = torch.logaddexp(
                    first_log_prob, proposal_log_prob
                ) - math.log(2.0)
            else:
                importance_log_prob = proposal_log_prob
            log_density_ratio = (
                target_log_prob - importance_log_prob
            ).detach().cpu().numpy()
            reward_logits = reward_logits + log_density_ratio
        if eligible_value is not None:
            eligible = (
                eligible_value.detach().cpu().numpy()
                if torch.is_tensor(eligible_value)
                else np.asarray(eligible_value)
            )
            if eligible.shape != expected:
                raise ValueError(
                    "eligible mask must match proposal costs; "
                    f"expected {tuple(expected)}, got {tuple(eligible.shape)}"
                )
            eligible = eligible.astype(bool, copy=False)
            if not np.all(np.any(eligible, axis=1)):
                raise ValueError(
                    "each proposal row must contain an eligible candidate"
                )
            reward_logits = np.where(eligible, reward_logits, -np.inf)
        weights_np = np.stack(
            [normalized_weights(row) for row in reward_logits], axis=0
        )
        weights = torch.as_tensor(
            weights_np, device=center.device, dtype=center.dtype
        )
        mean = torch.sum(weights[..., None, None] * candidates, dim=1)
        ess = 1.0 / np.sum(np.square(weights_np), axis=1)
        finite_costs = np.where(np.isfinite(costs), costs, np.inf)
        diagnostics = {
            "sampler": (
                "quadratic_smoothness_gaussian_residual_importance"
                if quadratic_smoothness_proposal is not None
                else (
                    "autoregressive_p99_truncated_importance"
                    if autoregressive
                    else "direct_importance"
                )
            ),
            "beta_schedule": [0.0, 1.0 / float(self.config.temperature)],
            "ess_per_stage": [ess.tolist()],
            "final_ess": ess.tolist(),
            "mala_acceptance_rate": None,
            "best_cost": np.min(finite_costs, axis=1).tolist(),
            "weighted_cost": np.sum(weights_np * costs, axis=1).tolist(),
            "wall_clock_seconds": time.perf_counter() - started,
            "full_cost_evaluation_calls": 1,
            "full_cost_particle_evaluations": int(
                candidates.shape[0] * candidates.shape[1]
            ),
            "gradient_cost_evaluation_calls": 0,
            "gradient_particle_evaluations": 0,
            "importance_density_correction": (
                target_center is not None or smoothness_log_density_ratio is not None
            ),
            "importance_proposal": (
                "fm_gaussian_times_quadratic_smoothness"
                if quadratic_smoothness_proposal is not None
                else (
                    "autoregressive_p99_hard_support"
                    if autoregressive
                    else (
                        "weighted_multi_center_gaussian_mixture"
                        if multi_center_mixture
                        else (
                            "half_policy_half_previous_blended"
                            if mixture_with_target
                            or proposal_mixture_center is not None
                            else (
                                "previous_blended"
                                if target_center is not None
                                else "policy"
                            )
                        )
                    )
                )
            ),
            "autoregressive_dims": (
                int(autoregressive_dims) if autoregressive else None
            ),
            "autoregressive_residual_bounds": (
                autoregressive_residual_bounds.detach().cpu().tolist()
                if autoregressive
                and torch.is_tensor(autoregressive_residual_bounds)
                else None
            ),
            "importance_policy_component_samples": (
                (
                    int(proposal_mixture_counts[0])
                    if multi_center_mixture
                    else proposal_count
                )
                if (
                    multi_center_mixture
                    or mixture_with_target
                    or proposal_mixture_center is not None
                    or target_center is None
                )
                else 0
            ),
            "importance_shifted_component_samples": (
                (
                    int(sum(proposal_mixture_counts[1:]))
                    if multi_center_mixture
                    else proposal_count
                )
                if target_center is not None
                else 0
            ),
            "importance_mixture_component_labels": (
                list(proposal_mixture_labels)
                if multi_center_mixture
                and proposal_mixture_labels is not None
                else None
            ),
            "importance_mixture_component_samples": (
                {
                    str(label): int(count)
                    for label, count in zip(
                        (
                            proposal_mixture_labels
                            if proposal_mixture_labels is not None
                            else tuple(
                                f"component_{index}"
                                for index in range(len(proposal_mixture_counts))
                            )
                        ),
                        proposal_mixture_counts,
                        strict=True,
                    )
                }
                if multi_center_mixture
                else None
            ),
            "importance_total_samples": int(candidates.shape[1]),
            "importance_target_scale": (
                None
                if target_center is None
                else float(
                    proposal_scale if target_scale is None else target_scale
                )
            ),
            "importance_log_density_ratio_min": (
                None
                if log_density_ratio is None
                else np.min(log_density_ratio, axis=1).tolist()
            ),
            "importance_log_density_ratio_mean": (
                None
                if log_density_ratio is None
                else np.mean(log_density_ratio, axis=1).tolist()
            ),
            "importance_log_density_ratio_max": (
                None
                if log_density_ratio is None
                else np.max(log_density_ratio, axis=1).tolist()
            ),
        }
        diagnostics.update(proposal_diagnostics)
        return ProposalResult(
            candidates=candidates,
            costs=costs,
            reward_logits=reward_logits,
            weights=weights,
            mean=mean,
            effective_sample_size=ess,
            proposal_scale=float(proposal_scale),
            diagnostics=diagnostics,
        )

    def _optimize_clean_trajectories_smc(
        self,
        center: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
        cost_fn: CleanTrajectoryCost,
        gradient_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        generator: torch.Generator,
        proposal_scale: float,
        proposal_count: int,
    ) -> ProposalResult:
        requested_mala_steps = int(self.config.smc_mala_steps)
        active_mala_steps = (
            requested_mala_steps if gradient_cost_fn is not None else 0
        )
        candidates = sample_trajectory_proposals(
            center,
            scale=float(proposal_scale),
            lower=lower,
            upper=upper,
            num_samples=int(proposal_count),
            sampler=self.config.proposal_sampler,
            generator=generator,
            include_center=False,
        )

        def full_cost(value: torch.Tensor):
            return _evaluate_cost_on_device(cost_fn, value)

        final_beta = (
            1.0 / float(self.config.temperature)
            if self.config.smc_final_beta is None
            else float(self.config.smc_final_beta)
        )
        result = adaptive_tempered_smc(
            candidates,
            center=center,
            scale=float(proposal_scale),
            lower=lower,
            upper=upper,
            full_cost_fn=full_cost,
            gradient_cost_fn=gradient_cost_fn,
            config=TemperedSMCConfig(
                final_beta=final_beta,
                target_ess_fraction=float(
                    self.config.smc_target_ess_fraction
                ),
                resample_ess_fraction=float(
                    self.config.smc_resample_ess_fraction
                ),
                resampling_method=str(self.config.smc_resampling_method),
                mala_steps=active_mala_steps,
                mala_step_size=float(self.config.smc_mala_step_size),
                mala_schedule=str(self.config.smc_mala_schedule),
                beta_tolerance=float(self.config.smc_beta_tolerance),
                max_tempering_stages=int(
                    self.config.smc_max_tempering_stages
                ),
            ),
            generator=generator,
        )
        result.diagnostics["mala_steps_requested"] = requested_mala_steps
        result.diagnostics["mala_disabled_no_gradient_cost"] = bool(
            requested_mala_steps and gradient_cost_fn is None
        )
        mean = torch.sum(
            result.weights[..., None, None] * result.particles, dim=1
        )
        costs = result.costs.detach().cpu().numpy()
        weights = result.weights.to(device=center.device, dtype=center.dtype)
        reward_logits = -float(final_beta) * costs
        ess = (
            1.0 / torch.sum(weights.square(), dim=1)
        ).detach().cpu().numpy()
        return ProposalResult(
            candidates=result.particles,
            costs=costs,
            reward_logits=reward_logits,
            weights=weights,
            mean=mean,
            effective_sample_size=ess,
            proposal_scale=float(proposal_scale),
            diagnostics=result.diagnostics,
        )

    def guide_score(
        self,
        x_t: torch.Tensor,
        policy_velocity: torch.Tensor,
        *,
        time_value: float,
        lower: torch.Tensor,
        upper: torch.Tensor,
        cost_fn: CleanTrajectoryCost,
        generator: torch.Generator,
        trajectory_coefficient: float,
        gradient_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        keypose_coefficient: float | None = None,
        proposals_per_particle: int | None = None,
        proposal_center: torch.Tensor | None = None,
        proposal_center_mixture: bool = False,
        proposal_mixture_centers: tuple[torch.Tensor, ...] | None = None,
        proposal_mixture_counts: tuple[int, ...] | None = None,
        proposal_mixture_labels: tuple[str, ...] | None = None,
        importance_target_center: torch.Tensor | None = None,
        importance_target_scale: float | None = None,
        autoregressive_initial: torch.Tensor | None = None,
        autoregressive_terminal: torch.Tensor | None = None,
        autoregressive_residual_bounds: torch.Tensor | None = None,
        autoregressive_dims: int = 7,
        quadratic_smoothness_proposal: (
            QuadraticSmoothnessGaussianProposal | None
        ) = None,
    ) -> ScoreGuidanceResult:
        """Guide a waypoint+terminal-keypose block using clean-space costs."""
        if x_t.shape != policy_velocity.shape or x_t.ndim != 3:
            raise ValueError("x_t and policy_velocity must share [P,W,D]")
        policy_clean = torch.clamp(
            x_t - float(time_value) * policy_velocity,
            min=lower,
            max=upper,
        )
        policy_score = rectified_flow_score(
            x_t, policy_velocity, time_value=time_value
        )
        if importance_target_center is not None:
            if importance_target_center.shape != policy_clean.shape:
                raise ValueError(
                    "importance_target_center must match x_t; expected "
                    f"{tuple(policy_clean.shape)}, got "
                    f"{tuple(importance_target_center.shape)}"
                )
            importance_target_center = importance_target_center.to(
                device=x_t.device, dtype=x_t.dtype
            )
        if importance_target_scale is not None:
            if importance_target_center is None:
                raise ValueError(
                    "importance_target_scale requires importance_target_center"
                )
            if float(importance_target_scale) <= 0.0:
                raise ValueError("importance_target_scale must be positive")
        if float(time_value) >= 1.0 - 1e-8:
            candidates = policy_clean[:, None, :, :]
            costs_value = cost_fn(candidates)
            if isinstance(costs_value, ProposalCostResult):
                costs_value = costs_value.costs
            costs = (
                costs_value.detach().cpu().numpy()
                if torch.is_tensor(costs_value)
                else np.asarray(costs_value)
            )
            weights = torch.ones(
                (x_t.shape[0], 1), device=x_t.device, dtype=x_t.dtype
            )
            proposals = ProposalResult(
                candidates=candidates,
                costs=costs,
                reward_logits=-costs / float(self.config.temperature),
                weights=weights,
                mean=policy_clean,
                effective_sample_size=np.ones(x_t.shape[0], dtype=np.float64),
                proposal_scale=0.0,
            )
            return ScoreGuidanceResult(
                guided_flow=policy_velocity,
                policy_score=policy_score,
                mbd_score=policy_score,
                guided_score=policy_score,
                proposals=proposals,
            )

        proposal_scale = flow_matching_clean_proposal_scale(
            time_value=time_value,
            noise_multiplier=float(self.config.proposal_std),
        )
        if proposal_center is None:
            if proposal_center_mixture:
                raise ValueError(
                    "proposal_center_mixture requires a shifted proposal center"
                )
            resolved_proposal_center = policy_clean
        else:
            if proposal_center.shape != policy_clean.shape:
                raise ValueError(
                    "proposal_center must match the clean policy endpoint; "
                    f"expected {tuple(policy_clean.shape)}, got "
                    f"{tuple(proposal_center.shape)}"
                )
            resolved_proposal_center = torch.clamp(
                proposal_center.to(device=x_t.device, dtype=x_t.dtype),
                min=lower,
                max=upper,
            )
        smc_starts_from_importance_target = bool(
            self.config.inference_sampler == "adaptive_smc"
            and importance_target_center is not None
        )
        optimize_center = (
            importance_target_center
            if smc_starts_from_importance_target
            else resolved_proposal_center
        )
        optimize_scale = (
            float(importance_target_scale)
            if smc_starts_from_importance_target
            and importance_target_scale is not None
            else proposal_scale
        )
        proposals = self.optimize_clean_trajectories(
            optimize_center,
            lower=lower,
            upper=upper,
            cost_fn=cost_fn,
            generator=generator,
            proposal_scale=optimize_scale,
            proposals_per_particle=proposals_per_particle,
            gradient_cost_fn=gradient_cost_fn,
            target_center=(
                None
                if smc_starts_from_importance_target
                else (
                    importance_target_center
                    if importance_target_center is not None
                    else (policy_clean if proposal_center is not None else None)
                )
            ),
            target_scale=(
                None
                if smc_starts_from_importance_target
                else importance_target_scale
            ),
            mixture_with_target=bool(
                proposal_center_mixture
                and importance_target_center is None
            ),
            proposal_mixture_center=(
                policy_clean
                if proposal_center_mixture
                and importance_target_center is not None
                else None
            ),
            proposal_mixture_centers=proposal_mixture_centers,
            proposal_mixture_counts=proposal_mixture_counts,
            proposal_mixture_labels=proposal_mixture_labels,
            autoregressive_initial=autoregressive_initial,
            autoregressive_terminal=autoregressive_terminal,
            autoregressive_residual_bounds=autoregressive_residual_bounds,
            autoregressive_dims=int(autoregressive_dims),
            quadratic_smoothness_proposal=quadratic_smoothness_proposal,
        )
        mbd_score = (
            (1.0 - float(time_value)) * proposals.mean - x_t
        ) / (float(time_value) ** 2)
        local_layout = TokenBlockLayout(
            trajectory_start=0, keypose_index=x_t.shape[1] - 1
        )
        guided_score = blend_policy_mbd_flows(
            policy_score,
            mbd_score,
            layout=local_layout,
            coefficients=FlowBlendCoefficients(
                action=float(trajectory_coefficient),
                trajectory=float(trajectory_coefficient),
                keypose=keypose_coefficient,
            ),
        )
        guided_flow = rectified_flow_from_score(
            x_t, guided_score, time_value=time_value
        )
        return ScoreGuidanceResult(
            guided_flow=guided_flow,
            policy_score=policy_score,
            mbd_score=mbd_score,
            guided_score=guided_score,
            proposals=proposals,
        )

    def guide_score_from_clean_mean(
        self,
        x_t: torch.Tensor,
        policy_velocity: torch.Tensor,
        *,
        time_value: float,
        lower: torch.Tensor,
        upper: torch.Tensor,
        clean_mean: torch.Tensor,
        cost_fn: CleanTrajectoryCost,
        trajectory_coefficient: float,
        keypose_coefficient: float | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> ScoreGuidanceResult:
        """Convert an externally computed clean posterior mean into guidance.

        This is the deterministic counterpart of :meth:`guide_score`.  It is
        useful when a caller can integrate ``q_t(y|x_t) exp(-C(y)/tau)``
        analytically (for example, a Gaussian target and quadratic cost).
        ``clean_mean`` is still clamped to the controller's executable bounds.
        """
        if x_t.shape != policy_velocity.shape or x_t.ndim != 3:
            raise ValueError("x_t and policy_velocity must share [P,W,D]")
        if clean_mean.shape != x_t.shape:
            raise ValueError("clean_mean must match x_t")
        mean = torch.maximum(
            torch.minimum(
                clean_mean.to(device=x_t.device, dtype=x_t.dtype), upper
            ),
            lower,
        )
        policy_score = rectified_flow_score(
            x_t, policy_velocity, time_value=time_value
        )
        candidates = mean[:, None, :, :]
        costs_value = cost_fn(candidates)
        if isinstance(costs_value, ProposalCostResult):
            costs_value = costs_value.costs
        costs = (
            costs_value.detach().cpu().numpy()
            if torch.is_tensor(costs_value)
            else np.asarray(costs_value)
        )
        expected = (x_t.shape[0], 1)
        if costs.shape != expected:
            raise ValueError(
                f"cost_fn must return {expected}, got {tuple(costs.shape)}"
            )
        weights = torch.ones(
            expected, device=x_t.device, dtype=x_t.dtype
        )
        proposal_scale = (
            0.0
            if float(time_value) >= 1.0 - 1e-8
            else flow_matching_clean_proposal_scale(
                time_value=time_value,
                noise_multiplier=float(self.config.proposal_std),
            )
        )
        proposal_diagnostics = {
            "sampler": "analytic_clean_posterior_mean",
            "final_ess": [1.0] * int(x_t.shape[0]),
            "best_cost": np.asarray(costs).reshape(-1).tolist(),
            "weighted_cost": np.asarray(costs).reshape(-1).tolist(),
            "full_cost_evaluation_calls": 1,
            "full_cost_particle_evaluations": int(x_t.shape[0]),
            "gradient_cost_evaluation_calls": 0,
            "gradient_particle_evaluations": 0,
        }
        if diagnostics is not None:
            proposal_diagnostics.update(diagnostics)
        proposals = ProposalResult(
            candidates=candidates,
            costs=np.asarray(costs),
            reward_logits=-np.asarray(costs) / float(self.config.temperature),
            weights=weights,
            mean=mean,
            effective_sample_size=np.ones(x_t.shape[0], dtype=np.float64),
            proposal_scale=float(proposal_scale),
            diagnostics=proposal_diagnostics,
        )
        if float(time_value) >= 1.0 - 1e-8:
            mbd_score = policy_score
        else:
            mbd_score = (
                (1.0 - float(time_value)) * mean - x_t
            ) / (float(time_value) ** 2)
        local_layout = TokenBlockLayout(
            trajectory_start=0, keypose_index=x_t.shape[1] - 1
        )
        guided_score = blend_policy_mbd_flows(
            policy_score,
            mbd_score,
            layout=local_layout,
            coefficients=FlowBlendCoefficients(
                action=float(trajectory_coefficient),
                trajectory=float(trajectory_coefficient),
                keypose=keypose_coefficient,
            ),
        )
        guided_flow = (
            policy_velocity
            if float(time_value) >= 1.0 - 1e-8
            else rectified_flow_from_score(
                x_t, guided_score, time_value=time_value
            )
        )
        return ScoreGuidanceResult(
            guided_flow=guided_flow,
            policy_score=policy_score,
            mbd_score=mbd_score,
            guided_score=guided_score,
            proposals=proposals,
        )

    def guide_keypose_then_waypoints(
        self,
        x_t: torch.Tensor,
        policy_velocity: torch.Tensor,
        *,
        time_value: float,
        lower: torch.Tensor,
        upper: torch.Tensor,
        keypose_cost_fn: CleanTrajectoryCost,
        waypoint_cost_fn: CleanTrajectoryCost,
        generator: torch.Generator,
        keypose_generator: torch.Generator | None = None,
        waypoint_generator: torch.Generator | None = None,
        keypose_gradient_cost_fn: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None,
        waypoint_gradient_cost_fn: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None,
        waypoint_coefficient: float,
        keypose_coefficient: float,
        keypose_proposals_per_particle: int | None = None,
        waypoint_proposals_per_particle: int | None = None,
        proposal_center: torch.Tensor | None = None,
        proposal_center_mixture: bool = False,
        keypose_proposal_mixture_centers: (
            tuple[torch.Tensor, ...] | None
        ) = None,
        keypose_proposal_mixture_counts: tuple[int, ...] | None = None,
        keypose_proposal_mixture_labels: tuple[str, ...] | None = None,
        waypoint_autoregressive_initial: torch.Tensor | None = None,
        waypoint_autoregressive_residual_bounds: torch.Tensor | None = None,
        waypoint_autoregressive_dims: int = 7,
        waypoint_smoothness_gaussian_physical_scale: torch.Tensor | None = None,
        waypoint_smoothness_gaussian_weight: float = 0.0,
        condition_waypoints_on_guided_keypose: bool = False,
    ) -> SequentialScoreGuidanceResult:
        """Guide the terminal keypose, then guide waypoints with it frozen.

        ``x_t`` contains zero or more waypoint tokens followed by exactly one
        terminal keypose token. The second MBD proposal set never resamples the
        keypose: every waypoint candidate is concatenated with the weighted
        proposal mean produced by the first keypose MBD step.
        """
        if x_t.shape != policy_velocity.shape or x_t.ndim != 3:
            raise ValueError("x_t and policy_velocity must share [P,W,D]")
        if int(x_t.shape[1]) < 1:
            raise ValueError("sequential guidance requires a terminal keypose")
        if proposal_center is not None and proposal_center.shape != x_t.shape:
            raise ValueError(
                "proposal_center must match the sequential guidance block; "
                f"expected {tuple(x_t.shape)}, got {tuple(proposal_center.shape)}"
            )

        resolved_keypose_generator = (
            generator if keypose_generator is None else keypose_generator
        )
        resolved_waypoint_generator = (
            generator if waypoint_generator is None else waypoint_generator
        )
        keypose = self.guide_score(
            x_t[:, -1:, :],
            policy_velocity[:, -1:, :],
            time_value=time_value,
            lower=lower,
            upper=upper,
            cost_fn=keypose_cost_fn,
            generator=resolved_keypose_generator,
            trajectory_coefficient=float(keypose_coefficient),
            gradient_cost_fn=keypose_gradient_cost_fn,
            keypose_coefficient=float(keypose_coefficient),
            proposals_per_particle=keypose_proposals_per_particle,
            proposal_center=(
                None if proposal_center is None else proposal_center[:, -1:, :]
            ),
            proposal_center_mixture=bool(proposal_center_mixture),
            proposal_mixture_centers=keypose_proposal_mixture_centers,
            proposal_mixture_counts=keypose_proposal_mixture_counts,
            proposal_mixture_labels=keypose_proposal_mixture_labels,
            importance_target_center=(
                keypose_proposal_mixture_centers[0]
                if keypose_proposal_mixture_centers is not None
                else None
            ),
        )
        fixed_keypose = keypose.proposals.mean
        guided_flow = policy_velocity.clone()
        guided_flow[:, -1:, :] = keypose.guided_flow
        guided_keypose = torch.maximum(
            torch.minimum(
                x_t[:, -1:, :]
                - float(time_value) * keypose.guided_flow,
                upper,
            ),
            lower,
        )
        waypoint_conditioning_keypose = (
            guided_keypose
            if condition_waypoints_on_guided_keypose
            else fixed_keypose
        )
        if int(x_t.shape[1]) == 1:
            return SequentialScoreGuidanceResult(
                guided_flow=guided_flow,
                fixed_keypose=fixed_keypose,
                waypoint_conditioning_keypose=waypoint_conditioning_keypose,
                keypose=keypose,
                waypoints=None,
            )

        def conditioned_cost(candidates: torch.Tensor):
            expanded_keypose = waypoint_conditioning_keypose[:, None, :, :].expand(
                -1, int(candidates.shape[1]), -1, -1
            )
            return waypoint_cost_fn(
                torch.cat((candidates, expanded_keypose), dim=-2)
            )

        conditioned_gradient_cost = None
        if waypoint_gradient_cost_fn is not None:
            def conditioned_gradient_cost(candidates: torch.Tensor):
                expanded_keypose = waypoint_conditioning_keypose[:, None, :, :].expand(
                    -1, int(candidates.shape[1]), -1, -1
                )
                return waypoint_gradient_cost_fn(
                    torch.cat((candidates, expanded_keypose), dim=-2)
                )

        smoothness_proposal = None
        if waypoint_smoothness_gaussian_physical_scale is not None:
            if waypoint_autoregressive_initial is None:
                raise ValueError(
                    "smoothness-absorbed waypoints require the current pose"
                )
            smoothness_proposal = QuadraticSmoothnessGaussianProposal(
                initial=waypoint_autoregressive_initial,
                terminal=waypoint_conditioning_keypose,
                physical_scale=waypoint_smoothness_gaussian_physical_scale,
                weight=float(waypoint_smoothness_gaussian_weight),
                residual_bounds=waypoint_autoregressive_residual_bounds,
                constrained_dims=int(waypoint_autoregressive_dims),
            )

        waypoints = self.guide_score(
            x_t[:, :-1, :],
            policy_velocity[:, :-1, :],
            time_value=time_value,
            lower=lower,
            upper=upper,
            cost_fn=conditioned_cost,
            generator=resolved_waypoint_generator,
            trajectory_coefficient=float(waypoint_coefficient),
            gradient_cost_fn=conditioned_gradient_cost,
            keypose_coefficient=float(waypoint_coefficient),
            proposals_per_particle=waypoint_proposals_per_particle,
            proposal_center=(
                None
                if waypoint_autoregressive_residual_bounds is not None
                else (
                    None
                    if proposal_center is None
                    else proposal_center[:, :-1, :]
                )
            ),
            proposal_center_mixture=(
                False
                if waypoint_autoregressive_residual_bounds is not None
                else bool(proposal_center_mixture)
            ),
            autoregressive_initial=(
                None
                if smoothness_proposal is not None
                else waypoint_autoregressive_initial
            ),
            autoregressive_terminal=(
                waypoint_conditioning_keypose
                if waypoint_autoregressive_residual_bounds is not None
                and smoothness_proposal is None
                else None
            ),
            autoregressive_residual_bounds=(
                None
                if smoothness_proposal is not None
                else waypoint_autoregressive_residual_bounds
            ),
            autoregressive_dims=int(waypoint_autoregressive_dims),
            quadratic_smoothness_proposal=smoothness_proposal,
        )
        guided_flow[:, :-1, :] = waypoints.guided_flow
        return SequentialScoreGuidanceResult(
            guided_flow=guided_flow,
            fixed_keypose=fixed_keypose,
            waypoint_conditioning_keypose=waypoint_conditioning_keypose,
            keypose=keypose,
            waypoints=waypoints,
        )


__all__ = [
    "CleanTrajectoryCost",
    "FlowBlendCoefficients",
    "ProposalResult",
    "ProposalCostResult",
    "RectifiedFlowMBD",
    "RectifiedFlowMBDConfig",
    "QuadraticSmoothnessGaussianProposal",
    "SequentialScoreGuidanceResult",
    "ScoreGuidanceResult",
    "TokenBlockLayout",
    "blend_policy_mbd_flows",
    "flow_matching_clean_likelihood_center",
    "flow_matching_clean_proposal_scale",
    "logmeanexp",
    "maximum_task_weight_under_mbd_kl",
    "memoryless_sde_kl_blocks",
    "normalized_weights",
    "rectified_flow_from_score",
    "rectified_flow_score",
    "sample_autoregressive_waypoint_proposals",
    "sample_quadratic_smoothness_gaussian_proposals",
    "sample_independent_truncated_gaussian",
    "sample_trajectory_proposals",
    "straight_line_cspace_path",
]
