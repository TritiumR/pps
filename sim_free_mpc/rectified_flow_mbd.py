"""Generic rectified-flow MBD guidance with externally supplied costs.

This module deliberately knows nothing about a robot, simulator, or task.  A
caller supplies normalized token bounds and a batched clean-trajectory cost
callback.  That keeps policy/MBD mechanics in PPS while allowing downstream
applications to own FK, collision, phase, and scene-state semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Any, Callable

import numpy as np
import torch

from .tempered_smc import (
    TemperedSMCConfig,
    adaptive_tempered_smc,
)


@dataclass(frozen=True)
class ProposalCostResult:
    """Proposal costs plus an optional hard eligibility mask."""

    costs: np.ndarray | torch.Tensor
    eligible: np.ndarray | torch.Tensor | None = None


CleanTrajectoryCost = Callable[
    [torch.Tensor], np.ndarray | torch.Tensor | ProposalCostResult
]
ProposalLogAcceptance = Callable[[torch.Tensor], torch.Tensor]
ProposalPrefilterCost = Callable[[torch.Tensor], torch.Tensor]
GaussianMeanShift = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, Any]]
    | tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
]
CandidateTransform = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, Any]],
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
    record_proposal_zero_candidate: bool = False
    replicated_snis_groups: int = 1
    replicated_snis_exact_quality_guard: bool = True
    replicated_snis_blend: float = 1.0
    first_order_proxy_exact_quality_guard: bool = True

    def validate(self) -> None:
        if int(self.proposals_per_particle) < 1:
            raise ValueError("proposals_per_particle must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(self.proposal_std) <= 0.0:
            raise ValueError("proposal_std must be positive")
        if int(self.replicated_snis_groups) < 1:
            raise ValueError("replicated_snis_groups must be positive")
        if not 0.0 <= float(self.replicated_snis_blend) <= 1.0:
            raise ValueError("replicated_snis_blend must be in [0, 1]")
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


@dataclass
class PersistentProposalBank:
    """Proposal samples and their generating bounded-Gaussian components.

    Samples are retained across related importance-sampling calls.  At every
    call the proposal density is the deterministic mixture of every component
    represented in the bank plus the newly drawn component, weighted by its
    sample count.  This is multiple importance sampling with the balance
    heuristic; no stale sample is treated as though it came from the current
    proposal distribution.
    """

    candidate_batches: list[torch.Tensor]
    component_centers: list[torch.Tensor]
    component_scales: list[float]
    component_counts: list[int]
    cost_batches: list[np.ndarray]
    eligible_batches: list[np.ndarray | None]

    @classmethod
    def empty(cls) -> "PersistentProposalBank":
        return cls([], [], [], [], [], [])

    @property
    def sample_count(self) -> int:
        return int(sum(self.component_counts))

    def append(
        self,
        candidates: torch.Tensor,
        *,
        center: torch.Tensor,
        scale: float,
        costs: np.ndarray,
        eligible: np.ndarray | None,
    ) -> None:
        expected_center = candidates.shape[:1] + candidates.shape[2:]
        if candidates.ndim != 4 or center.shape != expected_center:
            raise ValueError(
                "persistent proposal batch/center shapes are incompatible"
            )
        if float(scale) <= 0.0:
            raise ValueError("persistent proposal scale must be positive")
        expected_costs = tuple(candidates.shape[:2])
        if costs.shape != expected_costs:
            raise ValueError("persistent proposal costs have the wrong shape")
        if eligible is not None and eligible.shape != expected_costs:
            raise ValueError("persistent proposal eligibility has the wrong shape")
        self.candidate_batches.append(candidates.detach().clone())
        self.component_centers.append(center.detach().clone())
        self.component_scales.append(float(scale))
        self.component_counts.append(int(candidates.shape[1]))
        self.cost_batches.append(np.asarray(costs).copy())
        self.eligible_batches.append(
            None if eligible is None else np.asarray(eligible, dtype=bool).copy()
        )


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
    alternate_center: torch.Tensor | None = None
    alternate_mixture_fraction: float = 0.5
    defensive_mean_shift_fn: GaussianMeanShift | None = None
    defensive_mixture_fraction: float = 0.5
    difference_order: int = 2
    first_difference_weight: float = 0.0
    proposal_smoothness_weight_multiplier: float = 1.0


@dataclass(frozen=True)
class LocalQuadraticGaussianProposal:
    """Cost-local Gaussian used only as an exact-IS proposal.

    ``gradient`` and ``hessian`` describe a differentiable local surrogate at
    ``center``. The dense Hessian determines the shifted mean, while its
    diagonal determines a tractable axis-aligned covariance. The complete
    nonlinear cost is still used in every importance weight, and a defensive
    fraction of samples remains drawn from the original FM likelihood.
    """

    gradient: torch.Tensor
    hessian: torch.Tensor
    defensive_fraction: float = 0.25
    damping: float = 1.0e-4


@dataclass(frozen=True)
class FirstOrderGaussianProposal:
    """Defensive exact-IS proposal shaped by one local cost gradient.

    A linear cost tilt shifts an isotropic Gaussian mean by
    ``-variance * gradient / temperature``.  The shifted component is mixed
    with the original likelihood and the exact mixture density is included in
    every importance weight.  The gradient therefore changes finite-sample
    variance only: the estimator retains full support and converges to the
    same Tweedie posterior mean even when the gradient comes from a cheaper,
    imperfect dynamics model such as ARX.
    """

    gradient: torch.Tensor
    cost_at_reference: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    defensive_fraction: float = 0.25
    max_shift_standard_deviations: float = 0.75
    shifted_center_override: torch.Tensor | None = None
    shifted_scale_multiplier: float = 1.0
    use_control_variate: bool = True


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
    antithetic: bool = False,
    uniform_samples: torch.Tensor | None = None,
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
    if bool(torch.any(upper64 < lower64).item()):
        raise RuntimeError("truncated Gaussian has an empty support interval")
    fixed = upper64 == lower64
    if bool(torch.any(fixed).item()):
        # Constant training-action coordinates carry a unit point mass, as in
        # bounded_gaussian_log_prob. Sample only the nonconstant subspace.
        samples = lower64[None, None, :].expand(
            int(center.shape[0]), int(num_samples), int(center.shape[1])
        ).clone().to(dtype=center.dtype)
        varying = ~fixed
        if bool(torch.any(varying).item()):
            samples[..., varying] = sample_independent_truncated_gaussian(
                center[:, varying], scale=scale, lower=lower64[varying],
                upper=upper64[varying], num_samples=num_samples, generator=generator,
                antithetic=antithetic,
                uniform_samples=(
                    None if uniform_samples is None else uniform_samples[..., varying]
                ),
            )
        return samples
    scale64 = torch.as_tensor(float(scale), device=center.device, dtype=work_dtype)
    sqrt_two = math.sqrt(2.0)
    standardized_lower = (lower64[None, :] - center64) / scale64
    standardized_upper = (upper64[None, :] - center64) / scale64
    if bool(torch.any(standardized_upper <= standardized_lower).item()):
        raise RuntimeError("truncated Gaussian has an empty support interval")
    uniform_shape = (int(center.shape[0]), int(center.shape[1]))
    expected_uniform_shape = (
        uniform_shape[0], int(num_samples), uniform_shape[1]
    )
    if uniform_samples is not None:
        if tuple(uniform_samples.shape) != expected_uniform_shape:
            raise ValueError(
                "uniform_samples must match [particles, samples, dimensions]"
            )
        uniform = uniform_samples.to(device=center.device, dtype=work_dtype)
    elif antithetic and int(num_samples) >= 2:
        pair_count = int(num_samples) // 2
        primary = torch.rand(
            (uniform_shape[0], pair_count, uniform_shape[1]),
            device=center.device,
            dtype=work_dtype,
            generator=generator,
        )
        uniform = torch.cat((primary, 1.0 - primary), dim=1)
        if int(num_samples) % 2:
            uniform = torch.cat((
                uniform,
                torch.rand(
                    (uniform_shape[0], 1, uniform_shape[1]),
                    device=center.device,
                    dtype=work_dtype,
                    generator=generator,
                ),
            ), dim=1)
    else:
        uniform = torch.rand(
            (uniform_shape[0], int(num_samples), uniform_shape[1]),
            device=center.device,
            dtype=work_dtype,
            generator=generator,
        )
    epsilon = torch.finfo(work_dtype).eps
    uniform = uniform.clamp(epsilon, 1.0 - epsilon)

    def positive_tail_sample(
        tail_lower: torch.Tensor, tail_upper: torch.Tensor
    ) -> torch.Tensor:
        # Sample Q(z) uniformly over [Q(upper), Q(lower)] in log space.
        # log_ndtr avoids the 1-Phi(z) cancellation that begins near z=8.
        log_q_lower = torch.special.log_ndtr(-tail_lower)
        log_q_upper = torch.special.log_ndtr(-tail_upper)
        log_span = log_q_lower + torch.log1p(
            -torch.exp(log_q_upper - log_q_lower)
        )
        log_q = torch.logaddexp(
            log_q_upper[:, None, :],
            torch.log(uniform) + log_span[:, None, :],
        )
        # Invert log Q(z) with Newton iterations. This remains stable even
        # when Q(z) itself is below float64 normal range.
        value = torch.sqrt(torch.clamp(-2.0 * log_q, min=0.0))
        log_sqrt_two_pi = 0.5 * math.log(2.0 * math.pi)
        for _ in range(10):
            log_survival = torch.special.log_ndtr(-value)
            derivative = -torch.exp(
                -0.5 * value.square() - log_sqrt_two_pi - log_survival
            )
            value = value - (log_survival - log_q) / derivative
        return value

    positive = standardized_lower >= 0.0
    negative = standardized_upper <= 0.0
    central = ~(positive | negative)
    standard_normal = torch.empty_like(uniform)
    if bool(torch.any(positive).item()):
        positive_values = positive_tail_sample(
            standardized_lower, standardized_upper
        )
        standard_normal = torch.where(
            positive[:, None, :], positive_values, standard_normal
        )
    if bool(torch.any(negative).item()):
        negative_values = -positive_tail_sample(
            -standardized_upper, -standardized_lower
        )
        standard_normal = torch.where(
            negative[:, None, :], negative_values, standard_normal
        )
    if bool(torch.any(central).item()):
        lower_cdf = 0.5 * torch.erfc(-standardized_lower / sqrt_two)
        upper_cdf = 0.5 * torch.erfc(-standardized_upper / sqrt_two)
        quantiles = lower_cdf[:, None, :] + uniform * (
            upper_cdf - lower_cdf
        )[:, None, :]
        central_values = torch.special.ndtri(quantiles)
        standard_normal = torch.where(
            central[:, None, :], central_values, standard_normal
        )
    samples = center64[:, None, :] + scale64 * standard_normal
    if not bool(torch.all(torch.isfinite(samples)).item()):
        raise RuntimeError("truncated Gaussian produced non-finite samples")
    samples = torch.maximum(
        torch.minimum(samples, upper64[None, None, :]),
        lower64[None, None, :],
    )
    return samples.to(dtype=center.dtype)


def _sample_diagonal_truncated_gaussian(
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Exact inverse-CDF samples for per-particle diagonal Gaussians."""
    if center.ndim != 2 or scale.shape != center.shape:
        raise ValueError("diagonal Gaussian center/scale must have shape [P,L]")
    if bool(torch.any(scale <= 0.0).item()):
        raise ValueError("diagonal Gaussian scales must be positive")
    work_dtype = torch.float64
    mean = center.to(dtype=work_dtype)
    sigma = scale.to(device=center.device, dtype=work_dtype)
    lo = lower.to(device=center.device, dtype=work_dtype).reshape(1, -1)
    hi = upper.to(device=center.device, dtype=work_dtype).reshape(1, -1)
    sqrt_two = math.sqrt(2.0)
    cdf_lo = 0.5 * (1.0 + torch.erf((lo - mean) / sigma / sqrt_two))
    cdf_hi = 0.5 * (1.0 + torch.erf((hi - mean) / sigma / sqrt_two))
    if bool(torch.any(cdf_hi <= cdf_lo).item()):
        raise RuntimeError("local quadratic proposal has an empty CDF interval")
    epsilon = torch.finfo(work_dtype).eps
    uniform = torch.rand(
        (center.shape[0], int(num_samples), center.shape[1]),
        device=center.device,
        dtype=work_dtype,
        generator=generator,
    ).clamp(epsilon, 1.0 - epsilon)
    quantile = cdf_lo[:, None] + uniform * (cdf_hi - cdf_lo)[:, None]
    standardized = sqrt_two * torch.erfinv(
        2.0 * quantile.clamp(epsilon, 1.0 - epsilon) - 1.0
    )
    value = mean[:, None] + sigma[:, None] * standardized
    value = torch.maximum(torch.minimum(value, hi[:, None]), lo[:, None])
    return value.to(dtype=center.dtype)


def _diagonal_truncated_gaussian_log_prob(
    samples: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Normalized box-truncated diagonal Gaussian log density."""
    if samples.ndim != 3 or center.ndim != 2 or scale.shape != center.shape:
        raise ValueError("diagonal Gaussian density shapes are invalid")
    work_dtype = torch.float64
    value = samples.to(dtype=work_dtype)
    mean = center.to(device=samples.device, dtype=work_dtype)[:, None]
    sigma = scale.to(device=samples.device, dtype=work_dtype)[:, None]
    lo = lower.to(device=samples.device, dtype=work_dtype).reshape(1, 1, -1)
    hi = upper.to(device=samples.device, dtype=work_dtype).reshape(1, 1, -1)
    z = (value - mean) / sigma
    log_pdf = -0.5 * z.square() - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    z_lo = (lo - mean) / sigma
    z_hi = (hi - mean) / sigma
    log_cdf_lo = torch.special.log_ndtr(z_lo)
    log_cdf_hi = torch.special.log_ndtr(z_hi)
    log_ratio = torch.clamp(
        log_cdf_lo - log_cdf_hi, max=-torch.finfo(work_dtype).eps
    )
    log_normalizer = log_cdf_hi + torch.log1p(-torch.exp(log_ratio))
    return (log_pdf - log_normalizer).sum(dim=-1)


def sample_local_quadratic_gaussian_proposals(
    center: torch.Tensor,
    *,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    temperature: float,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    defensive_fraction: float = 0.25,
    damping: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample a defensive local-Gaussian mixture with exact IS correction."""
    if center.ndim != 3 or gradient.shape != center.shape:
        raise ValueError("local quadratic center/gradient must have shape [P,W,D]")
    particles, poses, dimensions = center.shape
    flat_dim = poses * dimensions
    if hessian.shape != (particles, flat_dim, flat_dim):
        raise ValueError("local quadratic Hessian must have shape [P,WD,WD]")
    if int(num_samples) < 2:
        raise ValueError("local quadratic proposals require at least two samples")
    if not 0.0 < float(defensive_fraction) < 1.0:
        raise ValueError("defensive fraction must lie in (0,1)")
    if float(damping) < 0.0 or float(temperature) <= 0.0 or float(scale) <= 0.0:
        raise ValueError("local quadratic temperature/scale/damping are invalid")

    base_mean = center.reshape(particles, flat_dim)
    flat_gradient = gradient.reshape(particles, flat_dim)
    symmetric = 0.5 * (hessian + hessian.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp_min(0.0)
    psd_hessian = torch.einsum(
        "pij,pj,pkj->pik", eigenvectors, eigenvalues, eigenvectors
    )
    identity = torch.eye(flat_dim, device=center.device, dtype=center.dtype)
    precision = (
        identity[None] / float(scale) ** 2
        + psd_hessian / float(temperature)
        + float(damping) * identity[None]
    )
    shifted_mean = base_mean - torch.linalg.solve(
        precision, flat_gradient[..., None] / float(temperature)
    )[..., 0]
    shifted_scale = torch.rsqrt(
        torch.diagonal(precision, dim1=-2, dim2=-1).clamp_min(1.0e-12)
    )
    flat_lower = lower.repeat(poses)
    flat_upper = upper.repeat(poses)
    defensive_count = max(1, min(
        int(num_samples) - 1,
        int(round(float(num_samples) * float(defensive_fraction))),
    ))
    shifted_count = int(num_samples) - defensive_count
    base_scale = torch.full_like(base_mean, float(scale))
    defensive = _sample_diagonal_truncated_gaussian(
        base_mean,
        base_scale,
        lower=flat_lower,
        upper=flat_upper,
        num_samples=defensive_count,
        generator=generator,
    )
    shifted = _sample_diagonal_truncated_gaussian(
        shifted_mean,
        shifted_scale,
        lower=flat_lower,
        upper=flat_upper,
        num_samples=shifted_count,
        generator=generator,
    )
    flat_candidates = torch.cat((defensive, shifted), dim=1)
    base_log = _diagonal_truncated_gaussian_log_prob(
        flat_candidates,
        center=base_mean,
        scale=base_scale,
        lower=flat_lower,
        upper=flat_upper,
    )
    shifted_log = _diagonal_truncated_gaussian_log_prob(
        flat_candidates,
        center=shifted_mean,
        scale=shifted_scale,
        lower=flat_lower,
        upper=flat_upper,
    )
    base_fraction = float(defensive_count) / float(num_samples)
    proposal_log = torch.logaddexp(
        base_log + math.log(base_fraction),
        shifted_log + math.log(1.0 - base_fraction),
    )
    log_density_ratio = base_log - proposal_log
    candidates = flat_candidates.reshape(
        particles, int(num_samples), poses, dimensions
    )
    diagnostics = {
        "local_quadratic_gaussian": True,
        "defensive_fraction": base_fraction,
        "defensive_count": defensive_count,
        "shifted_count": shifted_count,
        "hessian_min_eigenvalue": eigenvalues.amin(dim=-1).tolist(),
        "hessian_max_eigenvalue": eigenvalues.amax(dim=-1).tolist(),
        "base_variance": float(scale) ** 2,
        "shifted_variance_mean": shifted_scale.square().mean(dim=-1).tolist(),
        "mean_shift_rms": torch.sqrt(
            (shifted_mean - base_mean).square().mean(dim=-1)
        ).tolist(),
    }
    return candidates, log_density_ratio, diagnostics


def sample_first_order_gaussian_proposals(
    center: torch.Tensor,
    *,
    gradient: torch.Tensor,
    temperature: float,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    defensive_fraction: float = 0.25,
    max_shift_standard_deviations: float = 0.75,
    shifted_center_override: torch.Tensor | None = None,
    shifted_scale_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample a trust-region linear tilt with exact mixture correction.

    This is deliberately O(PWD): unlike the local-quadratic proposal it does
    not factor a dense Hessian.  That makes a cheap ARX gradient useful even
    when the proposal costs themselves come from MJX/MJWarp.
    """
    if center.ndim != 3 or gradient.shape != center.shape:
        raise ValueError("first-order center/gradient must have shape [P,W,D]")
    if int(num_samples) < 2:
        raise ValueError("first-order proposals require at least two samples")
    if not 0.0 < float(defensive_fraction) < 1.0:
        raise ValueError("first-order defensive fraction must lie in (0,1)")
    if (
        float(temperature) <= 0.0
        or float(scale) <= 0.0
        or float(max_shift_standard_deviations) <= 0.0
        or float(shifted_scale_multiplier) <= 0.0
    ):
        raise ValueError("first-order temperature/scale/trust region must be positive")
    if not bool(torch.all(torch.isfinite(gradient)).item()):
        raise ValueError("first-order gradient must be finite")

    raw_shift = -float(scale) ** 2 * gradient / float(temperature)
    flat_shift = raw_shift.flatten(start_dim=1)
    raw_shift_norm = torch.linalg.vector_norm(flat_shift, dim=1).clamp_min(1.0e-12)
    maximum_norm = float(max_shift_standard_deviations) * float(scale)
    trust_scale = torch.clamp(maximum_norm / raw_shift_norm, max=1.0)
    shift = raw_shift * trust_scale.reshape(-1, 1, 1)
    shifted_center = center + shift
    if shifted_center_override is not None:
        if shifted_center_override.shape != center.shape:
            raise ValueError("shifted center override must match proposal center")
        shifted_center = shifted_center_override.to(
            device=center.device, dtype=center.dtype
        )
        shift = shifted_center - center
    shifted_scale = float(scale) * float(shifted_scale_multiplier)

    defensive_count = max(1, min(
        int(num_samples) - 1,
        int(round(float(num_samples) * float(defensive_fraction))),
    ))
    shifted_count = int(num_samples) - defensive_count
    defensive = sample_trajectory_proposals(
        center,
        scale=float(scale),
        lower=lower,
        upper=upper,
        num_samples=defensive_count,
        sampler="truncated_gaussian",
        generator=generator,
        include_center=False,
        quasi_random=True,
    )
    tilted = sample_trajectory_proposals(
        shifted_center,
        scale=shifted_scale,
        lower=lower,
        upper=upper,
        num_samples=shifted_count,
        sampler="truncated_gaussian",
        generator=generator,
        include_center=False,
        quasi_random=True,
    )
    candidates = torch.cat((defensive, tilted), dim=1)
    base_log = bounded_gaussian_log_prob(
        candidates,
        center=center,
        scale=float(scale),
        lower=lower,
        upper=upper,
        sampler="truncated_gaussian",
    )
    shifted_log = bounded_gaussian_log_prob(
        candidates,
        center=shifted_center,
        scale=shifted_scale,
        lower=lower,
        upper=upper,
        sampler="truncated_gaussian",
    )
    base_fraction = float(defensive_count) / float(num_samples)
    proposal_log = torch.logaddexp(
        base_log + math.log(base_fraction),
        shifted_log + math.log(1.0 - base_fraction),
    )
    return candidates, base_log - proposal_log, {
        "first_order_gaussian": True,
        "randomized_qmc_within_components": "scrambled_sobol",
        "defensive_fraction": base_fraction,
        "defensive_count": defensive_count,
        "shifted_count": shifted_count,
        "base_variance": float(scale) ** 2,
        "shifted_variance": shifted_scale ** 2,
        "shifted_scale_multiplier": float(shifted_scale_multiplier),
        "shifted_center_override": shifted_center_override is not None,
        "raw_mean_shift_rms": torch.sqrt(
            flat_shift.square().mean(dim=1)
        ).tolist(),
        "mean_shift_rms": torch.sqrt(
            shift.flatten(start_dim=1).square().mean(dim=1)
        ).tolist(),
        "trust_scale": trust_scale.tolist(),
        "mean_shift_mahalanobis_norm": (
            torch.linalg.vector_norm(shift.flatten(start_dim=1), dim=1)
            / float(scale)
        ).tolist(),
        "max_shift_standard_deviations": float(max_shift_standard_deviations),
        "gradient_source": "caller_supplied_may_be_low_fidelity",
        "importance_target": "unchanged_base_gaussian",
    }


def _truncated_gaussian_partition_and_mean(
    center: torch.Tensor,
    *,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log box probability and mean for independent Gaussians."""
    dtype = torch.float64
    mean = center.to(dtype=dtype)
    lo = lower.to(device=center.device, dtype=dtype).reshape(1, 1, -1)
    hi = upper.to(device=center.device, dtype=dtype).reshape(1, 1, -1)
    sigma = torch.as_tensor(float(scale), device=center.device, dtype=dtype)
    fixed = hi == lo
    z_lo = (lo - mean) / sigma
    z_hi = (hi - mean) / sigma
    log_cdf_lo = torch.special.log_ndtr(z_lo)
    log_cdf_hi = torch.special.log_ndtr(z_hi)
    ratio = torch.clamp(
        log_cdf_lo - log_cdf_hi, max=-torch.finfo(dtype).eps
    )
    coordinate_log_z = log_cdf_hi + torch.log1p(-torch.exp(ratio))
    coordinate_log_z = torch.where(
        fixed, torch.zeros_like(coordinate_log_z), coordinate_log_z
    )
    log_phi_lo = -0.5 * z_lo.square() - 0.5 * math.log(2.0 * math.pi)
    log_phi_hi = -0.5 * z_hi.square() - 0.5 * math.log(2.0 * math.pi)
    standardized_mean = (
        torch.exp(log_phi_lo - coordinate_log_z)
        - torch.exp(log_phi_hi - coordinate_log_z)
    )
    truncated_mean = mean + sigma * standardized_mean
    truncated_mean = torch.where(fixed, lo.expand_as(mean), truncated_mean)
    return coordinate_log_z.flatten(start_dim=1).sum(dim=1), truncated_mean


def _cross_fitted_first_order_control_variate_mean(
    candidates: torch.Tensor,
    *,
    reward_logits: np.ndarray,
    log_base_over_proposal: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    temperature: float,
    proposal: FirstOrderGaussianProposal,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Estimate exact posterior moments with a linear-cost control variate.

    Regression coefficients are learned on the opposite parity fold, so each
    corrected unnormalized moment remains unbiased conditional on its training
    fold. The final ratio has only the ordinary finite-sample ratio bias and is
    consistent for the exact-cost posterior, regardless of surrogate quality.
    """
    count = int(candidates.shape[1])
    if count < 4 or proposal.cost_at_reference is None:
        return None, {"first_order_control_variate": False, "reason": "missing_moments"}
    reference = center if proposal.reference is None else proposal.reference
    if reference.shape != center.shape:
        raise ValueError("first-order reference must match the proposal center")
    cost_at_reference = proposal.cost_at_reference.reshape(-1).to(
        device=center.device, dtype=torch.float64
    )
    if int(cost_at_reference.numel()) != int(center.shape[0]):
        raise ValueError("first-order reference cost must have one value per particle")

    center64 = center.to(dtype=torch.float64)
    gradient = proposal.gradient.to(device=center.device, dtype=torch.float64)
    reference64 = reference.to(device=center.device, dtype=torch.float64)
    raw_shift = -float(scale) ** 2 * gradient / float(temperature)
    shift_norm = torch.linalg.vector_norm(
        raw_shift.flatten(start_dim=1), dim=1
    ).clamp_min(1.0e-12)
    maximum_norm = (
        float(proposal.max_shift_standard_deviations) * float(scale)
    )
    trust_scale = torch.clamp(maximum_norm / shift_norm, max=1.0)
    effective_gradient = gradient * trust_scale.reshape(-1, 1, 1)
    shifted_center = (
        center64
        - float(scale) ** 2 * effective_gradient / float(temperature)
    )
    base_log_z, _ = _truncated_gaussian_partition_and_mean(
        center64, scale=scale, lower=lower, upper=upper
    )
    shifted_log_z, shifted_mean = _truncated_gaussian_partition_and_mean(
        shifted_center, scale=scale, lower=lower, upper=upper
    )
    center_offset = (
        (center64 - reference64) * effective_gradient
    ).flatten(start_dim=1).sum(1)
    gradient_sq = effective_gradient.flatten(start_dim=1).square().sum(1)
    linear_log_partition = (
        -cost_at_reference / float(temperature)
        - center_offset / float(temperature)
        + 0.5 * float(scale) ** 2 * gradient_sq / float(temperature) ** 2
        + shifted_log_z
        - base_log_z
    )
    values = candidates.to(dtype=torch.float64).flatten(start_dim=2)
    linear_cost = cost_at_reference[:, None] + torch.sum(
        effective_gradient[:, None].flatten(start_dim=2)
        * (values - reference64[:, None].flatten(start_dim=2)),
        dim=-1,
    )
    log_v = (
        log_base_over_proposal.to(dtype=torch.float64)
        - linear_cost / float(temperature)
    )
    log_u = torch.as_tensor(
        reward_logits, device=center.device, dtype=torch.float64
    )
    corrected: list[torch.Tensor] = []
    denominator_values: list[float] = []
    fallback_particles: list[int] = []
    epsilon = torch.finfo(torch.float64).eps
    for particle in range(int(center.shape[0])):
        finite_u = log_u[particle][torch.isfinite(log_u[particle])]
        common = torch.max(torch.cat((
            finite_u,
            log_v[particle],
            linear_log_partition[particle, None],
        )))
        u = torch.exp(log_u[particle] - common)
        v = torch.exp(log_v[particle] - common)
        expected_v = torch.exp(linear_log_partition[particle] - common)
        y = values[particle]
        expected_vy = expected_v * shifted_mean[particle].reshape(-1)
        fold_estimates_b: list[torch.Tensor] = []
        fold_estimates_a: list[torch.Tensor] = []
        for parity in (0, 1):
            test_index = torch.arange(parity, count, 2, device=center.device)
            train_index = torch.arange(1 - parity, count, 2, device=center.device)
            train_v = v[train_index]
            train_u = u[train_index]
            centered_v = train_v - train_v.mean()
            variance_v = torch.sum(centered_v.square()).clamp_min(epsilon)
            beta_b = torch.sum(
                centered_v * (train_u - train_u.mean())
            ) / variance_v
            fold_estimates_b.append(
                torch.mean(u[test_index] - beta_b * (v[test_index] - expected_v))
            )
            train_h = v[train_index, None] * y[train_index]
            train_x = u[train_index, None] * y[train_index]
            centered_h = train_h - train_h.mean(dim=0)
            variance_h = torch.sum(centered_h.square(), dim=0).clamp_min(epsilon)
            beta_a = torch.sum(
                centered_h * (train_x - train_x.mean(dim=0)), dim=0
            ) / variance_h
            fold_estimates_a.append(torch.mean(
                u[test_index, None] * y[test_index]
                - beta_a * (v[test_index, None] * y[test_index] - expected_vy),
                dim=0,
            ))
        estimate_b = torch.stack(fold_estimates_b).mean()
        estimate_a = torch.stack(fold_estimates_a).mean(dim=0)
        if not bool(torch.isfinite(estimate_b).item()) or float(estimate_b.item()) <= 0.0:
            fallback_particles.append(particle)
            corrected.append(torch.empty(0, device=center.device, dtype=torch.float64))
            denominator_values.append(float("nan"))
        else:
            corrected.append(estimate_a / estimate_b)
            denominator_values.append(float(estimate_b.item()))
    if fallback_particles:
        return None, {
            "first_order_control_variate": False,
            "reason": "nonpositive_corrected_partition",
            "fallback_particles": fallback_particles,
        }
    mean = torch.stack(corrected).reshape_as(center64).to(dtype=center.dtype)
    return mean, {
        "first_order_control_variate": True,
        "method": "two_fold_cross_fitted_unnormalized_moments",
        "corrected_partition_scaled": denominator_values,
        "surrogate": "local_linear_cost_with_analytic_truncated_gaussian_moments",
        "asymptotic_target": "exact_cost_tweedie_posterior",
        "trust_scale": trust_scale.tolist(),
    }


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
    antithetic: bool = False,
    quasi_random: bool = False,
) -> torch.Tensor:
    """Sample ``[particle, proposal, pose, joint]`` clean trajectories."""
    if center.ndim != 3:
        raise ValueError("trajectory center must have shape [P,W,D]")
    particles, poses, action_dim = center.shape
    if sampler == "truncated_gaussian":
        uniform_samples = None
        if quasi_random:
            sobol_dimension = int(poses * action_dim)
            if sobol_dimension > 21_201:
                raise ValueError(
                    "Sobol proposal dimension exceeds Torch's 21201 limit"
                )
            particle_uniforms = []
            for _ in range(int(particles)):
                sobol_seed = int(torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    device=center.device,
                    generator=generator,
                ).item())
                values = torch.quasirandom.SobolEngine(
                    sobol_dimension, scramble=True, seed=sobol_seed
                ).draw(int(num_samples), dtype=torch.float64)
                particle_uniforms.append(values.reshape(
                    int(num_samples), int(poses), int(action_dim)
                ))
            uniform_samples = torch.stack(particle_uniforms).to(center.device)
            uniform_samples = uniform_samples.permute(0, 2, 1, 3).reshape(
                int(particles * poses), int(num_samples), int(action_dim)
            )
        flattened = sample_independent_truncated_gaussian(
            center.reshape(particles * poses, action_dim),
            scale=scale,
            lower=lower,
            upper=upper,
            num_samples=num_samples,
            generator=generator,
            antithetic=antithetic,
            uniform_samples=uniform_samples,
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


def sample_log_acceptance_tilted_mixture_proposals(
    centers: tuple[torch.Tensor, ...],
    *,
    mixture_weights: tuple[float, ...],
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    sampler: str,
    generator: torch.Generator,
    log_acceptance_fn: ProposalLogAcceptance,
    max_attempt_factor: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rejection-sample ``p(y) g(y)`` from a bounded Gaussian mixture.

    ``p`` is the complete existing proposal mixture and ``0 < g <= 1`` is a
    common tilt across mixture components. Consequently the unknown rejection
    normalizer is common to every retained sample and cancels exactly in
    self-normalized importance sampling; callers add ``-log(g)`` to the
    target-over-proposal log ratio.
    """
    if not centers or len(centers) != len(mixture_weights):
        raise ValueError("tilted proposal centers/weights must be nonempty and aligned")
    if num_samples < 1 or max_attempt_factor < 1:
        raise ValueError("tilted proposal sample/attempt counts must be positive")
    reference = centers[0]
    if reference.ndim != 3:
        raise ValueError("tilted proposal centers must have shape [P,W,D]")
    if any(center.shape != reference.shape for center in centers):
        raise ValueError("all tilted proposal centers must have identical shapes")
    weights = torch.as_tensor(
        mixture_weights, device=reference.device, dtype=torch.float64
    )
    if bool(torch.any(weights <= 0).item()) or not bool(torch.isfinite(weights).all().item()):
        raise ValueError("tilted proposal mixture weights must be finite and positive")
    weights = weights / weights.sum()
    center_stack = torch.stack(centers, dim=1)
    particles, _components, poses, dimensions = center_stack.shape
    accepted: list[list[torch.Tensor]] = [[] for _ in range(particles)]
    accepted_log: list[list[torch.Tensor]] = [[] for _ in range(particles)]
    accepted_count = [0 for _ in range(particles)]
    attempts = [0 for _ in range(particles)]
    maximum_attempts = int(max_attempt_factor) * int(num_samples)

    while min(accepted_count) < int(num_samples):
        largest_remaining = max(
            int(num_samples) - count for count in accepted_count
        )
        draw_count = min(
            maximum_attempts, max(32, 2 * largest_remaining)
        )
        labels = torch.multinomial(
            weights,
            num_samples=particles * draw_count,
            replacement=True,
            generator=generator,
        ).reshape(particles, draw_count)
        gather = labels[..., None, None].expand(
            particles, draw_count, poses, dimensions
        )
        selected_centers = torch.gather(center_stack, 1, gather)
        if sampler == "truncated_gaussian":
            flattened = sample_independent_truncated_gaussian(
                selected_centers.reshape(particles * draw_count * poses, dimensions),
                scale=float(scale),
                lower=lower,
                upper=upper,
                num_samples=1,
                generator=generator,
            )[:, 0, :]
            proposed = flattened.reshape(
                particles, draw_count, poses, dimensions
            )
        elif sampler == "clipped_gaussian":
            proposed = selected_centers + float(scale) * torch.randn(
                selected_centers.shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
            proposed = torch.maximum(
                torch.minimum(proposed, upper[None, None, None, :]),
                lower[None, None, None, :],
            )
        else:
            raise ValueError(f"Unknown proposal sampler {sampler!r}")
        log_acceptance = log_acceptance_fn(proposed)
        if log_acceptance.shape != (particles, draw_count):
            raise ValueError(
                "proposal log-acceptance callback must return [P,N]"
            )
        if not bool(torch.isfinite(log_acceptance).all().item()):
            raise ValueError("proposal log-acceptance must be finite")
        if bool(torch.any(log_acceptance > 1e-7).item()):
            raise ValueError("proposal log-acceptance must not exceed zero")
        uniform = torch.rand(
            (particles, draw_count),
            device=reference.device,
            dtype=torch.float64,
            generator=generator,
        ).clamp_min(torch.finfo(torch.float64).tiny)
        selected = torch.log(uniform) <= log_acceptance.to(torch.float64)
        for particle in range(particles):
            attempts[particle] += draw_count
            if bool(torch.any(selected[particle]).item()):
                accepted[particle].append(proposed[particle, selected[particle]])
                accepted_log[particle].append(
                    log_acceptance[particle, selected[particle]]
                )
                accepted_count[particle] += int(selected[particle].sum().item())
            if attempts[particle] >= maximum_attempts and accepted_count[particle] < num_samples:
                raise RuntimeError(
                    "eigengrasp proposal rejection acceptance is too low; "
                    f"particle={particle} accepted={accepted_count[particle]} "
                    f"attempted={attempts[particle]}"
                )

    candidates = torch.stack(
        [torch.cat(rows, dim=0)[:num_samples] for rows in accepted], dim=0
    )
    retained_log = torch.stack(
        [torch.cat(rows, dim=0)[:num_samples] for rows in accepted_log], dim=0
    )
    diagnostics = {
        "proposal_log_acceptance_tilt": True,
        "proposal_tilt_attempts": attempts,
        "proposal_tilt_acceptance_rate": [
            float(num_samples) / float(count) for count in attempts
        ],
        "proposal_tilt_log_acceptance_mean": retained_log.mean(dim=1).tolist(),
    }
    return candidates, retained_log, diagnostics


def resample_proxy_tilted_candidates(
    candidates: torch.Tensor,
    proxy_costs: torch.Tensor,
    *,
    retain_count: int,
    generator: torch.Generator,
    target_ess: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compressed SIR from q*exp(-lambda*proxy) with unique-particle stopping.

    Categorical draws continue until ``retain_count`` distinct pool entries
    have appeared. Duplicate draws are compressed into integer multiplicities;
    the caller adds log(multiplicity)-log(tilt) to the full-cost importance
    logits. Thus the expensive cost is evaluated once per distinct proposal.
    """
    if candidates.ndim != 4 or proxy_costs.shape != candidates.shape[:2]:
        raise ValueError("prefilter candidates/costs must have shapes [P,N,W,D]/[P,N]")
    pool_size = int(candidates.shape[1])
    if not 1 <= int(retain_count) <= pool_size:
        raise ValueError("prefilter retain count must lie within the proposal pool")
    if not 1.0 <= float(target_ess) <= float(pool_size):
        raise ValueError("prefilter target ESS must lie in [1, pool size]")
    if not bool(torch.isfinite(proxy_costs).all().item()):
        raise ValueError("proposal prefilter costs must be finite")
    if bool(torch.any(proxy_costs < 0.0).item()):
        raise ValueError("proposal prefilter costs must be nonnegative")

    lambdas = torch.zeros(
        (candidates.shape[0],), device=candidates.device, dtype=torch.float64
    )
    costs64 = proxy_costs.to(torch.float64)
    calibrated_targets: list[float] = []
    for particle in range(int(candidates.shape[0])):
        centered = costs64[particle] - costs64[particle].min()
        zero_count = int((centered <= 0.0).sum().item())
        calibrated_target = min(
            float(pool_size),
            max(float(target_ess), float(zero_count + retain_count)),
        )
        calibrated_targets.append(calibrated_target)
        low = 0.0
        high = 1.0

        def ess_at(value: float) -> float:
            weights = torch.softmax(-value * centered, dim=0)
            return float((1.0 / weights.square().sum()).item())

        while ess_at(high) > calibrated_target and high < 1.0e8:
            high *= 2.0
        for _ in range(48):
            middle = 0.5 * (low + high)
            if ess_at(middle) > calibrated_target:
                low = middle
            else:
                high = middle
        lambdas[particle] = high

    log_tilt = -lambdas[:, None] * costs64
    weights = torch.softmax(log_tilt, dim=1)
    retained_rows = []
    retained_log_rows = []
    multiplicity_rows = []
    retained_proxy_rows = []
    draw_counts = []
    max_draws = max(1000000, 100 * int(retain_count))
    for particle in range(int(candidates.shape[0])):
        order: list[int] = []
        counts: dict[int, int] = {}
        draws_used = 0
        while len(order) < int(retain_count):
            remaining = int(retain_count) - len(order)
            batch = min(max_draws - draws_used, max(256, 4 * remaining))
            if batch <= 0:
                raise RuntimeError(
                    "proxy resampling could not collect enough unique proposals; "
                    f"particle={particle} unique={len(order)} draws={draws_used}"
                )
            draws = torch.multinomial(
                weights[particle],
                num_samples=batch,
                replacement=True,
                generator=generator,
            ).tolist()
            for index in draws:
                index = int(index)
                counts[index] = counts.get(index, 0) + 1
                draws_used += 1
                if counts[index] == 1:
                    order.append(index)
                    if len(order) == int(retain_count):
                        break
        indices = torch.as_tensor(order, device=candidates.device, dtype=torch.long)
        multiplicity = torch.as_tensor(
            [counts[index] for index in order],
            device=candidates.device,
            dtype=torch.long,
        )
        retained_rows.append(candidates[particle, indices])
        retained_log_rows.append(log_tilt[particle, indices])
        multiplicity_rows.append(multiplicity)
        retained_proxy_rows.append(proxy_costs[particle, indices])
        draw_counts.append(draws_used)

    retained = torch.stack(retained_rows, dim=0)
    retained_log_tilt = torch.stack(retained_log_rows, dim=0)
    multiplicities = torch.stack(multiplicity_rows, dim=0)
    retained_proxy = torch.stack(retained_proxy_rows, dim=0)
    pool_ess = 1.0 / weights.square().sum(dim=1)
    diagnostics = {
        "proposal_prefilter_method": "compressed_sir_unique_stopping",
        "proposal_prefilter_pool_size": pool_size,
        "proposal_prefilter_retained_unique": int(retain_count),
        "proposal_prefilter_requested_target_ess": float(target_ess),
        "proposal_prefilter_calibrated_target_ess": calibrated_targets,
        "proposal_prefilter_lambda": lambdas.tolist(),
        "proposal_prefilter_pool_ess": pool_ess.tolist(),
        "proposal_prefilter_draw_count": draw_counts,
        "proposal_prefilter_duplicate_count": [
            int(draws - retain_count) for draws in draw_counts
        ],
        "proposal_prefilter_max_multiplicity": multiplicities.max(dim=1).values.tolist(),
        "proposal_prefilter_zero_count": (proxy_costs <= 0.0).sum(dim=1).tolist(),
        "proposal_prefilter_retained_zero_fraction": (
            retained_proxy <= 0.0
        ).to(torch.float32).mean(dim=1).tolist(),
        "proposal_prefilter_retained_proxy_mean": retained_proxy.mean(dim=1).tolist(),
        "proposal_prefilter_retained_proxy_max": retained_proxy.max(dim=1).values.tolist(),
    }
    return retained, retained_log_tilt, multiplicities, diagnostics


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


def _standard_normal_log_interval(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable log[Phi(upper)-Phi(lower)] using upper-tail symmetry."""
    if lower.dtype != torch.float64 or upper.dtype != torch.float64:
        raise ValueError("standard-normal interval inputs must be float64")
    reflect = lower >= 0.0
    lo = torch.where(reflect, -upper, lower)
    hi = torch.where(reflect, -lower, upper)
    log_lo = torch.special.log_ndtr(lo)
    log_hi = torch.special.log_ndtr(hi)
    delta = torch.clamp(
        log_lo - log_hi, max=-torch.finfo(torch.float64).eps
    )
    log_one_minus = torch.where(
        delta < -math.log(2.0),
        torch.log1p(-torch.exp(delta)),
        torch.log(-torch.expm1(delta)),
    )
    return log_hi + log_one_minus, log_lo, reflect


def _sample_truncated_standard_normal(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-CDF sampling stable in either Gaussian tail."""
    log_interval, log_lo, reflect = _standard_normal_log_interval(lower, upper)
    eps = torch.finfo(torch.float64).eps
    uniform = torch.rand(
        lower.shape,
        device=lower.device,
        dtype=torch.float64,
        generator=generator,
    ).clamp(eps, 1.0 - eps)
    log_probability = torch.logaddexp(log_lo, torch.log(uniform) + log_interval)
    probability = torch.exp(log_probability).clamp(
        torch.finfo(torch.float64).tiny, 1.0 - eps
    )
    standardized = torch.special.ndtri(probability)
    # Inverse CDF cannot represent probabilities below the float64 subnormal
    # range. Sample those one-sided tails exactly with Robert's exponential
    # rejection proposal, truncated at the finite upper endpoint when needed.
    transformed_lo = torch.where(reflect, -upper, lower)
    transformed_hi = torch.where(reflect, -lower, upper)
    tail = transformed_hi < -5.0
    if bool(torch.any(tail).item()):
        positive_lower = -transformed_hi
        positive_upper = -transformed_lo
        alpha = 0.5 * (
            positive_lower + torch.sqrt(positive_lower.square() + 4.0)
        )
        accepted = ~tail
        tail_sample = torch.zeros_like(standardized)
        for _ in range(256):
            if bool(torch.all(accepted).item()):
                break
            draw_u = torch.rand(
                lower.shape,
                device=lower.device,
                dtype=torch.float64,
                generator=generator,
            ).clamp(eps, 1.0 - eps)
            width = positive_upper - positive_lower
            exponential_mass = -torch.expm1(-alpha * width)
            proposal = positive_lower - torch.log1p(
                -draw_u * exponential_mass
            ) / alpha
            mode = torch.minimum(
                positive_upper, torch.maximum(positive_lower, alpha)
            )
            log_acceptance = (
                -0.5 * (proposal - alpha).square()
                + 0.5 * (mode - alpha).square()
            )
            draw_v = torch.rand(
                lower.shape,
                device=lower.device,
                dtype=torch.float64,
                generator=generator,
            ).clamp_min(torch.finfo(torch.float64).tiny)
            take = (~accepted) & (torch.log(draw_v) <= log_acceptance)
            tail_sample = torch.where(take, proposal, tail_sample)
            accepted |= take
        if not bool(torch.all(accepted).item()):
            raise RuntimeError("far-tail truncated Gaussian rejection did not converge")
        standardized = torch.where(tail, -tail_sample, standardized)
    standardized = torch.where(reflect, -standardized, standardized)
    return standardized, log_interval


def sample_joint_covariance_smoothness_gaussian_proposals(
    center: torch.Tensor,
    *,
    alternate_center: torch.Tensor,
    initial: torch.Tensor,
    previous: torch.Tensor | None,
    physical_scale: torch.Tensor,
    joint_covariance: torch.Tensor,
    dense_basis: torch.Tensor,
    dense_residual: torch.Tensor,
    smoothness_weight: float,
    velocity_smoothness_weight: float,
    boundary_acceleration_weight: float,
    difference_order: int,
    temperature: float,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    autoregressive_bounds_fn: Any,
    include_central_trajectory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample a full joint-covariance knot Gaussian with exact q density.

    The base covariance can vary by particle and knot and is combined with
    the physical first-difference smoothness quadratic in one ``K*D`` Gaussian.
    Joint/waypoint bounds are imposed through scalar Gaussian conditionals in
    knot-major order.  Their history-dependent normalizers are included in
    the returned deterministic-mixture proposal density.
    """
    if center.ndim != 3 or alternate_center.shape != center.shape:
        raise ValueError("joint-covariance centers must have shape [P,K,D]")
    particles, knots, action_dim = center.shape
    if joint_covariance.shape != (particles, knots, action_dim, action_dim):
        raise ValueError("joint covariance must have shape [P,K,D,D]")
    if int(num_samples) < 2 or float(scale) <= 0.0:
        raise ValueError("joint-covariance proposal requires 2+ samples and positive scale")
    if float(smoothness_weight) <= 0.0 or float(temperature) <= 0.0:
        raise ValueError("joint-covariance smoothness/temperature must be positive")
    if int(difference_order) not in (1, 3):
        raise ValueError("joint-covariance difference order must be 1 or 3")
    if float(velocity_smoothness_weight) < 0.0:
        raise ValueError("joint-covariance velocity weight must be nonnegative")
    if float(boundary_acceleration_weight) < 0.0:
        raise ValueError("joint-covariance boundary acceleration must be nonnegative")
    initial = initial.to(device=center.device, dtype=center.dtype).reshape(
        particles, action_dim
    )
    physical_scale = physical_scale.to(
        device=center.device, dtype=center.dtype
    ).reshape(action_dim)
    if dense_basis.ndim != 2 or dense_basis.shape[1] != knots:
        raise ValueError("dense knot basis must have shape [H,K]")
    horizon = int(dense_basis.shape[0])
    dense_basis = dense_basis.to(device=center.device, dtype=center.dtype)
    dense_residual = dense_residual.to(device=center.device, dtype=center.dtype)
    if dense_residual.shape != (particles, horizon, action_dim):
        raise ValueError("dense residual must have shape [P,H,D]")
    covariance_shape = 0.5 * (
        joint_covariance + joint_covariance.transpose(-1, -2)
    )
    base_precision = torch.linalg.inv(covariance_shape) / float(scale) ** 2
    dimensions = int(knots * action_dim)
    precision = torch.zeros(
        (particles, dimensions, dimensions),
        device=center.device,
        dtype=center.dtype,
    )
    rhs = torch.einsum("pkij,pkj->pki", base_precision, center).reshape(
        particles, dimensions
    )
    for knot_index in range(knots):
        start = knot_index * action_dim
        precision[:, start : start + action_dim, start : start + action_dim] = (
            base_precision[:, knot_index]
        )

    time_index = torch.arange(knots, device=center.device) * action_dim
    smooth_rhs = torch.zeros_like(center)

    def add_quadratic(
        operator: torch.Tensor,
        anchor: torch.Tensor,
        weight: float,
    ) -> None:
        if float(weight) <= 0.0:
            return
        rows = int(operator.shape[0])
        curvature = (
            2.0
            * float(weight)
            * physical_scale.square()
            / float(rows * action_dim * temperature)
        )
        temporal_precision = operator.T @ operator
        smooth_rhs.add_(
            -torch.einsum("d,ij,pjd->pid", curvature, operator.T, anchor)
        )
        for joint_index in range(action_dim):
            flat_index = time_index + joint_index
            precision[:, flat_index[:, None], flat_index[None, :]] += (
                curvature[joint_index] * temporal_precision
            )

    velocity_difference = torch.eye(
        horizon, device=center.device, dtype=center.dtype
    )
    if horizon > 1:
        dense_indices = torch.arange(1, horizon, device=center.device)
        velocity_difference[dense_indices, dense_indices - 1] = -1.0
    velocity_operator = velocity_difference @ dense_basis
    velocity_anchor = torch.einsum(
        "ij,pjd->pid", velocity_difference, dense_residual
    )
    velocity_anchor[:, 0] -= initial
    if int(difference_order) == 1:
        add_quadratic(velocity_operator, velocity_anchor, smoothness_weight)
    else:
        if horizon > 2:
            acceleration_difference = torch.zeros(
                (horizon - 2, horizon),
                device=center.device,
                dtype=center.dtype,
            )
            acceleration_index = torch.arange(
                horizon - 2, device=center.device
            )
            acceleration_difference[acceleration_index, acceleration_index] = 1.0
            acceleration_difference[acceleration_index, acceleration_index + 1] = -2.0
            acceleration_difference[acceleration_index, acceleration_index + 2] = 1.0
            add_quadratic(
                acceleration_difference @ dense_basis,
                torch.einsum(
                    "ij,pjd->pid", acceleration_difference, dense_residual
                ),
                smoothness_weight,
            )
        add_quadratic(
            velocity_operator, velocity_anchor, velocity_smoothness_weight
        )
    if float(boundary_acceleration_weight) > 0.0 and previous is not None:
        previous = previous.to(device=center.device, dtype=center.dtype).reshape(
            particles, action_dim
        )
        boundary_anchor = (
            dense_residual[:, :1] - 2.0 * initial[:, None] + previous[:, None]
        )
        add_quadratic(
            dense_basis[:1], boundary_anchor, boundary_acceleration_weight
        )
    rhs += smooth_rhs.reshape(particles, dimensions)
    precision = 0.5 * (precision + precision.transpose(-1, -2))
    posterior_covariance = torch.linalg.inv(precision)
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.transpose(-1, -2)
    )
    jitter = 8.0 * torch.finfo(center.dtype).eps
    identity = torch.eye(dimensions, device=center.device, dtype=center.dtype)
    cholesky = torch.linalg.cholesky(
        posterior_covariance + jitter * identity[None]
    )

    def posterior_mean(base_center: torch.Tensor) -> torch.Tensor:
        base_rhs = torch.einsum(
            "pkij,pkj->pki", base_precision, base_center
        ).reshape(particles, dimensions)
        base_rhs += smooth_rhs.reshape(particles, dimensions)
        return torch.linalg.solve(precision, base_rhs[..., None])[..., 0]

    nominal_mean = posterior_mean(center)
    alternate_mean = posterior_mean(alternate_center)
    alternate_count = int(num_samples) // 2
    nominal_count = int(num_samples) - alternate_count
    component_mean = torch.cat(
        (
            nominal_mean[:, None].expand(-1, nominal_count, -1),
            alternate_mean[:, None].expand(-1, alternate_count, -1),
        ),
        dim=1,
    )
    candidates_flat = torch.empty(
        (particles, int(num_samples), dimensions),
        device=center.device,
        dtype=center.dtype,
    )
    conditional_locations = component_mean.clone()
    sampled_lower = torch.empty_like(candidates_flat)
    sampled_upper = torch.empty_like(candidates_flat)
    previous = initial[:, None].expand(-1, int(num_samples), -1)
    proposal_log_density_generating = torch.zeros(
        (particles, int(num_samples)), device=center.device, dtype=torch.float64
    )
    for knot_index in range(knots):
        local_lower = lower[None, None, :].expand(
            particles, int(num_samples), action_dim
        )
        local_upper = upper[None, None, :].expand_as(local_lower)
        local_lower, local_upper = autoregressive_bounds_fn(
            knot_index, previous, local_lower, local_upper
        )
        knot_start = knot_index * action_dim
        knot_stop = knot_start + action_dim
        sampled_lower[:, :, knot_start:knot_stop] = local_lower
        sampled_upper[:, :, knot_start:knot_stop] = local_upper
        for joint_index in range(action_dim):
            flat_index = knot_index * action_dim + joint_index
            conditional_mean = conditional_locations[:, :, flat_index]
            conditional_std = cholesky[:, flat_index, flat_index][:, None]
            mean64 = conditional_mean.to(torch.float64)
            std64 = conditional_std.to(torch.float64)
            lo64 = local_lower[..., joint_index].to(torch.float64)
            hi64 = local_upper[..., joint_index].to(torch.float64)
            standardized, log_interval = _sample_truncated_standard_normal(
                (lo64 - mean64) / std64,
                (hi64 - mean64) / std64,
                generator=generator,
            )
            if include_central_trajectory:
                forced_value = center[:, knot_index, joint_index].to(torch.float64)
                forced_value = torch.maximum(
                    torch.minimum(forced_value, hi64[:, 0]), lo64[:, 0]
                )
                standardized[:, 0] = (
                    forced_value - mean64[:, 0]
                ) / std64[:, 0]
            sampled = mean64 + std64 * standardized
            sampled = torch.maximum(torch.minimum(sampled, hi64), lo64)
            candidates_flat[:, :, flat_index] = sampled.to(center.dtype)
            standardized_native = ((sampled - mean64) / std64).to(center.dtype)
            if flat_index + 1 < dimensions:
                conditional_locations[:, :, flat_index + 1:] += (
                    standardized_native[..., None]
                    * cholesky[:, None, flat_index + 1:, flat_index]
                )
            proposal_log_density_generating += (
                -0.5 * standardized.square()
                - torch.log(std64)
                - 0.5 * math.log(2.0 * math.pi)
                - log_interval
            )
        previous = candidates_flat[
            :, :, knot_index * action_dim : (knot_index + 1) * action_dim
        ]
    candidates = candidates_flat.reshape(
        particles, int(num_samples), knots, action_dim
    )

    bounds_lower_flat = sampled_lower.reshape(
        particles, int(num_samples), dimensions
    ).to(torch.float64)
    bounds_upper_flat = sampled_upper.reshape(
        particles, int(num_samples), dimensions
    ).to(torch.float64)
    diagonal = torch.diagonal(cholesky, dim1=-2, dim2=-1)
    # Both mixture components share one covariance. Solve all 2N right-hand
    # sides against the single factor instead of broadcasting that factor to
    # [P,N,D,D]. This removes the dominant 1024-sample memory traffic.
    component_means = torch.stack((nominal_mean, alternate_mean), dim=1)
    delta = candidates_flat[:, None] - component_means[:, :, None]
    solve_rhs = delta.permute(0, 3, 1, 2).reshape(
        particles, dimensions, 2 * int(num_samples)
    )
    component_innovations = torch.linalg.solve_triangular(
        cholesky, solve_rhs, upper=False
    ).reshape(
        particles, dimensions, 2, int(num_samples)
    ).permute(0, 2, 3, 1)
    # Given z=L^-1(x-m), each scalar conditional location is x_i-L_ii*z_i.
    # Therefore all conditional locations and normalization intervals can be
    # evaluated at once; no O(D) Python loop or prefix einsum is required.
    conditional_mean = (
        candidates_flat[:, None]
        - component_innovations * diagonal[:, None, None]
    ).to(torch.float64)
    conditional_std = diagonal[:, None, None].to(torch.float64)
    standardized_lower = (
        bounds_lower_flat[:, None] - conditional_mean
    ) / conditional_std
    standardized_upper = (
        bounds_upper_flat[:, None] - conditional_mean
    ) / conditional_std
    log_interval, _, _ = _standard_normal_log_interval(
        standardized_lower, standardized_upper
    )
    component_log_density_all = torch.sum(
        -0.5 * component_innovations.to(torch.float64).square()
        - torch.log(conditional_std)
        - 0.5 * math.log(2.0 * math.pi)
        - log_interval,
        dim=-1,
    )
    nominal_log_density = component_log_density_all[:, 0]
    alternate_log_density = component_log_density_all[:, 1]
    mixture_log_density = torch.logaddexp(
        nominal_log_density + math.log(float(nominal_count) / float(num_samples)),
        alternate_log_density + math.log(float(alternate_count) / float(num_samples)),
    )
    return candidates, mixture_log_density.to(center.dtype), {
        "full_joint_covariance": True,
        "smoothness_absorption": "exact_dense_quadratic_through_knot_basis",
        "difference_order": int(difference_order),
        "velocity_smoothness_weight": float(velocity_smoothness_weight),
        "boundary_acceleration_weight": float(boundary_acceleration_weight),
        "density": "exact_scalar_autoregressive_truncation_mixture",
        "optimized_dimensions": dimensions,
        "nominal_count": nominal_count,
        "alternate_count": alternate_count,
        "included_policy_center_trajectory": bool(include_central_trajectory),
        "policy_center_projection_rms": float(
            torch.sqrt(
                torch.mean((candidates[:, 0] - center).square())
            ).item()
        ) if include_central_trajectory else None,
        "generating_log_density_check_max_abs": float(
            torch.max(torch.abs(
                proposal_log_density_generating
                - torch.where(
                    torch.arange(int(num_samples), device=center.device)[None]
                    < nominal_count,
                    nominal_log_density,
                    alternate_log_density,
                )
            )).item()
        ),
        "posterior_variance_mean": float(
            posterior_covariance.diagonal(dim1=-2, dim2=-1).mean().item()
        ),
    }


def sample_quadratic_smoothness_gaussian_proposals(
    center: torch.Tensor,
    *,
    initial: torch.Tensor,
    terminal: torch.Tensor,
    physical_scale: torch.Tensor,
    smoothness_weight: float,
    first_difference_weight: float = 0.0,
    proposal_smoothness_weight_multiplier: float = 1.0,
    temperature: float,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    residual_bounds: torch.Tensor | None = None,
    constrained_dims: int | None = None,
    alternate_center: torch.Tensor | None = None,
    alternate_mixture_fraction: float = 0.5,
    defensive_mean_shift_fn: GaussianMeanShift | None = None,
    defensive_mixture_fraction: float = 0.5,
    difference_order: int = 2,
    include_central_trajectory: bool = True,
    return_proposal_log_density: bool = False,
    autoregressive_bounds_fn: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample ``q * exp(-quadratic smoothness / tau)`` exactly.

    The correlated conjugate Gaussian is sampled through its sequential
    conditionals. Each conditional is truncated by joint limits and the p99
    tube. The returned log density ratio corrects the history-dependent
    truncation normalizers, so the subsequent Tweedie estimate remains IS for
    the hard-supported conjugate target.

    ``difference_order=2`` is the anchored waypoint-Laplacian distribution.
    ``difference_order=1`` is the full action-chunk distribution whose cost is
    the mean squared physical jump from the live control to action 0 and then
    between every adjacent action. In that mode ``residual_bounds`` is the
    per-coordinate hard action-to-action jump support; ``terminal`` is ignored.
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
    if float(first_difference_weight) < 0.0:
        raise ValueError("quadratic first-difference weight must be nonnegative")
    if float(proposal_smoothness_weight_multiplier) <= 0.0:
        raise ValueError("proposal smoothness multiplier must be positive")
    if int(difference_order) not in (1, 2, 3):
        raise ValueError(
            "quadratic smoothness difference_order must be 1, 2, or 3"
        )
    initial = initial.to(device=center.device, dtype=center.dtype).reshape(
        particles, action_dim
    )
    terminal = terminal.to(device=center.device, dtype=center.dtype).reshape(
        particles, action_dim
    )
    physical_scale = physical_scale.to(
        device=center.device, dtype=center.dtype
    ).reshape(action_dim)

    # B maps free tokens to first or second differences of the anchored path.
    b_matrix = torch.zeros(
        (waypoints, waypoints), device=center.device, dtype=center.dtype
    )
    index = torch.arange(waypoints, device=center.device)
    if int(difference_order) == 1:
        b_matrix[index, index] = 1.0
        if waypoints > 1:
            neighbor = torch.arange(1, waypoints, device=center.device)
            b_matrix[neighbor, neighbor - 1] = -1.0
    elif int(difference_order) == 2:
        b_matrix[index, index] = -2.0
        if waypoints > 1:
            neighbor = torch.arange(waypoints - 1, device=center.device)
            b_matrix[neighbor, neighbor + 1] = 1.0
            b_matrix[neighbor + 1, neighbor] = 1.0
    elif waypoints > 2:
        second = torch.arange(2, waypoints, device=center.device)
        b_matrix[second, second] = 1.0
        b_matrix[second, second - 1] = -2.0
        b_matrix[second, second - 2] = 1.0
    anchor = torch.zeros(
        (particles, waypoints, action_dim),
        device=center.device,
        dtype=center.dtype,
    )
    if int(difference_order) == 1:
        anchor[:, 0, :] = -initial
    elif int(difference_order) == 2:
        anchor[:, 0, :] = initial
        anchor[:, -1, :] += terminal

    inverse_variance = 1.0 / float(scale) ** 2
    smoothness_steps = (
        max(waypoints - 2, 1)
        if int(difference_order) == 3
        else waypoints
    )
    gamma = (
        float(smoothness_weight)
        * physical_scale.square()
        / float(smoothness_steps * action_dim)
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
    if float(first_difference_weight) > 0.0:
        velocity_matrix = torch.zeros_like(b_matrix)
        velocity_matrix[index, index] = 1.0
        if waypoints > 1:
            neighbor = torch.arange(1, waypoints, device=center.device)
            velocity_matrix[neighbor, neighbor - 1] = -1.0
        velocity_anchor = torch.zeros_like(anchor)
        velocity_anchor[:, 0, :] = -initial
        velocity_gamma = (
            float(first_difference_weight)
            * physical_scale.square()
            / float(waypoints * action_dim)
        )
        velocity_curvature = 2.0 * velocity_gamma / float(temperature)
        velocity_btb = torch.matmul(velocity_matrix.T, velocity_matrix)
        precision = precision + (
            velocity_curvature[:, None, None] * velocity_btb[None, :, :]
        )
        rhs -= velocity_curvature[None, :, None] * torch.einsum(
            "ij,pjd->pdi", velocity_matrix.T, velocity_anchor
        )
    posterior_mean = torch.linalg.solve(
        precision[None, :, :, :], rhs[..., None]
    )[..., 0]
    covariance = torch.linalg.inv(precision)
    target_posterior_mean = posterior_mean
    target_covariance = covariance
    proposal_smoothness_multiplier = float(
        proposal_smoothness_weight_multiplier
    )
    if proposal_smoothness_multiplier != 1.0:
        precision = (
            inverse_variance * identity[None, :, :]
            + proposal_smoothness_multiplier
            * curvature[:, None, None]
            * btb[None, :, :]
        )
        rhs = inverse_variance * center.permute(0, 2, 1)
        rhs -= (
            proposal_smoothness_multiplier
            * curvature[None, :, None]
            * torch.einsum("ij,pjd->pdi", b_matrix.T, anchor)
        )
        if float(first_difference_weight) > 0.0:
            precision = precision + (
                proposal_smoothness_multiplier
                * velocity_curvature[:, None, None]
                * velocity_btb[None, :, :]
            )
            rhs -= (
                proposal_smoothness_multiplier
                * velocity_curvature[None, :, None]
                * torch.einsum(
                    "ij,pjd->pdi", velocity_matrix.T, velocity_anchor
                )
            )
        posterior_mean = torch.linalg.solve(
            precision[None, :, :, :], rhs[..., None]
        )[..., 0]
        covariance = torch.linalg.inv(precision)
    shifted_posterior_mean = None
    shifted_covariance = None
    shift_diagnostics: dict[str, Any] | None = None
    nominal_count = int(num_samples)
    shifted_count = 0
    if alternate_center is not None and defensive_mean_shift_fn is not None:
        raise ValueError(
            "alternate-center and defensive-shift mixtures are mutually exclusive"
        )
    mixture_fraction = float(defensive_mixture_fraction)
    if alternate_center is not None:
        alternate_center = alternate_center.to(
            device=center.device, dtype=center.dtype
        )
        if alternate_center.shape != center.shape:
            raise ValueError("alternate smoothness center must match center")
        mixture_fraction = float(alternate_mixture_fraction)
        alternate_rhs = rhs + inverse_variance * (
            alternate_center - center
        ).permute(0, 2, 1)
        shifted_posterior_mean = torch.linalg.solve(
            precision[None, :, :, :], alternate_rhs[..., None]
        )[..., 0]
        shifted_covariance = covariance
        shift_diagnostics = {
            "kind": "alternate_base_center",
            "base_center_rms_delta": float(
                torch.sqrt(torch.mean((alternate_center - center).square())).item()
            ),
        }
    elif defensive_mean_shift_fn is not None:
        mixture_fraction = float(defensive_mixture_fraction)
        conditioning = defensive_mean_shift_fn(
            posterior_mean.permute(0, 2, 1), covariance, terminal
        )
        if len(conditioning) == 2:
            shifted_waypoint_mean, shift_diagnostics = conditioning
            shifted_covariance = covariance
        elif len(conditioning) == 3:
            shifted_waypoint_mean, shifted_covariance, shift_diagnostics = conditioning
        else:
            raise ValueError("defensive Gaussian conditioning returned invalid tuple")
        if shifted_waypoint_mean.shape != center.shape:
            raise ValueError("defensive shifted mean must match waypoint center")
        if shifted_covariance.shape != covariance.shape:
            raise ValueError("defensive shifted covariance must match base covariance")
        shifted_covariance = shifted_covariance.to(
            device=center.device, dtype=center.dtype
        )
        shifted_posterior_mean = shifted_waypoint_mean.permute(0, 2, 1)
    if shifted_posterior_mean is not None:
        if not 0.0 < mixture_fraction < 1.0:
            raise ValueError("proposal mixture fraction must lie strictly in (0,1)")
        shifted_count = max(
            1, min(int(num_samples) - 1, int(round(
                float(num_samples) * mixture_fraction
            )))
        )
        nominal_count = int(num_samples) - shifted_count
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

    def component_conditional(
        mean: torch.Tensor,
        component_covariance: torch.Tensor,
        *,
        start: int,
        stop: int,
        waypoint_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(stop - start)
        conditional_mean = mean[:, :, waypoint_index][:, None, :].expand(
            -1, count, -1
        )
        conditional_variance = component_covariance[
            :, waypoint_index, waypoint_index
        ]
        if waypoint_index > 0:
            prefix_covariance = component_covariance[
                :, :waypoint_index, :waypoint_index
            ]
            cross_covariance = component_covariance[
                :, waypoint_index, :waypoint_index
            ]
            gain = torch.linalg.solve(
                prefix_covariance, cross_covariance[..., None]
            )[..., 0]
            prefix_delta = candidates[
                :, start:stop, :waypoint_index, :
            ].permute(0, 1, 3, 2) - mean[:, None, :, :waypoint_index]
            conditional_mean = conditional_mean + torch.sum(
                prefix_delta * gain[None, None, :, :], dim=-1
            )
            conditional_variance = conditional_variance - torch.sum(
                cross_covariance * gain, dim=-1
            )
        conditional_variance = conditional_variance[None, None, :].expand_as(
            conditional_mean
        )
        return conditional_mean, conditional_variance

    for waypoint_index in range(waypoints):
        nominal_mean, nominal_variance = component_conditional(
            posterior_mean,
            covariance,
            start=0,
            stop=nominal_count,
            waypoint_index=waypoint_index,
        )
        if shifted_posterior_mean is None:
            conditional_mean = nominal_mean
            conditional_variance = nominal_variance
        else:
            assert shifted_covariance is not None
            shifted_mean, shifted_variance = component_conditional(
                shifted_posterior_mean,
                shifted_covariance,
                start=nominal_count,
                stop=int(num_samples),
                waypoint_index=waypoint_index,
            )
            conditional_mean = torch.cat((nominal_mean, shifted_mean), dim=1)
            conditional_variance = torch.cat(
                (nominal_variance, shifted_variance), dim=1
            )
        conditional_std = torch.sqrt(torch.clamp(
            conditional_variance, min=torch.finfo(center.dtype).eps
        ))
        local_lower = lower[None, None, :].expand_as(conditional_mean)
        local_upper = upper[None, None, :].expand_as(conditional_mean)
        if bounds is not None:
            if int(difference_order) in (1, 3):
                tube_center = previous
            else:
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
        if autoregressive_bounds_fn is not None:
            local_lower, local_upper = autoregressive_bounds_fn(
                waypoint_index, previous, local_lower, local_upper
            )
            if local_lower.shape != conditional_mean.shape or local_upper.shape != conditional_mean.shape:
                raise ValueError("Autoregressive proposal bounds have an invalid shape")
            if bool(torch.any(local_lower > local_upper).item()):
                raise ValueError("Autoregressive proposal bounds have an empty interval")
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
        if include_central_trajectory:
            uniform[:, 0, :] = 0.5
        if include_central_trajectory and shifted_count:
            uniform[:, nominal_count, :] = 0.5
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
    if (
        shifted_posterior_mean is not None
        or proposal_smoothness_multiplier != 1.0
        or return_proposal_log_density
    ):
        log_two_pi = math.log(2.0 * math.pi)

        def component_log_density(
            component: torch.Tensor,
            component_covariance: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            log_gaussian = torch.zeros(
                (particles, int(num_samples)),
                device=center.device,
                dtype=torch.float64,
            )
            log_proposal = torch.zeros_like(log_gaussian)
            previous_value = initial[:, None, :].expand(
                -1, int(num_samples), -1
            )
            for waypoint_index in range(waypoints):
                conditional_mean_value = component[:, :, waypoint_index]
                conditional_variance = component_covariance[
                    :, waypoint_index, waypoint_index
                ]
                if waypoint_index > 0:
                    prefix_covariance = component_covariance[
                        :, :waypoint_index, :waypoint_index
                    ]
                    cross_covariance = component_covariance[
                        :, waypoint_index, :waypoint_index
                    ]
                    gain = torch.linalg.solve(
                        prefix_covariance, cross_covariance[..., None]
                    )[..., 0]
                    prefix_delta = (
                        candidates[:, :, :waypoint_index, :].permute(
                            0, 1, 3, 2
                        )
                        - component[:, None, :, :waypoint_index]
                    )
                    conditional_mean_value = (
                        conditional_mean_value[:, None, :]
                        + torch.sum(
                            prefix_delta * gain[None, None, :, :], dim=-1
                        )
                    )
                    conditional_variance = conditional_variance - torch.sum(
                        cross_covariance * gain, dim=-1
                    )
                else:
                    conditional_mean_value = conditional_mean_value[
                        :, None, :
                    ].expand(-1, int(num_samples), -1)
                conditional_std = torch.sqrt(torch.clamp(
                    conditional_variance,
                    min=torch.finfo(center.dtype).eps,
                ))[None, None, :]
                local_lower = lower[None, None, :].expand_as(
                    conditional_mean_value
                )
                local_upper = upper[None, None, :].expand_as(
                    conditional_mean_value
                )
                if bounds is not None:
                    if int(difference_order) in (1, 3):
                        tube_center = previous_value
                    else:
                        remaining = float(waypoints + 1 - waypoint_index)
                        tube_center = previous_value + (
                            goal - previous_value
                        ) / remaining
                    local_lower = local_lower.clone()
                    local_upper = local_upper.clone()
                    local_lower[..., :dims] = torch.maximum(
                        local_lower[..., :dims],
                        tube_center[..., :dims] - bounds,
                    )
                    local_upper[..., :dims] = torch.minimum(
                        local_upper[..., :dims],
                        tube_center[..., :dims] + bounds,
                    )
                if autoregressive_bounds_fn is not None:
                    local_lower, local_upper = autoregressive_bounds_fn(
                        waypoint_index, previous_value, local_lower, local_upper
                    )
                    if local_lower.shape != conditional_mean_value.shape or local_upper.shape != conditional_mean_value.shape:
                        raise ValueError("Autoregressive density bounds have an invalid shape")
                    if bool(torch.any(local_lower > local_upper).item()):
                        raise ValueError("Autoregressive density bounds have an empty interval")
                value64 = candidates[:, :, waypoint_index, :].to(torch.float64)
                mean64 = conditional_mean_value.to(torch.float64)
                std64 = conditional_std.to(torch.float64)
                standardized = (value64 - mean64) / std64
                step_gaussian = torch.sum(
                    -0.5 * standardized.square()
                    - torch.log(std64)
                    - 0.5 * log_two_pi,
                    dim=-1,
                )
                lo64 = local_lower.to(torch.float64)
                hi64 = local_upper.to(torch.float64)
                cdf_lo = 0.5 * (
                    1.0 + torch.erf((lo64 - mean64) / std64 / sqrt_two)
                )
                cdf_hi = 0.5 * (
                    1.0 + torch.erf((hi64 - mean64) / std64 / sqrt_two)
                )
                interval = torch.clamp(cdf_hi - cdf_lo, min=eps)
                log_gaussian += step_gaussian
                log_proposal += step_gaussian - torch.sum(
                    torch.log(interval), dim=-1
                )
                previous_value = candidates[:, :, waypoint_index, :]
            return log_gaussian, log_proposal

        # With the production/default proposal multiplier, the target and
        # nominal Gaussian are exactly the same distribution. Reuse one
        # conditional-density traversal instead of evaluating all H
        # conditionals twice over the full proposal bank. Samples and
        # importance weights are unchanged; only redundant work is removed.
        if proposal_smoothness_multiplier == 1.0:
            target_log_density, nominal_log_proposal = component_log_density(
                posterior_mean, covariance
            )
        else:
            target_log_density, _ = component_log_density(
                target_posterior_mean, target_covariance
            )
            _, nominal_log_proposal = component_log_density(
                posterior_mean, covariance
            )
        if shifted_posterior_mean is None:
            mixture_log_density = nominal_log_proposal
            log_density_ratio = target_log_density - mixture_log_density
        else:
            assert shifted_covariance is not None
            _, shifted_log_proposal = component_log_density(
                shifted_posterior_mean, shifted_covariance
            )
            nominal_fraction = float(nominal_count) / float(num_samples)
            shifted_fraction = float(shifted_count) / float(num_samples)
            mixture_log_density = torch.logaddexp(
                nominal_log_proposal + math.log(nominal_fraction),
                shifted_log_proposal + math.log(shifted_fraction),
            )
            log_density_ratio = target_log_density - mixture_log_density
    return candidates, log_density_ratio.to(center.dtype), {
        "conjugate_quadratic_smoothness": True,
        "difference_order": int(difference_order),
        "first_difference_weight": float(first_difference_weight),
        "smoothness_metric": (
            "mixed_physical_action_acceleration_and_velocity"
            if int(difference_order) == 3 and float(first_difference_weight) > 0.0
            else "mean_squared_physical_action_jump_from_current"
            if int(difference_order) == 1
            else (
                "mean_squared_physical_unanchored_action_acceleration"
                if int(difference_order) == 3
                else "mean_squared_physical_anchored_laplacian"
            )
        ),
        "included_central_trajectory": bool(include_central_trajectory),
        "target_smoothness_weight": float(smoothness_weight),
        "proposal_smoothness_weight_multiplier": proposal_smoothness_multiplier,
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
            (
                "joint_bounds_and_action_jump_p95"
                if int(difference_order) in (1, 3)
                else "joint_bounds_and_autoregressive_p99"
            )
            if residual_bounds is not None
            else "joint_bounds"
        ),
        "defensive_mixture": shifted_posterior_mean is not None,
        "mixture_kind": (
            "policy_previous_blended"
            if alternate_center is not None
            else ("defensive_shift" if shifted_posterior_mean is not None else None)
        ),
        "defensive_nominal_count": nominal_count,
        "defensive_shifted_count": shifted_count,
        "defensive_shift": shift_diagnostics,
        **(
            {"proposal_log_density": mixture_log_density}
            if return_proposal_log_density
            else {}
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
        proposal_mixture_include_centers: tuple[bool, ...] | None = None,
        autoregressive_initial: torch.Tensor | None = None,
        autoregressive_terminal: torch.Tensor | None = None,
        autoregressive_residual_bounds: torch.Tensor | None = None,
        autoregressive_dims: int = 7,
        quadratic_smoothness_proposal: (
            QuadraticSmoothnessGaussianProposal | None
        ) = None,
        first_order_proposal: FirstOrderGaussianProposal | None = None,
        local_quadratic_proposal: LocalQuadraticGaussianProposal | None = None,
        proposal_log_acceptance_fn: ProposalLogAcceptance | None = None,
        proposal_prefilter_cost_fn: ProposalPrefilterCost | None = None,
        proposal_prefilter_pool_size: int | None = None,
        proposal_prefilter_target_ess: float | None = None,
        evaluation_only_candidate: torch.Tensor | None = None,
        diagnostic_repeats: int = 1,
        persistent_proposal_bank: PersistentProposalBank | None = None,
        candidate_transform_fn: CandidateTransform | None = None,
    ) -> ProposalResult:
        """Optimize clean trajectories with direct IS or adaptive SMC/MALA."""
        if int(diagnostic_repeats) < 1:
            raise ValueError("diagnostic_repeats must be positive")
        if candidate_transform_fn is not None and (
            persistent_proposal_bank is not None
            or proposal_prefilter_cost_fn is not None
            or evaluation_only_candidate is not None
        ):
            raise ValueError(
                "candidate transforms do not support persistent proposals, "
                "prefiltering, or evaluation-only candidates"
            )
        if persistent_proposal_bank is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("persistent proposals require direct IS")
            if int(diagnostic_repeats) != 1:
                raise ValueError(
                    "persistent proposals do not support diagnostic repeats"
                )
            if target_center is None or target_scale is None:
                raise ValueError(
                    "persistent proposals require an explicit current target"
                )
        proposal_count = (
            int(self.config.proposals_per_particle)
            if proposals_per_particle is None
            else int(proposals_per_particle)
        )
        if proposal_count < 1:
            raise ValueError("proposals_per_particle must be positive")
        if proposal_prefilter_cost_fn is not None:
            if proposal_prefilter_pool_size is None:
                raise ValueError("proposal prefilter requires a pool size")
            if int(proposal_prefilter_pool_size) < proposal_count:
                raise ValueError(
                    "proposal prefilter pool size must be at least the retained count"
                )
        elif proposal_prefilter_pool_size is not None:
            raise ValueError("proposal prefilter pool size requires a cost callback")
        if proposal_prefilter_target_ess is not None and proposal_prefilter_cost_fn is None:
            raise ValueError("proposal prefilter target ESS requires a cost callback")
        if (
            evaluation_only_candidate is not None
            and evaluation_only_candidate.shape != center.shape
        ):
            raise ValueError(
                "evaluation_only_candidate must match the proposal center"
            )
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
            if proposal_mixture_include_centers is not None and len(
                proposal_mixture_include_centers
            ) != len(proposal_mixture_centers):
                raise ValueError(
                    "proposal center-injection flags must match mixture centers"
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
            or proposal_mixture_include_centers is not None
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
        if first_order_proposal is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("first-order proposals require direct IS")
            if self.config.proposal_sampler != "truncated_gaussian":
                raise ValueError(
                    "first-order proposals require truncated_gaussian"
                )
            if (
                autoregressive
                or quadratic_smoothness_proposal is not None
                or (
                    target_center is not None
                    and persistent_proposal_bank is None
                )
                or mixture_with_target
                or proposal_mixture_center is not None
                or multi_center_mixture
            ):
                raise ValueError(
                    "first-order proposal owns its proposal distribution"
                )
            if (
                persistent_proposal_bank is not None
                and first_order_proposal.use_control_variate
            ):
                raise ValueError(
                    "persistent first-order mixtures do not support the "
                    "local control variate"
                )
        if local_quadratic_proposal is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("local quadratic proposals require direct IS")
            if (
                autoregressive
                or quadratic_smoothness_proposal is not None
                or first_order_proposal is not None
                or target_center is not None
                or mixture_with_target
                or proposal_mixture_center is not None
                or multi_center_mixture
            ):
                raise ValueError(
                    "local quadratic proposal owns its proposal distribution"
                )
        if proposal_prefilter_cost_fn is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("proposal prefilters require direct IS")
            if (
                proposal_log_acceptance_fn is not None
                or autoregressive
                or quadratic_smoothness_proposal is not None
                or first_order_proposal is not None
                or local_quadratic_proposal is not None
            ):
                raise ValueError(
                    "proposal prefilters are exclusive with other specialized proposals"
                )
        if proposal_log_acceptance_fn is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("proposal rejection tilts require direct IS")
            if (
                autoregressive
                or quadratic_smoothness_proposal is not None
                or first_order_proposal is not None
                or local_quadratic_proposal is not None
            ):
                raise ValueError(
                    "proposal rejection tilts are exclusive with waypoint-specific proposals"
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
        first_order_log_density_ratio: torch.Tensor | None = None
        local_quadratic_log_density_ratio: torch.Tensor | None = None
        tilt_log_acceptance: torch.Tensor | None = None
        retained_proposal_count = (
            int(sum(proposal_mixture_counts))
            if multi_center_mixture
            else (
                2 * proposal_count
                if mixture_with_target or proposal_mixture_center is not None
                else proposal_count
            )
        )
        if (
            proposal_prefilter_cost_fn is not None
            and int(proposal_prefilter_pool_size) < retained_proposal_count
        ):
            raise ValueError("proposal prefilter pool cannot cover retained mixture")
        prefilter_log_tilt: torch.Tensor | None = None
        prefilter_multiplicities: torch.Tensor | None = None
        prefilter_pool_count = (
            proposal_count
            if proposal_prefilter_cost_fn is None
            else int(proposal_prefilter_pool_size)
        )
        if multi_center_mixture and proposal_prefilter_cost_fn is not None:
            original_total = int(sum(proposal_mixture_counts))
            scaled = [
                int(prefilter_pool_count * int(count) // original_total)
                for count in proposal_mixture_counts
            ]
            for index in range(prefilter_pool_count - sum(scaled)):
                scaled[index % len(scaled)] += 1
            sampling_mixture_counts = tuple(scaled)
        else:
            sampling_mixture_counts = proposal_mixture_counts
        two_center_sample_count = (
            proposal_count
            if proposal_prefilter_cost_fn is None
            else (prefilter_pool_count + 1) // 2
        )
        if proposal_log_acceptance_fn is not None:
            if multi_center_mixture:
                tilt_centers = tuple(proposal_mixture_centers)
                total = float(sum(proposal_mixture_counts))
                tilt_weights = tuple(
                    float(count) / total for count in proposal_mixture_counts
                )
                tilt_count = int(sum(proposal_mixture_counts))
            elif mixture_with_target or proposal_mixture_center is not None:
                first_center = (
                    target_center
                    if proposal_mixture_center is None
                    else proposal_mixture_center
                )
                tilt_centers = (first_center, center)
                tilt_weights = (0.5, 0.5)
                tilt_count = 2 * proposal_count
            else:
                tilt_centers = (center,)
                tilt_weights = (1.0,)
                tilt_count = proposal_count
            candidates, tilt_log_acceptance, proposal_diagnostics = (
                sample_log_acceptance_tilted_mixture_proposals(
                    tilt_centers,
                    mixture_weights=tilt_weights,
                    scale=float(proposal_scale),
                    lower=lower,
                    upper=upper,
                    num_samples=tilt_count,
                    sampler=self.config.proposal_sampler,
                    generator=generator,
                    log_acceptance_fn=proposal_log_acceptance_fn,
                )
            )
        elif first_order_proposal is not None:
            (
                candidates,
                first_order_log_density_ratio,
                proposal_diagnostics,
            ) = sample_first_order_gaussian_proposals(
                center,
                gradient=first_order_proposal.gradient,
                temperature=float(self.config.temperature),
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                generator=generator,
                defensive_fraction=float(
                    first_order_proposal.defensive_fraction
                ),
                max_shift_standard_deviations=float(
                    first_order_proposal.max_shift_standard_deviations
                ),
                shifted_center_override=(
                    first_order_proposal.shifted_center_override
                ),
                shifted_scale_multiplier=float(
                    first_order_proposal.shifted_scale_multiplier
                ),
            )
        elif local_quadratic_proposal is not None:
            (
                candidates,
                local_quadratic_log_density_ratio,
                proposal_diagnostics,
            ) = sample_local_quadratic_gaussian_proposals(
                center,
                gradient=local_quadratic_proposal.gradient,
                hessian=local_quadratic_proposal.hessian,
                temperature=float(self.config.temperature),
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=proposal_count,
                generator=generator,
                defensive_fraction=float(
                    local_quadratic_proposal.defensive_fraction
                ),
                damping=float(local_quadratic_proposal.damping),
            )
        elif quadratic_smoothness_proposal is not None:
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
                    num_samples=two_center_sample_count,
                    generator=generator,
                    residual_bounds=(
                        quadratic_smoothness_proposal.residual_bounds
                    ),
                    constrained_dims=(
                        quadratic_smoothness_proposal.constrained_dims
                    ),
                    alternate_center=(
                        quadratic_smoothness_proposal.alternate_center
                    ),
                    alternate_mixture_fraction=float(
                        quadratic_smoothness_proposal.alternate_mixture_fraction
                    ),
                    defensive_mean_shift_fn=(
                        quadratic_smoothness_proposal.defensive_mean_shift_fn
                    ),
                    defensive_mixture_fraction=float(
                        quadratic_smoothness_proposal.defensive_mixture_fraction
                    ),
                    difference_order=int(
                        quadratic_smoothness_proposal.difference_order
                    ),
                    first_difference_weight=float(
                        quadratic_smoothness_proposal.first_difference_weight
                    ),
                    proposal_smoothness_weight_multiplier=float(
                        quadratic_smoothness_proposal.proposal_smoothness_weight_multiplier
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
                num_samples=two_center_sample_count,
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
                        include_center=(
                            False
                            if proposal_mixture_include_centers is None
                            else bool(proposal_mixture_include_centers[index])
                        ),
                    )
                    for index, (mixture_center, count) in enumerate(zip(
                        proposal_mixture_centers,
                        sampling_mixture_counts,
                        strict=True,
                    ))
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
                num_samples=two_center_sample_count,
                sampler=self.config.proposal_sampler,
                generator=generator,
                include_center=False,
            )
            shifted_candidates = sample_trajectory_proposals(
                center,
                scale=float(proposal_scale),
                lower=lower,
                upper=upper,
                num_samples=two_center_sample_count,
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
                num_samples=prefilter_pool_count,
                sampler=self.config.proposal_sampler,
                generator=generator,
                include_center=target_center is None,
            )
        if candidate_transform_fn is not None:
            transformed, transform_diagnostics = candidate_transform_fn(
                candidates, lower, upper
            )
            if transformed.shape != candidates.shape:
                raise ValueError("candidate transform must preserve proposal shape")
            if not bool(torch.all(torch.isfinite(transformed)).item()):
                raise ValueError("candidate transform produced non-finite values")
            # Candidate transforms may solve their feasible centers in
            # float64 and cast back to float32. Match the bridge-side
            # roundoff allowance; the transform's own support audit remains
            # tighter than 2e-5.
            tolerance = 256.0 * torch.finfo(transformed.dtype).eps
            if bool(torch.any(transformed < lower - tolerance).item()) or bool(
                torch.any(transformed > upper + tolerance).item()
            ):
                lower_violation = float(
                    torch.clamp(lower - transformed, min=0.0).max().item()
                )
                upper_violation = float(
                    torch.clamp(transformed - upper, min=0.0).max().item()
                )
                raise ValueError(
                    "candidate transform violated hard joint bounds: "
                    f"lower={lower_violation:.6g}, "
                    f"upper={upper_violation:.6g}"
                )
            candidates = transformed
            proposal_diagnostics["candidate_transform"] = transform_diagnostics
        fresh_candidates = candidates
        persistent_component_centers: tuple[torch.Tensor, ...] | None = None
        persistent_component_scales: tuple[float, ...] | None = None
        persistent_component_counts: tuple[int, ...] | None = None
        persistent_previous_count = 0
        if persistent_proposal_bank is not None:
            if any(
                value is not None
                for value in (
                    proposal_log_acceptance_fn,
                    proposal_prefilter_cost_fn,
                    quadratic_smoothness_proposal,
                    local_quadratic_proposal,
                    autoregressive_initial,
                    proposal_mixture_center,
                )
            ) or mixture_with_target:
                raise ValueError(
                    "persistent proposals require bounded-Gaussian proposal "
                    "components without tilting, prefiltering, local "
                    "quadratics, autoregression, or implicit mixtures"
                )
            persistent_previous_count = persistent_proposal_bank.sample_count
            if persistent_proposal_bank.candidate_batches:
                candidates = torch.cat(
                    (*persistent_proposal_bank.candidate_batches, candidates),
                    dim=1,
                )
            if first_order_proposal is not None:
                raw_shift = (
                    -float(proposal_scale) ** 2
                    * first_order_proposal.gradient
                    / float(self.config.temperature)
                )
                raw_norm = torch.linalg.vector_norm(
                    raw_shift.flatten(start_dim=1), dim=1
                ).clamp_min(1.0e-12)
                maximum_norm = (
                    float(first_order_proposal.max_shift_standard_deviations)
                    * float(proposal_scale)
                )
                trust = torch.clamp(maximum_norm / raw_norm, max=1.0)
                shifted_center = center + raw_shift * trust.reshape(-1, 1, 1)
                if first_order_proposal.shifted_center_override is not None:
                    shifted_center = first_order_proposal.shifted_center_override.to(
                        device=center.device, dtype=center.dtype
                    )
                defensive_count = max(1, min(
                    int(proposal_count) - 1,
                    int(round(
                        float(proposal_count)
                        * float(first_order_proposal.defensive_fraction)
                    )),
                ))
                fresh_component_centers = (center, shifted_center)
                fresh_component_scales = (
                    float(proposal_scale),
                    float(proposal_scale)
                    * float(first_order_proposal.shifted_scale_multiplier),
                )
                fresh_component_counts = (
                    defensive_count,
                    int(proposal_count) - defensive_count,
                )
            else:
                fresh_component_centers = (
                    tuple(proposal_mixture_centers)
                    if multi_center_mixture
                    else (center,)
                )
                fresh_component_scales = tuple(
                    float(proposal_scale) for _ in fresh_component_centers
                )
                fresh_component_counts = (
                    tuple(int(count) for count in sampling_mixture_counts)
                    if multi_center_mixture
                    else (int(fresh_candidates.shape[1]),)
                )
            persistent_component_centers = tuple(
                (*persistent_proposal_bank.component_centers, *fresh_component_centers)
            )
            persistent_component_scales = tuple(
                (
                    *persistent_proposal_bank.component_scales,
                    *fresh_component_scales,
                )
            )
            persistent_component_counts = tuple(
                (*persistent_proposal_bank.component_counts, *fresh_component_counts)
            )
        if proposal_prefilter_cost_fn is not None:
            candidates = candidates[:, :prefilter_pool_count]
            proxy_costs = proposal_prefilter_cost_fn(candidates)
            if proxy_costs.shape != candidates.shape[:2]:
                raise ValueError("proposal prefilter callback must return [P,N]")
            resolved_prefilter_ess = (
                min(float(prefilter_pool_count), 4.0 * float(retained_proposal_count))
                if proposal_prefilter_target_ess is None
                else float(proposal_prefilter_target_ess)
            )
            candidates, prefilter_log_tilt, prefilter_multiplicities, prefilter_diagnostics = (
                resample_proxy_tilted_candidates(
                    candidates,
                    proxy_costs,
                    retain_count=retained_proposal_count,
                    generator=generator,
                    target_ess=resolved_prefilter_ess,
                )
            )
            proposal_diagnostics.update(prefilter_diagnostics)
        cost_candidates = (
            fresh_candidates
            if persistent_proposal_bank is not None
            else candidates
        )
        evaluated_candidates = cost_candidates
        if evaluation_only_candidate is not None:
            evaluated_candidates = torch.cat(
                (cost_candidates, evaluation_only_candidate[:, None]), dim=1
            )
        costs_value = cost_fn(evaluated_candidates)
        eligible_value = None
        if isinstance(costs_value, ProposalCostResult):
            eligible_value = costs_value.eligible
            costs_value = costs_value.costs
        if torch.is_tensor(costs_value):
            costs = costs_value.detach().cpu().numpy()
        else:
            costs = np.asarray(costs_value)
        evaluated_expected = evaluated_candidates.shape[:2]
        if costs.shape != evaluated_expected:
            raise ValueError(
                "cost_fn must match evaluated candidates; expected "
                f"{tuple(evaluated_expected)}, got {tuple(costs.shape)}"
            )
        evaluation_only_cost = None
        if evaluation_only_candidate is not None:
            evaluation_only_cost = costs[:, -1].copy()
            costs = costs[:, :-1]
            if eligible_value is not None:
                eligible_value = eligible_value[:, :-1]
        fresh_costs = np.asarray(costs).copy()
        fresh_eligible = None
        if eligible_value is not None:
            fresh_eligible = (
                eligible_value.detach().cpu().numpy()
                if torch.is_tensor(eligible_value)
                else np.asarray(eligible_value)
            ).astype(bool, copy=False)
        if persistent_proposal_bank is not None:
            costs = np.concatenate(
                (*persistent_proposal_bank.cost_batches, fresh_costs), axis=1
            )
            if len(fresh_component_counts) > 1:
                split_indices = np.cumsum(fresh_component_counts[:-1])
                fresh_eligibility_batches = tuple(
                    None
                    if fresh_eligible is None
                    else np.asarray(value, dtype=bool)
                    for value in (
                        (None,) * len(fresh_component_counts)
                        if fresh_eligible is None
                        else np.split(fresh_eligible, split_indices, axis=1)
                    )
                )
            else:
                fresh_eligibility_batches = (fresh_eligible,)
            eligibility_batches = (
                *persistent_proposal_bank.eligible_batches,
                *fresh_eligibility_batches,
            )
            if any(value is not None for value in eligibility_batches):
                eligible_value = np.concatenate(
                    tuple(
                        value
                        if value is not None
                        else np.ones(
                            (int(candidates.shape[0]), int(count)), dtype=bool
                        )
                        for value, count in zip(
                            eligibility_batches,
                            persistent_component_counts,
                            strict=True,
                        )
                    ),
                    axis=1,
                )
            else:
                eligible_value = None
        expected = candidates.shape[:2]
        reward_logits = -costs / float(self.config.temperature)
        log_density_ratio = None
        if prefilter_log_tilt is not None:
            if prefilter_multiplicities is None:
                raise RuntimeError("proxy prefilter lost multiplicities")
            proxy_ratio = (
                -prefilter_log_tilt
                + torch.log(prefilter_multiplicities.to(prefilter_log_tilt.dtype))
            ).detach().cpu().numpy()
            reward_logits = reward_logits + proxy_ratio
            log_density_ratio = proxy_ratio
        if tilt_log_acceptance is not None:
            # q(y) is proportional to p(y) * g(y); the common rejection
            # normalizer cancels under self-normalized importance sampling.
            log_density_ratio = (
                -tilt_log_acceptance.detach().cpu().numpy()
            )
            reward_logits = reward_logits + log_density_ratio
        if smoothness_log_density_ratio is not None:
            log_density_ratio = (
                smoothness_log_density_ratio.detach().cpu().numpy()
            )
            reward_logits = reward_logits + log_density_ratio
        if (
            first_order_log_density_ratio is not None
            and persistent_proposal_bank is None
        ):
            first_order_ratio = (
                first_order_log_density_ratio.detach().cpu().numpy()
            )
            reward_logits = reward_logits + first_order_ratio
            log_density_ratio = (
                first_order_ratio
                if log_density_ratio is None
                else log_density_ratio + first_order_ratio
            )
        if local_quadratic_log_density_ratio is not None:
            local_ratio = (
                local_quadratic_log_density_ratio.detach().cpu().numpy()
            )
            reward_logits = reward_logits + local_ratio
            log_density_ratio = (
                local_ratio
                if log_density_ratio is None
                else log_density_ratio + local_ratio
            )
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
            if persistent_component_centers is not None:
                total_samples = float(sum(persistent_component_counts))
                component_log_probs = torch.stack(
                    tuple(
                        bounded_gaussian_log_prob(
                            candidates,
                            center=component_center,
                            scale=float(component_scale),
                            lower=lower,
                            upper=upper,
                            sampler=self.config.proposal_sampler,
                        )
                        + math.log(float(count) / total_samples)
                        for component_center, component_scale, count in zip(
                            persistent_component_centers,
                            persistent_component_scales,
                            persistent_component_counts,
                            strict=True,
                        )
                    ),
                    dim=0,
                )
                importance_log_prob = torch.logsumexp(
                    component_log_probs, dim=0
                )
            elif multi_center_mixture:
                total_samples = float(sum(sampling_mixture_counts))
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
                            sampling_mixture_counts,
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
            target_ratio = (
                target_log_prob - importance_log_prob
            ).detach().cpu().numpy()
            reward_logits = reward_logits + target_ratio
            log_density_ratio = (
                target_ratio
                if log_density_ratio is None
                else log_density_ratio + target_ratio
            )
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
        direct_importance_mean = mean
        control_variate_diagnostics: dict[str, Any] = {
            "first_order_control_variate": False,
            "reason": "not_requested",
        }
        quality_guard_evaluations = 0
        proxy_guard_diagnostics: dict[str, Any] = {
            "first_order_proxy_quality_guard": "not_applicable",
        }
        if (
            bool(self.config.first_order_proxy_exact_quality_guard)
            and first_order_proposal is not None
            and first_order_proposal.shifted_center_override is not None
            and first_order_log_density_ratio is not None
            and persistent_proposal_bank is None
        ):
            # The first component is sampled directly from the unchanged FM
            # likelihood. Its standalone SNIS mean and the exact-mixture SNIS
            # mean are both consistent for the same Tweedie posterior. Select
            # between them with the authoritative nonlinear cost, preventing
            # a useful proxy mode from worsening finite-sample reward without
            # changing the asymptotic score.
            defensive_count = int(proposal_diagnostics["defensive_count"])
            defensive_candidates = candidates[:, :defensive_count]
            defensive_logits = (
                -np.asarray(costs[:, :defensive_count], dtype=np.float64)
                / float(self.config.temperature)
            )
            defensive_available = np.ones(center.shape[0], dtype=bool)
            if eligible_value is not None:
                defensive_eligible = np.asarray(
                    eligible_value[:, :defensive_count], dtype=bool
                )
                defensive_available = np.any(defensive_eligible, axis=1)
                defensive_logits = np.where(
                    defensive_eligible, defensive_logits, -np.inf
                )
            defensive_weights_np = np.stack(
                [normalized_weights(row) for row in defensive_logits], axis=0
            )
            defensive_weights = torch.as_tensor(
                defensive_weights_np, device=center.device, dtype=center.dtype
            )
            defensive_mean = torch.sum(
                defensive_weights[..., None, None] * defensive_candidates,
                dim=1,
            )
            comparison_candidates = torch.stack(
                (defensive_mean, direct_importance_mean), dim=1
            )
            comparison_value = cost_fn(comparison_candidates)
            comparison_eligible = None
            if isinstance(comparison_value, ProposalCostResult):
                comparison_eligible = comparison_value.eligible
                comparison_value = comparison_value.costs
            comparison_costs = (
                comparison_value.detach().cpu().numpy()
                if torch.is_tensor(comparison_value)
                else np.asarray(comparison_value)
            )
            if comparison_costs.shape != candidates.shape[:1] + (2,):
                raise ValueError("proxy quality guard cost must return [P,2]")
            mixture_eligible = np.ones(center.shape[0], dtype=bool)
            if comparison_eligible is not None:
                comparison_eligible = (
                    comparison_eligible.detach().cpu().numpy()
                    if torch.is_tensor(comparison_eligible)
                    else np.asarray(comparison_eligible)
                )
                if comparison_eligible.shape != comparison_costs.shape:
                    raise ValueError(
                        "proxy quality guard eligibility must return [P,2]"
                    )
                defensive_available &= comparison_eligible[:, 0].astype(
                    bool, copy=False
                )
                mixture_eligible = comparison_eligible[:, 1].astype(
                    bool, copy=False
                )
            select_defensive = (
                defensive_available
                & np.isfinite(comparison_costs[:, 0])
                & (
                    ~mixture_eligible
                    | (comparison_costs[:, 0] <= comparison_costs[:, 1] + 1.0e-12)
                )
            )
            mean = torch.where(
                torch.as_tensor(
                    select_defensive, device=center.device, dtype=torch.bool
                ).reshape(-1, 1, 1),
                defensive_mean,
                direct_importance_mean,
            )
            direct_importance_mean = mean
            quality_guard_evaluations += int(2 * center.shape[0])
            proxy_guard_diagnostics = {
                "first_order_proxy_quality_guard": "exact_cost_nonincrease",
                "first_order_proxy_defensive_cost": comparison_costs[:, 0].tolist(),
                "first_order_proxy_mixture_cost": comparison_costs[:, 1].tolist(),
                "first_order_proxy_selected_cost": np.where(
                    select_defensive, comparison_costs[:, 0], comparison_costs[:, 1]
                ).tolist(),
                "first_order_proxy_selected_defensive": select_defensive.tolist(),
                "first_order_proxy_defensive_samples": defensive_count,
            }
        replicated_groups = min(
            int(self.config.replicated_snis_groups), candidates.shape[1]
        )
        replicated_diagnostics: dict[str, Any] = {
            "replicated_snis_groups": replicated_groups,
            "replicated_snis_quality_guard_accepted": None,
        }
        if replicated_groups > 1:
            # Interleaving keeps every replicated estimate representative when
            # proposal components occupy contiguous slices of the population.
            group_means = []
            group_sizes = []
            for group_index in range(replicated_groups):
                group_logits = reward_logits[:, group_index::replicated_groups]
                group_candidates = candidates[:, group_index::replicated_groups]
                group_weights_np = np.stack(
                    [normalized_weights(row) for row in group_logits], axis=0
                )
                group_weights = torch.as_tensor(
                    group_weights_np, device=center.device, dtype=center.dtype
                )
                group_means.append(torch.sum(
                    group_weights[..., None, None] * group_candidates, dim=1
                ))
                group_sizes.append(int(group_candidates.shape[1]))
            replicated_mean = torch.stack(group_means, dim=1).mean(dim=1)
            blend = float(self.config.replicated_snis_blend)
            replicated_mean = (
                (1.0 - blend) * direct_importance_mean
                + blend * replicated_mean
            )
            replicated_diagnostics["replicated_snis_blend"] = blend
            if not self.config.replicated_snis_exact_quality_guard:
                mean = replicated_mean
                replicated_diagnostics.update({
                    "replicated_snis_group_sizes": group_sizes,
                    "replicated_snis_quality_guard": "disabled",
                })
            else:
                comparison_candidates = torch.stack(
                    (direct_importance_mean, replicated_mean), dim=1
                )
                comparison_value = cost_fn(comparison_candidates)
                comparison_eligible = None
                if isinstance(comparison_value, ProposalCostResult):
                    comparison_eligible = comparison_value.eligible
                    comparison_value = comparison_value.costs
                comparison_costs = (
                    comparison_value.detach().cpu().numpy()
                    if torch.is_tensor(comparison_value)
                    else np.asarray(comparison_value)
                )
                if comparison_costs.shape != candidates.shape[:1] + (2,):
                    raise ValueError("replicated SNIS guard cost must return [P,2]")
                direct_eligible = np.ones(center.shape[0], dtype=bool)
                replicated_eligible = np.ones(center.shape[0], dtype=bool)
                if comparison_eligible is not None:
                    comparison_eligible = (
                        comparison_eligible.detach().cpu().numpy()
                        if torch.is_tensor(comparison_eligible)
                        else np.asarray(comparison_eligible)
                    )
                    if comparison_eligible.shape != comparison_costs.shape:
                        raise ValueError(
                            "replicated SNIS guard eligibility must return [P,2]"
                        )
                    direct_eligible = comparison_eligible[:, 0].astype(
                        bool, copy=False
                    )
                    replicated_eligible = comparison_eligible[:, 1].astype(
                        bool, copy=False
                    )
                accept_replicated = (
                    replicated_eligible
                    & np.isfinite(comparison_costs[:, 1])
                    & (
                        ~direct_eligible
                        | (
                            comparison_costs[:, 1]
                            <= comparison_costs[:, 0] + 1.0e-12
                        )
                    )
                )
                mean = torch.where(
                    torch.as_tensor(
                        accept_replicated, device=center.device, dtype=torch.bool
                    ).reshape(-1, 1, 1),
                    replicated_mean,
                    direct_importance_mean,
                )
                quality_guard_evaluations += int(2 * center.shape[0])
                replicated_diagnostics.update({
                    "replicated_snis_group_sizes": group_sizes,
                    "replicated_snis_direct_cost": comparison_costs[:, 0].tolist(),
                    "replicated_snis_candidate_cost": comparison_costs[:, 1].tolist(),
                    "replicated_snis_selected_cost": np.where(
                        accept_replicated,
                        comparison_costs[:, 1],
                        comparison_costs[:, 0],
                    ).tolist(),
                    "replicated_snis_quality_guard_accepted": (
                        accept_replicated.tolist()
                    ),
                })
        if (
            first_order_proposal is not None
            and first_order_log_density_ratio is not None
            and first_order_proposal.use_control_variate
        ):
            corrected_mean, control_variate_diagnostics = (
                _cross_fitted_first_order_control_variate_mean(
                    candidates,
                    reward_logits=reward_logits,
                    log_base_over_proposal=first_order_log_density_ratio,
                    center=center,
                    scale=float(proposal_scale),
                    lower=lower,
                    upper=upper,
                    temperature=float(self.config.temperature),
                    proposal=first_order_proposal,
                )
            )
            if corrected_mean is not None:
                comparison_candidates = torch.stack(
                    (direct_importance_mean, corrected_mean), dim=1
                )
                comparison_value = cost_fn(comparison_candidates)
                comparison_eligible = None
                if isinstance(comparison_value, ProposalCostResult):
                    comparison_eligible = comparison_value.eligible
                    comparison_value = comparison_value.costs
                comparison_costs = (
                    comparison_value.detach().cpu().numpy()
                    if torch.is_tensor(comparison_value)
                    else np.asarray(comparison_value)
                )
                if comparison_costs.shape != candidates.shape[:1] + (2,):
                    raise ValueError(
                        "quality guard cost callback must return [P,2]"
                    )
                direct_eligible = np.ones(center.shape[0], dtype=bool)
                corrected_eligible = np.ones(center.shape[0], dtype=bool)
                if comparison_eligible is not None:
                    comparison_eligible = (
                        comparison_eligible.detach().cpu().numpy()
                        if torch.is_tensor(comparison_eligible)
                        else np.asarray(comparison_eligible)
                    )
                    if comparison_eligible.shape != comparison_costs.shape:
                        raise ValueError(
                            "quality guard eligibility must return [P,2]"
                        )
                    direct_eligible = comparison_eligible[:, 0].astype(
                        bool, copy=False
                    )
                    corrected_eligible = comparison_eligible[:, 1].astype(
                        bool, copy=False
                    )
                accept = (
                    corrected_eligible
                    & np.isfinite(comparison_costs[:, 1])
                    & (
                        ~direct_eligible
                        | (
                            comparison_costs[:, 1]
                            <= comparison_costs[:, 0] + 1.0e-12
                        )
                    )
                )
                mean = torch.where(
                    torch.as_tensor(
                        accept, device=center.device, dtype=torch.bool
                    ).reshape(-1, 1, 1),
                    corrected_mean,
                    direct_importance_mean,
                )
                quality_guard_evaluations += int(2 * center.shape[0])
                control_variate_diagnostics.update({
                    "quality_guard": "exact_cost_nonincrease",
                    "quality_guard_direct_cost": comparison_costs[:, 0].tolist(),
                    "quality_guard_corrected_cost": comparison_costs[:, 1].tolist(),
                    "quality_guard_selected_cost": np.where(
                        accept, comparison_costs[:, 1], comparison_costs[:, 0]
                    ).tolist(),
                    "quality_guard_direct_eligible": direct_eligible.tolist(),
                    "quality_guard_corrected_eligible": corrected_eligible.tolist(),
                    "quality_guard_accepted": accept.tolist(),
                })
        raw_variance = torch.var(candidates, dim=1, correction=0)
        weighted_variance = torch.sum(
            weights[..., None, None] * (candidates - mean[:, None]).square(),
            dim=1,
        )
        variance_physical_scale = (
            quadratic_smoothness_proposal.physical_scale
            if quadratic_smoothness_proposal is not None
            else None
        )
        if variance_physical_scale is not None:
            variance_physical_scale = variance_physical_scale.to(
                device=candidates.device, dtype=candidates.dtype
            ).reshape(1, 1, -1)
            raw_physical_variance = (
                raw_variance * variance_physical_scale.square()
            )
            weighted_physical_variance = (
                weighted_variance * variance_physical_scale.square()
            )
            raw_physical_rms_std_by_waypoint = torch.sqrt(
                raw_physical_variance.mean(dim=-1)
            )
            weighted_physical_rms_std_by_waypoint = torch.sqrt(
                weighted_physical_variance.mean(dim=-1)
            )
        else:
            raw_physical_rms_std_by_waypoint = None
            weighted_physical_rms_std_by_waypoint = None
        if prefilter_multiplicities is None:
            ess = 1.0 / np.sum(np.square(weights_np), axis=1)
        else:
            multiplicity_np = prefilter_multiplicities.detach().cpu().numpy()
            # Expanded-draw ESS: each compressed aggregate weight w_i stands
            # for n_i identical per-draw weights w_i / n_i.
            ess = 1.0 / np.sum(
                np.square(weights_np) / multiplicity_np, axis=1
            )
            proposal_diagnostics["proposal_prefilter_expanded_final_ess"] = (
                ess.tolist()
            )
        finite_costs = np.where(np.isfinite(costs), costs, np.inf)
        maximum_weight_indices = np.argmax(weights_np, axis=1)
        best_cost_indices = np.argmin(finite_costs, axis=1)
        defensive_shifted_center_index = (
            int(proposal_diagnostics["defensive_nominal_count"])
            if proposal_diagnostics.get("defensive_mixture", False)
            else None
        )
        diagnostics = {
            **replicated_diagnostics,
            "sampler": (
                "first_order_gaussian_exact_importance"
                if first_order_proposal is not None
                else (
                    "local_quadratic_gaussian_exact_importance"
                    if local_quadratic_proposal is not None
                    else (
                        "quadratic_smoothness_gaussian_residual_importance"
                        if quadratic_smoothness_proposal is not None
                        else (
                            "autoregressive_p99_truncated_importance"
                            if autoregressive
                            else "direct_importance"
                        )
                    )
                )
            ),
            "beta_schedule": [0.0, 1.0 / float(self.config.temperature)],
            "ess_per_stage": [ess.tolist()],
            "final_ess": ess.tolist(),
            "mala_acceptance_rate": None,
            "best_cost": np.min(finite_costs, axis=1).tolist(),
            "weighted_cost": np.sum(weights_np * costs, axis=1).tolist(),
            "evaluation_only_cost": (
                None
                if evaluation_only_cost is None
                else evaluation_only_cost.tolist()
            ),
            "maximum_weight": np.max(weights_np, axis=1).tolist(),
            "maximum_weight_index": maximum_weight_indices.tolist(),
            "best_cost_index": best_cost_indices.tolist(),
            "proposal_zero_weight": weights_np[:, 0].tolist(),
            "proposal_zero_cost": costs[:, 0].tolist(),
            "proposal_zero_is_maximum_weight": (
                maximum_weight_indices == 0
            ).tolist(),
            "proposal_zero_is_best_cost": (
                best_cost_indices == 0
            ).tolist(),
            "nonpositive_residual_cost_fraction": np.mean(
                costs <= 0.0, axis=1
            ).tolist(),
            "defensive_shifted_center_index": defensive_shifted_center_index,
            "defensive_shifted_center_cost": (
                None
                if defensive_shifted_center_index is None
                else costs[:, defensive_shifted_center_index].tolist()
            ),
            "defensive_shifted_center_weight": (
                None
                if defensive_shifted_center_index is None
                else weights_np[:, defensive_shifted_center_index].tolist()
            ),
            "raw_normalized_rms_std_by_waypoint": torch.sqrt(
                raw_variance.mean(dim=-1)
            ).detach().cpu().tolist(),
            "weighted_normalized_rms_std_by_waypoint": torch.sqrt(
                weighted_variance.mean(dim=-1)
            ).detach().cpu().tolist(),
            "raw_physical_rms_std_by_waypoint": (
                None
                if raw_physical_rms_std_by_waypoint is None
                else raw_physical_rms_std_by_waypoint.detach().cpu().tolist()
            ),
            "weighted_physical_rms_std_by_waypoint": (
                None
                if weighted_physical_rms_std_by_waypoint is None
                else weighted_physical_rms_std_by_waypoint.detach().cpu().tolist()
            ),
            "proposal_zero_candidate": (
                candidates[:, 0].detach().cpu()
                if self.config.record_proposal_zero_candidate
                else None
            ),
            "wall_clock_seconds": time.perf_counter() - started,
            "full_cost_evaluation_calls": 1 + (
                quality_guard_evaluations // max(2 * center.shape[0], 1)
            ),
            "full_cost_particle_evaluations": int(
                cost_candidates.shape[0] * cost_candidates.shape[1]
            ) + quality_guard_evaluations,
            "cached_cost_particle_evaluations": int(
                candidates.shape[0] * persistent_previous_count
            ),
            "gradient_cost_evaluation_calls": 0,
            "gradient_particle_evaluations": 0,
            "importance_density_correction": (
                target_center is not None
                or smoothness_log_density_ratio is not None
                or first_order_log_density_ratio is not None
                or local_quadratic_log_density_ratio is not None
                or tilt_log_acceptance is not None
            ),
            "importance_proposal": (
                "bounded_gaussian_mixture_times_log_acceptance_tilt"
                if tilt_log_acceptance is not None
                else "fm_gaussian_times_first_order_tilt_mixture"
                if first_order_proposal is not None
                else "fm_gaussian_times_quadratic_smoothness"
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
            "proposal_mixture_center_injection": (
                None
                if proposal_mixture_include_centers is None
                else tuple(bool(value) for value in proposal_mixture_include_centers)
            ),
            "importance_total_samples": int(candidates.shape[1]),
            "persistent_proposals": persistent_proposal_bank is not None,
            "persistent_previous_samples": int(persistent_previous_count),
            "persistent_fresh_samples": int(fresh_candidates.shape[1]),
            "persistent_component_count": (
                0
                if persistent_component_counts is None
                else len(persistent_component_counts)
            ),
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
        diagnostics.update(proxy_guard_diagnostics)
        diagnostics.update(control_variate_diagnostics)
        if persistent_proposal_bank is not None:
            offset = 0
            for component_center, component_scale, component_count in zip(
                fresh_component_centers,
                fresh_component_scales,
                fresh_component_counts,
                strict=True,
            ):
                stop = offset + int(component_count)
                persistent_proposal_bank.append(
                    fresh_candidates[:, offset:stop],
                    center=component_center,
                    scale=float(component_scale),
                    costs=fresh_costs[:, offset:stop],
                    eligible=(
                        None
                        if fresh_eligible is None
                        else fresh_eligible[:, offset:stop]
                    ),
                )
                offset = stop
        if int(diagnostic_repeats) > 1:
            repeat_started = time.perf_counter()
            repeated_means = [mean.detach()]
            repeated_ess = [np.asarray(ess, dtype=np.float64)]
            repeated_latencies = [float(diagnostics["wall_clock_seconds"])]
            repeated_best_costs = [diagnostics.get("best_cost")]
            repeated_weighted_costs = [diagnostics.get("weighted_cost")]
            modulus = 2**63 - 1
            base_seed = int(generator.initial_seed())
            for repeat_index in range(1, int(diagnostic_repeats)):
                repeat_generator = torch.Generator(device=center.device)
                repeat_generator.manual_seed(
                    (base_seed + 2_654_435_761 * repeat_index) % modulus
                )
                repeated = self.optimize_clean_trajectories(
                    center,
                    lower=lower,
                    upper=upper,
                    cost_fn=cost_fn,
                    generator=repeat_generator,
                    proposal_scale=float(proposal_scale),
                    proposals_per_particle=proposal_count,
                    gradient_cost_fn=gradient_cost_fn,
                    target_center=target_center,
                    target_scale=target_scale,
                    mixture_with_target=mixture_with_target,
                    proposal_mixture_center=proposal_mixture_center,
                    proposal_mixture_centers=proposal_mixture_centers,
                    proposal_mixture_counts=proposal_mixture_counts,
                    proposal_mixture_labels=proposal_mixture_labels,
                    autoregressive_initial=autoregressive_initial,
                    autoregressive_terminal=autoregressive_terminal,
                    autoregressive_residual_bounds=autoregressive_residual_bounds,
                    autoregressive_dims=int(autoregressive_dims),
                    quadratic_smoothness_proposal=quadratic_smoothness_proposal,
                    first_order_proposal=first_order_proposal,
                    local_quadratic_proposal=local_quadratic_proposal,
                    proposal_log_acceptance_fn=proposal_log_acceptance_fn,
            proposal_prefilter_cost_fn=proposal_prefilter_cost_fn,
            proposal_prefilter_pool_size=proposal_prefilter_pool_size,
            proposal_prefilter_target_ess=proposal_prefilter_target_ess,
                    evaluation_only_candidate=evaluation_only_candidate,
                    diagnostic_repeats=1,
                )
                repeated_means.append(repeated.mean.detach())
                repeated_ess.append(
                    np.asarray(repeated.effective_sample_size, dtype=np.float64)
                )
                repeated_diagnostics = repeated.diagnostics or {}
                repeated_best_costs.append(repeated_diagnostics.get("best_cost"))
                repeated_weighted_costs.append(
                    repeated_diagnostics.get("weighted_cost")
                )
                repeated_latencies.append(
                    float((repeated.diagnostics or {})["wall_clock_seconds"])
                )
            means = torch.stack(repeated_means, dim=0)
            empirical_variance = torch.var(means, dim=0, correction=1)
            normalized_rms = torch.sqrt(empirical_variance.mean(dim=-1))
            physical_rms = None
            if variance_physical_scale is not None:
                physical_variance = (
                    empirical_variance * variance_physical_scale.square()
                )
                physical_rms = torch.sqrt(physical_variance.mean(dim=-1))
            repeated_mean_candidates = means.permute(1, 0, 2, 3).contiguous()
            repeated_mean_cost_value = cost_fn(repeated_mean_candidates)
            if isinstance(repeated_mean_cost_value, ProposalCostResult):
                repeated_mean_cost_value = repeated_mean_cost_value.costs
            repeated_mean_cost = (
                repeated_mean_cost_value.detach().cpu().numpy()
                if torch.is_tensor(repeated_mean_cost_value)
                else np.asarray(repeated_mean_cost_value)
            )
            if repeated_mean_cost.shape != repeated_mean_candidates.shape[:2]:
                raise ValueError("repeated mean cost callback returned wrong shape")
            diagnostics.update({
                "x0_prediction_repeat_count": int(diagnostic_repeats),
                "x0_prediction_repeat_means_normalized": (
                    means.detach().cpu().tolist()
                ),
                "x0_prediction_repeat_best_cost": repeated_best_costs,
                "x0_prediction_repeat_weighted_cost": repeated_weighted_costs,
                "x0_prediction_repeat_mean_cost": repeated_mean_cost.tolist(),
                "x0_prediction_empirical_normalized_rms_std_by_waypoint": (
                    normalized_rms.detach().cpu().tolist()
                ),
                "x0_prediction_empirical_physical_rms_std_by_waypoint": (
                    None if physical_rms is None
                    else physical_rms.detach().cpu().tolist()
                ),
                "x0_prediction_repeat_ess": (
                    np.stack(repeated_ess, axis=0).tolist()
                ),
                "x0_prediction_repeat_latency_seconds": repeated_latencies,
                "x0_prediction_diagnostic_wall_clock_seconds": (
                    time.perf_counter() - repeat_started
                ),
            })
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
        first_order_gradient_cost_fn: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None,
        first_order_defensive_fraction: float = 0.25,
        first_order_max_shift_standard_deviations: float = 0.75,
        first_order_proposal_override: FirstOrderGaussianProposal | None = None,
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
        local_quadratic_proposal: LocalQuadraticGaussianProposal | None = None,
        proposal_log_acceptance_fn: ProposalLogAcceptance | None = None,
        proposal_prefilter_cost_fn: ProposalPrefilterCost | None = None,
        proposal_prefilter_pool_size: int | None = None,
        proposal_prefilter_target_ess: float | None = None,
        evaluation_only_candidate: torch.Tensor | None = None,
        diagnostic_repeats: int = 1,
        persistent_proposal_bank: PersistentProposalBank | None = None,
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
        if (
            first_order_gradient_cost_fn is not None
            and first_order_proposal_override is not None
        ):
            raise ValueError(
                "first-order gradient and cached proposal override are exclusive"
            )
        first_order_proposal = first_order_proposal_override
        first_order_gradient_seconds = 0.0
        if first_order_gradient_cost_fn is not None:
            if local_quadratic_proposal is not None:
                raise ValueError(
                    "first-order and local-quadratic proposals are exclusive"
                )
            if any(value is not None for value in (
                quadratic_smoothness_proposal,
                proposal_log_acceptance_fn,
                proposal_prefilter_cost_fn,
            )):
                raise ValueError(
                    "first-order gradients are exclusive with another "
                    "specialized proposal"
                )
            gradient_center = (
                importance_target_center
                if importance_target_center is not None
                else resolved_proposal_center
            )
            # The Gaussian mean may lie outside the executable box late in the
            # reverse solve. Linearize at its projection, but keep the original
            # mean as the exact importance target.
            gradient_point = torch.maximum(
                torch.minimum(gradient_center, upper), lower
            ).detach().requires_grad_(True)
            gradient_started = time.perf_counter()
            with torch.enable_grad():
                gradient_cost = first_order_gradient_cost_fn(
                    gradient_point[:, None]
                )
                if gradient_cost.shape != gradient_point.shape[:1] + (1,):
                    raise ValueError(
                        "first-order gradient callback must return [P,1]"
                    )
                gradient = torch.autograd.grad(
                    gradient_cost.sum(), gradient_point, create_graph=False
                )[0]
            first_order_gradient_seconds = time.perf_counter() - gradient_started
            first_order_proposal = FirstOrderGaussianProposal(
                gradient=gradient.detach(),
                cost_at_reference=gradient_cost.detach()[:, 0],
                reference=gradient_point.detach(),
                defensive_fraction=float(first_order_defensive_fraction),
                max_shift_standard_deviations=float(
                    first_order_max_shift_standard_deviations
                ),
            )
        first_order_owns_importance_target = bool(
            first_order_proposal is not None
            and importance_target_center is not None
        )
        local_quadratic_owns_importance_target = bool(
            local_quadratic_proposal is not None
            and importance_target_center is not None
        )
        smc_starts_from_importance_target = bool(
            self.config.inference_sampler == "adaptive_smc"
            and importance_target_center is not None
        )
        optimize_center = (
            importance_target_center
            if (
                smc_starts_from_importance_target
                or first_order_owns_importance_target
                or local_quadratic_owns_importance_target
            )
            else resolved_proposal_center
        )
        optimize_scale = (
            float(importance_target_scale)
            if (
                (
                    smc_starts_from_importance_target
                    or first_order_owns_importance_target
                    or local_quadratic_owns_importance_target
                )
                and importance_target_scale is not None
            )
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
                if (
                    smc_starts_from_importance_target
                    or first_order_owns_importance_target
                    or local_quadratic_owns_importance_target
                )
                else (
                    importance_target_center
                    if importance_target_center is not None
                    else (policy_clean if proposal_center is not None else None)
                )
            ),
            target_scale=(
                None
                if (
                    smc_starts_from_importance_target
                    or first_order_owns_importance_target
                    or local_quadratic_owns_importance_target
                )
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
            first_order_proposal=first_order_proposal,
            local_quadratic_proposal=local_quadratic_proposal,
            proposal_log_acceptance_fn=proposal_log_acceptance_fn,
            proposal_prefilter_cost_fn=proposal_prefilter_cost_fn,
            proposal_prefilter_pool_size=proposal_prefilter_pool_size,
            proposal_prefilter_target_ess=proposal_prefilter_target_ess,
            evaluation_only_candidate=evaluation_only_candidate,
            diagnostic_repeats=int(diagnostic_repeats),
            persistent_proposal_bank=persistent_proposal_bank,
        )
        if first_order_proposal is not None:
            if proposals.diagnostics is None:
                raise RuntimeError("first-order proposal diagnostics are missing")
            proposals.diagnostics["first_order_gradient_wall_clock_seconds"] = (
                first_order_gradient_seconds
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
        keypose_first_order_gradient_cost_fn: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None,
        keypose_first_order_defensive_fraction: float = 0.25,
        keypose_first_order_max_shift_standard_deviations: float = 0.75,
        keypose_first_order_proposal_override: (
            FirstOrderGaussianProposal | None
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
        keypose_proposal_log_acceptance_fn: ProposalLogAcceptance | None = None,
        keypose_proposal_prefilter_cost_fn: ProposalPrefilterCost | None = None,
        keypose_proposal_prefilter_pool_size: int | None = None,
        keypose_proposal_prefilter_target_ess: float | None = None,
        keypose_local_quadratic_proposal: (
            LocalQuadraticGaussianProposal | None
        ) = None,
        keypose_importance_target_center: torch.Tensor | None = None,
        keypose_importance_target_scale: float | None = None,
        keypose_x0_diagnostic_repeats: int = 1,
        keypose_persistent_proposal_bank: PersistentProposalBank | None = None,
        waypoint_autoregressive_initial: torch.Tensor | None = None,
        waypoint_autoregressive_residual_bounds: torch.Tensor | None = None,
        waypoint_autoregressive_dims: int = 7,
        waypoint_smoothness_gaussian_physical_scale: torch.Tensor | None = None,
        waypoint_smoothness_gaussian_weight: float = 0.0,
        waypoint_smoothness_gaussian_alternate_center: torch.Tensor | None = None,
        waypoint_smoothness_gaussian_alternate_mixture_fraction: float = 0.5,
        waypoint_smoothness_gaussian_defensive_mean_shift_fn: (
            GaussianMeanShift | None
        ) = None,
        waypoint_smoothness_gaussian_defensive_mixture_fraction: float = 0.5,
        waypoint_temperature: float | None = None,
        condition_waypoints_on_guided_keypose: bool = False,
        keypose_evaluation_only_candidate: torch.Tensor | None = None,
        waypoint_x0_diagnostic_repeats: int = 1,
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
        if waypoint_temperature is not None and waypoint_temperature <= 0.0:
            raise ValueError("waypoint_temperature must be positive")

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
            first_order_gradient_cost_fn=keypose_first_order_gradient_cost_fn,
            first_order_defensive_fraction=float(
                keypose_first_order_defensive_fraction
            ),
            first_order_max_shift_standard_deviations=float(
                keypose_first_order_max_shift_standard_deviations
            ),
            first_order_proposal_override=keypose_first_order_proposal_override,
            keypose_coefficient=float(keypose_coefficient),
            proposals_per_particle=keypose_proposals_per_particle,
            proposal_center=(
                None if proposal_center is None else proposal_center[:, -1:, :]
            ),
            proposal_center_mixture=bool(proposal_center_mixture),
            proposal_mixture_centers=keypose_proposal_mixture_centers,
            proposal_mixture_counts=keypose_proposal_mixture_counts,
            proposal_mixture_labels=keypose_proposal_mixture_labels,
            proposal_log_acceptance_fn=keypose_proposal_log_acceptance_fn,
            proposal_prefilter_cost_fn=keypose_proposal_prefilter_cost_fn,
            proposal_prefilter_pool_size=keypose_proposal_prefilter_pool_size,
            proposal_prefilter_target_ess=keypose_proposal_prefilter_target_ess,
            local_quadratic_proposal=keypose_local_quadratic_proposal,
            evaluation_only_candidate=keypose_evaluation_only_candidate,
            importance_target_center=(
                keypose_importance_target_center
                if keypose_importance_target_center is not None
                else (
                    keypose_proposal_mixture_centers[0]
                    if keypose_proposal_mixture_centers is not None
                    else None
                )
            ),
            importance_target_scale=keypose_importance_target_scale,
            diagnostic_repeats=int(keypose_x0_diagnostic_repeats),
            persistent_proposal_bank=keypose_persistent_proposal_bank,
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
                alternate_center=waypoint_smoothness_gaussian_alternate_center,
                alternate_mixture_fraction=float(
                    waypoint_smoothness_gaussian_alternate_mixture_fraction
                ),
                defensive_mean_shift_fn=(
                    waypoint_smoothness_gaussian_defensive_mean_shift_fn
                ),
                defensive_mixture_fraction=float(
                    waypoint_smoothness_gaussian_defensive_mixture_fraction
                ),
            )

        waypoint_mbd = self
        if (
            waypoint_temperature is not None
            and float(waypoint_temperature) != float(self.config.temperature)
        ):
            waypoint_mbd = RectifiedFlowMBD(
                replace(
                    self.config,
                    temperature=float(waypoint_temperature),
                )
            )

        waypoints = waypoint_mbd.guide_score(
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
            diagnostic_repeats=int(waypoint_x0_diagnostic_repeats),
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
    "FirstOrderGaussianProposal",
    "ProposalResult",
    "ProposalCostResult",
    "RectifiedFlowMBD",
    "RectifiedFlowMBDConfig",
    "LocalQuadraticGaussianProposal",
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
    "sample_first_order_gaussian_proposals",
    "sample_local_quadratic_gaussian_proposals",
    "sample_trajectory_proposals",
    "straight_line_cspace_path",
]
