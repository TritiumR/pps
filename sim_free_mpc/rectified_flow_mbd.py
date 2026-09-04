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
    if bool(torch.any(standardized_upper <= standardized_lower).item()):
        raise RuntimeError("truncated Gaussian has an empty support interval")
    uniform = torch.rand(
        (int(center.shape[0]), int(num_samples), int(center.shape[1])),
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
        if local_quadratic_proposal is not None:
            if self.config.inference_sampler != "direct":
                raise ValueError("local quadratic proposals require direct IS")
            if (
                autoregressive
                or quadratic_smoothness_proposal is not None
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
            fresh_component_centers = (
                tuple(proposal_mixture_centers)
                if multi_center_mixture
                else (center,)
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
                    *(float(proposal_scale) for _ in fresh_component_centers),
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
            if multi_center_mixture:
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
            "sampler": (
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
            "full_cost_evaluation_calls": 1,
            "full_cost_particle_evaluations": int(
                cost_candidates.shape[0] * cost_candidates.shape[1]
            ),
            "cached_cost_particle_evaluations": int(
                candidates.shape[0] * persistent_previous_count
            ),
            "gradient_cost_evaluation_calls": 0,
            "gradient_particle_evaluations": 0,
            "importance_density_correction": (
                target_center is not None
                or smoothness_log_density_ratio is not None
                or tilt_log_acceptance is not None
            ),
            "importance_proposal": (
                "bounded_gaussian_mixture_times_log_acceptance_tilt"
                if tilt_log_acceptance is not None
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
        if persistent_proposal_bank is not None:
            offset = 0
            for component_center, component_count in zip(
                fresh_component_centers,
                fresh_component_counts,
                strict=True,
            ):
                stop = offset + int(component_count)
                persistent_proposal_bank.append(
                    fresh_candidates[:, offset:stop],
                    center=component_center,
                    scale=float(proposal_scale),
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
                or local_quadratic_owns_importance_target
            )
            else resolved_proposal_center
        )
        optimize_scale = (
            float(importance_target_scale)
            if (
                (
                    smc_starts_from_importance_target
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
            local_quadratic_proposal=local_quadratic_proposal,
            proposal_log_acceptance_fn=proposal_log_acceptance_fn,
            proposal_prefilter_cost_fn=proposal_prefilter_cost_fn,
            proposal_prefilter_pool_size=proposal_prefilter_pool_size,
            proposal_prefilter_target_ess=proposal_prefilter_target_ess,
            evaluation_only_candidate=evaluation_only_candidate,
            diagnostic_repeats=int(diagnostic_repeats),
            persistent_proposal_bank=persistent_proposal_bank,
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
    "sample_local_quadratic_gaussian_proposals",
    "sample_trajectory_proposals",
    "straight_line_cspace_path",
]
