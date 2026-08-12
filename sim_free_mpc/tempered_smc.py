"""Adaptive tempered SMC and preconditioned MALA for clean trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch


FullCost = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]
GradientCost = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TemperedSMCConfig:
    final_beta: float = 10.0
    target_ess_fraction: float = 0.5
    resample_ess_fraction: float = 0.5
    resampling_method: str = "systematic"
    mala_steps: int = 0
    mala_step_size: float = 0.001
    mala_schedule: str = "every_resample"
    geometric_duplicate_tolerance: float = 1e-4
    beta_tolerance: float = 1e-4
    beta_bisection_steps: int = 32
    max_tempering_stages: int = 64

    def validate(self) -> None:
        if self.final_beta <= 0.0:
            raise ValueError("SMC final_beta must be positive")
        if not 0.0 < self.target_ess_fraction <= 1.0:
            raise ValueError("SMC target ESS fraction must lie in (0, 1]")
        if not 0.0 < self.resample_ess_fraction <= 1.0:
            raise ValueError("SMC resample ESS fraction must lie in (0, 1]")
        if self.resampling_method not in ("systematic", "stratified"):
            raise ValueError("SMC resampling must be systematic or stratified")
        if self.mala_steps < 0:
            raise ValueError("MALA steps must be nonnegative")
        if self.mala_step_size <= 0.0:
            raise ValueError("MALA step size must be positive")
        if self.mala_schedule not in ("every_resample", "final_beta"):
            raise ValueError(
                "MALA schedule must be every_resample or final_beta"
            )
        if self.geometric_duplicate_tolerance < 0.0:
            raise ValueError(
                "geometric duplicate tolerance must be nonnegative"
            )
        if self.beta_tolerance <= 0.0:
            raise ValueError("SMC beta tolerance must be positive")
        if self.beta_bisection_steps < 1 or self.max_tempering_stages < 1:
            raise ValueError("SMC iteration counts must be positive")


@dataclass(frozen=True)
class TemperedSMCResult:
    particles: torch.Tensor
    costs: torch.Tensor
    log_weights: torch.Tensor
    weights: torch.Tensor
    diagnostics: dict


def _normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    normalizer = torch.logsumexp(log_weights, dim=1, keepdim=True)
    if not torch.all(torch.isfinite(normalizer)):
        raise RuntimeError("all SMC particles received zero weight")
    return log_weights - normalizer


def _ess_from_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    weights = torch.exp(_normalize_log_weights(log_weights))
    return 1.0 / torch.sum(weights.square(), dim=1)


def _incremental_log_weights(
    costs: torch.Tensor,
    eligible: torch.Tensor | None,
    delta_beta: float,
) -> torch.Tensor:
    result = -float(delta_beta) * costs
    if eligible is not None:
        result = torch.where(
            eligible,
            result,
            torch.full_like(result, -torch.inf),
        )
    return result


def _conditional_ess(
    log_weights: torch.Tensor,
    costs: torch.Tensor,
    eligible: torch.Tensor | None,
    delta_beta: float,
) -> torch.Tensor:
    """ESS after applying an incremental bridge to current log weights."""
    return _ess_from_log_weights(
        log_weights + _incremental_log_weights(costs, eligible, delta_beta)
    )


def choose_next_beta(
    beta: float,
    *,
    final_beta: float,
    log_weights: torch.Tensor,
    costs: torch.Tensor,
    eligible: torch.Tensor | None,
    target_ess: float,
    tolerance: float,
    bisection_steps: int,
) -> tuple[float, torch.Tensor]:
    """Choose the largest beta whose incremental weights meet target CESS."""
    remaining = float(final_beta) - float(beta)
    if remaining <= float(tolerance):
        next_beta = float(final_beta)
        return next_beta, _conditional_ess(
            log_weights, costs, eligible,
            max(remaining, torch.finfo(costs.dtype).eps),
        )
    full_ess = _conditional_ess(log_weights, costs, eligible, remaining)
    if float(torch.min(full_ess).item()) >= float(target_ess):
        return float(final_beta), full_ess

    # A hard eligibility mask can make the requested CESS impossible for every
    # positive increment. Take one small positive step so resampling can remove
    # ineligible particles, rather than stalling at beta=0.
    tiny = min(remaining, float(tolerance))
    tiny_ess = _conditional_ess(log_weights, costs, eligible, tiny)
    if float(torch.min(tiny_ess).item()) < float(target_ess):
        return float(beta) + tiny, tiny_ess

    low = 0.0
    high = remaining
    for _ in range(int(bisection_steps)):
        middle = 0.5 * (low + high)
        middle_ess = _conditional_ess(
            log_weights, costs, eligible, middle
        )
        if float(torch.min(middle_ess).item()) >= float(target_ess):
            low = middle
        else:
            high = middle
    delta = max(low, min(remaining, float(tolerance)))
    next_beta = min(float(final_beta), float(beta) + delta)
    return next_beta, _conditional_ess(
        log_weights, costs, eligible, next_beta - float(beta)
    )


def resample_indices(
    weights: torch.Tensor,
    *,
    method: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Vectorized systematic or stratified resampling for [P, N] weights."""
    if weights.ndim != 2:
        raise ValueError("SMC weights must have shape [particle, proposal]")
    rows, count = weights.shape
    cdf = torch.cumsum(weights, dim=1)
    cdf[:, -1] = 1.0
    grid = torch.arange(count, device=weights.device, dtype=weights.dtype)[None]
    if method == "systematic":
        offsets = torch.rand(
            (rows, 1), device=weights.device, dtype=weights.dtype,
            generator=generator,
        )
        positions = (grid + offsets) / float(count)
    elif method == "stratified":
        offsets = torch.rand(
            (rows, count), device=weights.device, dtype=weights.dtype,
            generator=generator,
        )
        positions = (grid + offsets) / float(count)
    else:
        raise ValueError(f"unknown resampling method {method!r}")
    return torch.searchsorted(cdf.contiguous(), positions.contiguous(), right=False)


def _gather_population(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    suffix = (1,) * (values.ndim - 2)
    expanded = indices.reshape(*indices.shape, *suffix).expand(
        *indices.shape, *values.shape[2:]
    )
    return torch.gather(values, 1, expanded)


def _log_base_gaussian(
    values: torch.Tensor,
    center: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    residual = (values - center[:, None]) / float(scale)
    return -0.5 * torch.sum(residual.square(), dim=tuple(range(2, values.ndim)))


def _gradient_log_target(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: float,
    beta: float,
    gradient_cost_fn: GradientCost,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Robot rollouts run under torch.inference_mode(); locally disable it and
    # clone so the MALA-only cost remains differentiable.
    with torch.inference_mode(False), torch.enable_grad():
        variable = values.detach().clone().requires_grad_(True)
        differentiable_cost = gradient_cost_fn(variable)
        expected = variable.shape[:2]
        if differentiable_cost.shape != expected:
            raise ValueError(
                f"MALA gradient cost must return {tuple(expected)}, "
                f"got {tuple(differentiable_cost.shape)}"
            )
        if not torch.all(torch.isfinite(differentiable_cost)):
            raise FloatingPointError("MALA differentiable cost is non-finite")
        gradient_cost = torch.autograd.grad(
            torch.sum(differentiable_cost), variable
        )[0]
        gradient_log_q = -(variable - center[:, None]) / float(scale) ** 2
        gradient = gradient_log_q - float(beta) * gradient_cost
    return gradient.detach(), differentiable_cost.detach()



def _ancestor_diversity(ancestors: torch.Tensor) -> dict:
    """Genealogical diversity relative to the initial q population."""
    if ancestors.ndim != 2:
        raise ValueError("ancestor ids must have shape [particle, proposal]")
    rows, count = ancestors.shape
    counts = torch.zeros(
        (rows, count), device=ancestors.device, dtype=torch.float64
    )
    counts.scatter_add_(
        1, ancestors, torch.ones_like(ancestors, dtype=torch.float64)
    )
    probabilities = counts / float(count)
    positive = probabilities > 0.0
    entropy = -torch.sum(
        torch.where(
            positive,
            probabilities * torch.log(
                probabilities.clamp_min(torch.finfo(torch.float64).tiny)
            ),
            torch.zeros_like(probabilities),
        ),
        dim=1,
    )
    normalization = math.log(float(count)) if count > 1 else 1.0
    return {
        "unique_ancestors": torch.sum(counts > 0.0, dim=1).cpu().tolist(),
        "ancestor_entropy": entropy.cpu().tolist(),
        "normalized_ancestor_entropy": (
            entropy / normalization
        ).cpu().tolist(),
        "max_ancestor_fraction": (
            torch.amax(counts, dim=1) / float(count)
        ).cpu().tolist(),
    }


def _geometric_diversity(
    particles: torch.Tensor, *, duplicate_tolerance: float
) -> dict:
    """Population spread after flattening each clean trajectory."""
    if particles.ndim < 3:
        raise ValueError("geometric diversity expects [P,N,...] particles")
    with torch.no_grad():
        flat = particles.reshape(particles.shape[0], particles.shape[1], -1)
        centered = flat - torch.mean(flat, dim=1, keepdim=True)
        covariance_trace = torch.mean(
            torch.sum(centered.square(), dim=-1), dim=1
        )
        count = int(flat.shape[1])
        if count < 2:
            mean_distance = torch.zeros_like(covariance_trace)
            near_duplicate_fraction = torch.zeros_like(covariance_trace)
        else:
            distances = torch.cdist(
                flat, flat, compute_mode="donot_use_mm_for_euclid_dist"
            )
            off_diagonal = ~torch.eye(
                count, device=flat.device, dtype=torch.bool
            )[None]
            denominator = float(count * (count - 1))
            mean_distance = torch.sum(
                torch.where(
                    off_diagonal, distances, torch.zeros_like(distances)
                ),
                dim=(1, 2),
            ) / denominator
            near = torch.sum(
                off_diagonal
                & (distances <= float(duplicate_tolerance)),
                dim=(1, 2),
            )
            near_duplicate_fraction = near.to(flat.dtype) / denominator
    return {
        "mean_pairwise_distance": mean_distance.cpu().tolist(),
        "covariance_trace": covariance_trace.cpu().tolist(),
        "near_duplicate_fraction": near_duplicate_fraction.cpu().tolist(),
        "near_duplicate_tolerance": float(duplicate_tolerance),
    }

def mala_rejuvenate(
    particles: torch.Tensor,
    costs: torch.Tensor,
    eligible: torch.Tensor | None,
    *,
    center: torch.Tensor,
    scale: float,
    beta: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    full_cost_fn: FullCost,
    gradient_cost_fn: GradientCost,
    steps: int,
    step_size: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
    """Preconditioned MALA with exact forward/reverse MH correction."""
    if steps < 1:
        return particles, costs, eligible, {
            "accepted": 0,
            "proposed": 0,
            "full_cost_calls": 0,
            "full_cost_particle_evaluations": 0,
            "gradient_cost_calls": 0,
            "gradient_particle_evaluations": 0,
        }
    current = particles
    current_cost = costs
    current_eligible = eligible
    gradient, _ = _gradient_log_target(
        current, center=center, scale=scale, beta=beta,
        gradient_cost_fn=gradient_cost_fn,
    )
    gradient_calls = 1
    gradient_particle_evaluations = int(current.shape[0] * current.shape[1])
    full_calls = 0
    full_particle_evaluations = 0
    accepted = 0
    proposed_count = 0
    covariance_scale = float(step_size) * float(scale) ** 2
    noise_scale = math.sqrt(float(step_size)) * float(scale)
    reduce_dims = tuple(range(2, current.ndim))

    for _ in range(int(steps)):
        forward_mean = current + 0.5 * covariance_scale * gradient
        proposal = forward_mean + noise_scale * torch.randn(
            current.shape,
            device=current.device,
            dtype=current.dtype,
            generator=generator,
        )
        in_bounds = torch.all(
            (proposal >= lower) & (proposal <= upper), dim=reduce_dims
        )
        proposal_cost, proposal_eligible = full_cost_fn(proposal)
        full_calls += 1
        full_particle_evaluations += int(proposal.shape[0] * proposal.shape[1])
        proposal_gradient, _ = _gradient_log_target(
            proposal, center=center, scale=scale, beta=beta,
            gradient_cost_fn=gradient_cost_fn,
        )
        gradient_calls += 1
        gradient_particle_evaluations += int(
            proposal.shape[0] * proposal.shape[1]
        )
        reverse_mean = proposal + 0.5 * covariance_scale * proposal_gradient
        forward_residual = proposal - forward_mean
        reverse_residual = current - reverse_mean
        log_forward = -torch.sum(
            forward_residual.square(), dim=reduce_dims
        ) / (2.0 * covariance_scale)
        log_reverse = -torch.sum(
            reverse_residual.square(), dim=reduce_dims
        ) / (2.0 * covariance_scale)
        current_log_target = _log_base_gaussian(
            current, center, scale
        ) - float(beta) * current_cost
        proposal_log_target = _log_base_gaussian(
            proposal, center, scale
        ) - float(beta) * proposal_cost
        if current_eligible is not None:
            current_log_target = torch.where(
                current_eligible,
                current_log_target,
                torch.full_like(current_log_target, -torch.inf),
            )
        if proposal_eligible is not None:
            proposal_log_target = torch.where(
                proposal_eligible,
                proposal_log_target,
                torch.full_like(proposal_log_target, -torch.inf),
            )
        log_acceptance = (
            proposal_log_target - current_log_target
            + log_reverse - log_forward
        )
        log_uniform = torch.log(torch.rand(
            log_acceptance.shape,
            device=current.device,
            dtype=current.dtype,
            generator=generator,
        ).clamp_min(torch.finfo(current.dtype).tiny))
        accept = in_bounds & (log_uniform < torch.minimum(
            log_acceptance, torch.zeros_like(log_acceptance)
        ))
        accepted += int(torch.sum(accept).item())
        proposed_count += int(accept.numel())
        value_mask = accept.reshape(
            *accept.shape, *((1,) * (current.ndim - 2))
        )
        current = torch.where(value_mask, proposal, current).detach()
        current_cost = torch.where(accept, proposal_cost, current_cost).detach()
        gradient = torch.where(value_mask, proposal_gradient, gradient).detach()
        if current_eligible is not None or proposal_eligible is not None:
            old_eligible = (
                torch.ones_like(accept)
                if current_eligible is None else current_eligible
            )
            new_eligible = (
                torch.ones_like(accept)
                if proposal_eligible is None else proposal_eligible
            )
            current_eligible = torch.where(
                accept, new_eligible, old_eligible
            ).detach()

    return current, current_cost, current_eligible, {
        "accepted": accepted,
        "proposed": proposed_count,
        "full_cost_calls": full_calls,
        "full_cost_particle_evaluations": full_particle_evaluations,
        "gradient_cost_calls": gradient_calls,
        "gradient_particle_evaluations": gradient_particle_evaluations,
    }


def adaptive_tempered_smc(
    initial_particles: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    full_cost_fn: FullCost,
    gradient_cost_fn: GradientCost | None,
    config: TemperedSMCConfig,
    generator: torch.Generator,
) -> TemperedSMCResult:
    """Bridge q to q exp(-beta C) using adaptive ESS increments."""
    config.validate()
    if initial_particles.ndim != 4 or center.ndim != 3:
        raise ValueError("SMC expects particles [P,N,W,D] and center [P,W,D]")
    if initial_particles.shape[0] != center.shape[0]:
        raise ValueError("SMC particle rows and centers do not match")
    if config.mala_steps and gradient_cost_fn is None:
        raise ValueError("MALA requires a differentiable gradient cost callback")
    started = time.perf_counter()
    particles = initial_particles
    costs, eligible = full_cost_fn(particles)
    expected = particles.shape[:2]
    if costs.shape != expected:
        raise ValueError(
            f"full SMC cost must return {tuple(expected)}, got {tuple(costs.shape)}"
        )
    if eligible is not None and eligible.shape != expected:
        raise ValueError("SMC eligibility mask must match costs")
    if torch.any(torch.isnan(costs) | torch.isneginf(costs)):
        raise FloatingPointError("SMC full costs contain NaN or negative infinity")

    rows, count = int(expected[0]), int(expected[1])
    log_weights = torch.full(
        expected,
        -math.log(float(count)),
        device=particles.device,
        dtype=particles.dtype,
    )
    ancestors = torch.arange(
        count, device=particles.device, dtype=torch.long
    )[None].expand(rows, -1).clone()
    initial_ancestry = _ancestor_diversity(ancestors)
    initial_geometric_diversity = _geometric_diversity(
        particles,
        duplicate_tolerance=float(config.geometric_duplicate_tolerance),
    )

    beta = 0.0
    beta_schedule = [0.0]
    ess_per_stage = []
    conditional_ess_per_stage = []
    resampled_per_stage = []
    resampling_events = []
    mala_events = []
    pre_resample_final_ess = None
    final_pre_mala_ancestry = None
    final_pre_mala_geometric_diversity = None
    final_post_mala_geometric_diversity = None
    full_cost_calls = 1
    full_cost_particle_evaluations = rows * count
    gradient_cost_calls = 0
    gradient_particle_evaluations = 0
    mala_accepted = 0
    mala_proposed = 0

    for _ in range(int(config.max_tempering_stages)):
        if beta >= float(config.final_beta) - float(config.beta_tolerance):
            beta = float(config.final_beta)
            break
        next_beta, conditional_ess = choose_next_beta(
            beta,
            final_beta=float(config.final_beta),
            log_weights=log_weights,
            costs=costs,
            eligible=eligible,
            target_ess=float(config.target_ess_fraction) * count,
            tolerance=float(config.beta_tolerance),
            bisection_steps=int(config.beta_bisection_steps),
        )
        delta_beta = next_beta - beta
        log_weights = _normalize_log_weights(
            log_weights + _incremental_log_weights(
                costs, eligible, delta_beta
            )
        )
        beta = next_beta
        total_ess = _ess_from_log_weights(log_weights)
        beta_schedule.append(float(beta))
        conditional_ess_per_stage.append(
            conditional_ess.detach().cpu().tolist()
        )
        ess_per_stage.append(total_ess.detach().cpu().tolist())
        if beta >= float(config.final_beta) - float(config.beta_tolerance):
            pre_resample_final_ess = total_ess.detach().cpu().tolist()

        resample_threshold = float(config.resample_ess_fraction) * count
        # Bisection approaches the threshold from above; include a tiny
        # numerical margin so target-ESS stages resample immediately.
        resample_tolerance = max(1e-3, 1e-6 * float(count))
        should_resample = bool(torch.any(
            total_ess <= resample_threshold + resample_tolerance
        ).item())
        resampled_per_stage.append(should_resample)
        if not should_resample:
            continue

        ancestry_before = _ancestor_diversity(ancestors)
        resampling_event = {
            "beta": float(beta),
            "ess_before_resampling": total_ess.detach().cpu().tolist(),
            "ancestry_before_resampling": ancestry_before,
        }
        weights = torch.exp(log_weights)
        indices = resample_indices(
            weights, method=config.resampling_method, generator=generator
        )
        particles = _gather_population(particles, indices).detach()
        costs = torch.gather(costs, 1, indices).detach()
        ancestors = torch.gather(ancestors, 1, indices).detach()
        if eligible is not None:
            eligible = torch.gather(eligible, 1, indices).detach()
        log_weights = torch.full_like(log_weights, -math.log(float(count)))
        ancestry_after = _ancestor_diversity(ancestors)
        resampling_event.update({
            "ess_after_resampling": [float(count)] * rows,
            "ancestry_after_resampling": ancestry_after,
        })
        resampling_events.append(resampling_event)

        if (
            config.mala_steps
            and config.mala_schedule == "every_resample"
        ):
            geometric_before = _geometric_diversity(
                particles,
                duplicate_tolerance=float(
                    config.geometric_duplicate_tolerance
                ),
            )
            particles, costs, eligible, mala = mala_rejuvenate(
                particles, costs, eligible,
                center=center, scale=scale, beta=beta,
                lower=lower, upper=upper,
                full_cost_fn=full_cost_fn,
                gradient_cost_fn=gradient_cost_fn,
                steps=int(config.mala_steps),
                step_size=float(config.mala_step_size),
                generator=generator,
            )
            geometric_after = _geometric_diversity(
                particles,
                duplicate_tolerance=float(
                    config.geometric_duplicate_tolerance
                ),
            )
            mala_events.append({
                "schedule": "every_resample",
                "beta": float(beta),
                "ancestry_before_mala": ancestry_after,
                "geometric_diversity_before_mala": geometric_before,
                "geometric_diversity_after_mala": geometric_after,
                "accepted": int(mala["accepted"]),
                "proposed": int(mala["proposed"]),
                "acceptance_rate": (
                    float(mala["accepted"]) / float(mala["proposed"])
                    if mala["proposed"] else None
                ),
            })
            mala_accepted += int(mala["accepted"])
            mala_proposed += int(mala["proposed"])
            full_cost_calls += int(mala["full_cost_calls"])
            full_cost_particle_evaluations += int(
                mala["full_cost_particle_evaluations"]
            )
            gradient_cost_calls += int(mala["gradient_cost_calls"])
            gradient_particle_evaluations += int(
                mala["gradient_particle_evaluations"]
            )
    else:
        raise RuntimeError(
            "adaptive SMC exceeded max_tempering_stages before final beta"
        )

    if config.mala_steps and config.mala_schedule == "final_beta":
        final_pre_mala_ancestry = _ancestor_diversity(ancestors)
        final_pre_mala_geometric_diversity = _geometric_diversity(
            particles,
            duplicate_tolerance=float(config.geometric_duplicate_tolerance),
        )
        particles, costs, eligible, mala = mala_rejuvenate(
            particles, costs, eligible,
            center=center, scale=scale, beta=float(config.final_beta),
            lower=lower, upper=upper,
            full_cost_fn=full_cost_fn,
            gradient_cost_fn=gradient_cost_fn,
            steps=int(config.mala_steps),
            step_size=float(config.mala_step_size),
            generator=generator,
        )
        final_post_mala_geometric_diversity = _geometric_diversity(
            particles,
            duplicate_tolerance=float(config.geometric_duplicate_tolerance),
        )
        mala_events.append({
            "schedule": "final_beta",
            "beta": float(config.final_beta),
            "ancestry_before_mala": final_pre_mala_ancestry,
            "geometric_diversity_before_mala": (
                final_pre_mala_geometric_diversity
            ),
            "geometric_diversity_after_mala": (
                final_post_mala_geometric_diversity
            ),
            "accepted": int(mala["accepted"]),
            "proposed": int(mala["proposed"]),
            "acceptance_rate": (
                float(mala["accepted"]) / float(mala["proposed"])
                if mala["proposed"] else None
            ),
        })
        mala_accepted += int(mala["accepted"])
        mala_proposed += int(mala["proposed"])
        full_cost_calls += int(mala["full_cost_calls"])
        full_cost_particle_evaluations += int(
            mala["full_cost_particle_evaluations"]
        )
        gradient_cost_calls += int(mala["gradient_cost_calls"])
        gradient_particle_evaluations += int(
            mala["gradient_particle_evaluations"]
        )

    weights = torch.exp(_normalize_log_weights(log_weights))
    final_ess = _ess_from_log_weights(log_weights)
    weighted_cost = torch.sum(weights * costs, dim=1)
    finite_costs = torch.where(
        torch.isfinite(costs), costs, torch.full_like(costs, torch.inf)
    )
    diagnostics = {
        "sampler": "adaptive_tempered_smc",
        "beta_schedule": beta_schedule,
        "conditional_ess_per_stage": conditional_ess_per_stage,
        "ess_per_stage": ess_per_stage,
        "resampled_per_stage": resampled_per_stage,
        "resampling_events": resampling_events,
        "initial_ancestry": initial_ancestry,
        "final_ancestry": _ancestor_diversity(ancestors),
        "initial_geometric_diversity": initial_geometric_diversity,
        "final_geometric_diversity": _geometric_diversity(
            particles,
            duplicate_tolerance=float(config.geometric_duplicate_tolerance),
        ),
        "final_pre_mala_ancestry": final_pre_mala_ancestry,
        "final_pre_mala_geometric_diversity": (
            final_pre_mala_geometric_diversity
        ),
        "final_post_mala_geometric_diversity": (
            final_post_mala_geometric_diversity
        ),
        "pre_resample_final_ess": pre_resample_final_ess,
        "final_ess": final_ess.detach().cpu().tolist(),
        "mala_schedule": config.mala_schedule,
        "mala_events": mala_events,
        "mala_steps_per_event": int(config.mala_steps),
        "mala_acceptance_rate": (
            float(mala_accepted) / float(mala_proposed)
            if mala_proposed else None
        ),
        "mala_accepted": mala_accepted,
        "mala_proposed": mala_proposed,
        "best_cost": torch.amin(finite_costs, dim=1).detach().cpu().tolist(),
        "weighted_cost": weighted_cost.detach().cpu().tolist(),
        "wall_clock_seconds": time.perf_counter() - started,
        "full_cost_evaluation_calls": full_cost_calls,
        "full_cost_particle_evaluations": full_cost_particle_evaluations,
        "gradient_cost_evaluation_calls": gradient_cost_calls,
        "gradient_particle_evaluations": gradient_particle_evaluations,
        "target_ess_fraction": float(config.target_ess_fraction),
        "resample_ess_fraction": float(config.resample_ess_fraction),
        "resampling_method": config.resampling_method,
        "mala_step_size": float(config.mala_step_size),
        "proposal_covariance": f"{float(scale) ** 2:g} * I",
    }
    return TemperedSMCResult(
        particles=particles,
        costs=costs,
        log_weights=log_weights,
        weights=weights,
        diagnostics=diagnostics,
    )


__all__ = [
    "GradientCost",
    "TemperedSMCConfig",
    "TemperedSMCResult",
    "adaptive_tempered_smc",
    "choose_next_beta",
    "mala_rejuvenate",
    "resample_indices",
]
