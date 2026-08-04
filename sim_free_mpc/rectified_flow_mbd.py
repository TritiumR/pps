"""Generic rectified-flow MBD guidance with externally supplied costs.

This module deliberately knows nothing about a robot, simulator, or task.  A
caller supplies normalized token bounds and a batched clean-trajectory cost
callback.  That keeps policy/MBD mechanics in PPS while allowing downstream
applications to own FK, collision, phase, and scene-state semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch


CleanTrajectoryCost = Callable[[torch.Tensor], np.ndarray | torch.Tensor]


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


@dataclass(frozen=True)
class ProposalResult:
    candidates: torch.Tensor
    costs: np.ndarray
    weights: torch.Tensor
    reward_logits: np.ndarray
    mean: torch.Tensor
    effective_sample_size: np.ndarray
    proposal_scale: float

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
    candidates[:, 0, :, :] = center
    return candidates


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
    ) -> ProposalResult:
        candidates = sample_trajectory_proposals(
            center,
            scale=float(proposal_scale),
            lower=lower,
            upper=upper,
            num_samples=int(self.config.proposals_per_particle),
            sampler=self.config.proposal_sampler,
            generator=generator,
        )
        costs_value = cost_fn(candidates)
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
        weights_np = np.stack(
            [normalized_weights(row) for row in reward_logits], axis=0
        )
        weights = torch.as_tensor(
            weights_np, device=center.device, dtype=center.dtype
        )
        mean = torch.sum(weights[..., None, None] * candidates, dim=1)
        ess = 1.0 / np.sum(np.square(weights_np), axis=1)
        return ProposalResult(
            candidates=candidates,
            costs=costs,
            reward_logits=reward_logits,
            weights=weights,
            mean=mean,
            effective_sample_size=ess,
            proposal_scale=float(proposal_scale),
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
        keypose_coefficient: float | None = None,
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
        if float(time_value) >= 1.0 - 1e-8:
            candidates = policy_clean[:, None, :, :]
            costs_value = cost_fn(candidates)
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
        proposals = self.optimize_clean_trajectories(
            policy_clean,
            lower=lower,
            upper=upper,
            cost_fn=cost_fn,
            generator=generator,
            proposal_scale=proposal_scale,
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


__all__ = [
    "CleanTrajectoryCost",
    "FlowBlendCoefficients",
    "ProposalResult",
    "RectifiedFlowMBD",
    "RectifiedFlowMBDConfig",
    "ScoreGuidanceResult",
    "TokenBlockLayout",
    "blend_policy_mbd_flows",
    "flow_matching_clean_proposal_scale",
    "logmeanexp",
    "maximum_task_weight_under_mbd_kl",
    "memoryless_sde_kl_blocks",
    "normalized_weights",
    "rectified_flow_from_score",
    "rectified_flow_score",
    "sample_independent_truncated_gaussian",
    "sample_trajectory_proposals",
    "straight_line_cspace_path",
]
