from __future__ import annotations

from typing import Any

import torch

from .ddim import ddim_iteration_alphas


def _optimize_action_prox_chunk(
    planner: Any,
    x_t: torch.Tensor,
    policy_inputs: dict[str, Any],
    context: dict[str, Any],
) -> tuple[torch.Tensor, Any, int, float]:
    """Optimize clean action candidates around the current normalized action.

    This is an intentionally experimental MBD-score proposal rule. The original
    implementation centers clean candidates at ``x_t / sqrt(alpha_bar)`` with a
    DDIM-derived clean std. This variant treats the current noisy state itself as
    a normalized action proposal center and samples locally around it.
    """
    if x_t.shape[0] != 1:
        raise ValueError("MBD action-prox sampler currently expects batch size 1.")
    if planner.config.optimize_space != "action":
        raise ValueError("mbd_score_action_prox only supports score-space action optimization.")

    active_dims = min(planner.config.action_dims, x_t.shape[-1])
    horizon = x_t.shape[1]
    opt_horizon = planner._interpolation_knot_count(horizon)
    proposal_center = planner._control_point_resample(
        x_t.detach()[0, :, :active_dims],
        opt_horizon,
    )
    proposal_std = float(planner.config.noise)

    def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
        full_horizon_samples = planner._interpolate_control_points(samples, horizon)
        return planner._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

    result = planner.sampler.optimize_with_noise_scale(
        proposal_center,
        cost_from_positions,
        noise_scale=proposal_std,
    )
    x0_hat = x_t.detach().clone()
    x0_hat[:, :, :active_dims] = planner._interpolate_control_points(result.mean, horizon).unsqueeze(0)
    return x0_hat, result, active_dims, proposal_std


def estimate_mbd_score_action_prox(
    planner: Any,
    x_t: torch.Tensor,
    policy_inputs: dict[str, Any],
    context: dict[str, Any],
    *,
    iteration: int,
    num_iterations: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=num_iterations,
        num_train_timesteps=planner.config.ddim_num_train_timesteps,
    )
    x0_hat, result, active_dims, proposal_std = _optimize_action_prox_chunk(
        planner,
        x_t,
        policy_inputs,
        context,
    )

    alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
    beta = torch.clamp(1.0 - alpha, min=planner.config.flow_eps)
    sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=planner.config.flow_eps))

    score = torch.zeros_like(x_t)
    active_x = x_t.detach()[:, :, :active_dims]
    active_x0 = x0_hat[:, :, :active_dims]
    score[:, :, :active_dims] = (sqrt_alpha * active_x0 - active_x) / beta

    diagnostics = planner._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
    diagnostics.update(
        {
            "update_mode": "estimate_mbd_score_action_prox",
            "ddim_iteration": int(iteration),
            "ddim_num_iterations": int(num_iterations),
            "alpha_bar": float(alpha_bar),
            "alpha_bar_prev": float(alpha_bar_prev),
            "active_dims": int(active_dims),
            "proposal_center": "current_noisy_action",
            "proposal_noise_scale": float(proposal_std),
            "clean_sample_std": float(proposal_std),
        }
    )
    return score, diagnostics


def step_mbd_score_action_prox(
    planner: Any,
    x_t: torch.Tensor,
    policy_inputs: dict[str, Any],
    context: dict[str, Any],
    *,
    iteration: int,
    num_iterations: int,
    score_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=num_iterations,
        num_train_timesteps=planner.config.ddim_num_train_timesteps,
    )
    x0_hat, result, active_dims, proposal_std = _optimize_action_prox_chunk(
        planner,
        x_t,
        policy_inputs,
        context,
    )

    alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
    alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
    beta = torch.clamp(1.0 - alpha, min=planner.config.flow_eps)
    sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=planner.config.flow_eps))

    active_x = x_t.detach()[:, :, :active_dims]
    active_x0 = x0_hat[:, :, :active_dims]
    score_numerator = sqrt_alpha * active_x0 - active_x
    score = torch.zeros_like(x_t)
    score[:, :, :active_dims] = score_numerator / beta

    alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=planner.config.flow_eps), min=planner.config.flow_eps)
    active_prev = (active_x + float(score_scale) * score_numerator) / torch.sqrt(alpha_step)
    next_x = x_t.detach().clone()
    next_x[:, :, :active_dims] = active_prev

    diagnostics = planner._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
    diagnostics.update(
        {
            "update_mode": "mbd_score_action_prox",
            "ddim_iteration": int(iteration),
            "ddim_num_iterations": int(num_iterations),
            "alpha_bar": float(alpha_bar),
            "alpha_bar_prev": float(alpha_bar_prev),
            "proposal_center": "current_noisy_action",
            "proposal_noise_scale": float(proposal_std),
            "clean_sample_std": float(proposal_std),
            "score_scale": float(score_scale),
            "mbd_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
        }
    )
    return next_x, diagnostics
