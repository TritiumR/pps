from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .action_space import decode_model_action_chunks
from .costs import PriorityStateCost
from .ddim import ddim_clean_sample_std_scale, ddim_iteration_alphas
from .dial_sampler import DIALSampler, DIALSamplerConfig
from .fk import PandaFK, quat_mul_wxyz, transform_points_wxyz


@dataclass(frozen=True)
class SimFreeMPCConfig:
    task_name: str = "auto"
    num_samples: int = 64
    iterations: int = 2
    noise: float = 0.35
    temperature: float = 0.15
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    action_dims: int = 8
    flow_eps: float = 1e-3
    joint_delta_clip: float = 0.25
    ddim_num_train_timesteps: int = 100
    interpolate: bool = False
    control_frequency: float = 40.0
    interpolate_frequency: float = 5.0


class SimFreeMPC:
    """FK/cost-only MPC used as a geometric steering term for PPS."""

    def __init__(self, policy: Any, config: SimFreeMPCConfig):
        self.policy = policy
        self.config = config
        self.fk = PandaFK()
        self.sampler = DIALSampler(
            DIALSamplerConfig(
                num_samples=config.num_samples,
                iterations=config.iterations,
                noise=config.noise,
                temperature=config.temperature,
                beta_opt_iter=config.beta_opt_iter,
                beta_horizon=config.beta_horizon,
                action_dims=config.action_dims,
            )
        )
        self.cost = PriorityStateCost(config.task_name)

    def _interpolation_knot_count(self, horizon: int) -> int:
        if not self.config.interpolate or horizon <= 1:
            return horizon
        ratio = self.config.interpolate_frequency / max(self.config.control_frequency, self.config.flow_eps)
        knots = int(math.ceil(horizon * ratio))
        return max(2, min(horizon, knots))

    @staticmethod
    def _linear_resample(sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if sequence.shape[-2] == output_horizon:
            return sequence
        if sequence.shape[-2] == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])

        original_ndim = sequence.ndim
        if original_ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence [H,D] or [B,H,D], got {tuple(sequence.shape)}")

        resampled = F.interpolate(
            sequence.transpose(1, 2),
            size=output_horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        if original_ndim == 2:
            return resampled[0]
        return resampled

    def _optimize_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
    ):
        if x_t.shape[0] != 1:
            raise ValueError("SimFreeMPC MVP currently expects batch size 1.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        mean0 = self._linear_resample(x_t[0, :, :active_dims].detach(), opt_horizon)

        def cost_fn(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._linear_resample(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        result = self.sampler.optimize(mean0, cost_fn)
        target = x_t.detach().clone()
        target[:, :, :active_dims] = self._linear_resample(result.mean, horizon).unsqueeze(0)
        return target, result, active_dims

    def _cost_active_samples(
        self,
        samples: torch.Tensor,
        x_template: torch.Tensor,
        active_dims: int,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
    ) -> torch.Tensor:
        full = x_template.detach().repeat(samples.shape[0], 1, 1)
        full[:, :, :active_dims] = samples
        decoded = decode_model_action_chunks(
            self.policy,
            policy_inputs,
            full,
            current_joint_pos=context.get("joint_pos"),
            max_joint_delta=self.config.joint_delta_clip,
        )
        real = decoded.real_actions
        joints = real[..., :7]
        fk = self.fk.forward(joints)
        ee_pos = fk.ee_pos
        ee_quat = fk.ee_quat
        root_pos = context.get("robot_root_pos")
        root_quat = context.get("robot_root_quat")
        if root_pos is not None and root_quat is not None:
            if not torch.is_tensor(root_pos):
                root_pos = torch.as_tensor(root_pos)
            if not torch.is_tensor(root_quat):
                root_quat = torch.as_tensor(root_quat)
            ee_pos = transform_points_wxyz(
                root_pos.to(device=ee_pos.device, dtype=ee_pos.dtype),
                root_quat.to(device=ee_pos.device, dtype=ee_pos.dtype),
                ee_pos,
            )
            ee_quat = quat_mul_wxyz(
                root_quat.to(device=ee_quat.device, dtype=ee_quat.dtype),
                ee_quat,
            )
        return self.cost(real_actions=real, ee_pos=ee_pos, ee_quat=ee_quat, context=context)

    def _optimize_ddim_clean_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        alpha_bar: float,
    ):
        if x_t.shape[0] != 1:
            raise ValueError("SimFreeMPC MVP currently expects batch size 1.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        sqrt_alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype).sqrt()
        clean_center_full = x_t.detach()[0, :, :active_dims] / torch.clamp(
            sqrt_alpha,
            min=self.config.flow_eps,
        )
        clean_center = self._linear_resample(clean_center_full, opt_horizon)
        clean_std = self.config.noise * ddim_clean_sample_std_scale(alpha_bar)

        def cost_fn(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._linear_resample(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        result = self.sampler.optimize_with_noise_scale(clean_center, cost_fn, noise_scale=clean_std)
        x0_hat = x_t.detach().clone()
        x0_hat[:, :, :active_dims] = self._linear_resample(result.mean, horizon).unsqueeze(0)
        return x0_hat, result, active_dims, clean_std

    def _diagnostics(
        self,
        *,
        result,
        target: torch.Tensor,
        x_t: torch.Tensor,
        score: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        diagnostics = {
            "cost_min": float(result.costs.min().detach().cpu()),
            "cost_mean": float(result.costs.mean().detach().cpu()),
            "cost_weighted": float(torch.sum(result.costs * result.weights).detach().cpu()),
            "target_delta_norm": float(torch.linalg.vector_norm((target - x_t).detach()).cpu()),
            "interpolate": bool(self.config.interpolate),
        }
        if self.config.interpolate:
            diagnostics.update(
                {
                    "interpolate_horizon": int(x_t.shape[1]),
                    "interpolate_knot_count": int(self._interpolation_knot_count(x_t.shape[1])),
                    "interpolate_frequency": float(self.config.interpolate_frequency),
                    "control_frequency": float(self.config.control_frequency),
                }
            )
        if score is not None:
            diagnostics.update(
                {
                    "score_norm": float(torch.linalg.vector_norm(score.detach()).cpu()),
                    "score_abs_mean": float(score.detach().abs().mean().cpu()),
                    "noise_scale_min": float(result.noise_scale.min().detach().cpu()),
                    "noise_scale_max": float(result.noise_scale.max().detach().cpu()),
                }
            )
        return diagnostics

    def step(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        dt: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target, result, _ = self._optimize_chunk(x_t, policy_inputs, context)

        dt_abs = torch.abs(torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype))
        flow = -(target - x_t.detach()) / torch.clamp(dt_abs, min=self.config.flow_eps)

        return flow, self._diagnostics(result=result, target=target, x_t=x_t)

    def step_score_space(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        step_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target, result, active_dims = self._optimize_chunk(x_t, policy_inputs, context)
        variance = torch.clamp(
            self._linear_resample(
                result.noise_scale.to(device=x_t.device, dtype=x_t.dtype).view(-1, 1),
                x_t.shape[1],
            )
            .squeeze(-1)
            .pow(2),
            min=self.config.flow_eps,
        )
        score = torch.zeros_like(x_t)
        score[:, :, :active_dims] = (
            target[:, :, :active_dims] - x_t.detach()[:, :, :active_dims]
        ) / variance.view(1, -1, 1)
        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = (
            x_t.detach()[:, :, :active_dims]
            + step_scale * variance.view(1, -1, 1) * score[:, :, :active_dims]
        )
        diagnostics = self._diagnostics(result=result, target=target, x_t=x_t, score=score)
        diagnostics["update_mode"] = "score_space"
        return next_x, diagnostics

    def step_ddim(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        step_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        x0_hat, result, active_dims, clean_std = self._optimize_ddim_clean_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))
        sqrt_beta = torch.sqrt(beta)

        score = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        score[:, :, :active_dims] = (-active_x + sqrt_alpha * active_x0) / beta

        pred_epsilon = (active_x - sqrt_alpha * active_x0) / sqrt_beta
        active_prev = (
            torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * active_x0
            + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_epsilon
        )

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_x + float(step_scale) * (active_prev - active_x)

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "ddim",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "clean_sample_std": float(clean_std),
                "ddim_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
            }
        )
        return next_x, diagnostics

    def step_mbd_score(
        self,
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
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        x0_hat, result, active_dims, clean_std = self._optimize_ddim_clean_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        score_numerator = sqrt_alpha * active_x0 - active_x
        score[:, :, :active_dims] = score_numerator / beta

        alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=self.config.flow_eps), min=self.config.flow_eps)
        active_prev = (active_x + float(score_scale) * score_numerator) / torch.sqrt(alpha_step)

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "mbd_score",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "clean_sample_std": float(clean_std),
                "score_scale": float(score_scale),
                "mbd_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
            }
        )
        return next_x, diagnostics
