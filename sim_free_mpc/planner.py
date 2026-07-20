from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .action_space import decode_model_action_chunks, rebase_model_action_chunk
from .costs import PriorityStateCost
from .costs_capsule_flow import CapsuleFlowStateCost
from .costs_explore import ExploreStateCost
from .costs_grasp_flow import GraspFlowStateCost
from .costs_ref_style import RefStyleStateCost
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
    cost_style: str = "priority"
    optimize_space: str = "action"
    anneal_proposal: bool = False   # action_prox: scale proposal std by sqrt((1-a)/a) per step (opt-in)


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
        self._warm_action: torch.Tensor | None = None
        self._warm_state: torch.Tensor | None = None
        if config.cost_style == "ref_style":
            self.cost = RefStyleStateCost(config.task_name)
        elif config.cost_style == "explore":
            self.cost = ExploreStateCost(config.task_name)
        elif config.cost_style == "grasp_flow":
            self.cost = GraspFlowStateCost(config.task_name)
        elif config.cost_style == "capsule_flow":
            self.cost = CapsuleFlowStateCost(config.task_name)
        elif config.cost_style == "priority":
            self.cost = PriorityStateCost(config.task_name)
        else:
            raise ValueError(f"Unknown sim-free MPC cost_style: {config.cost_style!r}")
        if config.optimize_space not in ("action", "accel"):
            raise ValueError(f"Unknown sim-free MPC optimize_space: {config.optimize_space!r}")

    def reset_action_warm(self) -> None:
        self._warm_action = None
        self._warm_state = None

    def set_warm_action(
        self,
        action: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> None:
        if action.ndim != 3:
            raise ValueError(f"Expected warm action [B,H,D], got {tuple(action.shape)}")
        self._warm_action = action.detach().clone()
        self._warm_state = None if state is None else torch.as_tensor(state).detach().clone()

    def warm_start_noise(
        self,
        fallback_noise: torch.Tensor,
        *,
        shift_steps: int,
        current_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Shift and rebase the previous action as the next initial noise."""
        warm_action = self._warm_action
        if warm_action is None or warm_action.shape != fallback_noise.shape:
            return fallback_noise, False

        warm_action = warm_action.to(device=fallback_noise.device, dtype=fallback_noise.dtype)
        horizon = warm_action.shape[1]
        shift = min(max(int(shift_steps), 0), horizon)
        if shift == 0:
            shifted_action = warm_action.clone()
        elif shift == horizon:
            shifted_action = warm_action[:, -1:, :].expand_as(warm_action).clone()
        else:
            tail = warm_action[:, -1:, :].expand(-1, shift, -1)
            shifted_action = torch.cat((warm_action[:, shift:, :], tail), dim=1)

        warm_state = self._warm_state
        if warm_state is not None and current_state is not None:
            shifted_action = rebase_model_action_chunk(
                self.policy,
                shifted_action,
                previous_state=warm_state,
                current_state=current_state,
            )
        return shifted_action, True

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

    @staticmethod
    def _bspline_basis(
        num_control_points: int,
        output_horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        degree: int = 3,
    ) -> torch.Tensor:
        if num_control_points <= 0 or output_horizon <= 0:
            raise ValueError("B-spline interpolation expects positive horizon sizes.")
        if num_control_points == 1:
            return torch.ones(output_horizon, 1, device=device, dtype=dtype)

        degree = min(degree, num_control_points - 1)
        interior_count = num_control_points - degree - 1
        start = torch.zeros(degree + 1, device=device, dtype=dtype)
        end = torch.ones(degree + 1, device=device, dtype=dtype)
        if interior_count > 0:
            interior = torch.linspace(0.0, 1.0, interior_count + 2, device=device, dtype=dtype)[1:-1]
            knots = torch.cat([start, interior, end])
        else:
            knots = torch.cat([start, end])

        u = torch.linspace(0.0, 1.0, output_horizon, device=device, dtype=dtype)
        basis = torch.stack(
            [((u >= knots[i]) & (u < knots[i + 1])).to(dtype) for i in range(knots.numel() - 1)],
            dim=1,
        )
        for level in range(1, degree + 1):
            cols = basis.shape[1] - 1
            next_basis = []
            for i in range(cols):
                left_den = knots[i + level] - knots[i]
                if torch.abs(left_den) > 0:
                    left = (u - knots[i]) / left_den * basis[:, i]
                else:
                    left = torch.zeros_like(u)

                right_den = knots[i + level + 1] - knots[i + 1]
                if torch.abs(right_den) > 0:
                    right = (knots[i + level + 1] - u) / right_den * basis[:, i + 1]
                else:
                    right = torch.zeros_like(u)
                next_basis.append(left + right)
            basis = torch.stack(next_basis, dim=1)

        terminal = u == 1.0
        if terminal.any():
            basis[terminal] = 0.0
            basis[terminal, -1] = 1.0
        row_sum = torch.clamp(basis.sum(dim=1, keepdim=True), min=torch.finfo(dtype).eps)
        return basis / row_sum

    @classmethod
    def _bspline_resample(cls, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if sequence.shape[-2] == output_horizon:
            return sequence
        if sequence.shape[-2] == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])

        original_ndim = sequence.ndim
        if original_ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence [H,D] or [B,H,D], got {tuple(sequence.shape)}")

        basis = cls._bspline_basis(
            sequence.shape[1],
            output_horizon,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        resampled = torch.einsum("oh,bhd->bod", basis, sequence)
        if original_ndim == 2:
            return resampled[0]
        return resampled

    def _control_point_resample(self, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        return self._linear_resample(sequence, output_horizon)

    def _interpolate_control_points(self, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if not self.config.interpolate:
            return self._linear_resample(sequence, output_horizon)
        return self._bspline_resample(sequence, output_horizon)

    @staticmethod
    def _trajectory_to_accel_code(sequence: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory as a same-shaped acceleration-space code.

        The code is a linear, invertible coordinate transform:
        code[0] is the first position, code[1] is the first velocity, and
        code[2:] are second differences. MBD/DDIM updates must happen in this
        code space if sampling is done in acceleration space; updating the
        original action trajectory with an acceleration-space weighted mean
        mixes coordinates and gives inconsistent reverse steps.
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected sequence [H,D], got {tuple(sequence.shape)}")
        code = torch.zeros_like(sequence)
        code[0] = sequence[0]
        if sequence.shape[0] > 1:
            code[1] = sequence[1] - sequence[0]
        if sequence.shape[0] > 2:
            code[2:] = sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2]
        return code

    @staticmethod
    def _accel_code_to_trajectory(code: torch.Tensor) -> torch.Tensor:
        if code.ndim != 3:
            raise ValueError(f"Expected accel code [K,H,D], got {tuple(code.shape)}")
        horizon = code.shape[1]
        if horizon == 0:
            return code
        traj = [code[:, 0, :]]
        if horizon > 1:
            traj.append(traj[0] + code[:, 1, :])
        for step in range(2, horizon):
            traj.append(2.0 * traj[-1] - traj[-2] + code[:, step, :])
        return torch.stack(traj, dim=1)

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
        mean0 = self._control_point_resample(x_t[0, :, :active_dims].detach(), opt_horizon)

        def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        if self.config.optimize_space == "accel":
            mean_accel = self._trajectory_to_accel_code(mean0)

            def cost_fn(samples: torch.Tensor) -> torch.Tensor:
                positions = self._accel_code_to_trajectory(samples)
                return cost_from_positions(positions)

            result = self.sampler.optimize(mean_accel, cost_fn)
            result_mean = self._accel_code_to_trajectory(result.mean.unsqueeze(0))[0]
        else:
            result = self.sampler.optimize(mean0, cost_from_positions)
            result_mean = result.mean
        target = x_t.detach().clone()
        target[:, :, :active_dims] = self._interpolate_control_points(result_mean, horizon).unsqueeze(0)
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
            apply_clamp=False,
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
        if self.config.cost_style in ("ref_style", "explore", "grasp_flow", "capsule_flow"):
            return self.cost(real_actions=real, tcp_pos=ee_pos, tcp_quat=ee_quat, context=context)
        return self.cost(real_actions=real, ee_pos=ee_pos, ee_quat=ee_quat, context=context)

    def _last_cost_term_diagnostics(self, result) -> dict[str, Any]:
        cost = getattr(self, "cost", None)
        terms = getattr(cost, "last_terms", None)
        if not terms:
            return {}
        best_idx = int(torch.argmin(result.costs).detach().cpu())
        diagnostics: dict[str, Any] = {}
        stage = getattr(cost, "last_stage", None)
        if stage is not None:
            diagnostics["cost_stage"] = stage
        for name, values in terms.items():
            if values.ndim != 1 or values.shape[0] != result.costs.shape[0]:
                continue
            values = values.to(device=result.weights.device, dtype=result.weights.dtype)
            diagnostics[f"term_{name}_best"] = float(values[best_idx].detach().cpu())
            diagnostics[f"term_{name}_weighted"] = float(torch.sum(values * result.weights).detach().cpu())
        debug_values = getattr(cost, "last_debug", None)
        if debug_values:
            for name, values in debug_values.items():
                if values.ndim != 1 or values.shape[0] != result.costs.shape[0]:
                    continue
                values = values.to(device=result.weights.device, dtype=result.weights.dtype)
                diagnostics[f"debug_{name}_best"] = float(values[best_idx].detach().cpu())
                diagnostics[f"debug_{name}_weighted"] = float(torch.sum(values * result.weights).detach().cpu())
        return diagnostics

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
        active_x_opt = self._control_point_resample(x_t.detach()[0, :, :active_dims], opt_horizon)
        clean_std = self.config.noise * ddim_clean_sample_std_scale(alpha_bar)

        def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        if self.config.optimize_space == "accel":
            active_code = self._trajectory_to_accel_code(active_x_opt)
            mean_accel = active_code / torch.clamp(sqrt_alpha, min=self.config.flow_eps)

            def cost_fn(samples: torch.Tensor) -> torch.Tensor:
                positions = self._accel_code_to_trajectory(samples)
                return cost_from_positions(positions)

            result = self.sampler.optimize_with_noise_scale(mean_accel, cost_fn, noise_scale=clean_std)
            result_mean = self._accel_code_to_trajectory(result.mean.unsqueeze(0))[0]
        else:
            clean_center = active_x_opt / torch.clamp(sqrt_alpha, min=self.config.flow_eps)
            result = self.sampler.optimize_with_noise_scale(
                clean_center,
                cost_from_positions,
                noise_scale=clean_std,
            )
            result_mean = result.mean
        x0_hat = x_t.detach().clone()
        x0_hat[:, :, :active_dims] = self._interpolate_control_points(result_mean, horizon).unsqueeze(0)
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
            "cost_style": self.config.cost_style,
            "optimize_space": self.config.optimize_space,
        }
        if self.config.interpolate:
            diagnostics.update(
                {
                    "interpolate_horizon": int(x_t.shape[1]),
                    "interpolate_knot_count": int(self._interpolation_knot_count(x_t.shape[1])),
                    "interpolate_frequency": float(self.config.interpolate_frequency),
                    "control_frequency": float(self.config.control_frequency),
                    "interpolate_method": "clamped_cubic_bspline",
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
        diagnostics.update(self._last_cost_term_diagnostics(result))
        return diagnostics

    def _optimize_action_prox_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        alpha_bar: float,
    ):
        """Optimize action candidates directly around the current noisy action."""
        if x_t.shape[0] != 1:
            raise ValueError("MBD action-space sampler currently expects batch size 1.")
        if self.config.optimize_space != "action":
            raise ValueError("MBD action-prox/warm paths only support action optimize_space.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        proposal_center = self._control_point_resample(
            x_t.detach()[0, :, :active_dims],
            opt_horizon,
        )
        proposal_std = float(self.config.noise) * math.sqrt(
            max(1.0 - float(alpha_bar), 0.0)
        )

        def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            return self._cost_active_samples(
                full_horizon_samples,
                x_t,
                active_dims,
                policy_inputs,
                context,
            )

        result = self.sampler.optimize_with_noise_scale(
            proposal_center,
            cost_from_positions,
            noise_scale=proposal_std,
        )
        x0_hat = x_t.detach().clone()
        x0_hat[:, :, :active_dims] = self._interpolate_control_points(
            result.mean,
            horizon,
        ).unsqueeze(0)
        return x0_hat, result, active_dims, proposal_std

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
            self._interpolate_control_points(
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
        if self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            code_score = (-active_code + sqrt_alpha * clean_code) / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(code_score.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            pred_epsilon = (active_code - sqrt_alpha * clean_code) / sqrt_beta
            prev_code = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * clean_code
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_epsilon
            )
            active_prev = self._interpolate_control_points(
                self._accel_code_to_trajectory(prev_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
        else:
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

        alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=self.config.flow_eps), min=self.config.flow_eps)
        if self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            score_numerator_code = sqrt_alpha * clean_code - active_code
            score_code = score_numerator_code / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(score_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            prev_code = (
                active_code + float(score_scale) * score_numerator_code
            ) / torch.sqrt(alpha_step)
            active_prev = self._interpolate_control_points(
                self._accel_code_to_trajectory(prev_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
        else:
            score_numerator = sqrt_alpha * active_x0 - active_x
            score[:, :, :active_dims] = score_numerator / beta
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

    def step_mbd_score_action_prox(
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
        x0_hat, result, active_dims, proposal_std = self._optimize_action_prox_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        score_numerator = sqrt_alpha * active_x0 - active_x
        score = torch.zeros_like(x_t)
        score[:, :, :active_dims] = score_numerator / beta

        alpha_step = torch.clamp(
            alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
            min=self.config.flow_eps,
        )
        active_prev = (
            active_x + float(score_scale) * score_numerator
        ) / torch.sqrt(alpha_step)

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "mbd_score_action_prox",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "active_dims": int(active_dims),
                "proposal_center": "current_noisy_action",
                "proposal_noise_scale": float(proposal_std),
                "clean_sample_std": float(proposal_std),
                "score_scale": float(score_scale),
                "mbd_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
            }
        )
        return next_x, diagnostics

    def step_mbd_score_action_warm(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        score_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        next_x, diagnostics = self.step_mbd_score_action_prox(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
            score_scale=score_scale,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "mbd_score_action_warm"
        return next_x, diagnostics

    def estimate_mbd_score(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, _, diagnostics = self.estimate_mbd_score_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        return score, diagnostics

    def estimate_mbd_score_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        numerator = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]

        if self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            score_code = (sqrt_alpha * clean_code - active_code) / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(score_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            numerator[:, :, :active_dims] = beta * score[:, :, :active_dims]
        else:
            active_numerator = sqrt_alpha * active_x0 - active_x
            numerator[:, :, :active_dims] = active_numerator
            score[:, :, :active_dims] = active_numerator / beta

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "estimate_mbd_score",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "active_dims": int(active_dims),
                "clean_sample_std": float(clean_std),
            }
        )
        return score, numerator, diagnostics

    def estimate_mbd_score_action_prox(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, _, diagnostics = self.estimate_mbd_score_action_prox_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        return score, diagnostics

    def estimate_mbd_score_action_prox_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        x0_hat, result, active_dims, proposal_std = self._optimize_action_prox_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        numerator = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        active_numerator = sqrt_alpha * active_x0 - active_x
        numerator[:, :, :active_dims] = active_numerator
        score[:, :, :active_dims] = active_numerator / beta

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
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
        return score, numerator, diagnostics

    def estimate_mbd_score_action_warm(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, diagnostics = self.estimate_mbd_score_action_prox(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "estimate_mbd_score_action_warm"
        return score, diagnostics

    def estimate_mbd_score_action_warm_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        score, numerator, diagnostics = self.estimate_mbd_score_action_prox_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "estimate_mbd_score_action_warm"
        return score, numerator, diagnostics

    def step_from_mbd_residual(
        self,
        x_t: torch.Tensor,
        base_numerator: torch.Tensor,
        residual_score: torch.Tensor,
        *,
        iteration: int,
        num_iterations: int,
        base_scale: float = 1.0,
        residual_scale: float = 1.0,
        active_dims: int | None = None,
    ) -> torch.Tensor:
        """Apply score residual while preserving the direct MBD base update."""
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        if base_numerator.shape != x_t.shape or residual_score.shape != x_t.shape:
            raise ValueError(
                "base_numerator and residual_score must match x_t: "
                f"base={tuple(base_numerator.shape)}, residual={tuple(residual_score.shape)}, "
                f"x_t={tuple(x_t.shape)}"
            )
        if active_dims is None:
            active_dims = min(self.config.action_dims, x_t.shape[-1])
        active_dims = min(int(active_dims), x_t.shape[-1])

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        alpha_step = torch.clamp(
            alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
            min=self.config.flow_eps,
        )

        active_x = x_t.detach()[:, :, :active_dims]
        active_prev = (
            active_x
            + float(base_scale) * base_numerator[:, :, :active_dims]
            + float(residual_scale) * beta * residual_score[:, :, :active_dims]
        ) / torch.sqrt(alpha_step)
        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev
        return next_x

    def step_from_score(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        *,
        iteration: int,
        num_iterations: int,
        update_mode: str,
        score_scale: float = 1.0,
        active_dims: int | None = None,
    ) -> torch.Tensor:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        if score.shape != x_t.shape:
            raise ValueError(
                f"score shape must match x_t shape, got {tuple(score.shape)} vs {tuple(x_t.shape)}"
            )
        if active_dims is None:
            active_dims = min(self.config.action_dims, x_t.shape[-1])
        active_dims = min(int(active_dims), x_t.shape[-1])

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)

        active_x = x_t.detach()[:, :, :active_dims]
        active_score = score[:, :, :active_dims]
        next_x = x_t.detach().clone()

        if update_mode == "mbd_score":
            alpha_step = torch.clamp(
                alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
                min=self.config.flow_eps,
            )
            active_prev = (
                active_x + float(score_scale) * beta * active_score
            ) / torch.sqrt(alpha_step)
        elif update_mode == "ddim":
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))
            sqrt_beta = torch.sqrt(beta)
            x0_hat = (active_x + beta * active_score) / sqrt_alpha
            eps_hat = -sqrt_beta * active_score
            active_prev = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps_hat
            )
            active_prev = active_x + float(score_scale) * (active_prev - active_x)
        else:
            raise ValueError(
                "External score updates support update_mode='ddim' or 'mbd_score', "
                f"got {update_mode!r}."
            )

        next_x[:, :, :active_dims] = active_prev
        return next_x
