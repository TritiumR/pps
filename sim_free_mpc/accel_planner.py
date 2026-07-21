from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .costs import PriorityStateCost
from .costs_explore import ExploreStateCost
from .costs_grasp_flow import GraspFlowStateCost
from .costs_grasp_flow_ex import GraspFlowStateCost as ExtendedGraspFlowStateCost
from .costs_grasp_flow_fake import GraspFlowStateCost as FakeGraspFlowStateCost
from .costs_grasp_flow_loose import GraspFlowStateCost as LooseGraspFlowStateCost
from .costs_ref_style import RefStyleStateCost
from .ddim import ddim_clean_sample_std_scale, ddim_iteration_alphas
from .dial_sampler import DIALSampler, DIALSamplerConfig
from .fk import PANDA_JOINT_LIMITS, PandaFK, quat_mul_wxyz, transform_points_wxyz


@dataclass(frozen=True)
class AccelMPCConfig:
    task_name: str = "auto"
    num_samples: int = 512
    iterations: int = 8
    noise: float = 0.5
    temperature: float = 0.7
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    control_frequency: float = 15.0
    cost_style: str = "ref_style"
    clamp_joint_limits: bool = True
    ddim_num_train_timesteps: int = 100
    flow_eps: float = 1e-6


class AccelActionMPC:
    """Pure MPPI planner over real joint accelerations.

    The decision variable is joint acceleration u[t] in action space. Samples
    are integrated to q trajectories before FK/cost evaluation, and the final
    output is an executable [H,8] action chunk.
    """

    def __init__(self, config: AccelMPCConfig):
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
                action_dims=8,
            )
        )
        if config.cost_style == "ref_style":
            self.cost = RefStyleStateCost(config.task_name)
        elif config.cost_style == "explore":
            self.cost = ExploreStateCost(config.task_name)
        elif config.cost_style == "grasp_flow":
            self.cost = GraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_ex":
            self.cost = ExtendedGraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_fake":
            self.cost = FakeGraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_loose":
            self.cost = LooseGraspFlowStateCost(config.task_name)
        elif config.cost_style == "priority":
            self.cost = PriorityStateCost(config.task_name)
        else:
            raise ValueError(f"Unknown accel MPC cost_style: {config.cost_style!r}")

    @staticmethod
    def _context_tensor(
        context: dict[str, Any],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        value = context.get(key)
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=dtype)
        if value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        return value

    @staticmethod
    def _joint_limits(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        limits = torch.as_tensor(PANDA_JOINT_LIMITS, device=device, dtype=dtype)
        return limits[:, 0], limits[:, 1]

    def _current_joint_state(
        self,
        context: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q0 = self._context_tensor(context, "joint_pos", device=device, dtype=dtype)
        if q0 is None:
            raise ValueError("AccelActionMPC requires context['joint_pos'].")
        q0 = q0.flatten()[:7]
        if q0.shape[0] != 7:
            raise ValueError(f"Expected 7 arm joints in context['joint_pos'], got {tuple(q0.shape)}")

        qd0 = self._context_tensor(context, "joint_vel", device=device, dtype=dtype)
        if qd0 is None:
            qd0 = torch.zeros_like(q0)
        else:
            qd0 = qd0.flatten()[:7]
            if qd0.shape[0] != 7:
                qd0 = torch.zeros_like(q0)
        return q0, qd0

    def integrate(self, u: torch.Tensor, q0: torch.Tensor, qd0: torch.Tensor) -> torch.Tensor:
        if u.ndim != 3 or u.shape[-1] != 7:
            raise ValueError(f"Expected acceleration samples [B,H,7], got {tuple(u.shape)}")
        batch = u.shape[0]
        dt = 1.0 / max(float(self.config.control_frequency), 1e-6)
        q = q0.to(device=u.device, dtype=u.dtype).view(1, 7).expand(batch, 7).clone()
        qd = qd0.to(device=u.device, dtype=u.dtype).view(1, 7).expand(batch, 7).clone()
        if self.config.clamp_joint_limits:
            q_lo, q_hi = self._joint_limits(u.device, u.dtype)
        traj = []
        for step in range(u.shape[1]):
            qd = qd + u[:, step, :] * dt
            q = q + qd * dt
            if self.config.clamp_joint_limits:
                q = torch.clamp(q, q_lo, q_hi)
            traj.append(q)
        return torch.stack(traj, dim=1)

    @staticmethod
    def _gripper_horizon(gripper_traj: torch.Tensor) -> int:
        if gripper_traj.ndim == 2 and gripper_traj.shape[-1] == 1:
            return gripper_traj.shape[0]
        if gripper_traj.ndim == 1:
            return gripper_traj.shape[0]
        raise ValueError(f"Expected gripper_traj [H] or [H,1], got {tuple(gripper_traj.shape)}")

    def _prepare_gripper(
        self,
        gripper_traj: torch.Tensor,
        *,
        device: torch.device | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, torch.device]:
        if not torch.is_tensor(gripper_traj):
            gripper_traj = torch.as_tensor(gripper_traj)
        horizon = self._gripper_horizon(gripper_traj)
        if device is None:
            device = gripper_traj.device
        return gripper_traj.to(device=device, dtype=dtype), horizon, device

    def _world_fk(
        self,
        q_traj: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fk = self.fk.forward(q_traj)
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
        return ee_pos, ee_quat

    def _cost_q_traj(
        self,
        q_traj: torch.Tensor,
        gripper_traj: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        if gripper_traj.ndim == 1:
            gripper = gripper_traj.view(1, -1, 1).expand(q_traj.shape[0], -1, -1)
        elif gripper_traj.ndim == 2:
            gripper = gripper_traj.view(1, gripper_traj.shape[0], gripper_traj.shape[1]).expand(
                q_traj.shape[0], -1, -1
            )
        elif gripper_traj.ndim == 3 and gripper_traj.shape[0] == q_traj.shape[0]:
            gripper = gripper_traj
        else:
            raise ValueError(f"Unexpected gripper trajectory shape: {tuple(gripper_traj.shape)}")
        if gripper.shape[1] != q_traj.shape[1]:
            raise ValueError(
                f"Gripper horizon {gripper.shape[1]} does not match q horizon {q_traj.shape[1]}."
            )
        real_actions = torch.cat([q_traj, gripper[..., :1]], dim=-1)
        ee_pos, ee_quat = self._world_fk(q_traj, context)
        if self.config.cost_style in (
            "ref_style",
            "explore",
            "grasp_flow",
            "grasp_flow_ex",
            "grasp_flow_fake",
            "grasp_flow_loose",
        ):
            return self.cost(real_actions=real_actions, tcp_pos=ee_pos, tcp_quat=ee_quat, context=context)
        return self.cost(real_actions=real_actions, ee_pos=ee_pos, ee_quat=ee_quat, context=context)

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

    @staticmethod
    def _initial_hybrid_mean(horizon: int, gripper_traj: torch.Tensor, *, device, dtype) -> torch.Tensor:
        mean = torch.zeros(horizon, 8, device=device, dtype=dtype)
        mean[:, 7:8] = torch.clamp(gripper_traj.reshape(horizon, -1)[..., :1], 0.0, 1.0)
        return mean

    def _hybrid_samples_to_actions(
        self,
        samples: torch.Tensor,
        q0: torch.Tensor,
        qd0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if samples.ndim != 3 or samples.shape[-1] != 8:
            raise ValueError(f"Expected hybrid accel/gripper samples [B,H,8], got {tuple(samples.shape)}")
        q_traj = self.integrate(samples[..., :7], q0, qd0)
        gripper = torch.clamp(samples[..., 7:8], 0.0, 1.0)
        return q_traj, gripper

    def plan(
        self,
        *,
        context: dict[str, Any],
        gripper_traj: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        gripper_traj, horizon, device = self._prepare_gripper(gripper_traj, device=device, dtype=dtype)
        q0, qd0 = self._current_joint_state(context, device=device, dtype=dtype)
        mean_z = self._initial_hybrid_mean(horizon, gripper_traj, device=device, dtype=dtype)

        def cost_fn(samples: torch.Tensor) -> torch.Tensor:
            q_traj, gripper = self._hybrid_samples_to_actions(samples, q0, qd0)
            return self._cost_q_traj(q_traj, gripper, context)

        result = self.sampler.optimize(mean_z, cost_fn)
        q_star, gripper_star = self._hybrid_samples_to_actions(result.mean.unsqueeze(0), q0, qd0)
        q_star = q_star[0]
        gripper_star = gripper_star[0]
        action_chunk = torch.cat([q_star, gripper_star], dim=-1)
        weighted_cost = torch.sum(result.costs * result.weights)
        diagnostics = {
            "update_mode": "accel_mppi",
            "cost_min": float(result.costs.min().detach().cpu()),
            "cost_mean": float(result.costs.mean().detach().cpu()),
            "cost_weighted": float(weighted_cost.detach().cpu()),
            "accel_norm": float(torch.linalg.vector_norm(result.mean[..., :7].detach()).cpu()),
            "gripper_mean": float(gripper_star.detach().mean().cpu()),
            "action_delta_norm": float(torch.linalg.vector_norm((q_star[0] - q0).detach()).cpu()),
            "target_delta_norm": float(torch.linalg.vector_norm((q_star - q0.view(1, 7)).detach()).cpu()),
            "score_norm": 0.0,
            "control_frequency": float(self.config.control_frequency),
            "cost_style": self.config.cost_style,
            "optimize_space": "accel_action",
        }
        diagnostics.update(self._last_cost_term_diagnostics(result))
        return action_chunk, diagnostics

    def plan_mbd_score(
        self,
        *,
        context: dict[str, Any],
        gripper_traj: torch.Tensor,
        num_iterations: int,
        score_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
        gripper_traj, horizon, device = self._prepare_gripper(gripper_traj, device=device, dtype=dtype)
        q0, qd0 = self._current_joint_state(context, device=device, dtype=dtype)
        if int(num_iterations) <= 0:
            raise ValueError("num_iterations must be positive for plan_mbd_score.")
        alpha_start, _ = ddim_iteration_alphas(
            iteration=0,
            num_iterations=int(num_iterations),
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        sqrt_alpha_start = torch.sqrt(torch.as_tensor(alpha_start, device=device, dtype=dtype))
        sqrt_beta_start = torch.sqrt(torch.clamp(1.0 - torch.as_tensor(alpha_start, device=device, dtype=dtype), min=0.0))
        z_t = self._initial_hybrid_mean(horizon, gripper_traj, device=device, dtype=dtype)
        z_t[:, :7] = torch.randn(horizon, 7, device=device, dtype=dtype) * float(self.config.noise)
        z_t[:, 7:8] = (
            sqrt_alpha_start * z_t[:, 7:8]
            + sqrt_beta_start
            * torch.randn(horizon, 1, device=device, dtype=dtype)
            * min(float(self.config.noise), 0.35)
        )
        last_result = None
        last_score = None
        last_clean = None
        last_clean_std = None
        last_alpha_bar = None
        last_alpha_bar_prev = None

        for iteration in range(int(num_iterations)):
            alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=int(num_iterations),
                num_train_timesteps=self.config.ddim_num_train_timesteps,
            )
            alpha = torch.as_tensor(alpha_bar, device=device, dtype=dtype)
            alpha_prev = torch.as_tensor(alpha_bar_prev, device=device, dtype=dtype)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))
            beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
            clean_center = z_t / sqrt_alpha
            clean_std = float(self.config.noise) * ddim_clean_sample_std_scale(alpha_bar)

            def cost_fn(samples: torch.Tensor) -> torch.Tensor:
                q_traj, gripper = self._hybrid_samples_to_actions(samples, q0, qd0)
                return self._cost_q_traj(q_traj, gripper, context)

            result = self.sampler.optimize_with_noise_scale(
                clean_center,
                cost_fn,
                noise_scale=clean_std,
            )
            z0_hat = result.mean.to(device=device, dtype=dtype)
            score_numerator = sqrt_alpha * z0_hat - z_t
            score = score_numerator / beta
            alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=self.config.flow_eps), min=self.config.flow_eps)
            z_t = (z_t + float(score_scale) * score_numerator) / torch.sqrt(alpha_step)
            z_t[:, 7:8] = torch.clamp(z_t[:, 7:8], 0.0, 1.0)

            last_result = result
            last_score = score
            last_clean = z0_hat
            last_clean_std = clean_std
            last_alpha_bar = alpha_bar
            last_alpha_bar_prev = alpha_bar_prev

        q_star, gripper_star = self._hybrid_samples_to_actions(z_t.unsqueeze(0), q0, qd0)
        q_star = q_star[0]
        gripper_star = gripper_star[0]
        action_chunk = torch.cat([q_star, gripper_star], dim=-1)
        weighted_cost = torch.sum(last_result.costs * last_result.weights)
        diagnostics = {
            "update_mode": "accel_mbd_score",
            "cost_min": float(last_result.costs.min().detach().cpu()),
            "cost_mean": float(last_result.costs.mean().detach().cpu()),
            "cost_weighted": float(weighted_cost.detach().cpu()),
            "accel_norm": float(torch.linalg.vector_norm(z_t[..., :7].detach()).cpu()),
            "clean_accel_norm": float(torch.linalg.vector_norm(last_clean[..., :7].detach()).cpu()),
            "gripper_mean": float(gripper_star.detach().mean().cpu()),
            "action_delta_norm": float(torch.linalg.vector_norm((q_star[0] - q0).detach()).cpu()),
            "target_delta_norm": float(torch.linalg.vector_norm((q_star - q0.view(1, 7)).detach()).cpu()),
            "score_norm": float(torch.linalg.vector_norm(last_score.detach()).cpu()),
            "score_abs_mean": float(last_score.detach().abs().mean().cpu()),
            "clean_sample_std": float(last_clean_std),
            "alpha_bar": float(last_alpha_bar),
            "alpha_bar_prev": float(last_alpha_bar_prev),
            "ddim_num_iterations": int(num_iterations),
            "score_scale": float(score_scale),
            "control_frequency": float(self.config.control_frequency),
            "cost_style": self.config.cost_style,
            "optimize_space": "accel_action",
        }
        diagnostics.update(self._last_cost_term_diagnostics(last_result))
        return action_chunk, diagnostics
