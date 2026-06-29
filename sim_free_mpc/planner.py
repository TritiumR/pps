from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .action_space import decode_model_action_chunks
from .costs import PriorityStateCost
from .dial_sampler import DIALSampler, DIALSamplerConfig
from .fk import PandaFK, transform_points_wxyz


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

    def step(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        dt: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if x_t.shape[0] != 1:
            raise ValueError("SimFreeMPC MVP currently expects batch size 1.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        mean0 = x_t[0, :, :active_dims].detach()

        def cost_fn(samples: torch.Tensor) -> torch.Tensor:
            full = x_t.detach().repeat(samples.shape[0], 1, 1)
            full[:, :, :active_dims] = samples
            decoded = decode_model_action_chunks(self.policy, policy_inputs, full)
            real = decoded.real_actions
            joints = real[..., :7]
            fk = self.fk.forward(joints)
            ee_pos = fk.ee_pos
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
            return self.cost(real_actions=real, ee_pos=ee_pos, context=context)

        result = self.sampler.optimize(mean0, cost_fn)
        target = x_t.detach().clone()
        target[:, :, :active_dims] = result.mean.unsqueeze(0)

        dt_abs = torch.abs(torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype))
        flow = -(target - x_t.detach()) / torch.clamp(dt_abs, min=self.config.flow_eps)

        diagnostics = {
            "cost_min": float(result.costs.min().detach().cpu()),
            "cost_mean": float(result.costs.mean().detach().cpu()),
            "cost_weighted": float(torch.sum(result.costs * result.weights).detach().cpu()),
            "target_delta_norm": float(torch.linalg.vector_norm((target - x_t).detach()).cpu()),
        }
        return flow, diagnostics
