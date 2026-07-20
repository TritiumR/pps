from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DIALSamplerConfig:
    num_samples: int = 64
    iterations: int = 2
    noise: float = 0.35
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    temperature: float = 0.15
    action_dims: int = 8
    joint_limit_margin: float = 0.0


@dataclass(frozen=True)
class DIALResult:
    mean: torch.Tensor
    costs: torch.Tensor
    weights: torch.Tensor
    samples: torch.Tensor
    noise_scale: torch.Tensor


class DIALSampler:
    """Minimal DIAL-style annealed sampler in PyTorch.

    This implements the useful DIAL-MPC sampling/update shape without depending
    on Hydrax's rollout abstraction.
    """

    def __init__(self, config: DIALSamplerConfig):
        self.config = config
        self.proposal_fn = None

    def optimize(
        self,
        initial_mean: torch.Tensor,
        cost_fn,
        *,
        generator: torch.Generator | None = None,
    ) -> DIALResult:
        if initial_mean.ndim != 2:
            raise ValueError(f"Expected initial_mean [H,D], got {tuple(initial_mean.shape)}")

        mean = initial_mean
        horizon = mean.shape[0]
        last_costs = None
        last_weights = None
        last_samples = None
        last_noise_scale = None

        for opt_iter in range(self.config.iterations):
            horizon_idx = torch.arange(horizon, device=mean.device, dtype=mean.dtype)
            noise_level = self.config.noise * torch.exp(
                -torch.as_tensor(opt_iter, device=mean.device, dtype=mean.dtype)
                / (self.config.beta_opt_iter * max(self.config.iterations, 1))
                - (horizon - 1 - horizon_idx)
                / (self.config.beta_horizon * max(horizon, 1))
            )
            if self.proposal_fn is None:
                noise = torch.randn(
                    (self.config.num_samples, *mean.shape),
                    device=mean.device,
                    dtype=mean.dtype,
                    generator=generator,
                )
                samples = mean.unsqueeze(0) + noise * noise_level[None, :, None]
            else:
                samples = self.proposal_fn(
                    mean,
                    noise_level,
                    int(self.config.num_samples),
                    generator,
                )
                if samples.shape != (self.config.num_samples, *mean.shape):
                    raise ValueError(
                        "proposal_fn returned "
                        f"{tuple(samples.shape)}, expected "
                        f"{(self.config.num_samples, *mean.shape)}"
                    )
                samples = samples.to(device=mean.device, dtype=mean.dtype)
            costs = cost_fn(samples)
            if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                raise ValueError(
                    "cost_fn must return [num_samples], got "
                    f"{tuple(costs.shape)} for {self.config.num_samples} samples"
                )
            weights = torch.softmax(-costs / max(self.config.temperature, 1e-6), dim=0)
            mean = torch.sum(weights[:, None, None] * samples, dim=0)

            last_costs = costs
            last_weights = weights
            last_samples = samples
            last_noise_scale = noise_level

        assert (
            last_costs is not None
            and last_weights is not None
            and last_samples is not None
            and last_noise_scale is not None
        )
        return DIALResult(
            mean=mean,
            costs=last_costs,
            weights=last_weights,
            samples=last_samples,
            noise_scale=last_noise_scale,
        )

    def optimize_with_noise_scale(
        self,
        initial_mean: torch.Tensor,
        cost_fn,
        *,
        noise_scale: torch.Tensor | float,
        generator: torch.Generator | None = None,
    ) -> DIALResult:
        """Optimize samples around a clean DDIM center with a fixed proposal std.

        Unlike `optimize`, this does not apply the legacy exponential
        opt-iteration/horizon noise schedule. The caller supplies the DDIM
        posterior clean-sample scale for the current reverse step.
        """
        if initial_mean.ndim != 2:
            raise ValueError(f"Expected initial_mean [H,D], got {tuple(initial_mean.shape)}")

        mean = initial_mean
        scale = torch.as_tensor(noise_scale, device=mean.device, dtype=mean.dtype)
        if scale.ndim == 0:
            scale_view = scale.view(1, 1, 1)
            returned_scale = scale.expand(mean.shape[0])
        elif scale.ndim == 1:
            if scale.shape[0] != mean.shape[0]:
                raise ValueError(
                    f"Expected noise_scale [H] with H={mean.shape[0]}, got {tuple(scale.shape)}"
                )
            scale_view = scale.view(1, mean.shape[0], 1)
            returned_scale = scale
        elif scale.shape == mean.shape:
            scale_view = scale.unsqueeze(0)
            returned_scale = scale.mean(dim=-1)
        else:
            raise ValueError(
                "noise_scale must be scalar, [H], or [H,D], got "
                f"{tuple(scale.shape)} for mean {tuple(mean.shape)}"
            )

        last_costs = None
        last_weights = None
        last_samples = None

        for _ in range(self.config.iterations):
            if self.proposal_fn is None:
                noise = torch.randn(
                    (self.config.num_samples, *mean.shape),
                    device=mean.device,
                    dtype=mean.dtype,
                    generator=generator,
                )
                samples = mean.unsqueeze(0) + noise * scale_view
                samples[0] = mean
            else:
                samples = self.proposal_fn(
                    mean,
                    scale,
                    int(self.config.num_samples),
                    generator,
                )
                if samples.shape != (self.config.num_samples, *mean.shape):
                    raise ValueError(
                        "proposal_fn returned "
                        f"{tuple(samples.shape)}, expected "
                        f"{(self.config.num_samples, *mean.shape)}"
                    )
                samples = samples.to(device=mean.device, dtype=mean.dtype)
            costs = cost_fn(samples)
            if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                raise ValueError(
                    "cost_fn must return [num_samples], got "
                    f"{tuple(costs.shape)} for {self.config.num_samples} samples"
                )
            weights = torch.softmax(-costs / max(self.config.temperature, 1e-6), dim=0)
            mean = torch.sum(weights[:, None, None] * samples, dim=0)

            last_costs = costs
            last_weights = weights
            last_samples = samples

        assert last_costs is not None and last_weights is not None and last_samples is not None
        return DIALResult(
            mean=mean,
            costs=last_costs,
            weights=last_weights,
            samples=last_samples,
            noise_scale=returned_scale,
        )
