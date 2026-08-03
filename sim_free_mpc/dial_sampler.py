from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DIALSamplerConfig:
    num_samples: int = 64
    # Softmax restricted to candidates within this many temperature units of the best before the
    # weighted mean, so averaging happens INSIDE a mode instead of across two. inf = old behaviour.
    mode_window: float = float("inf")
    iterations: int = 2
    noise: float = 0.35
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    temperature: float = 0.15
    action_dims: int = 8
    joint_limit_margin: float = 0.0
    logit_norm: str = "raw"  # "raw" | "std" (DIAL-style scale-free logits)


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

    def _call_proposal(self, mean, scale, generator):
        """Run proposal_fn, accepting either `samples` or `(samples, log_importance)`.

        Returning the log-importance is how a non-default proposal keeps the estimator's target
        fixed; a proposal that returns bare samples is asserting it drew from the default Gaussian.
        """
        out = self.proposal_fn(mean, scale, int(self.config.num_samples), generator)
        samples, log_importance = out if isinstance(out, tuple) else (out, None)
        expected = (self.config.num_samples, *mean.shape)
        if samples.shape != expected:
            raise ValueError(f"proposal_fn returned {tuple(samples.shape)}, expected {expected}")
        samples = samples.to(device=mean.device, dtype=mean.dtype)
        if log_importance is not None:
            log_importance = log_importance.to(device=mean.device, dtype=mean.dtype)
            if log_importance.shape != (self.config.num_samples,):
                raise ValueError(
                    f"proposal_fn log_importance {tuple(log_importance.shape)}, "
                    f"expected {(self.config.num_samples,)}")
        return samples, log_importance

    def _mode_restrict(self, weights: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """Average WITHIN the best mode, not across modes.

        The weighted mean of a bimodal population lands in the valley between the modes -- a
        trajectory neither supports -- which is why a proposed branch is re-absorbed immediately.

        The literature's repair is MCMC with Metropolis correction per level, but that needs the
        energy OF THE MEAN and would double the cost calls. Restricting the softmax to candidates
        within `mode_window` temperature units of the best is mode-preserving without that
        evaluation, and inert on unimodal populations. mode_window=inf reproduces the old behaviour.
        """
        window = float(getattr(self.config, "mode_window", float("inf")))
        if not (window > 0.0) or window == float("inf") or costs.numel() < 2:
            return weights
        keep = costs <= (costs.min() + window * max(self.config.temperature, 1e-6))
        if not bool(keep.any()):
            return weights
        w = weights * keep.to(weights.dtype)
        total = w.sum()
        # A window that admits no probability mass would divide by ~0; fall back rather than emit NaN.
        return w / total if float(total) > 1e-12 else weights

    def _weights_from_costs(self, costs: torch.Tensor, log_importance=None) -> torch.Tensor:
        """Softmax weights. logit_norm='std' divides by the batch cost std
        (DIAL dial_core.py:126), making temperature a sharpness in std units,
        constant across noise levels. DIAL's baseline subtraction is omitted:
        softmax is shift-invariant, so it has no effect on the weights.

        log_importance is log p_proposal_default(x) - log q(x) for a proposal_fn that did not draw
        from the default Gaussian. Without it a mixture proposal silently moves the target: the
        estimator would be over exp(-J)*q instead of exp(-J)*p, so injecting expert-centred rows
        hands the expert region prior mass the base never assigned it, and rho stops being a pure
        support knob. Omitting it is only correct when q IS the default Gaussian.
        """
        denom = max(self.config.temperature, 1e-6)
        if self.config.logit_norm == "std":
            denom = costs.std(unbiased=False).clamp_min(1e-6) * denom
        logits = -costs / denom
        if log_importance is not None:
            logits = logits + log_importance
        return torch.softmax(logits, dim=0)

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
                log_importance = None
            else:
                samples, log_importance = self._call_proposal(mean, noise_level, generator)
            costs = cost_fn(samples)
            if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                raise ValueError(
                    "cost_fn must return [num_samples], got "
                    f"{tuple(costs.shape)} for {self.config.num_samples} samples"
                )
            weights = self._weights_from_costs(costs, log_importance)
            weights = self._mode_restrict(weights, costs)
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
                log_importance = None
            else:
                samples, log_importance = self._call_proposal(mean, scale, generator)
            costs = cost_fn(samples)
            if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                raise ValueError(
                    "cost_fn must return [num_samples], got "
                    f"{tuple(costs.shape)} for {self.config.num_samples} samples"
                )
            weights = self._weights_from_costs(costs, log_importance)
            weights = self._mode_restrict(weights, costs)
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
