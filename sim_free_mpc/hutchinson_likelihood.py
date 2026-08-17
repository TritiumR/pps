"""Conditional probability-flow likelihood with a Hutchinson divergence estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import Callable

import torch


@dataclass(frozen=True)
class FlowLikelihoodEstimate:
    log_prob: torch.Tensor
    prior_log_prob: torch.Tensor
    divergence_integral: torch.Tensor
    terminal: torch.Tensor


def rademacher_probes(
    shape: tuple[int, ...],
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    bits = torch.randint(0, 2, (count, *shape), generator=generator, device=device)
    return bits.to(dtype=dtype).mul_(2).sub_(1)


def hutchinson_divergence(
    velocity: torch.Tensor,
    state: torch.Tensor,
    probes: torch.Tensor,
) -> torch.Tensor:
    """Estimate ``tr(d velocity / d state)`` independently for each batch item."""
    if velocity.shape != state.shape:
        raise ValueError(f"velocity/state shape mismatch: {velocity.shape} vs {state.shape}")
    if probes.shape[1:] != state.shape:
        raise ValueError(f"probe/state shape mismatch: {probes.shape} vs {state.shape}")
    estimates = []
    for index, probe in enumerate(probes):
        vector_jacobian = torch.autograd.grad(
            outputs=(velocity * probe).sum(),
            inputs=state,
            retain_graph=index + 1 < probes.shape[0],
            create_graph=False,
        )[0]
        estimates.append((vector_jacobian * probe).flatten(1).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def estimate_probability_flow_log_likelihood(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    *,
    num_steps: int = 32,
    num_probes: int = 1,
    seed: int | Sequence[int] = 0,
    time_start: float = 1.0e-3,
    time_end: float = 1.0 - 1.0e-3,
) -> FlowLikelihoodEstimate:
    """Integrate data-to-noise and accumulate ``integral div(v) dt``.

    OpenPI uses ``x_t=(1-t)x_0+t*epsilon`` and learns ``dx/dt``. Therefore
    ``log p(x_0)=log N(x_1)+integral_0^1 div(v_t) dt``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0.0 <= time_start < time_end <= 1.0:
        raise ValueError(f"invalid integration interval [{time_start}, {time_end}]")
    x = initial.detach()
    if isinstance(seed, Sequence):
        if len(seed) != x.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} probe seeds, got {len(seed)}")
        probes = torch.cat(
            [
                rademacher_probes(
                    tuple(x[index : index + 1].shape),
                    count=num_probes,
                    device=x.device,
                    dtype=x.dtype,
                    seed=int(item_seed),
                )
                for index, item_seed in enumerate(seed)
            ],
            dim=1,
        )
    else:
        probes = rademacher_probes(
            tuple(x.shape), count=num_probes, device=x.device, dtype=x.dtype, seed=seed
        )
    dt = (time_end - time_start) / num_steps
    divergence_integral = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    for step in range(num_steps):
        time_value = time_start + (step + 0.5) * dt
        x = x.detach().requires_grad_(True)
        time = torch.full(
            (x.shape[0],), time_value, dtype=torch.float32, device=x.device
        )
        velocity = velocity_fn(x, time)
        divergence = hutchinson_divergence(velocity, x, probes)
        divergence_integral += float(dt) * divergence.detach().to(torch.float32)
        # Midpoint time with an Euler state update is deliberate: it avoids the singular
        # endpoints while keeping one expensive transformer/JVP evaluation per ODE step.
        x = x.detach() + float(dt) * velocity.detach()

    terminal = x.to(torch.float32)
    dimensions = terminal[0].numel()
    prior_log_prob = -0.5 * (
        terminal.flatten(1).square().sum(dim=1) + dimensions * math.log(2.0 * math.pi)
    )
    return FlowLikelihoodEstimate(
        log_prob=prior_log_prob + divergence_integral,
        prior_log_prob=prior_log_prob,
        divergence_integral=divergence_integral,
        terminal=terminal,
    )


__all__ = [
    "FlowLikelihoodEstimate",
    "estimate_probability_flow_log_likelihood",
    "hutchinson_divergence",
    "rademacher_probes",
]
