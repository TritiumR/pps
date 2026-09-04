"""Normalize trajectory costs with randomized Sobol quadrature.

The cost is interpreted as an energy, ``p(a | s) = exp(-C(a)/temperature) / Z(s)``.
The integration domain is the executable action support: each arm target is within
``joint_delta`` of the preceding target (and within Panda limits), while gripper targets
are in ``[0, 1]``.  Sobol points are mapped autoregressively into that domain and the
map's exact Jacobian is included in the quadrature weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .fk import PANDA_JOINT_LIMITS


@dataclass(frozen=True)
class QuadratureEstimate:
    log_partition: float
    log_partition_std_error: float
    log_partition_by_scramble: tuple[float, ...]
    num_points: int
    num_scrambles: int


def map_unit_to_action_chunks(
    unit: torch.Tensor,
    current_joint_pos: torch.Tensor,
    *,
    joint_delta: float = 0.15,
    joint_limits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``[N,H,8]`` unit-cube points to executable chunks.

    Returns physical actions and ``log |da/du|``.  The latter varies near a joint
    limit because the next target's interval is conditional on the previous target.
    """
    if unit.ndim != 3 or unit.shape[-1] != 8:
        raise ValueError(f"Expected unit points [N,H,8], got {tuple(unit.shape)}")
    if joint_delta <= 0.0:
        raise ValueError("joint_delta must be positive")
    if torch.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("unit points must lie in [0, 1]")

    device, dtype = unit.device, unit.dtype
    limits = torch.as_tensor(
        PANDA_JOINT_LIMITS if joint_limits is None else joint_limits,
        device=device,
        dtype=dtype,
    )
    if limits.shape != (7, 2):
        raise ValueError(f"Expected joint limits [7,2], got {tuple(limits.shape)}")
    current = torch.as_tensor(current_joint_pos, device=device, dtype=dtype).flatten()[:7]
    if current.shape != (7,):
        raise ValueError(f"Expected seven current joints, got {tuple(current.shape)}")

    previous = current.unsqueeze(0).expand(unit.shape[0], -1)
    rows = []
    log_jacobian = torch.zeros(unit.shape[0], device=device, dtype=dtype)
    for step in range(unit.shape[1]):
        lower = torch.maximum(limits[:, 0], previous - joint_delta)
        upper = torch.minimum(limits[:, 1], previous + joint_delta)
        width = upper - lower
        if torch.any(width <= 0.0):
            raise ValueError("Action support has a non-positive conditional width")
        joints = lower + unit[:, step, :7] * width
        gripper = unit[:, step, 7:8]
        rows.append(torch.cat((joints, gripper), dim=-1))
        log_jacobian += torch.log(width).sum(dim=-1)
        previous = joints
    return torch.stack(rows, dim=1), log_jacobian


def _cost_in_batches(
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
    actions: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    costs = []
    for start in range(0, actions.shape[0], batch_size):
        value = cost_fn(actions[start : start + batch_size])
        value = torch.as_tensor(value, device=actions.device, dtype=actions.dtype).flatten()
        expected = min(batch_size, actions.shape[0] - start)
        if value.shape != (expected,):
            raise ValueError(f"cost_fn returned {tuple(value.shape)}, expected {(expected,)}")
        costs.append(value)
    return torch.cat(costs)


@torch.no_grad()
def estimate_log_partition_sobol(
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
    current_joint_pos: torch.Tensor,
    *,
    horizon: int = 15,
    joint_delta: float = 0.15,
    temperature: float = 1.0,
    num_points: int = 16384,
    num_scrambles: int = 4,
    batch_size: int = 2048,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    joint_limits: torch.Tensor | None = None,
) -> QuadratureEstimate:
    """Estimate ``log integral exp(-C(a)/temperature) da``.

    Independent Owen-scrambled Sobol rules provide a replicate-based numerical error
    estimate. ``num_points`` must be a power of two to preserve the Sobol balance rule.
    """
    if horizon <= 0 or temperature <= 0.0 or batch_size <= 0 or num_scrambles <= 0:
        raise ValueError("horizon, temperature, batch_size, and num_scrambles must be positive")
    if num_points <= 0 or num_points & (num_points - 1):
        raise ValueError("num_points must be a positive power of two")
    device = torch.device(device or torch.as_tensor(current_joint_pos).device)
    dimension = horizon * 8
    power = int(math.log2(num_points))
    estimates = []
    for replicate in range(num_scrambles):
        engine = torch.quasirandom.SobolEngine(
            dimension, scramble=True, seed=int(seed) + replicate * 104729
        )
        unit = engine.draw_base2(power).to(device=device, dtype=dtype).reshape(
            num_points, horizon, 8
        )
        actions, log_jacobian = map_unit_to_action_chunks(
            unit,
            current_joint_pos,
            joint_delta=joint_delta,
            joint_limits=joint_limits,
        )
        costs = _cost_in_batches(cost_fn, actions, batch_size)
        log_weights = log_jacobian - costs / float(temperature)
        estimates.append(torch.logsumexp(log_weights.to(torch.float64), dim=0) - math.log(num_points))

    values = torch.stack(estimates)
    standard_error = (
        values.std(unbiased=True) / math.sqrt(num_scrambles)
        if num_scrambles > 1
        else values.new_tensor(float("nan"))
    )
    return QuadratureEstimate(
        log_partition=float(values.mean().item()),
        log_partition_std_error=float(standard_error.item()),
        log_partition_by_scramble=tuple(float(value.item()) for value in values),
        num_points=num_points,
        num_scrambles=num_scrambles,
    )


def action_log_likelihood(cost: float, log_partition: float, temperature: float = 1.0) -> float:
    """Return normalized physical-coordinate log density for one action chunk."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return -float(cost) / float(temperature) - float(log_partition)
