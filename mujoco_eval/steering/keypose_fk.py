"""Keypose FK steering: the trained policy denoises, a keypose-row cost steers it.

Cory's `sample_actions_fk_keypose` adapted to our single-arm stack. Per denoise level, for each
of P particles:

  1. the POLICY takes its own step, giving a clean-chunk estimate x0
  2. a proposal cloud is drawn around x0's KEYPOSE ROW only (truncated Gaussian, sigma shrinking
     with the noise level)
  3. proposals are costed; softmax(-cost/T) gives a cost-weighted mean -- the guided keypose
  4. logmeanexp of the same logits is the Feynman-Kac potential; parents are resampled by it
  5. the guidance displacement is KL-capped, then the action rows are optionally pulled along a
     straight joint path toward the guided keypose
  6. policy and guided chunks are blended PER BLOCK, and children are emitted in
     [base, guided] pairs so pure-policy diversity survives resampling

Three adaptations, each deliberate:
  * he uses rectified flow and blends velocities; we use DDIM and blend the ITERATE via
    `_blend_blocks` -- the same lerp one step later in the chain, and the form whose gamma=0
    identity is already md5-verified.
  * no ARX tracker gradient: our actions ARE joint targets, so the straight-line path in joint
    space is already the feasible path his learned tracker had to recover.
  * two blocks, not three. His middle block is 5 supervised AWE waypoints; ours would be plain
    consecutive action rows, and steering a block with no distinct semantics is what made the
    final-action-as-keypose test vacuous. `traj_start == keypose_row` leaves it empty by
    construction rather than fake.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import fk as fk_mod
from .chunk_cost import chunk_costs


def sample_proposals(center, scale, count, generator, lower=-4.0, upper=4.0):
    """Truncated-Gaussian cloud around one keypose row: [count, D]."""
    noise = torch.randn(count, *center.shape, generator=generator, dtype=center.dtype)
    return torch.clamp(center.unsqueeze(0) + scale * noise, lower, upper)


def guided_from_costs(proposals, costs, temperature):
    """Cost-weighted mean of the proposals, plus the FK log-potential of the cloud.

    The mean is the MBD/Tweedie estimate the guidance moves toward; the logmeanexp is the
    particle's weight under the Feynman-Kac potential.
    """
    logits = -torch.as_tensor(costs, dtype=torch.float32).reshape(-1) / max(temperature, 1e-6)
    weights = torch.softmax(logits, dim=0)
    # proposals is [count, D] for a single keypose row, or [count, R, D] when the whole goal
    # block is perturbed (Cory: the AWE waypoints are MBD-estimated too, not just the
    # keypose). Broadcast over whatever trailing shape it has -- identical to weights[:, None]
    # in the 2-D case.
    guided = (weights.view(-1, *([1] * (proposals.ndim - 1))) * proposals).sum(dim=0)
    potential = torch.logsumexp(logits, dim=0) - math.log(len(logits))
    return guided, float(potential)


def kl_capped(delta, sigma, max_kl):
    """Scale a guidance displacement so its Gaussian KL stays under max_kl.

    raw_kl = 0.5 * ||delta / sigma||^2 ; scale = min(1, sqrt(max_kl / raw_kl)).
    max_kl <= 0 disables the guidance entirely, which is the identity control.
    """
    if max_kl <= 0.0:
        return torch.zeros_like(delta), 0.0
    raw = 0.5 * float(torch.sum((delta / max(sigma, 1e-6)) ** 2))
    scale = min(1.0, math.sqrt(max_kl / max(raw, 1e-30)))
    return scale * delta, raw * scale * scale


def action_l1_pull(chunk, keypose_row, q0_model, step_size, action_rows=None):
    """Pull action rows toward the straight joint-space path from q0 to the guided keypose.

    His `_apply_action_l1_gradient` with the learned-tracker term dropped: our actions are joint
    targets, so linear interpolation IS the feasible path and no dynamics model is needed. Acts
    on the arm channels only -- the gripper is near-binary and would dominate an L1 pull.

    `keypose_row` is the GOAL (what to pull toward); `action_rows` is how many leading rows are
    executable and may be moved. They were one argument, which forced a choice between pulling
    toward the right target and writing to the right rows: passing keypose_row also rewrote the
    AWE waypoint rows, while passing action_rows silently retargeted the pull at W1 instead of the
    key pose. Both are wrong; they are separate quantities.
    """
    if step_size <= 0.0:
        return chunk
    rows = int(keypose_row if action_rows is None else action_rows)
    if rows <= 0 or keypose_row <= 0:
        return chunk
    frac = torch.linspace(1.0 / rows, 1.0, rows, dtype=chunk.dtype).unsqueeze(-1)
    goal = chunk[keypose_row]
    path = q0_model.unsqueeze(0) + frac * (goal - q0_model).unsqueeze(0)
    out = chunk.clone()
    out[:rows, :7] = out[:rows, :7] + step_size * (path[:, :7] - out[:rows, :7])
    return out


def resample_parents(potentials, count, generator):
    """Systematic resampling of parent indices under the FK potentials."""
    log_w = torch.as_tensor(potentials, dtype=torch.float64)
    idx = fk_mod.systematic_resample(log_w, generator)
    if count <= len(idx):
        return idx[:count]
    return idx[torch.arange(count) % len(idx)]


def ess_of(potentials):
    """Effective sample size of the particle population."""
    return fk_mod.ess(torch.as_tensor(potentials, dtype=torch.float64))
