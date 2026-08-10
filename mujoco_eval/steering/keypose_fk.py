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
    """Truncated-Gaussian cloud around one keypose row: [count, D].

    `lower`/`upper` accept scalars or per-coordinate tensors, so a data-derived proposal box
    drops in without changing the call sites that still want the +/-4 default.
    """
    noise = torch.randn(count, *center.shape, generator=generator, dtype=center.dtype)
    return torch.clamp(center.unsqueeze(0) + scale * noise, lower, upper)


def cost_weights(costs, temperature):
    """Softmax weights over proposal costs and their effective sample size.

    ESS is 1/sum(w^2) over the cloud -- his `ess_by_step`. It says whether the softmax selects
    (ESS << count) or averages (ESS ~ count); the mode's whole claim rests on the former.
    """
    logits = -torch.as_tensor(costs, dtype=torch.float32).reshape(-1) / max(temperature, 1e-6)
    weights = torch.softmax(logits, dim=0)
    ess = float(1.0 / torch.clamp(weights.pow(2).sum(), min=1e-12))
    return weights, ess, logits


def guided_from_costs(proposals, costs, temperature):
    """Cost-weighted mean of the proposals, plus the FK log-potential of the cloud.

    The mean is the MBD/Tweedie estimate the guidance moves toward; the logmeanexp is the
    particle's weight under the Feynman-Kac potential.
    """
    weights, _, logits = cost_weights(costs, temperature)
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


def blend_goal_block(policy_block, mbd_block, gamma_traj, gamma_keypose):
    """Per-block clean-space blend c_g = c_pol + gamma * (c_mbd - c_pol) over the goal rows.

    His `_blend_policy_mbd_flows` acts on SCORES; the DDIM x0 is an affine function of the score
    at fixed (x_t, t), so blending clean endpoints with the same coefficients is the identical
    operator one parameterisation removed. The trailing row of the block is the keypose and takes
    its own coefficient (0.9 in production); the waypoint rows take the trajectory one (0.6).
    """
    gamma = torch.full((policy_block.shape[0], 1), float(gamma_traj), dtype=policy_block.dtype)
    gamma[-1, 0] = float(gamma_keypose)
    return policy_block + gamma * (mbd_block - policy_block)


def l1_step_to_row(chunk, target_row, action_rows, step_size, arm_dims=7):
    """His clipped-L1 action pull: per coordinate, sign(d) * min(|d|, step * ramp).

    Returns the DISPLACEMENT for the leading `action_rows`, not the moved chunk, so the caller can
    scale it by the action-block coefficient (his blend applies gamma_action to the flow the pull
    induces, which in clean space is exactly gamma_action * this step).

    The ramp is (i+1)/n over the rows: the first executable row may move a fraction of a full step,
    the last a whole one, so an early row is not teleported at the goal. Arm channels only -- the
    gripper is near-binary and an L1 pull on it just drags it to the goal row's value.
    """
    n = int(action_rows)
    if float(step_size) <= 0.0 or n <= 0:
        return torch.zeros_like(chunk[:max(n, 0)])
    delta = chunk[int(target_row)].unsqueeze(0) - chunk[:n]
    ramp = torch.linspace(1.0 / n, 1.0, n, dtype=chunk.dtype).unsqueeze(-1)
    step = torch.sign(delta) * torch.minimum(delta.abs(), float(step_size) * ramp)
    step[:, arm_dims:] = 0.0
    return step


def select_goal_row(goal_tcp, current_tcp, wrist_m, fallback_m):
    """Index of the goal row the action pull aims at, and the threshold that selected it.

    His selector: the LAST (furthest-along) waypoint whose wrist sits within `wrist_m` of the
    current wrist, so the pull tracks progress instead of always aiming at W1. When nothing is
    near he clamps to the first waypoint; we widen to `fallback_m` first and only then fall back
    to the keypose (the last row), which is the one target that always exists.
    """
    d = torch.linalg.vector_norm(
        torch.as_tensor(goal_tcp, dtype=torch.float32)
        - torch.as_tensor(current_tcp, dtype=torch.float32).reshape(1, 3), dim=-1)
    for thr in (float(wrist_m), float(fallback_m)):
        if thr > 0.0 and bool((d <= thr).any()):
            return int(torch.nonzero(d <= thr).max()), thr
    return int(d.shape[0]) - 1, None


def consistency_penalty(keypose_real, previous_real, weight):
    """His cross-replan keypose consistency term: weight * mean((kp - prev)^2), physical units.

    Without it the ranked keypose is redrawn from scratch every replan and the arm chases a target
    that moves as fast as the cloud does.
    """
    if previous_real is None or float(weight) <= 0.0:
        return None
    kp = torch.as_tensor(np.asarray(keypose_real), dtype=torch.float32)
    prev = torch.as_tensor(np.asarray(previous_real), dtype=torch.float32).reshape(1, -1)
    return float(weight) * ((kp - prev[..., : kp.shape[-1]]) ** 2).mean(dim=-1)


def proposal_box(checkpoint, mean, std, default=4.0):
    """Model-space (lower, upper) proposal bounds, from the checkpoint's stats when it ships them.

    His bounds are the dataset's own action_min/action_max mapped through the training
    normalisation, so the cloud can never propose an action the data never contains. Our
    action_norm_stats.json currently ships mean/std only, so this returns the +/-default box it
    always used and says which source it took.
    """
    import json
    import pathlib

    lo = hi = None
    if checkpoint:
        base = pathlib.Path(checkpoint)
        for cand in (base / "action_norm_stats.json", base.parent / "action_norm_stats.json"):
            if not cand.exists():
                continue
            raw = json.loads(cand.read_text())
            if "min" in raw and "max" in raw:
                lo, hi = np.asarray(raw["min"], np.float32), np.asarray(raw["max"], np.float32)
            break
    if lo is None:
        return torch.tensor(-float(default)), torch.tensor(float(default)), "default"
    m = np.asarray(mean, np.float32)
    s = np.asarray(std, np.float32) + 1e-6
    return (torch.as_tensor((lo - m) / s, dtype=torch.float32),
            torch.as_tensor((hi - m) / s, dtype=torch.float32), "action_norm_stats")


def draw_index(costs, temperature, generator):
    """Categorical draw over softmax(-cost/T) -- his final particle pick, not an argmin.

    At T=0.01 the draw is an argmin almost surely; the difference only shows when two particles
    genuinely tie, where argmin's index order is an arbitrary tie-break.
    """
    weights, ess, _ = cost_weights(costs, temperature)
    return int(torch.multinomial(weights, 1, generator=generator)), ess


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
