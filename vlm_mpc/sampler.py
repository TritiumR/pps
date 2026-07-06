"""Sampling-based MPC over joint accelerations (a DIAL-MPC sampler; mirrors hydrax's `algs/dial.py`).

What it does on each planning step:
  1. sample many random acceleration sequences,
  2. turn each into a short joint trajectory (integrate accel -> velocity -> position),
  3. score every trajectory with the cost,
  4. set the new plan to the cost-weighted average of the samples (low-cost ones count more),
then repeat for a few iterations, shrinking the random spread each time so the plan settles into the
low-cost region.

The random spread (`sigma`) is shaped two ways:
  - it SHRINKS across iterations  -> explore widely at first, fine-tune later;
  - it GROWS along the horizon    -> sample the next step tightly, distant steps loosely
                                     (we're less certain about the far future).
The `beta_*` knobs set how strong each effect is; very large betas make the spread constant (plain MPPI).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

CostFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]
# plan(mean_a, q0, qd0, ctx, gen) -> (mean_a[H,7], q_traj[H,7], score, ess)
PlanFn = Callable[..., tuple[torch.Tensor, torch.Tensor, float, float]]


def make_accel_sampler(
    cost_fn: CostFn,
    q_lo: Sequence[float] | torch.Tensor,
    q_hi: Sequence[float] | torch.Tensor,
    dt: float,
    H: int = 8,
    num_samples: int = 512,
    iterations: int = 8,
    accel_std: float = 6.0,
    accel_clip: float = 15.0,
    temperature: float = 0.2,
    beta_opt_iter: float = 1.0,
    beta_horizon: float = 1.0,
    clamp_limits: bool = True,
    w_consist: float = 0.0,
    grip_std: float = 0.3,
    param: str = "accel",
    delta_std: float = 0.1,
    delta_clip: float = 0.2,
    device: str = "cuda:0",
) -> PlanFn:
    """Builds one planning step of the sampler.

    Args:
        cost_fn: scores a batch of candidate trajectories; ``cost_fn(q_traj[K,H,7], q_cur[7], ctx) ->
            costs[K]``. Lower is better. See `costs`.
        q_lo: per-joint lower position limits ``[7]``.
        q_hi: per-joint upper position limits ``[7]``.
        dt: control timestep (seconds).
        H: how many steps ahead the plan covers (the horizon).
        num_samples: how many random trajectories to try per iteration.
        iterations: how many refine passes to run inside one plan() call.
        accel_std: base size of the random spread on the accelerations (before annealing).
        accel_clip: hard cap on the sampled acceleration noise, so no single sample can fling a joint.
        temperature: how sharply to favor low-cost samples when averaging (smaller = greedier).
        beta_opt_iter: strength of the shrink-the-spread-each-iteration effect (larger = weaker; >=~1e6
            turns it off).
        beta_horizon: strength of the wider-spread-for-later-steps effect (larger = weaker).
        clamp_limits: keep every integrated position within ``[q_lo, q_hi]`` (default). Turn off only to
            study what happens without the joint-limit guard -- positions may then leave the robot's range.
        w_consist: weight on the consistency term -- penalizes a candidate trajectory for deviating from
            the warm-start trajectory (the previous step's plan, shifted). 0 disables it. This is ReKep's
            "keep the solution close to the previous one" cost; it damps the step-to-step plan wander that
            shows up as execution jitter when re-planning over a flat cost basin.
        device: torch device.

    Returns:
        ``plan(mean_a[H,7], q0[7], qd0[7], ctx, gen) -> (mean_a[H,7], q_traj[H,7], score, ess)``.
        ``score`` is the cost of the chosen plan; ``ess`` is the effective sample size -- roughly how many
        samples meaningfully contributed (a low value means the result leaned on just a few).
    """
    q_lo = torch.as_tensor(q_lo, device=device, dtype=torch.float32)
    q_hi = torch.as_tensor(q_hi, device=device, dtype=torch.float32)
    num_candidates = num_samples
    # Per-step spread multiplier: later horizon steps get a wider spread (we're less sure that far out).
    knot_idx = torch.arange(H, device=device, dtype=torch.float32)
    horizon_factor = torch.exp(-(H - 1 - knot_idx) / (beta_horizon * H))  # [H]
    # Parameterization: "accel" (decision var = joint accelerations, double-integrated -> smooth by
    # construction) or "delta" (decision var = per-step joint-position deltas, single-integrated, clamped
    # -- emulates the collaborator's direct-target + delta-clip setup, where smoothness is only a soft
    # cost). The sampling std/clip switch accordingly so both explore a comparable position range.
    base_std = accel_std if param == "accel" else delta_std
    clip = accel_clip if param == "accel" else delta_clip

    def integrate(u: torch.Tensor, q0: torch.Tensor, qd0: torch.Tensor) -> torch.Tensor:
        """Turns the sampled decision variable into a joint-position trajectory, step by step.

        accel mode: ``u`` = accelerations, double-integrated (semi-implicit Euler) -> velocity -> position.
        delta mode: ``u`` = per-step position deltas, single-integrated (q[t] = q[t-1] + u[t]).
        Positions are kept within the joint limits either way.
        """
        batch = u.shape[0]
        q = q0.expand(batch, 7).clone()
        traj = []
        if param == "accel":
            qd = qd0.expand(batch, 7).clone()
            for t in range(H):
                qd = qd + u[:, t, :] * dt
                q = q + qd * dt
                if clamp_limits:
                    q = torch.clamp(q, q_lo, q_hi)  # never command a joint past its physical range
                traj.append(q)
        else:  # delta: each step's target is a free variable (bounded only by the sampled delta clamp)
            for t in range(H):
                q = q + u[:, t, :]
                if clamp_limits:
                    q = torch.clamp(q, q_lo, q_hi)
                traj.append(q)
        return torch.stack(traj, dim=1)  # [B,H,7]

    def plan(mean_a, q0, qd0, ctx, gen, g_mean=None):
        # The sampler never looks inside ctx -- it just forwards it to cost_fn, which unpacks it (a target,
        # a (target, center) tuple, keypoints, etc.). Keeps the sampler independent of what's being grasped.
        # Gripper channel (opt-in): pass g_mean[H,1] to also sample the gripper as an 8th decision variable
        # -- a direct per-step command in [0,1] (NOT integrated), scored by cost_fn's gripper term and
        # cost-weighted like the arm. Enables PPS to steer the grasp. g_mean=None -> arm-only (unchanged).
        mean_a = mean_a.to(device)
        q0, qd0 = q0.to(device), qd0.to(device)
        use_grip = g_mean is not None
        if use_grip:
            g_mean = g_mean.to(device)  # [H,1] gripper-command mean in [0,1]
        # The warm-start trajectory IS the previous step's plan (shifted) -- the consistency reference.
        q_ws = integrate(mean_a.unsqueeze(0), q0, qd0)[0] if w_consist > 0.0 else None  # [H,7]
        weights = None
        for i in range(iterations):
            # Spread shrinks as the iterations progress, so the plan refines instead of jumping around.
            sigma = base_std * math.exp(-i / (beta_opt_iter * iterations)) * horizon_factor  # [H]
            eps = torch.randn(num_candidates, H, 7, generator=gen, device=device) * sigma[None, :, None]
            eps = torch.clamp(eps, -clip, clip)
            accels = mean_a.unsqueeze(0) + eps  # [K,H,7] random sequences around the current plan
            if param == "delta":  # hard-bound the per-step joint delta (collaborator's mpc_joint_delta_clip)
                accels = torch.clamp(accels, -delta_clip, delta_clip)
            q_traj = integrate(accels, q0, qd0)
            if use_grip:
                g_eps = torch.randn(num_candidates, H, 1, generator=gen, device=device) * grip_std
                g_samp = torch.clamp(g_mean.unsqueeze(0) + g_eps, 0.0, 1.0)  # [K,H,1] direct commands
                costs = cost_fn(q_traj, q0, ctx, g_samp)  # [K]
            else:
                costs = cost_fn(q_traj, q0, ctx)  # [K]
            if q_ws is not None:
                # Consistency: pull candidates toward the previous plan, so the executed command doesn't
                # wander step-to-step (ReKep's consistency cost).
                costs = costs + w_consist * (q_traj - q_ws).pow(2).sum(dim=(1, 2))
            # Guard: a VLM-authored constraint can produce NaN/inf (e.g. a 0/0 in an arccos "upright"
            # term). Map those to a huge finite cost so bad candidates are discarded, never poisoning the
            # weighted mean (a single NaN would otherwise NaN the whole plan -> NaN actions -> sim hang).
            costs = torch.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12)
            # New plan = average of the samples, weighted toward the low-cost ones.
            weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
            mean_a = torch.einsum("k,khj->hj", weights, accels)
            if use_grip:
                g_mean = torch.einsum("k,khj->hj", weights, g_samp)  # [H,1]
        q_best = integrate(mean_a.unsqueeze(0), q0, qd0)[0]  # the trajectory the final plan produces
        ess = float(1.0 / (weights * weights).sum())
        if use_grip:
            g_best = torch.clamp(g_mean, 0.0, 1.0)  # [H,1]
            score = float(cost_fn(q_best.unsqueeze(0), q0, ctx, g_best.unsqueeze(0))[0])
            return mean_a, q_best, g_best, score, ess
        score = float(cost_fn(q_best.unsqueeze(0), q0, ctx)[0])
        return mean_a, q_best, score, ess

    return plan
