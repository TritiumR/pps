"""Feynman-Kac / SMC particle steering over MBD denoise chains.

Every other steering mode acts WITHIN one chain: it edits the score field (additive), the
candidate pool (inject), or the softmax weights (tilt). This one acts ACROSS chains -- K chains
denoise independently and are duplicated or killed by a potential built from the proxy.

The measured motivation is that the candidate softmax already runs near uniform (ESS ~3456 of
4096), so within-chain selection pressure is weak; particle resampling adds a second, coarser
selection axis that no other mode has.

Pure functions only: no proxy client, no planner, no I/O, so they are testable on CPU.
"""

from __future__ import annotations

import torch

from sim_free_mpc.planner import task_tilt_penalty


def log_potential(x0_est, target, weight, dims=7):
    """Return the per-particle log potential toward the proxy's clean chunk.

    Negated `task_tilt_penalty` -- the same Gaussian functional form the tilt mode applies to
    candidates, evaluated here per particle so lambda means the same thing in both modes.
    """
    return -task_tilt_penalty(x0_est, target, weight, dims=dims)


def normalize_logits(g):
    """Standardize potentials so --fk_lambda is scale free across denoise levels.

    The raw penalty spans orders of magnitude between the first and last level, which would make
    one lambda winner-take-all at one end and inert at the other.
    """
    if g.shape[0] < 2:
        return torch.zeros_like(g)
    std = g.std()
    if not torch.isfinite(std) or float(std) < 1e-9:
        return torch.zeros_like(g)
    return (g - g.mean()) / std


def ess(log_w):
    """Return the effective sample size of a log-weight vector."""
    w = torch.softmax(log_w.double(), dim=0)
    return float(1.0 / torch.clamp(w.pow(2).sum(), min=1e-12))


def systematic_resample(log_w, generator):
    """Low-variance systematic resampling; returns the parent index for each slot."""
    k = int(log_w.shape[0])
    w = torch.softmax(log_w.double(), dim=0)
    offset = torch.rand(1, generator=generator, dtype=torch.float64)
    positions = (torch.arange(k, dtype=torch.float64) + offset) / k
    edges = torch.cumsum(w, dim=0)
    edges[-1] = 1.0
    return torch.searchsorted(edges, positions.clamp(max=1.0)).clamp(max=k - 1)
