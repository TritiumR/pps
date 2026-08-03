"""Simulation-independent helpers for grasp recovery.

Search recovery probes nearby poses after a failed grasp instead of repeatedly
retrying the same estimate.
"""

from __future__ import annotations

import numpy as np


def probe_pattern(r_max, rings=3, per_ring=6):
    """Return bounded XY probe offsets, starting at the origin."""
    pts = [np.zeros(3)]
    for i in range(1, rings + 1):
        r = r_max * i / rings
        # Stagger adjacent rings to avoid radial alignment.
        phase = (np.pi / per_ring) * (i % 2)
        for j in range(per_ring):
            a = 2.0 * np.pi * j / per_ring + phase
            pts.append(np.array([r * np.cos(a), r * np.sin(a), 0.0]))
    return pts


def descent_stalled(z_hist, k, eps, lookback=8, min_drop=0.02):
    """Return whether a prior descent has stopped within the recent window."""
    if len(z_hist) < k + 1:
        return False
    if (z_hist[-k - 1] - z_hist[-1]) >= eps:
        return False
    look = min(len(z_hist), lookback + 1)
    return (max(z_hist[-look:]) - z_hist[-1]) > min_drop


def debounce_gripper(
    actions,
    execute_steps,
    hold_n,
    held,
    open_run,
    close_val,
    thresh=0.5,
):
    """Suppress brief open commands while an object is held.

    Returns the suppressed commands, consecutive raw-open count, and latest
    executed close value.
    """
    suppressed = []
    for i in range(min(int(execute_steps), len(actions))):
        raw = float(actions[i][7])
        if raw > thresh:
            open_run = 0
            close_val = raw
            continue
        open_run += 1
        if held and open_run < hold_n:
            actions[i][7] = close_val
            suppressed.append((i, raw))
    return suppressed, open_run, close_val