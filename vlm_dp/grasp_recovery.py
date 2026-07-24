"""Recovery from a failed grasp, kept out of the bridge so it stays sim-free and testable.

A closed-empty at the grasp target is evidence the estimate is off. The reopen recovery re-closes on the
same point, an absorbing state that hammers (measured: 42 closes at one pose on a weight seed). Search
recovery instead steps a bounded feel-around, so each retry probes a different nearby pose. This module
holds the probe geometry, numpy-only, so the bridge imports it and the tests exercise it without a sim.
"""
from __future__ import annotations

import numpy as np


def probe_pattern(r_max, rings=3, per_ring=6):
    """Feel-around offsets [N,3] around the estimate: the origin first, then concentric xy rings out to
    r_max. Bounded on purpose, so the search stays a sensible neighbourhood and cannot drift to a
    distractor. Top-down (z is zero), since grasp height is owned by tip_z, not the search.
    """
    pts = [np.zeros(3)]
    for i in range(1, rings + 1):
        r = r_max * i / rings
        phase = (np.pi / per_ring) * (i % 2)          # stagger alternate rings so points do not align
        for j in range(per_ring):
            a = 2.0 * np.pi * j / per_ring + phase
            pts.append(np.array([r * np.cos(a), r * np.sin(a), 0.0]))
    return pts
