"""Rollout metrics shared by the base drivers: motion smoothness, scene disturbance, and reach.

All take plain lists/arrays (no torch), so a driver builds them from its recorded history without
duplicating the numpy math.
"""
import numpy as np


def motion_smoothness(q_hist):
    """Executed-joint history -> dict(jerk, speed, cos, tv, n); empty dict for < 3 steps.

    jerk = mean |2nd difference|, speed = mean step norm, cos = mean direction consistency of
    consecutive steps (1 = smooth), tv = per-joint total variation.
    """
    q = np.array(q_hist)
    if len(q) <= 2:
        return {}
    dq = np.diff(q, axis=0)
    step = np.linalg.norm(dq, axis=1)
    cos = (dq[1:] * dq[:-1]).sum(axis=1) / (step[1:] * step[:-1] + 1e-9)
    jerk = np.linalg.norm(q[2:] - 2 * q[1:-1] + q[:-2], axis=1).mean()
    return {"jerk": float(jerk), "speed": float(step.mean()), "cos": float(cos.mean()),
            "tv": np.abs(dq).sum(axis=0), "n": len(q)}


def scene_disturbance(env, obj0, names):
    """GT displacement of each of names from the initial snapshot obj0 -> (moved, sum, max) (m)."""
    moved = [float(np.linalg.norm(env.object_pose(n)[0] - obj0[n])) for n in names]
    return moved, sum(moved), (max(moved) if moved else 0.0)


def reach_stats(dist_hist):
    """TCP->target distance history -> dict(min, final) (m); None if empty."""
    if not dist_hist:
        return None
    d = np.array(dist_hist)
    return {"min": float(d.min()), "final": float(d[-1])}
