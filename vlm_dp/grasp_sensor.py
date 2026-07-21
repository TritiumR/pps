"""Grasp detection from the finger joint: a close that meets an object stalls short of the
free-close angle (calibrated: air 0.785 rad, pear 0.258, apple 0.166). Settling is judged on
the angle, never joint_vel (a held object reads a steady nonzero velocity).
"""
from __future__ import annotations

import collections

import numpy as np


class ApertureGraspSensor:
    """Is something between the fingers, and if so which object? ``observe`` once per control
    step; ``holding``/``held_object`` report the verdict."""

    def __init__(self, q_free=0.7854, stall_margin=0.15, q_touch=0.05, settle_steps=3,
                 settle_eps=0.01, close_steps=3, proximity=0.10):
        self.q_free = q_free                  # angle a free close settles at (the commanded close angle)
        self.stall_margin = stall_margin      # how far short of q_free counts as blocked
        self.q_touch = q_touch                # the fingers must have travelled; guards a still-opening hand
        self.settle_steps = settle_steps      # steps the angle must hold steady before it is read
        self.settle_eps = settle_eps          # rad of drift allowed inside the settle window
        self.close_steps = close_steps        # steps the close must have been commanded for
        self.proximity = proximity            # m: how near the TCP an object must be to be the one held
        self._q = collections.deque(maxlen=max(settle_steps + 1, 2))
        self._closed_for = 0

    def observe(self, env, commanded_close: bool) -> None:
        """Record one control step. ``commanded_close`` is the gripper command that was just applied."""
        self._closed_for = self._closed_for + 1 if commanded_close else 0
        self._q.append(env.gripper_q())

    def holding(self) -> bool:
        """Something is between the fingers: a settled close that stalled short of the free-close angle."""
        if self._closed_for < self.close_steps or len(self._q) < self.settle_steps + 1:
            return False
        window = list(self._q)[-(self.settle_steps + 1):]
        if max(window) - min(window) > self.settle_eps:   # still travelling: the transient of any close,
            return False                                  # empty ones included, so it must not be read yet
        return self.q_touch < window[-1] < self.q_free - self.stall_margin

    def held_object(self, positions: dict, tcp) -> str | None:
        """Nearest estimated centroid to the TCP when the fingers report a hold (else None).

        Proximity names the object and rejects closes blocked by the table or the arm itself.
        """
        if not self.holding() or not positions:
            return None
        tcp = np.asarray(tcp, dtype=np.float64)
        name, dist = min(((n, float(np.linalg.norm(np.asarray(p, dtype=np.float64) - tcp)))
                          for n, p in positions.items()), key=lambda kv: kv[1])
        return name if dist <= self.proximity else None

    def aperture(self) -> float:
        """Current finger angle. Larger is more closed; it also encodes the held object's width."""
        return self._q[-1] if self._q else 0.0
