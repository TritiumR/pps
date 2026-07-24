"""Grasp detection from the finger joint. A close that meets an object stalls short of the free-close
angle (calibrated: air 0.785 rad, pear 0.258, apple 0.166). Settling is judged on the angle, never
joint_vel, since a held object reads a steady nonzero velocity.
"""
from __future__ import annotations

import collections

import numpy as np


class ApertureGraspSensor:
    """Reports whether something is between the fingers and which object it is.

    Call observe once per control step. holding and held_object report the verdict.
    """

    def __init__(self, q_free=0.7854, stall_margin=0.15, q_touch=0.05, settle_steps=3,
                 settle_eps=0.01, close_steps=3, proximity=0.10):
        self.q_free = q_free                  # rad, angle a free close settles at (commanded close angle)
        self.stall_margin = stall_margin      # rad short of q_free that counts as blocked
        self.q_touch = q_touch                # rad the fingers must travel, guards a still-opening hand
        self.settle_steps = settle_steps      # steps the angle must hold steady before it is read
        self.settle_eps = settle_eps          # rad of drift allowed inside the settle window
        self.close_steps = close_steps        # steps the close must have been commanded for
        self.proximity = proximity            # m, how near the TCP an object must be to be the one held
        self._q = collections.deque(maxlen=max(settle_steps + 1, 2))
        self._closed_for = 0

    def observe(self, env, commanded_close: bool) -> None:
        """Record one control step. commanded_close is the gripper command that was just applied."""
        self._closed_for = self._closed_for + 1 if commanded_close else 0
        self._q.append(env.gripper_q())

    # The finger angle distinguishes exactly three states, each named here so callers express intent
    # rather than re-deriving a band from q_free/stall_margin/q_touch. A caller that rebuilds the band
    # inline can invert it (measured: a press advance read as closed for a fully open hand because it
    # dropped the q_touch bound) and the arithmetic gives no hint that it is wrong.

    def is_open(self) -> bool:
        """The fingers have not travelled: nothing has been closed on."""
        return self.aperture() <= self.q_touch

    def closed_on_air(self) -> bool:
        """The close ran to the free-close angle: the fingers met nothing."""
        return self.aperture() >= self.q_free - self.stall_margin

    def closed(self) -> bool:
        """The commanded close has completed: the fingers travelled past q_touch and settled.

        Deliberately weaker than holding: it says the close finished, not that anything is between the
        fingers. A press contact closes on a thin or articulated part whose angle can run near the
        free-close value, so the sensor cannot certify it, and a press advance may assert only this much.
        """
        if self._closed_for < self.close_steps or len(self._q) < self.settle_steps + 1:
            return False
        window = list(self._q)[-(self.settle_steps + 1):]
        if max(window) - min(window) > self.settle_eps:   # still travelling, the transient of any close
            return False                                  # (empty ones included), so do not read it yet
        return not self.is_open()

    def holding(self) -> bool:
        """Certifiable width between the fingers: a settled close that stalled short of the free-close
        angle. The strongest claim the angle supports."""
        return self.closed() and not self.closed_on_air()

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
        """Current finger angle in rad. Larger is more closed, and it encodes the held object's width."""
        return self._q[-1] if self._q else 0.0

    def released(self) -> bool:
        """Nothing held: fingers open, or run to the free-close angle (either side of the stall band)."""
        return self.is_open() or self.closed_on_air()
