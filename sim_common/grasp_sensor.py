"""Grasp detection from the gripper's own finger joint: proprioception, not scene state.

A close that meets an object stalls short of the free-close angle; an empty close reaches it. That gap is
the grasp signal, and it is the same signal a real parallel gripper reports from its encoder.

This replaces a simulator grasp flag computed from the object's true pose plus "the fingers moved at all",
which therefore certified an empty close near the object as a grasp. A stall cannot do that, so the
scaffolding that existed to catch those false positives -- carry the object N chunks, then check whether it
rose -- is not needed and is removed with it.

Calibrated on this gripper (agent_tests/_probe_gripper.py):

    close on air     0.785 rad   (exactly the commanded angle; settles within ~3 steps)
    close on pear    0.258 rad
    close on apple   0.166 rad

so held and empty are separated by more than half a radian, and neither fruit creeps out of the fingers.

Settling is judged on the finger ANGLE, never on the reported joint velocity. While the gripper holds an
object the drive keeps pushing against it through the underactuated linkage, so ``joint_vel`` reads a
steady -0.5 to -0.9 rad/s even though the angle is constant to within a milliradian. A velocity gate would
never fire on a held object.
"""
from __future__ import annotations

import collections

import numpy as np


class ApertureGraspSensor:
    """Is something between the fingers, and if so which object?

    ``observe`` is called once per executed control step with the gripper command that was just applied;
    ``holding`` and ``held_object`` report the current verdict.
    """

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
        """Which object is in the hand: the nearest estimated centroid, when the fingers report a hold.

        The stall says *something* is held, not *what*. Proximity to the TCP names it, and it is also what
        stops a close blocked by the table edge or by the arm's own link from reading as a grasp.
        ``positions`` are estimates; nothing here reads ground truth.
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
