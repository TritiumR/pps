"""Detect grasps from the finger joint angle.

A grasp stalls below the free-close angle. Settling is measured from joint
position because a held object may report a steady nonzero joint velocity.
"""

from __future__ import annotations

import collections

import numpy as np


class ApertureGraspSensor:
    """Detects whether the gripper is holding an object.

    Call observe() once per control step. Window sizes are measured in control
    steps and are designed to remain stable across observation frequencies.
    """

    def __init__(
        self,
        q_free=0.7854,
        stall_margin=0.15,
        q_touch=0.05,
        settle_steps=12,
        settle_eps=0.01,
        close_steps=12,
        proximity=0.10,
        settle_probe_stride=4,
        close_duty=0.75,
        legacy=False,
        stall_margin_enter=None,
        stall_margin_exit=None,
    ):
        # Restore the original continuous-close and dense-settling gates.
        self.legacy = bool(legacy)
        if self.legacy:
            settle_steps, close_steps, settle_probe_stride = 3, 3, 1

        self.q_free = q_free
        self.stall_margin = stall_margin

        # Separate acquisition and release thresholds to prevent boundary oscillation.
        self.hysteresis = (
            stall_margin_enter is not None or stall_margin_exit is not None
        )
        self.stall_margin_enter = (
            stall_margin
            if stall_margin_enter is None
            else float(stall_margin_enter)
        )
        self.stall_margin_exit = (
            stall_margin
            if stall_margin_exit is None
            else float(stall_margin_exit)
        )

        self.q_touch = q_touch
        self.settle_steps = settle_steps
        self.settle_eps = settle_eps
        self.close_steps = close_steps
        self.proximity = proximity

        # Decimation keeps the settling test independent of observation frequency.
        self.settle_probe_stride = max(int(settle_probe_stride), 1)

        # A duty cycle tolerates brief open commands during an otherwise active close.
        self.close_duty = float(close_duty)

        self._q = collections.deque(maxlen=max(settle_steps + 1, 2))
        self._cmd = collections.deque(maxlen=max(close_steps, 1))
        self._closed_for = 0

    def observe(self, env, commanded_close: bool) -> None:
        """Record the gripper state after one control step."""
        self._closed_for = self._closed_for + 1 if commanded_close else 0
        self._cmd.append(bool(commanded_close))
        self._q.append(env.gripper_q())

    def is_open(self) -> bool:
        """Return whether the fingers have not moved beyond the touch threshold."""
        return self.aperture() <= self.q_touch

    def closed_on_air(self) -> bool:
        """Return whether the fingers reached the free-close range."""
        return self.aperture() >= self.q_free - self.stall_margin_exit

    def closed(self) -> bool:
        """Return whether a commanded close has travelled and settled."""
        if not self._close_commanded() or len(self._q) < self.settle_steps + 1:
            return False
        if self._settle_spread() > self.settle_eps:
            return False
        return not self.is_open()

    def _close_commanded(self) -> bool:
        """Return whether closing was commanded for enough of the window."""
        if self.legacy:
            return self._closed_for >= self.close_steps
        if len(self._cmd) < self.close_steps:
            return False
        return (sum(self._cmd) / len(self._cmd)) >= self.close_duty

    def _settle_spread(self) -> float:
        """Return the peak-to-peak aperture across decimated settling probes."""
        window = list(self._q)[-(self.settle_steps + 1) :]
        probes = window[::-1][:: self.settle_probe_stride][::-1]
        return max(probes) - min(probes)

    def holding(self) -> bool:
        """Return whether a settled close stalled below the free-close range."""
        return (
            self.closed()
            and self.aperture() < self.q_free - self.stall_margin_enter
        )

    def hold_lost(self) -> bool:
        """Return whether a latched grasp has opened or reached free close."""
        return self.is_open() or (self.hysteresis and self.closed_on_air())

    def held_object(self, positions: dict, tcp) -> str | None:
        """Return the nearest object within reach when a grasp is detected."""
        if not self.holding() or not positions:
            return None

        tcp = np.asarray(tcp, dtype=np.float64)
        name, dist = min(
            (
                (
                    name,
                    float(
                        np.linalg.norm(
                            np.asarray(position, dtype=np.float64) - tcp
                        )
                    ),
                )
                for name, position in positions.items()
            ),
            key=lambda item: item[1],
        )
        return name if dist <= self.proximity else None

    def close_age(self) -> int:
        """Return consecutive applied control steps carrying a close command."""
        return self._closed_for

    def aperture(self) -> float:
        """Return the latest finger joint angle in radians."""
        return self._q[-1] if self._q else 0.0

    def released(self) -> bool:
        """Return whether the gripper is open or closed without an obstruction."""
        return self.is_open() or self.closed_on_air()
