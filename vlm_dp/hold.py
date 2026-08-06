"""Hold-state helpers shared by the bridge and world models."""

from __future__ import annotations


class HoldLatch:
    """Maintain a latched hold across control steps."""

    # Radian-scale default, from the Isaac gripper convention.
    _SLIP_MARGIN = 0.18

    def __init__(self, sensor, slip_margin=None):
        self.sensor = sensor
        self._held = None
        self._grip_aperture = None
        self._slip_margin = (
            self._SLIP_MARGIN if slip_margin is None else float(slip_margin)
        )

    def held(self):
        return self._held

    def update(self, positions, tcp, candidates=None):
        """Update the latched hold from the current sensor state."""
        if (
            self._held is not None
            and self._grip_aperture is not None
            and not self.sensor.hold_lost()
            and self.sensor.aperture()
            < self._grip_aperture + self._slip_margin
        ):
            held = self._held
        else:
            near = (
                {
                    name: pos
                    for name, pos in positions.items()
                    if name in candidates
                }
                if candidates
                else positions
            )
            held = self.sensor.held_object(near, tcp)

        if held != self._held:
            self._grip_aperture = (
                self.sensor.aperture()
                if held is not None
                else None
            )
            self._held = held

        return self._held


def payload_held(payload, authority, world, sensor, tcp, pos):
    """Return whether the requested payload is currently held."""
    latched = getattr(world, "held", None)
    if authority == "latched" and callable(latched):
        return latched() == payload

    return sensor.held_object({payload: pos}, tcp) == payload