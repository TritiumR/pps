"""CPU tests pinning what the finger angle means (no simulator needed).

The angle's direction is counter-intuitive: larger is more closed, so a fully open hand reads about 0 and a
close that met nothing runs UP to the free-close angle. A caller that re-derives a band from
``q_free``/``stall_margin``/``q_touch`` can invert it and the arithmetic looks fine: measured, the press
advance asserted "closed" for a fully open hand because it dropped the ``q_touch`` bound, so a rollout
advanced past the grasp empty-handed and then chased its own held keypoints for 700 steps.

So the states are named on the sensor and pinned here against the calibration in its own docstring.
Run: ``python -m vlm_dp.tests.test_grasp_sensor``.
"""
from __future__ import annotations

from vlm_dp.grasp_sensor import ApertureGraspSensor

# Finger angles from the sensor's docstring calibration, plus the open hand the bug turned on.
OPEN, APPLE, PEAR, AIR = 0.0, 0.166, 0.258, 0.785

# state -> (is_open, closed_on_air, closed, holding, released)
EXPECTED = {
    "open hand": (OPEN, (True, False, False, False, True)),
    "apple held": (APPLE, (False, False, True, True, False)),
    "pear held": (PEAR, (False, False, True, True, False)),
    "closed on air": (AIR, (False, True, True, False, True)),
}


class _Env:
    """Minimal env stub: the sensor reads only ``gripper_q``."""

    def __init__(self, q):
        self._q = q

    def gripper_q(self):
        return self._q


def _settled(q, steps=8):
    """A sensor that has been commanded closed and has settled at angle ``q``."""
    sensor = ApertureGraspSensor()
    for _ in range(steps):
        sensor.observe(_Env(q), commanded_close=True)
    return sensor


def test_predicates_match_the_calibration():
    """Each named state reads exactly as the calibration says it should."""
    wrong = {}
    for label, (q, expect) in EXPECTED.items():
        s = _settled(q)
        got = (s.is_open(), s.closed_on_air(), s.closed(), s.holding(), s.released())
        if got != expect:
            wrong[label] = {"angle": q, "got": got, "expected": expect}
    assert not wrong, f"(is_open, closed_on_air, closed, holding, released) disagree: {wrong}"


def test_an_open_hand_is_never_closed():
    """The regression this file exists for: an open hand must not satisfy the press advance."""
    s = _settled(OPEN)
    assert s.is_open(), "a hand at the rest angle must read open"
    assert not s.closed(), "an OPEN hand must never read as a completed close -- a press advance uses this"
    assert not s.holding(), "an open hand holds nothing"


def test_holding_is_strictly_stronger_than_closed():
    """``holding`` must imply ``closed``: it is the same close, plus stalling on an object."""
    for label, (q, _) in EXPECTED.items():
        s = _settled(q)
        assert not s.holding() or s.closed(), f"{label}: holding must imply closed"
    assert _settled(AIR).closed() and not _settled(AIR).holding(), \
        "closing on air completes a close but holds nothing -- that gap is why press needs `closed`"


def test_a_close_is_not_read_before_it_settles():
    """Mid-travel the angle passes through the stall band, and reading it then certifies a grasp that
    has not happened."""
    sensor = ApertureGraspSensor()
    for q in (0.0, 0.05, 0.12, 0.20):                 # still travelling toward a free close
        sensor.observe(_Env(q), commanded_close=True)
    assert not sensor.closed(), "an unsettled close must not read as completed"
    assert not sensor.holding(), "an unsettled close must not certify a hold"


def test_the_states_partition_the_angle():
    """open, stalled and on-air are mutually exclusive and cover the range, so no angle is two states."""
    for q in [i / 100.0 for i in range(0, 101)]:
        s = _settled(q)
        assert not (s.is_open() and s.closed_on_air()), f"angle {q} reads both open and closed-on-air"
        assert s.released() == (s.is_open() or s.closed_on_air()), f"angle {q}: released must be the complement"
        assert s.holding() != s.released(), f"angle {q}: a settled close either holds or does not"


_TESTS = [test_predicates_match_the_calibration, test_an_open_hand_is_never_closed,
          test_holding_is_strictly_stronger_than_closed, test_a_close_is_not_read_before_it_settles,
          test_the_states_partition_the_angle]


def main():
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report harness/import errors, don't hide them
            failures += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILED'} ({len(_TESTS)} tests)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
