"""CPU tests for the bounded grasp feel-around pattern (no simulator).

The pattern is what makes a failed grasp retry differently instead of hammering one point. These pin the
two properties the recovery relies on: it starts at the estimate, and it never leaves a bounded
neighbourhood. The stepping and target-offset wiring need the sensor, so they are checked in a sim run.

Run: python -m vlm_dp.tests.test_grasp_search.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.grasp_recovery import probe_pattern as _probe_pattern

_R = 0.045


def test_the_first_probe_is_the_estimate():
    """Probe 0 is the estimate itself, so search mode starts exactly where reopen mode grasps."""
    pts = _probe_pattern(_R)
    assert np.allclose(pts[0], np.zeros(3)), f"first probe must be the origin, got {pts[0]}"


def test_every_probe_stays_within_the_radius():
    """The search is a bounded neighbourhood: no probe wanders past r_max, so it cannot drift to a
    distractor the way the tracker did."""
    pts = _probe_pattern(_R)
    for p in pts:
        assert np.linalg.norm(p) <= _R + 1e-9, f"probe {p} exceeds r_max={_R}"


def test_the_probes_are_top_down():
    """Offsets are in the grasp (xy) plane. Grasp height is owned by tip_z, not the search."""
    for p in _probe_pattern(_R):
        assert abs(float(p[2])) < 1e-12, f"probe {p} has a z component"


def test_the_pattern_actually_spreads():
    """More than the origin, and the outer ring reaches the radius, so retries land at distinct poses."""
    pts = _probe_pattern(_R)
    assert len(pts) > 1, "a single point would reproduce the hammer"
    assert max(float(np.linalg.norm(p)) for p in pts) > 0.9 * _R, "the search never reaches out to r_max"
    d = [tuple(np.round(p, 4)) for p in pts]
    assert len(set(d)) == len(d), "probes must be distinct, or two retries repeat the same pose"


def test_the_pattern_is_deterministic():
    """Same radius gives the same sequence: the search is reproducible across replans and seeds."""
    assert all(np.allclose(a, b) for a, b in zip(_probe_pattern(_R), _probe_pattern(_R)))


def test_the_radius_scales_the_whole_pattern():
    """A larger search radius scales every offset, so the knob means what it says."""
    small, big = _probe_pattern(0.02), _probe_pattern(0.04)
    assert max(np.linalg.norm(p) for p in big) > 1.9 * max(np.linalg.norm(p) for p in small)


_TESTS = [test_the_first_probe_is_the_estimate, test_every_probe_stays_within_the_radius,
          test_the_probes_are_top_down, test_the_pattern_actually_spreads,
          test_the_pattern_is_deterministic, test_the_radius_scales_the_whole_pattern]


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
