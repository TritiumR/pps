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

from vlm_dp.grasp_sensor import ApertureGraspSensor, adaptive_stall_margin

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


def _settled(q, steps=None):
    """A sensor that has been commanded closed and has settled at angle ``q``.

    Length is derived from the sensor's own windows so the helper stays correct if they are retuned.
    """
    sensor = ApertureGraspSensor()
    for _ in range(steps if steps is not None else sensor.close_steps + sensor.settle_steps + 1):
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


def test_the_window_spans_a_whole_chunk_of_control_steps():
    """The windows are control steps, and must be long enough to outlast a chunk's worth of travel.

    Sampling moved from once-per-replan to once-per-control-step; the window constants did not, so the
    debounce silently shrank by the chunk length and closes were certified mid-travel. A slow drift
    that stays inside settle_eps for only a few consecutive steps must still not read as settled.
    """
    sensor = ApertureGraspSensor()
    assert sensor.close_steps >= 8 and sensor.settle_steps >= 8, \
        "windows count control steps now; a per-replan value here is 4x too short"
    for i in range(sensor.close_steps + sensor.settle_steps):   # 2 mrad/step: under settle_eps per
        sensor.observe(_Env(0.10 + 0.002 * i), commanded_close=True)   # step, over it across the window
    assert not sensor.closed(), "a close still drifting across the settle window must not be certified"


def test_the_settle_test_is_density_invariant():
    """The regression the drops came from: peak-to-peak grows with sample count, so a DENSE window
    made settle_eps mean something different from the tuned per-replan regime. Probing at a fixed
    stride makes the verdict depend on the physics and not on how often observe() is called.

    Measured consequence of getting this wrong (real traces, base vs the two replacements):
      per-replan 3/3  FP 3.0%  precision 0.722   <- tuned
      per-step   3/3  FP 6.6%  precision 0.655   <- false holds -> advance -> lift nothing -> regrasp
      per-step 12/12  FP 1.9%  recall 0.073 in closed loop  <- never advances, thrashes
    """
    import random
    rng = random.Random(0)
    # A settled-but-noisy angle: jitter well under settle_eps per step, no net drift.
    trace = [0.26 + rng.uniform(-0.0015, 0.0015) for _ in range(60)]

    def spread(stride):
        s = ApertureGraspSensor(settle_steps=12, settle_probe_stride=stride)
        for q in trace:
            s.observe(_Env(q), commanded_close=True)
        return s._settle_spread()

    dense, decimated = spread(1), spread(4)
    assert decimated <= dense + 1e-12, (
        f"decimated probes must not exceed the dense peak-to-peak (got {decimated:.5f} > {dense:.5f})")
    # The point: the decimated statistic is what was tuned, and it is stable under call-rate changes.
    s4 = ApertureGraspSensor(settle_steps=12, settle_probe_stride=4)
    for q in trace:
        s4.observe(_Env(q), commanded_close=True)
    probes = list(s4._q)[-13:][::-1][::4][::-1]
    assert len(probes) == 4, f"span 12 at stride 4 must give 4 probes, got {len(probes)}"
    assert probes[-1] == trace[-1], "the newest sample must always be a probe"


def test_a_brief_open_does_not_destroy_the_close_gate():
    """The second aliasing path. _closed_for was an UNBROKEN run, so one stray open zeroed it and a
    hold could never certify at the observed flip rate (median close-run 4-8 steps; only 13-25% of
    runs reach 12). A duty cycle over the same span tolerates single-step flips.

    Self-reinforcing if wrong: no certification -> payload None -> carry_hold inert -> nothing opposes
    opening -> more flips -> still no certification.
    """
    s = ApertureGraspSensor()
    n = s.close_steps + s.settle_steps + 1
    for i in range(n):                       # settled on the pear, with ONE stray open near the end
        s.observe(_Env(PEAR), commanded_close=(i != n - 3))
    assert s.closed(), (
        "a single open command inside the close window destroyed the hold; the close gate is still an "
        "unbroken run rather than a duty cycle")
    assert s.holding(), "the same, via holding()"


def test_a_mostly_open_window_does_not_certify():
    """The duty cycle must not become a rubber stamp: a hand that is mostly open is not holding."""
    s = ApertureGraspSensor()
    n = s.close_steps + s.settle_steps + 1
    for i in range(n):
        s.observe(_Env(PEAR), commanded_close=(i % 3 == 0))     # ~33% duty, below close_duty
    assert not s.closed(), "a window that is only ~33% commanded closed must not certify a close"


def test_the_states_partition_the_angle():
    """open, stalled and on-air are mutually exclusive and cover the range, so no angle is two states."""
    for q in [i / 100.0 for i in range(0, 101)]:
        s = _settled(q)
        assert not (s.is_open() and s.closed_on_air()), f"angle {q} reads both open and closed-on-air"
        assert s.released() == (s.is_open() or s.closed_on_air()), f"angle {q}: released must be the complement"
        assert s.holding() != s.released(), f"angle {q}: a settled close either holds or does not"


def test_thin_declared_geometry_adapts_the_air_boundary():
    """A 10mm handle must not share the coarse-object empty-hand threshold."""
    margin = adaptive_stall_margin(0.005, default=0.15)
    assert abs(margin - 0.0465) < 1e-6
    sensor = ApertureGraspSensor(stall_margin=0.15,
                                 stall_margin_enter=margin, stall_margin_exit=0.02)
    # The calibrated aperture for a 10mm object is 0.692: hold under the adaptive
    # boundary, but closed-on-air under the old 0.15 threshold.
    for _ in range(13):
        sensor.observe(_Env(0.692), True)
    assert sensor.holding()
    assert not sensor.closed_on_air()
    # A marginal contact may relax after acquisition without being mistaken for free close.
    for _ in range(13):
        sensor.observe(_Env(0.75), True)
    assert not sensor.holding()
    assert not sensor.hold_lost()
    sensor.observe(_Env(AIR), True)
    assert sensor.hold_lost()


_TESTS = [test_predicates_match_the_calibration, test_an_open_hand_is_never_closed,
          test_holding_is_strictly_stronger_than_closed, test_a_close_is_not_read_before_it_settles,
          test_the_window_spans_a_whole_chunk_of_control_steps,
          test_the_settle_test_is_density_invariant,
          test_a_brief_open_does_not_destroy_the_close_gate,
          test_a_mostly_open_window_does_not_certify, test_the_states_partition_the_angle,
          test_thin_declared_geometry_adapts_the_air_boundary]


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
