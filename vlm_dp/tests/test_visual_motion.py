"""CPU tests pinning that the visual tracker reports MOTION, not a substituted position.

The defect: the tracker projects an object's estimated CENTRE to a pixel, follows that pixel, then
back-projects it to a point on the object's visible SURFACE -- and wrote that surface point into the
belief as if it were the centre. Those are different geometric quantities, and no amount of tracking
accuracy reconciles them, so every correction injected a fixed surface-to-centre bias.

Measured on a static apple: 1.4 mm centre error before the first visual correction, 34.3 mm after,
of which ~20 mm was +z. That is the same magnitude as the descent shortfall the CONTROLLER was then
blamed for -- the arm was descending accurately toward a belief that had been moved upward, and the
cost terms were retuned repeatedly to compensate for it.

A displacement is well posed where a position substitution is not: if an object translates by d, its
surface point and its centre both translate by d. These tests pin that property, including under a
static object (the case that exposed it) and across a re-perception rebase.

Run: ``python -m vlm_dp.tests.test_visual_motion``.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.visual_tracker import VisualTracker

CENTRE = np.array([0.50, 0.10, 0.2309])      # believed object centre
SURFACE = CENTRE + np.array([0.0, 0.0, 0.030])   # camera sees the TOP: +30 mm in z


def _tracker():
    """A VisualTracker with the CoTracker model bypassed -- only _lift's arithmetic is under test."""
    t = VisualTracker.__new__(VisualTracker)
    t.names = ["apple"]
    t.vis_thresh = 0.5
    t.H, t.W = 480, 640
    t.sx = t.sy = 1.0
    t._last = {"apple": CENTRE.copy()}
    t._ref_centre = {"apple": CENTRE.copy()}
    t._ref_surface = {}
    return t


def _lift(t, surface_point, vis=1.0):
    """Drive _lift with a world-point image whose tracked pixel holds surface_point."""
    import torch
    points = np.zeros((t.H, t.W, 3), dtype=np.float64)
    points[10, 20] = surface_point
    uv = torch.as_tensor(np.array([[20.0, 10.0]]))
    return t._lift(uv, torch.as_tensor(np.array([vis])), points)


def test_the_first_sighting_is_an_anchor_not_a_correction():
    """With no prior surface reference there is no displacement to report yet, so emitting one would
    be emitting the raw surface point -- exactly the bug."""
    t = _tracker()
    out = _lift(t, SURFACE)
    assert out == {}, (
        f"the first visible sighting emitted a correction ({out}); it must only anchor, or the "
        "surface point is written into the belief as if it were the centre")
    assert "apple" in t._ref_surface, "the first sighting must anchor the surface reference"


def test_a_static_object_is_not_moved():
    """THE regression. A stationary object must receive a zero-displacement correction. Position
    substitution instead reports the surface point, biasing the belief by surface-to-centre (+30 mm
    z here; measured ~20 mm on the real apple)."""
    t = _tracker()
    _lift(t, SURFACE)                       # anchor
    out = _lift(t, SURFACE)                 # same place: the object has not moved
    assert "apple" in out, "a visible static object should still produce a (zero) correction"
    err = float(np.linalg.norm(out["apple"] - CENTRE))
    assert err < 1e-9, (
        f"a static object's belief moved {err * 1000:.1f} mm. The tracker is substituting the "
        f"surface point for the centre rather than reporting displacement")


def test_a_translation_is_reported_exactly():
    """The property that makes displacement well posed: object moves by d, surface moves by d, so
    the centre must move by d."""
    t = _tracker()
    _lift(t, SURFACE)
    d = np.array([0.04, -0.02, 0.01])
    out = _lift(t, SURFACE + d)
    assert np.allclose(out["apple"], CENTRE + d, atol=1e-9), (
        f"expected the centre to move by {d}, got {out['apple'] - CENTRE}")


def test_the_surface_to_centre_offset_never_enters_the_belief():
    """Whatever the offset between the tracked surface point and the centre, it must cancel."""
    for off in ([0, 0, 0.03], [0.025, 0, 0], [0.01, -0.02, 0.045]):
        t = _tracker()
        surf = CENTRE + np.asarray(off, dtype=np.float64)
        _lift(t, surf)
        d = np.array([0.03, 0.03, -0.01])
        out = _lift(t, surf + d)
        assert np.allclose(out["apple"], CENTRE + d, atol=1e-9), (
            f"surface offset {off} leaked into the belief: {out['apple'] - (CENTRE + d)}")


def test_occlusion_offers_no_correction():
    t = _tracker()
    _lift(t, SURFACE)
    assert _lift(t, SURFACE + np.array([0.05, 0, 0]), vis=0.0) == {}, (
        "an occluded track must offer no correction; dead-reckoning stands")


def test_rebase_prevents_double_counting_a_reperception():
    """Re-perception moves the belief by means the tracker did not observe. Without re-anchoring, the
    next displacement is added to a stale reference and the jump is counted twice."""
    t = _tracker()
    _lift(t, SURFACE)
    moved = CENTRE + np.array([0.10, 0.0, 0.0])          # re-perceived somewhere new
    t.rebase({"apple": moved})
    _lift(t, SURFACE + np.array([0.10, 0.0, 0.0]))       # re-anchor at the new surface
    d = np.array([0.0, 0.01, 0.0])
    out = _lift(t, SURFACE + np.array([0.10, 0.0, 0.0]) + d)
    assert np.allclose(out["apple"], moved + d, atol=1e-9), (
        f"expected {moved + d}, got {out['apple']} -- the re-perception jump was double-counted")


_TESTS = [test_the_first_sighting_is_an_anchor_not_a_correction,
          test_a_static_object_is_not_moved,
          test_a_translation_is_reported_exactly,
          test_the_surface_to_centre_offset_never_enters_the_belief,
          test_occlusion_offers_no_correction,
          test_rebase_prevents_double_counting_a_reperception]


def main():
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILED'} ({len(_TESTS)} tests)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
