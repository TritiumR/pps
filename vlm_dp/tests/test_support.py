"""CPU tests for the perception quality gates: DERIVED support surface + mask acceptance.

``support`` was a hand-authored role naming a scene entity, and only one of five tasks declared one --
so every other task fell back to the visible-extent midpoint, which sits ~1-2cm too HIGH because an
object's top is better seen than its occluded bottom. ``Perception.support_height`` measures the
surface instead, from a ring of scene points just outside the object's own footprint.

Run: ``python -m vlm_dp.tests.test_support``.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.perception import Perception
from vlm_dp.sim_helpers import center_from_points

RNG = np.random.default_rng(0)


def _scene(surface_z=0.80, obj_c=(0.5, 0.0), obj_r=0.03, obj_h=0.06, extra=()):
    """A flat surface with one object standing on it, as a back-projected point cloud.

    Only the object's TOP HALF is in the cloud: a top-down camera cannot see down to where it meets
    the surface, which is exactly the bias the derived support exists to correct.
    """
    g = np.stack(np.meshgrid(np.linspace(0.2, 0.8, 220), np.linspace(-0.3, 0.3, 220)), -1).reshape(-1, 2)
    table = np.column_stack([g, np.full(len(g), surface_z)])
    ang = RNG.uniform(0, 2 * np.pi, 1500)
    rad = obj_r * np.sqrt(RNG.uniform(0, 1, 1500))
    obj = np.column_stack([obj_c[0] + rad * np.cos(ang), obj_c[1] + rad * np.sin(ang),
                           RNG.uniform(surface_z + obj_h / 2, surface_z + obj_h, 1500)])
    return np.vstack([table, obj] + list(extra)), obj


class _P(Perception):
    """Perception with the vision stack bypassed: points and one object's mask supplied directly."""

    def __init__(self, scene, obj_pts):
        self.points = scene
        self._obj = obj_pts
        self._fixture_points = {}
        self.support, self.support_names = None, frozenset({"obj"})

    def object_points(self, name):
        return self._obj if name == "obj" else None


def test_support_is_measured_from_the_ring_not_the_object():
    """The derived height is the surface, even though no cloud point of the object touches it."""
    scene, obj = _scene(surface_z=0.80)
    got = _P(scene, obj).support_height("obj")
    assert got is not None, "a flat surface around the object must be measurable"
    assert abs(got - 0.80) < 2e-3, f"support should be the surface at 0.800, got {got:.4f}"
    assert obj[:, 2].min() > 0.82, "fixture check: the object cloud must NOT reach the surface"


def test_it_fixes_the_high_bias_in_the_grasp_centre():
    """The whole point: without a support the centre lands high, with it on the object's mid-height."""
    surface, height = 0.80, 0.06
    scene, obj = _scene(surface_z=surface, obj_h=height)
    truth = surface + height / 2.0

    naive = center_from_points(obj)[2]
    derived = center_from_points(obj, _P(scene, obj).support_height("obj"))[2]
    assert abs(derived - truth) < abs(naive - truth), (
        f"derived support must beat the visible-extent midpoint: derived={derived:.4f} "
        f"naive={naive:.4f} truth={truth:.4f}")
    assert naive - truth > 0.005, "fixture check: the naive centre must actually be biased HIGH"
    assert abs(derived - truth) < 0.005, f"derived centre should land within 5mm, got {derived:.4f}"


def test_a_taller_neighbour_does_not_lift_the_estimate():
    """A neighbour clipping the ring must not raise the surface (median + below-the-base cut)."""
    scene, obj = _scene(surface_z=0.80)
    tall = np.column_stack([RNG.uniform(0.56, 0.60, 900), RNG.uniform(-0.02, 0.02, 900),
                            RNG.uniform(0.80, 1.05, 900)])            # a bottle inside the ring
    got = _P(np.vstack([scene, tall]), obj).support_height("obj")
    assert abs(got - 0.80) < 5e-3, f"a tall neighbour must not lift the surface, got {got:.4f}"


def test_it_tracks_a_raised_surface():
    """An object on a tray reports the tray top, not the table, and no entity is named either way."""
    for z in (0.80, 0.86, 0.92):
        scene, obj = _scene(surface_z=z)
        got = _P(scene, obj).support_height("obj")
        assert abs(got - z) < 2e-3, f"surface at {z} measured as {got:.4f}"


def test_it_declines_rather_than_guesses():
    """Too few ring points -> None, so the caller falls back instead of using a fabricated height."""
    _, obj = _scene()
    assert _P(obj, obj).support_height("obj") is None, "object points alone are not a surface"
    assert _P(np.zeros((0, 3)), obj).support_height("obj") is None, "an empty scene must decline"


# ------------------------------------------------------------------------- mask acceptance gate
def _box_mask(x0, y0, x1, y1, h=200, w=200):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_boundary_bleed_is_scale_free():
    """The SAME 3px bleed must measure ~3px on a tiny box and on a large one.

    An area FRACTION would not: 3px reads as 20% of a 50px box but 2% of a 500px one, so a fractional
    threshold silently demands more precision from small objects than large ones.
    """
    from vlm_dp.perception import _mask_escape

    tiny = _mask_escape(_box_mask(7, 7, 23, 23), (10, 10, 20, 20))
    small = _mask_escape(_box_mask(47, 47, 103, 103), (50, 50, 100, 100))
    large = _mask_escape(_box_mask(7, 7, 183, 183, 400, 400), (10, 10, 180, 180))
    for label, got in (("tiny", tiny), ("small", small), ("large", large)):
        assert 2.0 < got < 5.0, f"a 3px bleed on a {label} box measured {got:.1f}px"
    assert max(tiny, small, large) - min(tiny, small, large) < 1.5, \
        f"the measure must not depend on box size: {tiny:.1f} {small:.1f} {large:.1f}"


def test_a_bled_mask_is_rejected_and_a_good_one_is_not():
    """A mask that left its own box is rejected, ordinary boundary wander is not."""
    from vlm_dp.perception import _MASK_ESCAPE_TOL, _mask_escape

    box = (50, 50, 100, 100)
    good = {"exactly the box": _box_mask(*box), "inside the box": _box_mask(60, 60, 90, 90),
            "3px boundary bleed": _box_mask(47, 47, 103, 103)}
    bad = {"covers the scene": _box_mask(0, 0, 200, 200), "a different object": _box_mask(120, 120, 180, 180)}
    for label, m in good.items():
        assert _mask_escape(m, box) <= _MASK_ESCAPE_TOL, f"{label} must be accepted"
    for label, m in bad.items():
        assert _mask_escape(m, box) > _MASK_ESCAPE_TOL, f"{label} must be rejected"


def test_a_degenerate_mask_or_box_is_rejected_not_crashed():
    """Empty masks and zero-area boxes decline, nothing raises."""
    from vlm_dp.perception import _mask_escape

    assert _mask_escape(np.zeros((200, 200), bool), (50, 50, 100, 100)) == float("inf")
    assert _mask_escape(_box_mask(50, 50, 100, 100), (50, 50, 50, 50)) == float("inf")
    assert _mask_escape(_box_mask(50, 50, 100, 100), (-90, -90, -10, -10)) == float("inf")


# --------------------------------------------------------- unnamed obstacles from scene geometry
def _blob(centre, half=0.03, n=400):
    return RNG.uniform(-half, half, (n, 3)) + np.asarray(centre)


def test_unnamed_geometry_becomes_object_shaped_obstacles():
    """Leftover cloud clusters into blobs carrying the same (centre, extents) a named object has.

    This is what keeps distractors avoidable once the vocabulary names only the instruction's
    referents: the keepout terms exclude by NAME, so an anonymous blob is always an obstacle and can
    never be selected as a target.
    """
    from vlm_dp.perception import _cluster_blobs

    got = _cluster_blobs(np.vstack([_blob((0.5, 0.0, 0.9)), _blob((0.7, 0.2, 0.9)), _blob((0.3, -0.2, 0.9))]))
    assert len(got) == 3, f"three separated objects should give three blobs, got {len(got)}"
    for centre, ext in got:
        assert len(ext) == 3 and ext[0] <= ext[1], "extents must be (narrow, wide, half-height)"
        assert 0.02 < ext[1] < 0.05, f"a 3cm blob should measure ~3cm, got {ext[1]:.3f}"
    assert {tuple(np.round(c, 1)) for c, _ in got} == {(0.5, 0.0, 0.9), (0.7, 0.2, 0.9), (0.3, -0.2, 0.9)}


def test_structure_is_not_turned_into_a_keepout():
    """A counter edge or wall must not become an obstacle cylinder, it would wall off the workspace."""
    from vlm_dp.perception import _cluster_blobs

    slab = RNG.uniform(-1, 1, (3000, 3)) * np.array([0.4, 0.02, 0.02]) + np.array([0.5, 0.5, 0.9])
    got = _cluster_blobs(np.vstack([_blob((0.5, 0.0, 0.9)), slab]))
    kept = [(c, e) for c, e in got if e[1] <= 0.10]
    assert len(got) == 2 and len(kept) == 1, f"the slab must be filtered out, kept {len(kept)} of {len(got)}"
    assert np.allclose(np.round(kept[0][0], 1), (0.5, 0.0, 0.9)), "the object, not the slab, survives"


def test_noise_specks_are_not_obstacles():
    """A handful of stray points is not a thing to avoid."""
    from vlm_dp.perception import _cluster_blobs

    got = _cluster_blobs(np.vstack([_blob((0.5, 0.0, 0.9)), _blob((0.9, 0.4, 0.9), half=0.01, n=5)]))
    assert len(got) == 1, f"a 5-point speck must not become an obstacle, got {len(got)} blobs"


# ------------------------------------------------------ co-placement seat shift (weight second object)
def test_shifted_seat_clears_a_placed_object():
    """The second fruit's seat is pushed clear of the first, the whole of weight co-placement.

    This is the ONLY co-placement mechanism the weight config enables (`carry_clear` is not in its term
    list), so the earlier double-stacking worry does not apply: there is nothing stacked with it.
    """
    from vlm_dp.grounding.gt import _SEAT_CLEARANCE, _shifted_seat

    seat = np.array([0.5, 0.5, 0.8])
    pear = (np.array([0.5, 0.5, 0.8]), 0.03)                 # first fruit, at the nominal seat
    out = _shifted_seat(seat, 0.03, [pear])
    need = 0.03 + 0.03 + _SEAT_CLEARANCE
    assert np.linalg.norm(out[:2] - pear[0][:2]) >= need - 1e-6, "the second seat must clear the first"
    assert out[2] == seat[2], "only xy is nudged; the seat height is unchanged"


def test_shifted_seat_clears_every_prior_it_is_given():
    """Sequential shifting converges, so the final seat clears all priors in any order or arrangement.

    Pinned because the shift mutates the seat per prior with no re-check, which looks like it could
    push back into an earlier one, but each push is radially outward, so across collinear, opposed and
    surrounding layouts the result still clears them all. Adversarial cases were searched for and none
    broke it. If a future layout does, this test is where it will surface.
    """
    from vlm_dp.grounding.gt import _SEAT_CLEARANCE, _shifted_seat

    need = 0.06 + _SEAT_CLEARANCE
    layouts = {
        "collinear": [(np.array([0.47, 0.5, 0.8]), 0.03), (np.array([0.53, 0.5, 0.8]), 0.03)],
        "opposed": [(np.array([0.5, 0.53, 0.8]), 0.03), (np.array([0.5, 0.47, 0.8]), 0.03)],
        "surrounding": [(np.array([0.50, 0.46, 0.8]), 0.03), (np.array([0.53, 0.52, 0.8]), 0.03),
                        (np.array([0.47, 0.52, 0.8]), 0.03)],
    }
    for name, priors in layouts.items():
        out = _shifted_seat(np.array([0.5, 0.5, 0.8]), 0.03, priors)
        for pos, r in priors:
            assert np.linalg.norm(out[:2] - pos[:2]) >= need - 1e-6, f"{name}: final seat overlaps a prior"


_TESTS = [test_support_is_measured_from_the_ring_not_the_object,
          test_it_fixes_the_high_bias_in_the_grasp_centre,
          test_a_taller_neighbour_does_not_lift_the_estimate,
          test_it_tracks_a_raised_surface,
          test_it_declines_rather_than_guesses,
          test_boundary_bleed_is_scale_free,
          test_a_bled_mask_is_rejected_and_a_good_one_is_not,
          test_a_degenerate_mask_or_box_is_rejected_not_crashed,
          test_unnamed_geometry_becomes_object_shaped_obstacles,
          test_structure_is_not_turned_into_a_keepout,
          test_noise_specks_are_not_obstacles,
          test_shifted_seat_clears_a_placed_object,
          test_shifted_seat_clears_every_prior_it_is_given]


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
