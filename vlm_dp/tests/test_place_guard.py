"""CPU tests for the two cost defects behind the video failure modes.

Both are the same shape as the hold-authority bug: a rule stated once and not applied where it
decides behaviour.

  1. center_region re-derived its dead zone as ``center_scale * radius`` instead of calling
     ``grasp_slack()``, which the codebase documents as "the single definition of at-the-grasp-pose"
     and which the stage machine already uses. So ``grasp_dead_zone`` moved the stage machine's
     notion of being at the grasp pose while the COST that decides where the gripper closes stayed
     on the legacy form -- precisely the drift the helper's docstring says it exists to prevent.
     Measured on the pear: holds beginning 24-29 mm from the centre survived, those beyond ~35 mm
     slipped on the lift, and the cost granted 12 mm of free slack on a ~17 mm centre estimate.

  2. Keepout penalties were squared. carry_clear worked out that this is wrong ("a squared cm-scale
     penetration is dwarfed by the sub-goal and the payload plows through") and went linear; collision
     and clear never followed. A 2 cm penetration at weight 40 costs 0.016 squared against ~0.4 from
     the attractor -- twenty-five times too little to bend the path.

Run: ``python -m vlm_dp.tests.test_place_guard``.
"""
from __future__ import annotations

import types

import torch

from vlm_dp.cost.terms import _keepout, grasp_slack

# Robotiq-scale gripper geometry, matching parity20/parity25.
GEOM = dict(open_half=0.04, aperture_margin=0.006, close_xy_floor=0.008, center_scale=0.40)
PEAR_R = 0.030          # measured pear half-width


def _geom(**kw):
    return types.SimpleNamespace(**{**GEOM, **kw})


def test_the_aperture_dead_zone_is_tighter_than_the_proportional_one_for_the_pear():
    """The behaviour being bought. 12 mm of free slack is most of the gap between a hold that
    survives and one that slips."""
    prop = grasp_slack(_geom(grasp_dead_zone="proportional"), PEAR_R)
    aper = grasp_slack(_geom(grasp_dead_zone="aperture"), PEAR_R)
    assert abs(prop - 0.012) < 1e-9, f"proportional must be center_scale*radius = 12 mm, got {prop}"
    assert aper < prop, (
        f"the aperture form must tighten the pear's dead zone, got {aper:.4f} >= {prop:.4f}")
    assert aper >= GEOM["close_xy_floor"] - 1e-9, (
        f"never below close_xy_floor, or the gate is unsatisfiable against grounding wobble: {aper}")


def test_the_dead_zone_shrinks_as_the_object_grows():
    """The physics the proportional form inverts: a fat object must be centred MORE precisely,
    because the fingers have less room around it. center_scale*radius does the opposite."""
    g = _geom(grasp_dead_zone="aperture")
    slacks = [grasp_slack(g, r) for r in (0.010, 0.020, 0.030)]
    assert slacks == sorted(slacks, reverse=True), (
        f"aperture slack must DECREASE as the object grows, got {slacks}")
    gp = _geom(grasp_dead_zone="proportional")
    props = [grasp_slack(gp, r) for r in (0.010, 0.020, 0.030)]
    assert props == sorted(props), (
        f"fixture: the proportional form should grow with radius (that is the defect), got {props}")


def test_center_region_uses_the_single_definition():
    """The bypass itself. center_region must track grasp_slack under BOTH settings, or the cost and
    the stage machine disagree about where the grasp pose is."""
    from vlm_dp.cost import terms
    import inspect
    # Code only: the comments explaining the fix necessarily name the constant it removed, and
    # matching those would make this assert fire on its own rationale.
    code = "\n".join(l.split("#", 1)[0] for l in inspect.getsource(terms.center_region).splitlines())
    assert "grasp_slack(" in code, (
        "center_region does not call grasp_slack: it is re-deriving the dead zone, so "
        "grasp_dead_zone cannot reach the cost that decides where the gripper closes")
    assert "center_scale" not in code, (
        "center_region still references center_scale directly; the whole point is that the dead "
        "zone has one definition")


def test_a_linear_keepout_can_argue_with_the_attractor():
    """The arithmetic that decides whether the hand routes around a seated object or through it."""
    pen = torch.tensor(0.02)                       # 2 cm penetration
    w, attractor = 40.0, 0.4                       # collision weight; terminal_reach 40 over 0.1 m
    squared = float(_keepout(pen, _geom(keepout_shape="squared")) * w)
    linear = float(_keepout(pen, _geom(keepout_shape="linear")) * w)
    assert squared < attractor / 10.0, (
        f"fixture: the squared form should be negligible against the attractor, got {squared:.4f}")
    assert linear >= attractor / 2.0, (
        f"the linear keepout must be the same order as the attractor it opposes, got {linear:.4f} "
        f"vs {attractor}; below that the gripper still plows through the seated object")


def test_linear_is_the_default_and_squared_is_still_reachable():
    pen = torch.tensor(0.02)
    assert float(_keepout(pen, _geom())) == float(pen), "linear must be the default"
    squared = float(_keepout(pen, _geom(keepout_shape="squared")))
    assert abs(squared - float(pen) ** 2) < 1e-9, (            # float32 tensor vs float64 literal
        f"the squared form must remain available for an A/B against the old arms, got {squared}")


def test_the_keepout_gradient_is_constant_not_vanishing():
    """Why linear works: a squared penalty's gradient goes to zero exactly where the object is about
    to be touched, so the last centimetre is unopposed."""
    g = _geom()
    grads = []
    for d in (0.002, 0.010, 0.020):
        pen = torch.tensor(d, requires_grad=True)
        _keepout(pen, g).backward()
        grads.append(float(pen.grad))
    assert all(abs(x - grads[0]) < 1e-6 for x in grads), (
        f"the linear keepout must have a constant gradient, got {grads}")
    sq_grads = []
    for d in (0.002, 0.020):
        pen = torch.tensor(d, requires_grad=True)
        _keepout(pen, _geom(keepout_shape="squared")).backward()
        sq_grads.append(float(pen.grad))
    assert sq_grads[0] < sq_grads[1] / 5.0, (
        f"fixture: the squared form should vanish near contact (that is the defect), got {sq_grads}")


def test_a_work_surface_is_not_a_keepout_for_either_guard():
    """The regression this file gained after the fact: enabling collision blocked the grasp itself.

    collision uses cylinders precisely so a flat support does not over-approximate in XY, but the
    margin is isotropic -- ee_r is a LATERAL gripper radius and it inflates the cylinder vertically
    too. A board then becomes a no-go slab (ee_r + coll_clear = 55 mm) that every top-down grasp
    must penetrate. Measured with the board included: the TCP stalled 51 mm ABOVE the pear centre
    and never descended, against 12 mm BELOW it on the last good run.

    So both guards must apply the fixture rule, and both must still keep a compact placed object.
    """
    import inspect
    from vlm_dp.cost import terms
    for name in ("collision", "clear"):
        code = "\n".join(l.split("#", 1)[0]
                         for l in inspect.getsource(getattr(terms, name)).splitlines())
        assert "clear_max_radius" in code, (
            f"{name} does not filter fixtures by clear_max_radius, so a work surface is a keepout "
            f"and top-down grasps are blocked by the support the object rests on")
        assert "DEFAULT_EXTENT" in code, f"{name} must consult extents to apply the fixture rule"


def test_the_fixture_rule_still_keeps_a_placed_pear_as_an_obstacle():
    """The filter must not undo the thing collision was enabled FOR. A pear is compact (~30 mm),
    far below clear_max_radius (100 mm), so it survives the rule; a board does not."""
    max_r = 0.10
    assert PEAR_R <= max_r, (
        f"a {PEAR_R * 1000:.0f} mm pear must remain an obstacle under the fixture rule")
    board_r = 0.20                       # measured board half-width, comfortably a fixture
    assert board_r > max_r, (
        "fixture check: the board must be excluded by the same rule, or the grasp stays blocked")


_TESTS = [test_the_aperture_dead_zone_is_tighter_than_the_proportional_one_for_the_pear,
          test_a_work_surface_is_not_a_keepout_for_either_guard,
          test_the_fixture_rule_still_keeps_a_placed_pear_as_an_obstacle,
          test_the_dead_zone_shrinks_as_the_object_grows,
          test_center_region_uses_the_single_definition,
          test_a_linear_keepout_can_argue_with_the_attractor,
          test_linear_is_the_default_and_squared_is_still_reachable,
          test_the_keepout_gradient_is_constant_not_vanishing]


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
