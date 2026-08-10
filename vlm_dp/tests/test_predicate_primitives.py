"""CPU tests pinning the trusted primitives completion predicates are written in.

The primitives are the whole point of the refactor: every stage of every task's plan used to
carry its own hand-inlined copy of the same sensing (an aperture band, an 8-frame window, a
slip ratio, a quiescence speed), which is duplication that drifts silently. These tests pin
the behaviour that duplication used to encode, on synthetic histories, with no simulator:

  * grasped() keeps TWO modes and they are not interchangeable. Acquisition demands POSITIVE
    evidence of co-motion (a hand that moved and an object that came along); maintenance
    demands only the ABSENCE of slip, so a still hand keeps a hold rather than losing it.
    Collapsing them would let a pick-up stage fire on an object resting on the board.
  * released() fires on either departure or opening, and stationary() on speed alone.
  * sustained() is worst-of-window, not last-frame, so one good sample cannot advance a stage.
  * every primitive returns False on a history too short to judge, so the frames just after a
    reset never fire a stage.
  * grasp_tolerance() resolves a DECLARED local grasp feature and never launders the owner's
    whole-body extent in its place (the capsule lid-rim defect).
  * clearance_margin() REFUSES a fixture-owned declared feature loudly instead of answering
    with a carried body's number.
  * the advanceability preflight and the rollout run the SAME primitive objects.

Run: ``python -m vlm_dp.tests.test_predicate_primitives``.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.grounding.predicates import (
    CompletionPredicates,
    HistoryBuffer,
    PredicateContractError,
    PredicateRuntime,
)

W = PredicateRuntime.WINDOW
S = PredicateRuntime.SUSTAIN


# ---------------------------------------------------------------------------------------------
# Synthetic histories. One keypoint (index 0) plus a decoy (index 1) that never moves.


def history(n, *, aperture, hand_step, obj_step, obj0=(0.5, 0.0, 0.9), eef0=(0.5, 0.0, 0.9)):
    """Build ``n`` frames in which the hand and the object each advance by a fixed step."""
    buf = HistoryBuffer(maxlen=max(n, 1), dt=1.0 / 15.0)
    for t in range(n):
        eef = np.asarray(eef0, dtype=np.float64) + np.asarray(hand_step, dtype=np.float64) * t
        obj = np.asarray(obj0, dtype=np.float64) + np.asarray(obj_step, dtype=np.float64) * t
        buf.push(np.stack([obj, np.array([0.0, 0.0, 0.0])]), eef,
                 aperture[t] if isinstance(aperture, (list, tuple, np.ndarray)) else aperture)
    return buf.view()


def runtime(**kw):
    kw.setdefault("owner_of", lambda i: {0: "pear", 1: "scale", 2: "capsule"}.get(int(i)))
    kw.setdefault("grasp_ext_of", {"pear": 0.035})
    kw.setdefault("extents", {"pear": (0.04, 0.04, 0.04), "scale": (0.1, 0.1, 0.02)})
    kw.setdefault("carried", {"pear"})
    return PredicateRuntime(**kw)


def evaluate(fn, hist, rt=None):
    """Call one primitive against an installed frame, returning ``(value, components)``."""
    rt = rt or runtime()
    comps = {}
    rt.begin(hist, np.asarray(hist["eef"][-1]), np.asarray(hist["kp"][-1]), {}, comps)
    return fn(rt), comps


# ---------------------------------------------------------------------------------------------
# grasped(): acquisition


def test_grasp_acquire_positive():
    """Fingers stalled on something, hand travelled, object kept its offset -> acquired."""
    hist = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    got, comps = evaluate(lambda rt: rt.grasped(0, "pear_in_hand", "acquire"), hist)
    print(f"  acquire positive: {comps}")
    assert got is True
    assert comps["pear_in_hand"] and comps["closed_on_object"] and comps["rides_with_hand"]
    assert comps["at_hand"], "the object is at the TCP in this history"
    assert comps["margin_hand_disp_m"] > PredicateRuntime.CARRY_HAND_DISP_M
    assert comps["margin_slip_ratio"] < PredicateRuntime.SLIP_RATIO_MAX


def test_grasp_acquire_negative_still_hand():
    """A hand that never moved proves nothing: acquisition must NOT fire on a resting object.

    This is the case that separates the two modes. The fingers can be stalled on the object and
    the object can sit exactly where it always sat; without hand travel there is no evidence the
    gripper owns it. maintain() passes here on purpose, acquire() must not.
    """
    hist = history(W, aperture=0.30, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0))
    acq, comps = evaluate(lambda rt: rt.grasped(0, "pear_in_hand", "acquire"), hist)
    keep, _ = evaluate(lambda rt: rt.grasped(0, "still_held", "maintain"), hist)
    print(f"  still hand: acquire={acq} maintain={keep} ratio={comps['margin_slip_ratio']}")
    assert acq is False, "acquisition on a still hand would fire on an untouched object"
    assert keep is True, "maintenance on a still hand must not drop a hold"


def test_grasp_acquire_negative_object_left_behind():
    """Hand travelled, object stayed: the ratio is ~1 and nothing was acquired."""
    hist = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0))
    got, comps = evaluate(lambda rt: rt.grasped(0, "pear_in_hand", "acquire"), hist)
    print(f"  left behind: ratio={comps['margin_slip_ratio']:.2f}")
    assert got is False and comps["rides_with_hand"] is False
    assert comps["margin_slip_ratio"] > 0.9


def test_grasp_acquire_negative_open_fingers():
    """A commanded close on empty air sits below the band and can never read as a grasp."""
    hist = history(W, aperture=0.02, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    got, comps = evaluate(lambda rt: rt.grasped(0, "pear_in_hand", "acquire"), hist)
    print(f"  open fingers: aperture={comps['margin_aperture']}")
    assert got is False and comps["closed_on_object"] is False


def test_grasp_acquire_negative_out_of_reach():
    """Co-moving but far from the TCP is a coincidence, not a grasp."""
    hist = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0),
                   obj0=(0.5, 0.5, 0.9))
    got, comps = evaluate(lambda rt: rt.grasped(0, "pear_in_hand", "acquire"), hist)
    print(f"  out of reach: reach={comps['margin_reach_m']:.2f}")
    assert got is False and comps["at_hand"] is False


def test_grasp_acquire_band_literals_unchanged():
    """The aperture band is exactly (0.08, 0.60): the sensor is NOT the source of it here."""
    assert (PredicateRuntime.APERTURE_LO, PredicateRuntime.APERTURE_HI) == (0.08, 0.60)
    rt = runtime(sensor={"q_free": 0.7854, "stall_margin": 0.15, "q_touch": 0.05})
    print(f"  {rt.describe()}")
    assert "NOT used for the band" in rt.describe(), "the divergence must be logged, not hidden"
    for q, expect in ((0.07, False), (0.09, True), (0.59, True), (0.61, False)):
        hist = history(W, aperture=q, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
        got, _ = evaluate(lambda rt_: rt_.grasped(0, "g", "acquire"), hist, rt)
        assert got is expect, f"aperture {q} should read {expect}"


# ---------------------------------------------------------------------------------------------
# grasped(): maintenance


def test_grasp_maintain_positive_while_carrying():
    """Carrying: the hand moved a long way and the object held its offset."""
    hist = history(W, aperture=0.30, hand_step=(0.02, 0.0, 0.01), obj_step=(0.02, 0.0, 0.01))
    got, comps = evaluate(lambda rt: rt.grasped(0, "still_held", "maintain"), hist)
    print(f"  maintain carry: {comps['margin_slip_ratio']:.3f}")
    assert got is True and comps["still_held"] is True


def test_grasp_maintain_negative_on_slip():
    """The object fell out mid-carry: it stops tracking the hand and the hold is gone."""
    hist = history(W, aperture=0.30, hand_step=(0.02, 0.0, 0.0), obj_step=(0.0, 0.0, -0.02))
    got, comps = evaluate(lambda rt: rt.grasped(0, "still_held", "maintain"), hist)
    print(f"  maintain slip: ratio={comps['margin_slip_ratio']:.2f}")
    assert got is False and comps["rides_with_hand"] is False


def test_grasp_mode_must_be_named():
    """An unknown mode is a plan bug and is refused, not silently defaulted."""
    hist = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    try:
        evaluate(lambda rt: rt.grasped(0, "g", "held"), hist)
    except PredicateContractError as exc:
        print(f"  refused: {exc}")
        return
    raise AssertionError("grasped() must refuse a mode it does not implement")


# ---------------------------------------------------------------------------------------------
# released() / stationary()


def test_release_by_opening():
    """The fingers opened: released, even though the hand never went anywhere."""
    hist = history(W, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0))
    got, comps = evaluate(lambda rt: rt.released(0, "hand_let_go"), hist)
    print(f"  open release: open_frac={comps['margin_open_frac']}")
    assert got is True and comps["margin_open_frac"] >= PredicateRuntime.OPEN_FRAC_MIN


def test_release_by_departure():
    """The hand left with the fingers still stalled; the object stayed on the scale."""
    hist = history(W, aperture=0.30, hand_step=(0.02, 0.0, 0.02), obj_step=(0.0, 0.0, 0.0))
    got, comps = evaluate(lambda rt: rt.released(0, "hand_let_go"), hist)
    print(f"  departure: hand={comps['margin_hand_disp_m']:.3f} ratio={comps['margin_slip_ratio']:.2f}")
    assert got is True and comps["margin_slip_ratio"] > PredicateRuntime.DEPART_RATIO_MIN


def test_release_negative_while_still_carrying():
    """Fingers stalled and the object tracking the hand is the opposite of a release."""
    hist = history(W, aperture=0.30, hand_step=(0.02, 0.0, 0.0), obj_step=(0.02, 0.0, 0.0))
    got, _ = evaluate(lambda rt: rt.released(0, "hand_let_go"), hist)
    assert got is False


def test_stationary_vs_moving():
    """Quiescence is a statement about speed and nothing else."""
    still = history(W, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0))
    moving = history(W, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.005, 0.0, 0.0))
    quiet, cq = evaluate(lambda rt: rt.stationary(0, "object_quiescent"), still)
    busy, cb = evaluate(lambda rt: rt.stationary(0, "object_quiescent"), moving)
    print(f"  speeds: still={cq['margin_speed_mps']:.4f} moving={cb['margin_speed_mps']:.4f}")
    assert quiet is True and busy is False
    assert cb["margin_speed_mps"] > PredicateRuntime.QUIESCENT_MPS


# ---------------------------------------------------------------------------------------------
# sustained()


def test_sustained_is_worst_of_window():
    """One satisfied frame is not enough: the WORST frame in the window decides."""
    # z climbs 1cm per frame; the cost is (0.95 - z), so only the last frames satisfy it.
    hist = history(S + 2, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.01),
                   obj0=(0.5, 0.0, 0.90))
    cost = lambda ee, kp: 0.95 - kp[0][..., 2]          # noqa: E731 - the plan's own idiom
    got, comps = evaluate(lambda rt: rt.sustained(cost, "at_carry_height"), hist)
    last = 0.95 - hist["kp"][-1][0][2]
    print(f"  worst={comps['margin_at_carry_height']:.4f} last_frame={last:.4f}")
    assert last <= 0, "fixture: the final frame alone satisfies the cost"
    assert got is False, "a sustained condition must not fire off the last frame alone"
    assert comps["margin_at_carry_height"] > PredicateRuntime.BELIEF_TOL


def test_sustained_fires_when_whole_window_holds():
    """Held for the whole window, within the belief tolerance -> fires."""
    hist = history(S + 2, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0),
                   obj0=(0.5, 0.0, 1.00))
    got, comps = evaluate(lambda rt: rt.sustained(lambda ee, kp: 0.95 - kp[0][..., 2], "high"), hist)
    print(f"  worst={comps['margin_high']:.4f} tol={PredicateRuntime.BELIEF_TOL}")
    assert got is True and comps["high"] is True


def test_sustained_belief_tolerance_is_one_sided_slack():
    """BELIEF_TOL is slack for noise, not a second threshold: just inside passes, well outside fails."""
    tol = PredicateRuntime.BELIEF_TOL
    for excess, expect in ((tol * 0.5, True), (tol * 4.0, False)):
        hist = history(S, aperture=0.0, hand_step=(0.0, 0.0, 0.0), obj_step=(0.0, 0.0, 0.0))
        got, _ = evaluate(lambda rt: rt.sustained(lambda ee, kp: excess, "c"), hist)
        assert got is expect, f"cost {excess} vs tol {tol}"


# ---------------------------------------------------------------------------------------------
# short history


def test_short_history_is_false_for_every_primitive():
    """Frames just after a reset must never fire a stage, whatever the geometry says."""
    for n in range(0, W):
        hist = history(max(n, 1), aperture=0.30, hand_step=(0.02, 0.0, 0.0),
                       obj_step=(0.02, 0.0, 0.0))
        if n == 0:
            hist = {"kp": [], "eef": [], "gripper_aperture": [], "dt": 1.0 / 15.0}
        rt = runtime()
        comps = {}
        rt.begin(hist, np.zeros(3), np.zeros((2, 3)), {}, comps)
        if n == 0:
            continue                     # an empty buffer is never handed to a predicate
        results = {
            "acquire": rt.grasped(0, "g", "acquire"),
            "maintain": rt.grasped(0, "h", "maintain"),
            "released": rt.released(0, "r"),
            "stationary": rt.stationary(0, "s"),
            "sustained": rt.sustained(lambda ee, kp: -1.0, "c"),
        }
        # maintain and sustained legitimately fire once their own (shorter) window is full.
        strict = {k: v for k, v in results.items() if k in ("acquire", "released", "stationary")}
        print(f"  n={n}: {results}")
        assert not any(strict.values()), f"{n} frames fired {strict}"
        if n < S:
            assert results["sustained"] is False, f"sustained fired on {n} < {S} frames"


# ---------------------------------------------------------------------------------------------
# tolerances


def test_grasp_tolerance_uses_the_declared_feature_never_the_whole_body():
    """The capsule defect, as a tolerance: a 10mm lid rim must not resolve to a 0.3175m machine.

    The rim is a DECLARED grasp feature on a fixture; the geometry that matters is the feature's,
    and falling back to the owner's whole-body extent is what sized every grasp term for a
    coffee machine instead of a lip.
    """
    rt = runtime(owner_of=lambda i: {7: "capsule", 0: "pear"}.get(int(i)),
                 grasp_ext_of={"capsule": 0.010, "pear": 0.035},
                 extents={"capsule": (0.3175, 0.3175, 0.20), "pear": (0.04, 0.04, 0.04)},
                 carried={"pod"}, declared={7})
    got = rt.grasp_tolerance(7)
    print(f"  declared rim -> {got * 1e3:.1f}mm (owner whole-body extent {0.3175 * 1e3:.0f}mm)")
    assert abs(got - 0.010) < 1e-9, "the declared local extent must win"
    assert got < 0.05, "the whole-object extent must never be laundered through this"
    assert abs(rt.grasp_tolerance(0) - 0.035) < 1e-9


def test_grasp_tolerance_falls_back_only_without_a_grasp_feature():
    """A body with no measured or declared grasp feature at all still resolves, from its extent."""
    rt = runtime(owner_of=lambda i: "apple", grasp_ext_of={}, extents={"apple": (0.039, 0.04, 0.04)})
    assert abs(rt.grasp_tolerance(0) - 0.039) < 1e-9


def test_grasp_tolerance_refuses_an_unowned_keypoint():
    rt = runtime(owner_of=lambda i: None)
    try:
        rt.grasp_tolerance(3)
    except PredicateContractError as exc:
        print(f"  refused: {exc}")
        return
    raise AssertionError("a tolerance on an unowned keypoint must be refused")


def test_clearance_margin_allows_a_carried_body():
    rt = runtime()
    assert rt.clearance_margin(0) == PredicateRuntime.CLEARANCE_M


def test_clearance_margin_refuses_a_fixture_owned_declared_feature():
    """The capsule-lip class. A fixture's clearance is the PLAN's to state, never this to invent.

    Routing such a tolerance through clearance_margin() would hand back a carried body's number
    for a machine's lid recess. It refuses instead, and the message names the owner so the plan
    author sees which keypoint was wrong.
    """
    rt = runtime(owner_of=lambda i: {7: "capsule", 0: "pear"}.get(int(i)),
                 carried={"pear"}, declared={7})
    try:
        rt.clearance_margin(7)
    except PredicateContractError as exc:
        print(f"  refused: {exc}")
        assert "capsule" in str(exc) and "DECLARED" in str(exc)
        return
    raise AssertionError("clearance_margin() must refuse a fixture-owned declared feature")


def test_clearance_margin_refuses_a_declared_feature_on_a_carried_owner():
    """The bypass the ownership test alone leaves open, closed.

    ``carried`` is derived from the plan's grasp keypoints, so a fixture the plan grasps a
    declared feature ON lands in it -- the capsule machine owns both its lid rim and the grasp
    performed on that rim. Testing ownership alone would then answer for the rim with a carried
    body's number, which is the exact defect the narrow scope exists to prevent.
    """
    rt = runtime(owner_of=lambda i: "capsule", carried={"capsule", "pod"}, declared={7})
    try:
        rt.clearance_margin(7)
    except PredicateContractError as exc:
        print(f"  refused: {exc}")
        assert "DECLARED" in str(exc)
        return
    raise AssertionError("a declared feature must be refused even when its owner is carried")


def test_clearance_margin_refuses_a_fixture_the_plan_never_carries():
    rt = runtime()
    try:
        rt.clearance_margin(1)                # the scale: a fixture, never carried
    except PredicateContractError as exc:
        print(f"  refused: {exc}")
        assert "scale" in str(exc)
        return
    raise AssertionError("clearance_margin() must refuse a fixture keypoint")


# ---------------------------------------------------------------------------------------------
# binding: one implementation, two callers


PLAN = '''
def stage1_completion(history, end_effector, keypoints, robot_state, components):
    """Probe: record the primitive objects this predicate actually sees."""
    seen = components.setdefault("seen", {})
    seen["grasped"] = grasped
    seen["sustained"] = sustained
    seen["clearance_margin"] = clearance_margin
    held = grasped(0, "pear_in_hand", "acquire")
    return bool(held)
'''


def test_preflight_and_runtime_share_one_primitive_implementation():
    """The bound primitives a predicate sees are the runtime's own methods, on both paths.

    The advanceability preflight and the rollout both go through ``CompletionPredicates.evaluate``
    with the same bound runtime, so a preflight cannot certify a rule that differs from the one
    that runs. Identity, not equality: a re-implementation could agree with a copy of the rule
    while disagreeing with the rule.
    """
    comp = CompletionPredicates.from_text(PLAN, 1)
    rt = runtime()
    comp.bind(rt)
    assert comp.runtime is rt

    carry = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    fired_run, comps_run = comp.evaluate(0, carry)
    # The preflight path: a different synthesized history, same bound predicate object.
    synth = history(W, aperture=0.30, hand_step=(0.008, 0.0, 0.0), obj_step=(0.008, 0.0, 0.0))
    fired_pre, comps_pre = comp.evaluate(0, synth)
    print(f"  rollout={fired_run} preflight={fired_pre}")
    assert fired_run and fired_pre, "a genuine carry must fire on both paths"
    for key in ("grasped", "sustained", "clearance_margin"):
        a, b = comps_run["seen"][key], comps_pre["seen"][key]
        assert a == b, f"{key} differs between the two callers"
        assert a.__self__ is rt, f"{key} is not the bound runtime's own method"
        assert a.__func__ is getattr(PredicateRuntime, key), f"{key} is not the shipped implementation"


def test_unbound_predicate_fails_loudly_rather_than_silently():
    """A plan in the primitive vocabulary with no runtime bound must not read as 'not yet done'.

    It returns False (a shadow signal may never take down a rollout) but records the NameError,
    which the advanceability preflight turns into a refusal to roll out.
    """
    comp = CompletionPredicates.from_text(PLAN, 1)
    carry = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    fired, comps = comp.evaluate(0, carry)
    print(f"  unbound: fired={fired} error={comps.get('error')}")
    assert fired is False and "NameError" in comps.get("error", "")


def test_contract_error_inside_a_predicate_is_reported_not_swallowed():
    """A tolerance used on the wrong keypoint surfaces as a contract error, not a quiet False."""
    plan = ('def stage1_completion(history, end_effector, keypoints, robot_state, components):\n'
            '    """Probe."""\n'
            '    ok = clearance_margin(1) > 0.0\n'
            '    return bool(ok)\n')
    comp = CompletionPredicates.from_text(plan, 1).bind(runtime())
    carry = history(W, aperture=0.30, hand_step=(0.01, 0.0, 0.0), obj_step=(0.01, 0.0, 0.0))
    fired, comps = comp.evaluate(0, carry)
    print(f"  {comps.get('error')}")
    assert fired is False and comps.get("error", "").startswith("PredicateContractError")


# ---------------------------------------------------------------------------------------------


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n== {fn.__name__}")
        fn()
    print(f"\nOK: {len(tests)} tests")


if __name__ == "__main__":
    main()
