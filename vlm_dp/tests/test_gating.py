"""CPU tests pinning which cost terms are active on each substage (no simulator needed).

The composite cost is one weighted sum over every configured term, and a term applies to a substage
only by self-gating to zeros. That gate is the whole substage-selector, alpha = w * 1[applies], so it
is the thing worth pinning. Each scenario below is one substage archetype: it declares the terms that
must be non-zero, and every other registered term must be exactly zero. Getting this wrong is silent in
a rollout (a term that should have stood down just quietly fights the plan), which is why it is a test
and not a comment.

Run: python -m vlm_dp.tests.test_gating.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import torch

from vlm_dp.cost.base_cost import DEFAULT_GEOM
from vlm_dp.cost.terms import TERMS, CostInputs

K, H = 2, 3

# Two candidates: one AT the workpiece (engages the near-side gates), one en route (engages the
# far-side gates). Terms are asserted on max|value| over K, so one candidate per branch suffices.
_NEAR = [[0.50, 0.00, 0.30], [0.51, 0.01, 0.29], [0.52, 0.02, 0.28]]
_FAR = [[0.80, 0.30, 0.34], [0.79, 0.29, 0.33], [0.78, 0.28, 0.32]]
EE = torch.tensor([_NEAR, _FAR])                                          # [K,H,3]

# Deliberately off-axis: an axis-aligned pose would zero `orientation`/`yaw` for the wrong reason.
_Q = torch.tensor([0.13, 0.96, 0.20, 0.15])
EE_QUAT = (_Q / _Q.norm()).view(1, 1, 4).expand(K, H, 4).contiguous()     # [K,H,4] wxyz

_JOINTS = torch.tensor([[0.00, -0.30, 0.00, -2.00, 0.00, 1.90, 0.79],     # varies over H -> smooth > 0
                        [0.02, -0.28, 0.01, -1.98, 0.01, 1.92, 0.80],
                        [0.04, -0.26, 0.02, -1.96, 0.02, 1.94, 0.81]])
_GRIP = torch.tensor([0.50, 0.60, 0.40])                                  # varies -> gripper_smooth > 0
ACTIONS = torch.cat([_JOINTS, _GRIP[:, None]], -1).view(1, H, 8).expand(K, H, 8).contiguous()

JOINT_POS = _JOINTS[0] + 0.10                                             # != actions -> joint_delta > 0
PLAN_REF = _JOINTS + 0.05                                                 # [H,7] -> consistency > 0

_EEF = np.array([0.50, 0.00, 0.30], np.float32)        # measured TCP == near candidate, step 0
_PEAR = np.array([0.50, 0.00, 0.29], np.float32)       # workpiece / payload
_APPLE = np.array([0.53, 0.03, 0.29], np.float32)      # obstacle inside the keepout
_SCALE = np.array([0.52, 0.02, 0.20], np.float32)      # destination, under the near candidate
_CABBAGE = np.array([0.20, -0.40, 0.29], np.float32)   # already placed
_LIFT_POINT = np.array([0.50, 0.00, 0.45], np.float32)
Z_TABLE = 0.31                                         # above the fingertips, so `floor` engages

EXTENTS = {"pear": (0.04, 0.04, 0.05), "apple": (0.04, 0.04, 0.04),
           "scale": (0.15, 0.15, 0.02), "cabbage": (0.04, 0.04, 0.04)}
_AXES = {"pear": (1.0, 0.0, 0.0)}                      # a measured narrow axis -> grasp_axis has theta
OBJECTS = {n: {"pos": torch.as_tensor(p), "extents": EXTENTS[n], "axis": _AXES.get(n),
               "grasp_extent": None, "grasp_region": None}
           for n, p in (("pear", _PEAR), ("apple", _APPLE), ("scale", _SCALE), ("cabbage", _CABBAGE))}

_TALL_SCALE = np.array([0.52, 0.02, 0.36], np.float32)   # a tall fixture, top above the carried payload

# Local grasp: a 10mm graspable feature on an 80mm body (the teapot handle), the only regime in
# which grasp_approach_corridor can fire -- see the note on _CONTACT.
_BIG_BODY = (0.05, 0.08, 0.06)
_KNOB = np.array([0.50, 0.00, 0.33], np.float32)         # the feature, above the body centre
_LOCAL_GRASP_OBJECTS = {**OBJECTS, "pear": {**OBJECTS["pear"], "extents": _BIG_BODY,
                                            "grasp_extent": 0.010}}

KEYPOINTS = np.array([[0.50, 0.00, 0.29], [0.52, 0.02, 0.27]], np.float32)


def _subgoal(ee, kp):
    """Relational sub-goal [K,H]: bring keypoint 0 onto keypoint 1."""
    return torch.linalg.vector_norm(kp[0] - kp[1], dim=-1)


def _path(ee, kp):
    """Running path constraint [K,H]: stay above 0.40 (violated by the fixture, so it scores)."""
    return 0.40 - ee[..., 2]


GEOM = types.SimpleNamespace(**DEFAULT_GEOM)


def _ctx(**over):
    """Base replan context with no substage. Each scenario overrides only its own fields."""
    base = {"objects": OBJECTS, "target": _PEAR, "eef_pos": _EEF,
            "grasp_obj": None, "payload": None, "place_target": None,
            "contact": "pinch", "orient": "down", "place_mode": "surface",
            "gripper_intent": "close", "joint_pos": JOINT_POS, "z_table": Z_TABLE,
            "plan_ref": PLAN_REF, "placed": frozenset(), "subtasks": {}}
    base.update(over)
    return base


# ------------------------------------------------------------------ the substage -> active-set spec
# not_hold is ungated on purpose: smooth, joint_delta and consistency are always-on penalties on
# MOTION with nothing balancing them, so the do-nothing plan scored cheaper than the demonstration
# at every stage. not_hold is that counterweight, so it belongs in the same always-on set.
_ALWAYS = {"smooth", "gripper_smooth", "joint_delta", "consistency", "collision", "clear", "floor",
           "not_hold"}
_PINCH = {"straddle", "grasp_axis", "center_region", "aperture_region"}       # pinch certification
# Contact stack: approach invariants, gated on the grasp frame rather than pinch-vs-press, so a lid
# approach still may not dive or overshoot its standoff. grasp_approach_corridor is NOT here -- with
# no grasp_extent its radius always exceeds body_r, making the term unfirable outside the local-grasp
# regime that grasp_corridor_cut pins.
_CONTACT = {"grasp_descend_rate", "grasp_standoff"}
_GRASP = {"tip_z", "yaw", "grasp_region"} | _PINCH | _CONTACT
_REACH = {"reach", "terminal_reach"}
_LIFT = {"lift_xy", "lift_z", "lift_terminal", "lift_reach"}
_PLACE = {"place_reach", "place_xy", "place_z", "place_carry_height", "place_descent",
          "place_terminal", "carry_liftoff", "carry_altitude", "carry_clear"}


# Measured articulation metadata, shaped like grounding/mg_gt's drawer and lid: a slide along -y whose
# handle bar runs along x, and a press straight down. Numbers are the MimicGen cabinet/lid constants.
def _pull_spec(**over):
    spec = {"axis": (0.0, -1.0, 0.0), "point": (0.50, 0.00, 0.30), "goal": (0.50, -0.136, 0.30),
            "bar": (1.0, 0.0, 0.0), "span": 0.05, "tol": 0.031, "stroke": 0.136}
    spec.update(over)
    return spec


def _press_spec(**over):
    spec = {"axis": (0.0, 0.0, -1.0), "point": (0.50, 0.00, 0.30), "depth": 0.058, "tol": 0.057,
            "reach": 0.024}
    spec.update(over)
    return spec


# Measured insertion metadata, shaped like grounding/mg_gt's threading ring and square peg: a seat
# under the workpiece, entered along +z, whose cone narrows from 50mm at 50mm up to a 10mm fit.
def _insert_spec(**over):
    spec = {"seat": (0.50, 0.00, 0.20), "axis": (0.0, 0.0, 1.0), "r_seat": 0.010,
            "r_mouth": 0.050, "height": 0.050, "capture": 0.15}
    spec.update(over)
    return spec


@dataclasses.dataclass
class _Scenario:
    """One substage archetype: its context, and the terms that must be non-zero on it."""
    name: str
    why: str
    context: dict
    active: set
    ee: torch.Tensor | None = None      # override the shared candidates when the archetype IS a pose
    extents: dict | None = None         # override when the archetype needs a different body size


SCENARIOS = [
    _Scenario(
        "grasp_pinch", "the ordinary pick: everything that certifies a pinch is on",
        _ctx(grasp_obj="pear", target=_PEAR),
        _ALWAYS | _REACH | _GRASP | {"orientation", "close_gripper", "grasp_commit"}),
    _Scenario(
        "grasp_press", "a lid/thin part: pinch certification stands down, reach-and-close stays",
        _ctx(grasp_obj="pear", target=_PEAR, contact="press"),
        _ALWAYS | _REACH | (_GRASP - _PINCH) | {"orientation", "close_gripper", "grasp_commit"}),
    _Scenario(
        "release_open", "reopen recovery owns the gripper channel; the closing terms stand down",
        _ctx(grasp_obj="pear", target=_PEAR, gripper_intent="open"),
        _ALWAYS | _REACH | _GRASP | {"orientation", "release_gripper"}),
    _Scenario(
        "carry_lift", "payload held, no destination yet: the lift column, not the place seat",
        _ctx(payload="pear", target=_LIFT_POINT),
        _ALWAYS | _LIFT | {"orientation", "carry_hold", "carry_clear"}),
    _Scenario(
        "place_surface", "payload over a surface: set-down is on",
        _ctx(payload="pear", place_target="scale", target=_SCALE),
        _ALWAYS | _PLACE | {"orientation", "carry_hold", "place_setdown"}),
    _Scenario(
        "place_container", "placing INSIDE: a container's top is its rim, so set-down stands down",
        _ctx(payload="pear", place_target="scale", target=_SCALE, place_mode="container"),
        _ALWAYS | _PLACE | {"orientation", "carry_hold"}),
    _Scenario(
        "pour_rotation", "the VLM commands the rotation: the downward tool-axis prior stands down",
        _ctx(payload="pear", place_target="scale", target=_SCALE, orient="free",
             constraint=_subgoal, path_fns=(_path,), keypoints=KEYPOINTS,
             held_idx=(0,), held_offset=np.array([[0.0, 0.0, -0.05]], np.float32)),
        _ALWAYS | _PLACE | {"carry_hold", "place_setdown", "rekep_subgoal", "rekep_path"}),
    _Scenario(
        "place_side_approach", "a TALL destination: the payload is outside its footprint and below its "
        "top face -- the only geometry in which the arm can strike the fixture's side. Set-down stands "
        "down here (it gates on being at the seat), which is exactly why nothing else guarded this.",
        _ctx(payload="pear", place_target="scale", target=_SCALE,
             objects={**OBJECTS, "scale": {**OBJECTS["scale"], "pos": torch.as_tensor(_TALL_SCALE)}}),
        _ALWAYS | _PLACE | {"orientation", "carry_hold", "place_approach_above"}),
    _Scenario(
        "place_guarded_press", "the stall-release press approach (ctx overshoot_on present): the "
        "guarded move caps the carried payload's speed inside the descend gate, so contact is a "
        "press and not the measured ballistic bounce.",
        _ctx(payload="pear", place_target="scale", target=_SCALE, overshoot_on=True),
        _ALWAYS | _PLACE | {"orientation", "carry_hold", "place_approach_above",
                            "place_approach_rate", "place_setdown"}),
    _Scenario(
        "place_released", "release completed at the seat (the latch): every attractor stands down "
        "-- holding the hand at the seated payload is how it gets tipped off -- carry_hold holds "
        "open, and place_retreat takes the hand vertically out.",
        _ctx(payload="pear", place_target="scale", target=_SCALE, overshoot_on=False,
             place_released=True, released=True),
        _ALWAYS | {"orientation", "carry_hold", "place_retreat", "release_retreat",
                   "release_rise_first"}),
    _Scenario(
        "departure", "the stage AFTER a set-down: the pear is seated and in `placed`, the hand is "
        "empty and the objective has already become the next grasp. No archetype covered this, which "
        "is where the failure lived -- measured across three arms, 554-765 mm of lateral travel with "
        "a PEAK rise of 0.0 mm and the pear knocked back off in every episode that seated it. Both "
        "retreat guards must still be live here, long after `released` went false.",
        _ctx(grasp_obj="apple", target=_APPLE, placed=frozenset({"pear"})),
        # grasp_axis needs a measured narrow axis and only the pear fixture has one; collision has no
        # obstacle left once the apple is the grasp target and the pear is `placed`. Both are inert
        # here for fixture reasons, not gate reasons -- grasp_pinch already pins them.
        (_ALWAYS | _REACH | _GRASP | {"orientation", "close_gripper", "grasp_commit",
                                      "release_retreat", "release_rise_first"})
        - {"grasp_axis", "collision"}),
    _Scenario(
        "grasp_hold_grace", "the churn window: the grasp stage certified a hold, so the bridge opens "
        "the regrasp grace and giving the grip up costs something. Its own archetype because "
        "hold_grace is bridge evidence, absent from every other context -- which is exactly why the "
        "term is inert on the shipped configs.",
        _ctx(grasp_obj="pear", target=_PEAR, hold_grace=1.0),
        _ALWAYS | _REACH | _GRASP | {"orientation", "close_gripper", "grasp_commit",
                                     "regrasp_penalty"}),
    _Scenario(
        "grasp_corridor_cut", "local grasp (a 10mm handle on an 80mm body, the teapot case): the palm "
        "is inside the BODY footprint and below the grasp point but outside the pinch corridor -- the "
        "hand clipping the object on the way in. Only reachable here: with no grasp_extent the "
        "corridor radius exceeds the body radius, so the term is structurally dead on a plain pick. "
        "straddle stands down: the open fingers clear a 10mm feature, which is the point of a local "
        "grasp.",
        _ctx(grasp_obj="pear", target=_KNOB, objects=_LOCAL_GRASP_OBJECTS),
        _ALWAYS | _REACH | _CONTACT | {"grasp_approach_corridor", "tip_z", "yaw", "grasp_axis",
                                       "grasp_region", "center_region", "aperture_region",
                                       "orientation", "close_gripper", "grasp_commit"},
        ee=torch.tensor([[[0.56, 0.0, 0.36], [0.56, 0.0, 0.33], [0.56, 0.0, 0.30]],
                         [[0.50, 0.0, 0.40], [0.50, 0.0, 0.38], [0.50, 0.0, 0.36]]]),
        extents={**EXTENTS, "pear": _BIG_BODY}),
    _Scenario(
        "articulated_pull", "a drawer pull with the open hand: the objective is the SLIDE, which no FK "
        "rollout can move, so the hand's own trajectory carries it. Its own archetype because the pull "
        "metadata is measured grounding absent from every free-body stage -- which is exactly why the "
        "term is inert on the shipped configs. No pinch here: the fingers straddle the bar.",
        _ctx(gripper_intent="open", contact="press", pull=_pull_spec()),
        _ALWAYS | _REACH | {"orientation", "release_gripper", "hook_pull"}),
    _Scenario(
        "articulated_press", "a lid or button press with the open hand: the goal sits beyond the "
        "surface (the force surrogate) and the approach is confined to a cone about the direction the "
        "contact point travels.",
        _ctx(gripper_intent="open", contact="press", press=_press_spec()),
        _ALWAYS | _REACH | {"orientation", "release_gripper", "press_axis"}),
    _Scenario(
        "insertion", "placing INTO a receptacle whose fit is tight: the admissible set is a cone, not "
        "the place attractor's ball. Its own archetype because the corridor is measured grounding "
        "absent from every set-down stage -- which is exactly why both terms are inert on the shipped "
        "configs.",
        _ctx(payload="pear", place_target="scale", target=_SCALE, place_mode="container",
             insert=_insert_spec()),
        _ALWAYS | _PLACE | {"orientation", "carry_hold", "insert_funnel", "descend_gate"}),
]


def _evaluate(scenario):
    """Score every registered term at weight 1 on one substage -> {name: max |value| over candidates}."""
    ee = EE if scenario.ee is None else scenario.ee
    extents = EXTENTS if scenario.extents is None else scenario.extents
    inputs = CostInputs(ACTIONS, ee, EE_QUAT, scenario.context, extents, GEOM)
    return {name: float(fn(inputs).abs().max()) for name, fn in TERMS.items()}


def _check(scenario):
    """Active terms must score, every other registered term must be exactly zero."""
    scores = _evaluate(scenario)
    unknown = scenario.active - set(TERMS)
    assert not unknown, f"{scenario.name}: spec names unregistered terms {sorted(unknown)}"
    silent = sorted(n for n in scenario.active if scores[n] == 0.0)
    assert not silent, (f"{scenario.name} ({scenario.why}): terms expected to apply scored zero "
                        f"{silent} -- either the gate is wrong or the fixture no longer exercises them")
    leaked = sorted(n for n, v in scores.items() if n not in scenario.active and v != 0.0)
    assert not leaked, (f"{scenario.name} ({scenario.why}): terms that should have stood down are "
                        f"still scoring {leaked}")

# Keyed by name, not position: a scenario inserted mid-list silently re-points every test after it,
# so test_grasp_corridor_cut once checked whichever archetype happened to land on index 10.
SCENARIO = {s.name: s for s in SCENARIOS}


def test_grasp_pinch():
    _check(SCENARIO["grasp_pinch"])


def test_grasp_press():
    _check(SCENARIO["grasp_press"])


def test_release_open():
    _check(SCENARIO["release_open"])


def test_carry_lift():
    _check(SCENARIO["carry_lift"])


def test_place_surface():
    _check(SCENARIO["place_surface"])


def test_place_container():
    _check(SCENARIO["place_container"])


def test_pour_rotation():
    _check(SCENARIO["pour_rotation"])


def test_place_side_approach():
    _check(SCENARIO["place_side_approach"])


def test_grasp_corridor_cut():
    _check(SCENARIO["grasp_corridor_cut"])


def test_grasp_hold_grace():
    _check(SCENARIO["grasp_hold_grace"])


def test_departure():
    _check(SCENARIO["departure"])


def test_articulated_pull():
    _check(SCENARIO["articulated_pull"])


def test_articulated_press():
    _check(SCENARIO["articulated_press"])


def test_insertion():
    _check(SCENARIO["insertion"])


def test_every_term_is_reachable():
    """No registered term may be unreachable, each must apply to at least one archetype.

    This is the guard that makes the spec self-maintaining: a new term forces its author to say which
    substage it belongs to, instead of it silently defaulting to always-on.
    """
    covered = set().union(*(s.active for s in SCENARIOS))
    missing = sorted(set(TERMS) - covered)
    assert not missing, (f"terms no archetype activates {missing} -- add them to a scenario's active "
                         f"set, or add the archetype they belong to")


def test_press_only_relaxes_pinch_certification():
    """press vs pinch differ in exactly the four certification terms, nothing else moves."""
    pinch, press = _evaluate(SCENARIO["grasp_pinch"]), _evaluate(SCENARIO["grasp_press"])
    differing = {n for n in TERMS if pinch[n] != press[n]}
    assert differing == _PINCH, f"contact=press changed {sorted(differing)}, expected {sorted(_PINCH)}"


def test_orientation_stands_down_only_when_the_vlm_rotates():
    """A downward tool-axis prior left on during a pour fights the commanded tilt."""
    down = _evaluate(_Scenario("d", "", _ctx(payload="pear", place_target="scale"), set()))
    free = _evaluate(_Scenario("f", "", _ctx(payload="pear", place_target="scale", orient="free"), set()))
    assert down["orientation"] > 0.0, "orient='down' must keep the downward tool-axis prior on"
    assert free["orientation"] == 0.0, "orient='free' must stand the downward tool-axis prior down"


def test_declared_orientation_axis_has_stage_local_authority():
    """A plan may strengthen only its articulated tilt without changing the global config."""
    base_ctx = _ctx(payload="pear", place_target="scale", orient="axis",
                    approach_axis=np.array([1.0, 0.0, 0.0], np.float32),
                    orientation_scale=1.0)
    strong_ctx = dict(base_ctx, orientation_scale=6.0)
    base = _evaluate(_Scenario("axis", "", base_ctx, set()))["orientation"]
    strong = _evaluate(_Scenario("axis-strong", "", strong_ctx, set()))["orientation"]
    assert base > 0.0
    assert abs(strong - 6.0 * base) < 1e-5


# ------------------------------------------------------------------- the pinch/press contact criterion
# Narrow horizontal half-extents from each object's own perceived cloud, on clean-mask looks. The
# criterion compares them against the gripper's 40 mm aperture half-width.
_MEASURED_GRIP = {"pear": 0.030, "apple": 0.027, "mango": 0.028, "egg": 0.016, "can": 0.017,
                  "teacup": 0.027, "teapot": 0.090, "cover": 0.125, "capsule": 0.089}
_EXPECT_CONTACT = {"pear": "pinch", "apple": "pinch", "mango": "pinch", "egg": "pinch", "can": "pinch",
                   "teacup": "pinch", "teapot": "press", "cover": "press", "capsule": "press"}

# Whole-object AABBs are the wrong input: a pear's widest slice is 46 mm, past the aperture, yet the
# hand grasps it at the neck. The criterion asks whether the gripper can take the object as a unit;
# an AABB answers whether its widest slice fits. Pinned so a whole-body rung trips this, not a run.
_USD_GRIP_MARGINAL = {"pear": 0.046, "capsule": 0.050}


def _grounding():
    from vlm_dp.grounding.rekep import RekepGrounding
    return RekepGrounding(task_key="_test", place_obj="_test")


def test_contact_criterion_on_measured_geometry():
    """pinch vs press follows the gripper actually fitting, on real measured extents.

    The regression: the old plan-structural criterion called the pot's `cover` an articulation because
    the PLAN looked like one (grasp it, then displace it, no external destination). Geometry says a
    250 mm lid cannot be taken by an 80 mm hand, whatever the plan looks like.
    """
    grounding = _grounding()
    extents = {n: (g, g, g) for n, g in _MEASURED_GRIP.items()}
    wrong = {n: grounding._contact_for(n, extents)[0] for n in _MEASURED_GRIP
             if grounding._contact_for(n, extents)[0] != _EXPECT_CONTACT[n]}
    assert not wrong, f"contact mode disagrees with the measured geometry: {wrong}"


def test_contact_criterion_has_margin_on_clean_clouds():
    """On a clean cloud measurement the split is wide, not a knife edge."""
    aperture = _grounding().open_half
    pinchable = max(g for n, g in _MEASURED_GRIP.items() if _EXPECT_CONTACT[n] == "pinch")
    pressed = min(g for n, g in _MEASURED_GRIP.items() if _EXPECT_CONTACT[n] == "press")
    assert pinchable < aperture < pressed, (
        f"the {aperture * 1e3:.0f}mm aperture must separate the two classes; measured "
        f"largest pinchable {pinchable * 1e3:.0f}mm, smallest pressed {pressed * 1e3:.0f}mm")
    assert pressed / pinchable > 2.0, (
        f"separation is only {pressed / pinchable:.1f}x -- too tight for cloud noise to be safe")


def test_contact_criterion_marginal_band_is_recorded():
    """Whole-body AABBs collapse the margin, with pear 46 mm and capsule 50 mm straddling a 40 mm aperture.

    Both land on the press side, so a whole-body rung would call a pear a press. This is a real limit
    of the criterion, not a bug to paper over with a fudge factor: a 1.15x vs 1.25x split cannot be
    separated by one. It is pinned here so the ambiguity is visible before someone runs that rung.
    """
    grounding = _grounding()
    extents = {n: (g, g, g) for n, g in _USD_GRIP_MARGINAL.items()}
    verdicts = {n: grounding._contact_for(n, extents)[0] for n in _USD_GRIP_MARGINAL}
    assert verdicts == {"pear": "press", "capsule": "press"}, (
        f"the recorded marginal band moved: {verdicts}. If the pear now reads pinch the criterion "
        f"changed -- re-derive the band rather than editing this expectation")
    assert min(_USD_GRIP_MARGINAL.values()) / grounding.open_half < 1.3, (
        "the marginal cases must stay within 30% of the aperture; further out they are no longer marginal")


# --------------------------------------------------------- the shared at-the-grasp-pose tolerance
def test_grasp_slack_is_the_slack_left_in_the_aperture():
    """The tolerance is what the fingers have room for, and it shrinks as the object grows."""
    from vlm_dp.cost.terms import grasp_slack

    # Floor off, so the geometry is visible: slack is strictly decreasing in object size.
    g = types.SimpleNamespace(open_half=0.04, aperture_margin=0.006, close_xy_floor=1e-4,
                              center_scale=0.40)
    slacks = [grasp_slack(g, r) for r in (0.005, 0.010, 0.020, 0.031)]
    assert all(a > b for a, b in zip(slacks, slacks[1:])), \
        f"slack must DECREASE as the object grows, got {[round(s, 4) for s in slacks]}"
    assert abs(grasp_slack(g, 0.010) - 0.024) < 1e-6, "a 10mm lip leaves 40-10-6 = 24mm of aperture"

    # Floor on: an object that fills the aperture clamps rather than going <= 0 (monotone, not strict).
    floored = types.SimpleNamespace(**{**vars(g), "close_xy_floor": 0.008})
    assert grasp_slack(floored, 0.040) == 0.008, "an object filling the aperture must clamp to the floor"
    assert grasp_slack(floored, 0.060) == 0.008, "an object WIDER than the aperture must not go negative"


def _config_geom(name):
    """Geometry from a REAL config on disk. A hand-made namespace here once hid a 4x regression: the
    test used center_scale=0.12 while every shipped config uses 0.40, so it passed on a fiction."""
    import os
    import yaml

    from vlm_dp import config_paths

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(config_paths.resolve(name, root), encoding="utf-8") as f:
        return types.SimpleNamespace(**yaml.safe_load(f)["cost"]["geometry"])


def test_grasp_slack_against_the_shipped_configs():
    """Per-task effect of the aperture dead-zone against the shipped configs.

    The capsule lid is the motivating case: 4mm to 24mm, because the proportional form gives the
    thinnest object the tightest tolerance. Weight tightens 12.4mm to 3.0mm, so it is pinned here
    rather than assumed and needs a rollout A/B before the aperture form counts as validated there.
    """
    from vlm_dp.cost.terms import grasp_slack

    cases = {   # config, measured radius (m), (legacy mm, aperture mm)
        "base.yaml": (0.031, 12.4, 3.0),                    # pear, cloud wide half-extent
        "capsule_gt_diag.yaml": (0.010, 4.0, 24.0),         # lid lip
        "vlm_driven_localgrasp.yaml": (0.090, 36.0, 8.0),   # teapot, cloud
    }
    for cfg, (radius, legacy_mm, aperture_mm) in cases.items():
        geom = _config_geom(cfg)
        legacy = types.SimpleNamespace(**{**vars(geom), "grasp_dead_zone": "proportional"})
        assert abs(grasp_slack(legacy, radius) * 1e3 - legacy_mm) < 0.1, f"{cfg}: legacy value moved"
        assert abs(grasp_slack(geom, radius) * 1e3 - aperture_mm) < 0.1, f"{cfg}: aperture value moved"

    assert cases["capsule_gt_diag.yaml"][2] > 5 * cases["capsule_gt_diag.yaml"][1], \
        "the thin lid must gain a much larger tolerance -- that is the defect being fixed"
    assert cases["base.yaml"][2] < cases["base.yaml"][1], \
        "weight TIGHTENS; this is a real behaviour change on the one working task, not a no-op"


# ------------------------------------------------- commitment-form release (carry_hold, opt-in obs-1 fix)
def _carry_state(payload_z, grip, release_commit):
    """One carry state for carry_hold: payload at height payload_z, gripper channel grip (1 closed, 0
    open), seat top at z=0.80. eef and payload share the seat xy, so distance-to-seat is the z gap."""
    seat_xy, seat_top = (0.50, 0.10), 0.80
    ee = torch.tensor([[seat_xy[0], seat_xy[1], payload_z]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    actions = torch.zeros(1, 4, 8)
    actions[..., 7] = grip
    ctx = {"payload": "obj", "place_target": "scale",
           "objects": {"obj": {"pos": torch.tensor([seat_xy[0], seat_xy[1], payload_z]),
                               "extents": (0.03, 0.03, 0.03)},
                       "scale": {"pos": torch.tensor([seat_xy[0], seat_xy[1], seat_top]),
                                 "extents": (0.15, 0.15, 0.0)}},
           "eef_pos": np.array([seat_xy[0], seat_xy[1], payload_z], np.float32)}
    geom = types.SimpleNamespace(release_xy=0.06, release_z=0.02, close_xy_floor=0.008,
                                 place_release_clearance=0.0, release_commit=release_commit)
    ext = {"obj": (0.03, 0.03, 0.03), "scale": (0.15, 0.15, 0.0)}
    return float(TERMS["carry_hold"](CostInputs(actions, ee, None, ctx, ext, geom)))


# Seat top 0.80 + payload half-height 0.03 -> dest 0.83. At-seat vs 30 cm above.
_AT_SEAT, _FAR = 0.83, 1.10


def test_release_commit_commits_smoothly():
    """The obs-1 fix: opening is rewarded at the seat and penalised far, and the open-vs-closed
    preference crosses the approach exactly once, so the MBD average never dithers at a hard boundary."""
    at_open, at_closed = _carry_state(_AT_SEAT, 0.0, True), _carry_state(_AT_SEAT, 1.0, True)
    far_open, far_closed = _carry_state(_FAR, 0.0, True), _carry_state(_FAR, 1.0, True)
    assert at_open < at_closed, f"at the seat, opening must win: open {at_open:.3f} vs closed {at_closed:.3f}"
    assert far_closed < far_open, f"far, holding closed must win: closed {far_closed:.3f} vs open {far_open:.3f}"
    assert at_open < far_open, "opening must be more rewarded at the seat than far away"
    prefers_open = [_carry_state(0.80 + 0.01 * i, 0.0, True) < _carry_state(0.80 + 0.01 * i, 1.0, True)
                    for i in range(40)]
    crossings = sum(1 for a, b in zip(prefers_open, prefers_open[1:]) if a != b)
    assert crossings == 1, f"open-preference must cross once (a smooth commit), crossed {crossings} times"


def test_release_commit_default_off_is_the_hard_gate():
    """release_commit=False must reproduce the existing hard AND-threshold release exactly."""
    dest_z = 0.80 + 0.0 + 0.03                      # seat top + place half-height + payload half-height
    for z, g in ((_AT_SEAT, 0.0), (_AT_SEAT, 1.0), (_FAR, 0.0), (_FAR, 1.0)):
        got = _carry_state(z, g, False)
        release = 1.0 if (z - dest_z) < 0.02 else 0.0     # xy is 0 here, so only the z term decides
        assert abs(got - (g - (1.0 - release)) ** 2) < 1e-6, f"parity broken at z={z} grip={g}: {got:.4f}"


# ------------------------------------------------------- measured-gate grasp_commit (opt-in variant)
def _grasp_commit_state(cand_z, meas_z, grip, measured_gate):
    """grasp_commit at a grasp center z=0.30: candidate tip at cand_z, measured TCP at meas_z (xy aligned).
    measured_gate on modulates the candidate reward by measured proximity, so a candidate that plans to
    be closed-at-the-grasp scores no reward until the measured TCP is also in the neighbourhood."""
    ee = torch.tensor([[0.50, 0.0, cand_z]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    actions = torch.zeros(1, 4, 8)
    actions[..., 7] = grip
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(1, 4, 4).contiguous()
    ctx = {"grasp_obj": "pear", "payload": None, "target": np.array([0.50, 0.0, 0.30], np.float32),
           "objects": {"pear": {"pos": torch.tensor([0.50, 0.0, 0.30]), "extents": (0.03, 0.03, 0.03),
                                "grasp_extent": None}},
           "eef_pos": np.array([0.50, 0.0, meas_z], np.float32), "gripper_intent": "close"}
    geom = types.SimpleNamespace(open_half=0.04, aperture_margin=0.006, close_xy_floor=0.008,
                                 center_scale=0.4, tcp_to_tip=0.0, commit_measured_gate=measured_gate,
                                 commit_measured_band=3.0)
    return float(TERMS["grasp_commit"](CostInputs(actions, ee, q, ctx, {"pear": (0.03, 0.03, 0.03)}, geom)))


def test_grasp_commit_measured_gate_blocks_premature_close():
    """The premature-close thrash fix: with the measured gate, a candidate planning closed-at-the-grasp earns no
    reward while the arm is far, but the full reward once the arm arrives. Default-off is unchanged."""
    at, far = 0.30, 0.45
    # Default (pure candidate): rewards closing even with the arm far -> premature close -> thrash.
    assert _grasp_commit_state(at, far, 1.0, False) < -0.5, "default grasp_commit rewards closing on the candidate"
    # Measured gate on: far gives no reward, near gives full reward.
    assert abs(_grasp_commit_state(at, far, 1.0, True)) < 0.05, "measured gate must kill the reward when the arm is far"
    assert _grasp_commit_state(at, at, 1.0, True) < -0.5, "measured gate must keep the reward once the arm arrives"
    # Steerable within the neighbourhood: with the arm near, opening still scores above closing.
    assert _grasp_commit_state(at, at, 0.0, True) > _grasp_commit_state(at, at, 1.0, True)


# ------------------------------- anisotropic commitment release (carry_hold, opt-in place-commit fix)
def _carry_aniso(xy_off, payload_z, grip, *, aniso=True, measured_gate=False, meas_off=None):
    """carry_hold with the per-axis commitment. The candidate holds the payload xy_off from the seat xy
    at height payload_z; seat top 0.80 + payload half-height 0.03 -> dest z 0.83. meas_off displaces the
    MEASURED payload only (the candidate still plans to be at the seat), which is what the measured gate
    is meant to catch."""
    seat_xy, seat_top = (0.50, 0.10), 0.80
    px = seat_xy[0] + xy_off
    ee = torch.tensor([[px, seat_xy[1], payload_z]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    actions = torch.zeros(1, 4, 8)
    actions[..., 7] = grip
    # eef_pos == measured payload: the payload is in the hand, so carried == the candidate TCP.
    mx, mz = (px, payload_z) if meas_off is None else (seat_xy[0] + meas_off[0], meas_off[1])
    ctx = {"payload": "obj", "place_target": "scale",
           "objects": {"obj": {"pos": torch.tensor([mx, seat_xy[1], mz]), "extents": (0.03, 0.03, 0.03)},
                       "scale": {"pos": torch.tensor([seat_xy[0], seat_xy[1], seat_top]),
                                 "extents": (0.15, 0.15, 0.0)}},
           "eef_pos": np.array([mx, seat_xy[1], mz], np.float32)}
    geom = types.SimpleNamespace(release_xy=0.06, release_z=0.02, close_xy_floor=0.008,
                                 place_release_clearance=0.0, release_commit=not aniso,
                                 release_commit_aniso=aniso, commit_measured_gate=measured_gate,
                                 release_commit_band=3.0)
    ext = {"obj": (0.03, 0.03, 0.03), "scale": (0.15, 0.15, 0.0)}
    return float(TERMS["carry_hold"](CostInputs(actions, ee, None, ctx, ext, geom)))


def _prefers_open(xy_off, z, **kw):
    return _carry_aniso(xy_off, z, 0.0, **kw) < _carry_aniso(xy_off, z, 1.0, **kw)


def test_release_aniso_commits_smoothly():
    """Same commitment property as the isotropic form: opening wins at the seat, holding wins far, and
    the preference crosses the approach exactly once, so a candidate average cannot sit mid-open."""
    assert _prefers_open(0.0, _AT_SEAT), "at the seat, opening must win"
    assert not _prefers_open(0.0, _FAR), "far above the seat, holding closed must win"
    pref = [_prefers_open(0.0, 0.80 + 0.01 * i) for i in range(40)]
    crossings = sum(1 for a, b in zip(pref, pref[1:]) if a != b)
    assert crossings == 1, f"open-preference must cross once (a smooth commit), crossed {crossings} times"


def test_release_aniso_is_tighter_in_z_than_in_xy():
    """The point of the per-axis form: the axes are not interchangeable. The SAME 4 cm displacement is
    inside the xy tolerance (open) but 2x the z tolerance (hold), because opening 4 cm up drops the
    payload. The isotropic form cannot express this -- it opens in both, which is the drop risk."""
    assert _prefers_open(0.04, _AT_SEAT), "4 cm sideways is within the seat footprint: opening must win"
    assert not _prefers_open(0.0, _AT_SEAT + 0.04), \
        "4 cm ABOVE the seat must not open: that is a drop, and the pear rolls off"
    # Contrast: the isotropic commit treats both as one 4 cm distance and opens either way.
    assert _prefers_open(0.0, _AT_SEAT + 0.04, aniso=False), \
        "isotropic release_commit is expected to open high -- the behaviour this form replaces"


def test_release_aniso_boundary_sits_on_the_tolerance():
    """A smooth replacement for a hard gate must keep the gate's DECISION BOUNDARY and soften only its
    sharpness. Ordering and single-crossing (above) do not pin WHERE the crossing is, and that gap let a
    shape through whose crossing sat at 1.83x tolerance: it released the payload up to 3.7 cm above the
    seat and dropped it. Pin the flip on each axis independently."""
    assert _prefers_open(0.0, _AT_SEAT + 0.018), "just inside release_z (0.02) must open"
    assert not _prefers_open(0.0, _AT_SEAT + 0.022), \
        "just outside release_z must NOT open -- that is a drop, not a set-down"
    assert _prefers_open(0.055, _AT_SEAT), "just inside release_xy (0.06) must open"
    assert not _prefers_open(0.065, _AT_SEAT), "just outside release_xy must NOT open"


def test_release_aniso_decides_on_the_measured_payload_not_the_candidate():
    """The release decision must not depend on where a CANDIDATE plans to be.

    A candidate-evaluated release couples the gripper decision to position: near the seat it demands an
    open hand, so a closed-hand candidate is penalised and the sampler can escape either by opening or
    by retreating. At this term's weight, retreating wins and the payload hovers above the destination
    for the rest of the episode instead of setting down. Deciding on the measured payload makes the
    term one scalar per replan, so it cannot push the arm anywhere -- which this pins: with the payload
    measurably 20 cm away, no candidate pose may command the hand open."""
    far = (0.0, _AT_SEAT + 0.20)
    assert not _prefers_open(0.0, _AT_SEAT, meas_off=far), \
        "a candidate planning the seat must not open while the payload is measurably 20 cm away"
    # ...and with the payload measurably AT the seat, the candidate's own height cannot veto the open.
    at = (0.0, _AT_SEAT)
    assert _prefers_open(0.0, _FAR, meas_off=at), \
        "the measured payload is seated, so the decision must not depend on the candidate's height"


def _best_gripper(xy, z, **kw):
    """Gripper command the term actually prefers: argmin of the cost over the channel."""
    grid = [i / 20.0 for i in range(21)]
    return min(grid, key=lambda g: _carry_aniso(xy, z, g, **kw))


def test_release_aniso_is_decisive_not_half_open():
    """The term must PIN the gripper, not merely lean.

    A reward linear in the gripper channel leaves the cost-weighted mean of 4096 candidates floating
    between open and closed; measured, that commanded a half-open hand on 89% of holding steps (58%
    under the quadratic gate) and the payload fell out. Tracking a target quadratically makes the
    preferred command saturate, so away from the boundary it is essentially fully open or fully shut.
    """
    assert _best_gripper(0.0, _AT_SEAT) <= 0.05, "at the seat the hand must be commanded fully open"
    assert _best_gripper(0.0, _FAR) >= 0.95, "far from the seat the hand must be commanded fully shut"
    # And the indecisive band must be narrow: one tolerance out is already essentially shut.
    assert _best_gripper(0.0, _AT_SEAT + 0.03) >= 0.90, \
        "1.5x the z tolerance must already be a firm hold, not a half-open hand"


def test_release_aniso_absent_leaves_the_hard_gate_untouched():
    """Adding the branch must not perturb the shipped default: with both commit flags absent, carry_hold
    is still the hard AND-threshold release, so parity13 and every existing config behave exactly as
    before. (_carry_state builds the geom without either flag.)"""
    dest_z = 0.80 + 0.03                            # seat top + payload half-height; xy is 0, so z decides
    for z, g in ((_AT_SEAT, 0.0), (_AT_SEAT, 1.0), (_FAR, 0.0), (_FAR, 1.0)):
        got = _carry_state(z, g, False)
        release = 1.0 if (z - dest_z) < 0.02 else 0.0
        assert abs(got - (g - (1.0 - release)) ** 2) < 1e-6, f"default path moved at z={z} grip={g}"


# ------------------------------------------- stall release (carry_hold, opt-in contact-evidence fix)
def _carry_stall(xy_off, payload_z, grip, contact):
    """carry_hold with release_on_stall: the release decision is xy-in-tolerance AND the bridge's
    seat_contact datum. payload_z is deliberately part of the sweep to pin that z NEVER decides."""
    seat_xy, seat_top = (0.50, 0.10), 0.80
    px = seat_xy[0] + xy_off
    ee = torch.tensor([[px, seat_xy[1], payload_z]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    actions = torch.zeros(1, 4, 8)
    actions[..., 7] = grip
    ctx = {"payload": "obj", "place_target": "scale", "seat_contact": contact,
           "objects": {"obj": {"pos": torch.tensor([px, seat_xy[1], payload_z]),
                               "extents": (0.03, 0.03, 0.03)},
                       "scale": {"pos": torch.tensor([seat_xy[0], seat_xy[1], seat_top]),
                                 "extents": (0.15, 0.15, 0.0)}},
           "eef_pos": np.array([px, seat_xy[1], payload_z], np.float32)}
    geom = types.SimpleNamespace(release_xy=0.06, release_z=0.02, close_xy_floor=0.008,
                                 place_release_clearance=0.0, release_on_stall=True,
                                 place_overshoot=0.08)
    ext = {"obj": (0.03, 0.03, 0.03), "scale": (0.15, 0.15, 0.0)}
    return float(TERMS["carry_hold"](CostInputs(actions, ee, None, ctx, ext, geom)))


def test_stall_release_needs_contact_not_altitude():
    """The premature-release fix: without contact evidence the hand holds at ANY altitude, including
    exactly at the seat (where the biased-z gate used to open 5-8cm early); with contact it opens."""
    for z in (_AT_SEAT, _FAR):
        assert _carry_stall(0.0, z, 1.0, 0.0) < _carry_stall(0.0, z, 0.0, 0.0), \
            f"no contact: holding must win at z={z}"
        assert _carry_stall(0.0, z, 0.0, 1.0) < _carry_stall(0.0, z, 1.0, 1.0), \
            f"contact: opening must win at z={z} (z must not gate the decision)"


def test_stall_release_still_requires_xy():
    """Contact alone is not enough: stalled far from the seat xy (resting on some other surface or
    blocked) must not release."""
    assert _carry_stall(0.30, _AT_SEAT, 1.0, 1.0) < _carry_stall(0.30, _AT_SEAT, 0.0, 1.0), \
        "contact but xy far: holding must win"


def test_stall_release_absent_leaves_aniso_untouched():
    """Without the flag, a seat_contact datum in the context must be ignored: the shipped aniso
    configs cannot change behaviour just because the bridge starts emitting the new key."""
    def aniso_with_contact(z, g, contact):
        seat_xy, seat_top = (0.50, 0.10), 0.80
        ee = torch.tensor([[seat_xy[0], seat_xy[1], z]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
        actions = torch.zeros(1, 4, 8)
        actions[..., 7] = g
        ctx = {"payload": "obj", "place_target": "scale", "seat_contact": contact,
               "objects": {"obj": {"pos": torch.tensor([seat_xy[0], seat_xy[1], z]),
                                   "extents": (0.03, 0.03, 0.03)},
                           "scale": {"pos": torch.tensor([seat_xy[0], seat_xy[1], seat_top]),
                                     "extents": (0.15, 0.15, 0.0)}},
               "eef_pos": np.array([seat_xy[0], seat_xy[1], z], np.float32)}
        geom = types.SimpleNamespace(release_xy=0.06, release_z=0.02, close_xy_floor=0.008,
                                     place_release_clearance=0.0, release_commit=False,
                                     release_commit_aniso=True, commit_measured_gate=False,
                                     release_commit_band=3.0)
        ext = {"obj": (0.03, 0.03, 0.03), "scale": (0.15, 0.15, 0.0)}
        return float(TERMS["carry_hold"](CostInputs(actions, ee, None, ctx, ext, geom)))
    for z, g in ((_AT_SEAT, 0.0), (_AT_SEAT, 1.0), (_FAR, 0.0), (_FAR, 1.0)):
        assert abs(aniso_with_contact(z, g, 1.0) - aniso_with_contact(z, g, 0.0)) < 1e-9, \
            f"aniso path must ignore seat_contact at z={z} grip={g}"


def test_stall_overshoot_lowers_only_the_seat_aim():
    """place_frame: inside the descend gate the aim drops by place_overshoot; outside (transit at
    carry altitude) it is unchanged, so the approach cannot be dragged into the destination's body.
    The overshoot is a press aid: with the hand no longer commanded closed (overshoot_on False,
    post-release) the aim returns to the seat, so the open hand lifts off the seated payload."""
    from vlm_dp.cost.terms import _place_frame
    def frame(xy_off, stall, overshoot_on=True):
        seat_xy, seat_top = (0.50, 0.10), 0.80
        px = seat_xy[0] + xy_off
        ee = torch.tensor([[px, seat_xy[1], 1.0]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
        ctx = {"payload": "obj", "place_target": "scale", "overshoot_on": overshoot_on,
               "objects": {"obj": {"pos": torch.tensor([px, seat_xy[1], 1.0]),
                                   "extents": (0.03, 0.03, 0.03)},
                           "scale": {"pos": torch.tensor([seat_xy[0], seat_xy[1], seat_top]),
                                     "extents": (0.15, 0.15, 0.0)}},
               "eef_pos": np.array([px, seat_xy[1], 1.0], np.float32)}
        geom = types.SimpleNamespace(place_release_clearance=0.0, release_on_stall=stall,
                                     place_overshoot=0.08, place_descend_radius=0.08)
        ext = {"obj": (0.03, 0.03, 0.03), "scale": (0.15, 0.15, 0.0)}
        f = _place_frame(CostInputs(torch.zeros(1, 4, 8), ee, None, ctx, ext, geom))
        return float(f[1].mean())
    inside_on, inside_off = frame(0.0, True), frame(0.0, False)
    outside_on, outside_off = frame(0.30, True), frame(0.30, False)
    assert abs((inside_on - inside_off) - 0.08) < 1e-6, \
        f"inside the gate the aim must drop by exactly the overshoot, got {inside_on - inside_off:.4f}"
    assert abs(outside_on - outside_off) < 1e-6, "outside the gate the transit aim must be unchanged"
    released = frame(0.0, True, overshoot_on=False)
    assert abs(released - inside_off) < 1e-6, \
        "post-release (overshoot_on False) the aim must return to the seat"


# --------------------------------------------------------- grasp-hold churn (all three parts opt-in)
def _regrasp(grip, grace, intent="close"):
    """regrasp_penalty on one grasp-stage candidate: gripper channel grip, window grace."""
    ee = torch.tensor([[0.50, 0.0, 0.30]]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    actions = torch.zeros(1, 4, 8)
    actions[..., 7] = grip
    ctx = {"gripper_intent": intent, "grasp_obj": "pear", "payload": None}
    if grace is not None:
        ctx["hold_grace"] = grace
    return float(TERMS["regrasp_penalty"](CostInputs(actions, ee, None, ctx, {}, GEOM)))


def test_regrasp_penalty_charges_opening_only_inside_the_window():
    """Opening costs while the window is open, holding is free, and both are free outside it."""
    assert _regrasp(0.0, 1.0) > _regrasp(1.0, 1.0), "inside the window, opening must cost more"
    assert _regrasp(1.0, 1.0) == 0.0, "holding closed must be free"
    assert _regrasp(0.0, 0.0) == 0.0, "window shut: opening is the ordinary command again"
    assert _regrasp(0.0, None) == 0.0, "no hold_grace key (every shipped context): term is inert"
    assert _regrasp(0.0, 1.0, intent="open") == 0.0, "the reopen recovery owns the channel"


# ------------------------------------------------------- articulated pull / press (both opt-in terms)
def _artic_score(term, key, spec, ee, eef=(0.50, 0.00, 0.30)):
    """One articulation term on a single candidate whose TCP path is ee [H,3]. spec None omits the key."""
    ee = torch.as_tensor(ee, dtype=torch.float32).view(1, -1, 3)
    ctx = {"eef_pos": np.asarray(eef, np.float32)}
    if spec is not None:
        ctx[key] = spec
    return float(TERMS[term](CostInputs(torch.zeros(1, ee.shape[1], 8), ee, None, ctx, {}, GEOM)))


_HOOKED = [[0.50, 0.00, 0.30], [0.50, -0.02, 0.30], [0.50, -0.04, 0.30]]   # on the line, sweeping open
_DWELL = [[0.50, 0.00, 0.30]] * 3                                          # on the line, going nowhere
_OFF_LINE = [[0.50, 0.00, 0.34], [0.50, -0.02, 0.34], [0.50, -0.04, 0.34]]  # same sweep, 40mm high
_ALONG_BAR = [[0.58, 0.00, 0.30], [0.58, -0.02, 0.30], [0.58, -0.04, 0.30]]  # 80mm along a 50mm bar


def test_hook_pull_charges_the_endpoint_and_the_line():
    """Progress is terminal, the hook is per-step, and the bar span is genuinely free."""
    swept = _artic_score("hook_pull", "pull", _pull_spec(), _HOOKED)
    assert swept < _artic_score("hook_pull", "pull", _pull_spec(), _DWELL), \
        "a chunk that sweeps toward the open pose must beat one that dwells on the handle"
    assert _artic_score("hook_pull", "pull", _pull_spec(), _OFF_LINE) > swept, \
        "leaving the handle's travel line must cost -- that is the hand coming off the bar"
    assert _artic_score("hook_pull", "pull", _pull_spec(span=0.10), _ALONG_BAR) < \
        _artic_score("hook_pull", "pull", _pull_spec(), _ALONG_BAR), \
        "inside the measured span, where along the bar the hand hooks must be free"
    assert _artic_score("hook_pull", "pull", _pull_spec(), _HOOKED,
                        eef=(0.50, 0.50, 0.30)) < 0.01 * swept, \
        "off-band the reach terms own the approach: the measured gate shuts the term"
    assert _artic_score("hook_pull", "pull", None, _HOOKED) == 0.0, \
        "no pull metadata (every free-body stage, every shipped config): the term is inert"


_PRESS_PART = [[0.50, 0.00, 0.30], [0.50, 0.00, 0.28], [0.50, 0.00, 0.26]]   # 40mm along the normal
_PRESS_FULL = [[0.50, 0.00, 0.30], [0.50, 0.00, 0.27], [0.50, 0.00, 0.242]]  # 58mm = the depth
_PRESS_DEEP = [[0.50, 0.00, 0.30], [0.50, 0.00, 0.24], [0.50, 0.00, 0.18]]   # 120mm, well past it
_PRESS_SIDE = [[0.50, 0.08, 0.30], [0.50, 0.08, 0.27], [0.50, 0.08, 0.242]]  # same depth, 80mm off


def test_press_axis_aims_beyond_the_surface_inside_a_cone():
    """Reaching the penetration target is what pays; going deeper pays nothing more; sideways costs."""
    part = _artic_score("press_axis", "press", _press_spec(), _PRESS_PART)
    full = _artic_score("press_axis", "press", _press_spec(), _PRESS_FULL)
    assert full < part, "the goal is BEYOND the surface: a press that stops short must cost more"
    assert _artic_score("press_axis", "press", _press_spec(), _PRESS_DEEP) == full, \
        "the depth is a penetration surrogate, not a reward: pressing past it buys nothing"
    assert _artic_score("press_axis", "press", _press_spec(), _PRESS_SIDE) > full, \
        "lateral error outside the measured cone must cost at equal depth"
    assert _artic_score("press_axis", "press", None, _PRESS_FULL) == 0.0, \
        "no press metadata (every free-body stage, every shipped config): the term is inert"


# ------------------------------------------------------------- insertion corridor (both opt-in terms)
def _insert_score(term, spec, ee, payload=(0.50, 0.00, 0.30), **ctx_over):
    """One insertion term on a single candidate. The measured payload sits at the measured TCP, so
    the carried path IS ee [H,3]; spec None omits the key."""
    ee = torch.as_tensor(ee, dtype=torch.float32).view(1, -1, 3)
    pos = np.asarray(payload, np.float32)
    ctx = {"eef_pos": pos, "payload": "pear",
           "objects": {"pear": {"pos": torch.as_tensor(pos), "extents": EXTENTS["pear"],
                                "axis": None, "grasp_extent": None, "grasp_region": None}}}
    if spec is not None:
        ctx["insert"] = spec
    ctx.update(ctx_over)
    return float(TERMS[term](CostInputs(torch.zeros(1, ee.shape[1], 8), ee, None, ctx, {}, GEOM)))


_ON_AXIS = [[0.50, 0.00, 0.30], [0.50, 0.00, 0.27], [0.50, 0.00, 0.24]]    # down the cone's axis
_WIDE_DOWN = [[0.62, 0.00, 0.30], [0.62, 0.00, 0.27], [0.62, 0.00, 0.24]]  # 120mm off, descending
_WIDE_HOLD = [[0.62, 0.00, 0.30]] * 3                                      # 120mm off, holding
_WIDE_LOW = [[0.62, 0.00, 0.18]] * 3                                       # 120mm off, below the seat
_WIDER_DOWN = [[0.70, 0.00, 0.30], [0.70, 0.00, 0.27], [0.70, 0.00, 0.24]]


def test_insert_funnel_charges_only_outside_the_cone():
    """Inside the corridor the attractor is left alone; outside, further out costs more."""
    assert _insert_score("insert_funnel", _insert_spec(), _ON_AXIS) == 0.0, \
        "a descent down the axis is inside the cone at every height: the corridor must be silent"
    wide = _insert_score("insert_funnel", _insert_spec(), _WIDE_DOWN)
    assert wide > 0.0, "a descent 120mm off a 50mm mouth is outside the corridor and must cost"
    assert _insert_score("insert_funnel", _insert_spec(), _WIDER_DOWN) > wide, \
        "further outside the cone must cost more"
    assert _insert_score("insert_funnel", _insert_spec(height=0.02), _WIDE_DOWN) < wide, \
        "the same mouth reached in less axial distance is a wider cone: the pair IS the slope"
    assert _insert_score("insert_funnel", _insert_spec(), _WIDE_DOWN,
                         payload=(0.50, 0.60, 0.30)) < 0.01 * wide, \
        "off-capture the place attractor owns the transit: the measured gate shuts the term"
    assert _insert_score("insert_funnel", None, _WIDE_DOWN) == 0.0, \
        "no insert metadata (every set-down stage, every shipped config): the term is inert"


def test_descend_gate_prices_descent_and_altitude_only_outside_the_cone():
    """Descending inside the corridor is free; outside it, the drop and the altitude both cost."""
    assert _insert_score("descend_gate", _insert_spec(), _ON_AXIS) == 0.0, \
        "the insertion descent itself must be free -- that is what the cone is for"
    down = _insert_score("descend_gate", _insert_spec(), _WIDE_DOWN)
    assert down > _insert_score("descend_gate", _insert_spec(), _WIDE_HOLD), \
        "sinking while outside the corridor must beat holding altitude there"
    assert _insert_score("descend_gate", _insert_spec(), _WIDE_LOW) > 0.0, \
        "the floor charges position too: hovering low and still outside the corridor is not free"
    assert _insert_score("descend_gate", _insert_spec(), _WIDE_HOLD, carry_z=0.34) > 0.0, \
        "carry_z above the seat raises the floor, so the carry altitude is the one enforced"
    assert _insert_score("descend_gate", _insert_spec(), _WIDE_DOWN,
                         payload=(0.50, 0.60, 0.30)) < 0.01 * down, \
        "off-capture the term is shut by the same measured gate as the funnel"
    assert _insert_score("descend_gate", None, _WIDE_DOWN) == 0.0, \
        "no insert metadata (every set-down stage, every shipped config): the term is inert"


class _FakeEnv:
    """Minimum surface ApertureGraspSensor.observe reads."""

    def __init__(self, q):
        self.q = q

    def gripper_q(self):
        return self.q


def _sensor_at(aperture, **kw):
    """A sensor settled at one aperture, in MG's metre units (q_free 0.078)."""
    from vlm_dp.grasp_sensor import ApertureGraspSensor
    s = ApertureGraspSensor(q_free=0.078, stall_margin=0.012, q_touch=0.008, settle_eps=0.004,
                            settle_steps=12, close_steps=12, **kw)
    for _ in range(16):
        s.observe(_FakeEnv(aperture), True)
    return s


# Measured on results/{stack,can}/base50: a held 40mm cube reads ~0.040, and the 1440 logged
# "empty hand" verdicts read 0.066-0.079 (median 0.069). The band below is between the two.
_HELD, _EMPTY = 0.040, 0.069


def test_hold_band_default_is_the_single_threshold():
    """Absent enter/exit keys, every predicate is exactly the old stall_margin arithmetic."""
    for ap in (0.0, 0.010, _HELD, 0.065, 0.066, _EMPTY, 0.078):
        s = _sensor_at(ap)
        assert s.closed_on_air() == (ap >= 0.078 - 0.012), f"closed_on_air moved at {ap}"
        assert s.holding() == (s.closed() and not s.closed_on_air()), f"holding moved at {ap}"
        assert s.hold_lost() == s.is_open(), f"the latch exit test moved at {ap}"


def test_hold_band_hysteresis_separates_acquire_from_lose():
    """enter tight / exit loose: a held cube certifies, a free close is lost, and the band between
    them does neither -- which is what stops the latch and the invariant contradicting each other."""
    kw = dict(stall_margin_enter=0.018, stall_margin_exit=0.010)   # thresholds 0.060 / 0.068
    held, empty = _sensor_at(_HELD, **kw), _sensor_at(_EMPTY, **kw)
    assert held.holding() and not held.hold_lost(), "a 40mm cube must certify and stay latched"
    assert empty.hold_lost() and not empty.holding(), "a free close must drop the latch"
    band = _sensor_at(0.064, **kw)
    assert not band.holding() and not band.hold_lost(), \
        "inside the band neither predicate may fire -- that dead zone IS the hysteresis"
    # The default cannot express it: one threshold makes the same reading both certify and lose.
    plain = _sensor_at(0.064)
    assert plain.holding() and not plain.closed_on_air(), \
        "single-threshold: 0.064 certifies, so a 5mm wobble flips it to empty hand"


# --------------------------------------------- placed payloads as obstacles (departure clipping fix)
def _collision_near_placed(include_placed):
    """Gripper body directly at a PLACED object's position; collision must repel iff the flag is on."""
    pos = (0.50, 0.10, 0.30)
    ee = torch.tensor([list(pos)]).view(1, 1, 3).expand(1, 4, 3).contiguous()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(1, 4, 4).contiguous()
    ctx = {"grasp_obj": "apple", "payload": None, "place_target": None, "destination": "scale",
           "placed": frozenset({"pear"}),
           "objects": {"pear": {"pos": torch.tensor(pos), "extents": (0.046, 0.052, 0.062)},
                       "apple": {"pos": torch.tensor([0.11, 1.41, 0.28]), "extents": (0.037, 0.041, 0.038)}}}
    geom = types.SimpleNamespace(ee_r=0.035, coll_clear=0.02, finger_r=0.012, open_half=0.04,
                                 tool_back=0.06, tcp_to_tip=0.0,
                                 collision_include_placed=include_placed)
    ext = {"pear": (0.046, 0.052, 0.062), "apple": (0.037, 0.041, 0.038)}
    return float(TERMS["collision"](CostInputs(torch.zeros(1, 4, 8), ee, q, ctx, ext, geom)))


def test_placed_payloads_become_obstacles_with_the_flag():
    """Default keeps the shipped exclusion; the flag makes a placed payload repel the gripper again
    (the departing hand knocked the seated pear off in every probe seed with the exclusion on)."""
    assert _collision_near_placed(False) == 0.0, "default: placed excluded, no repulsion"
    assert _collision_near_placed(True) > 0.0, "flag on: the placed pear must repel the gripper"


_TESTS = [test_placed_payloads_become_obstacles_with_the_flag,
          test_grasp_pinch, test_grasp_press, test_release_open, test_carry_lift,
          test_place_surface, test_place_container, test_pour_rotation, test_place_side_approach,
          test_grasp_corridor_cut, test_departure,
          test_grasp_hold_grace, test_regrasp_penalty_charges_opening_only_inside_the_window,
          test_articulated_pull, test_articulated_press,
          test_hook_pull_charges_the_endpoint_and_the_line,
          test_press_axis_aims_beyond_the_surface_inside_a_cone,
          test_insertion, test_insert_funnel_charges_only_outside_the_cone,
          test_descend_gate_prices_descent_and_altitude_only_outside_the_cone,
          test_hold_band_default_is_the_single_threshold,
          test_hold_band_hysteresis_separates_acquire_from_lose,
          test_every_term_is_reachable, test_press_only_relaxes_pinch_certification,
          test_orientation_stands_down_only_when_the_vlm_rotates,
          test_contact_criterion_on_measured_geometry, test_contact_criterion_has_margin_on_clean_clouds,
          test_contact_criterion_marginal_band_is_recorded,
          test_grasp_slack_is_the_slack_left_in_the_aperture,
          test_grasp_slack_against_the_shipped_configs,
          test_release_commit_commits_smoothly, test_release_commit_default_off_is_the_hard_gate,
          test_grasp_commit_measured_gate_blocks_premature_close,
          test_release_aniso_commits_smoothly, test_release_aniso_is_tighter_in_z_than_in_xy,
          test_release_aniso_boundary_sits_on_the_tolerance,
          test_release_aniso_decides_on_the_measured_payload_not_the_candidate,
          test_release_aniso_is_decisive_not_half_open,
          test_release_aniso_absent_leaves_the_hard_gate_untouched,
          test_stall_release_needs_contact_not_altitude, test_stall_release_still_requires_xy,
          test_stall_release_absent_leaves_aniso_untouched, test_stall_overshoot_lowers_only_the_seat_aim]


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
