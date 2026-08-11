"""CPU tests pinning the geometry and the hold rule a DECLARED grasp feature implies.

One root cause, two runtime defects, both reproduced here against the real code:

  1. A declared grasp keypoint (the capsule machine's lid rim) was propagated as its owner's grasp
     POINT but not as its owner's grasp EXTENT, so ``terms._grasp_frame`` fell back to the owner's
     whole-object keepout radius: 0.3175m for a 10mm lip. Every grasp term was then sized for a
     machine -- aperture_region floored at 6.4 weighted, a 33cm straddle repulsion centred on the
     rim the fingers were told to stand on, and a 12.7cm centre dead zone and close gate.
  2. The hold latch resolved against object CENTROID beliefs while the stage grasped a point
     0.25-0.30m away from one, so ApertureGraspSensor.held_object -- proximity gate 0.10m -- could
     not see a physically perfect grasp, and the stage was unadvanceable by construction.

Plus the reopen recovery, which cleared only on ``is_open()`` and therefore never cleared when the
fingers wedged part-way, and the two preflights that certified the run anyway (the grasp-geometry
one measured a different quantity than the cost uses; the advanceability one ran against a stub
sensor that answered "held" unconditionally).

Run: ``python -m vlm_dp.tests.test_grasp_geometry``.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------
# The scene, as the failing run measured it (results/.../motion_uni3_capsule, seed 6).
RIM = np.array([0.35, 0.10, 0.62])          # declared lid-rim keypoint, the point stage 1 grasps
CENTROID = RIM + np.array([0.0, 0.24, -0.15])   # the machine's centroid belief, 0.283m away
TCP_ON_RIM = RIM + np.array([0.0012, 0.0, 0.0])  # seed 6: TCP 1.2mm from the rim
TCP_HOVER = RIM + np.array([0.16, 0.0, 0.0])     # seeds 1/2: parked on the hover shell, dxy 0.16
CAPSULE_EXT = (0.29, 0.3175, 0.30)          # (grip, keepout, half-height) from perception
RIM_HALF_W = 0.010                          # the declared feature's own half-width
STALLED_AP = 0.25                           # seed 6: aperture stalled in the held band

GEOM = {
    "open_half": 0.04, "finger_r": 0.012, "tcp_to_tip": 0.0, "tool_back": 0.06,
    "center_scale": 0.4, "grasp_dead_zone": "proportional", "close_xy_floor": 0.008,
    "aperture_margin": 0.006, "close_z_scale": 0.025, "preshape_z": 0.025,
    "preshape_xy_tol": 0.02, "close_gate_split_axes": True,
}


def _stub_sim_modules():
    """Install the import-time stubs the sim-only dependencies need on a CPU box.

    Only the module OBJECTS are stubbed; every line of vlm_dp under test is the real one.
    """
    for name in ("sim_common", "sim_common.envs"):
        sys.modules.setdefault(name, types.ModuleType(name))
    droid = types.ModuleType("sim_common.envs.droid")
    droid.DroidEnv = object
    sys.modules.setdefault("sim_common.envs.droid", droid)
    for name, attrs in (("rekep.grounding", ("propose_keypoints",)),
                        ("rekep.constraint_generation", ("ConstraintGenerator",)),
                        ("rekep.keypoint_tracking", ("KeypointTracker",)),
                        ("rekep.utils", ("get_callable_grasping_cost_fn", "load_default_config"))):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, object)
        sys.modules[name] = mod


_stub_sim_modules()

from vlm_dp.cost import terms                                       # noqa: E402
from vlm_dp.grasp_sensor import ApertureGraspSensor                 # noqa: E402
from vlm_dp.grounding import SceneObject, Stage                     # noqa: E402
from vlm_dp.world import SensedWorld                                # noqa: E402


# ---------------------------------------------------------------------------------------------
# Fixtures


class _FK:
    def __init__(self, owner):
        self.owner = owner

    def grasp_point(self, _q, _offset):
        p = torch.tensor(self.owner.tcp_xyz, dtype=torch.float64).reshape(1, 3)
        return p, torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)


class _Env:
    """Minimal env: a finger angle and a hand position, both driven by the test."""

    def __init__(self, q=0.0, tcp=(0.0, 0.0, 0.0)):
        self.q = float(q)
        self.tcp_xyz = list(tcp)
        self.fk = _FK(self)

    def gripper_q(self):
        return self.q

    def tcp(self):
        return np.asarray(self.tcp_xyz, dtype=np.float64)

    def q0(self):
        return torch.zeros(7, dtype=torch.float64)


class _Perception:
    """Perception that never re-observes: the test owns the beliefs."""

    distrust = set()

    def observe(self, _env):
        return {}

    def object_extents(self, _name):
        return CAPSULE_EXT


def _inputs(tcp, radius, *, grip_cmd=1.0):
    """Build the exact CostInputs a grasp stage presents, with one feasibility radius."""
    ee = torch.as_tensor(tcp, dtype=torch.float32).reshape(1, 1, 3)
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])            # identity: closing +y, approach +z
    actions = torch.zeros(1, 1, 8)
    actions[..., 7] = grip_cmd
    objects = {"capsule": {"grasp_extent": radius}} if radius is not None else {"capsule": {}}
    ctx = {"target": RIM, "grasp_obj": "capsule", "objects": objects, "payload": None,
           "eef_pos": np.asarray(tcp, dtype=np.float32), "contact": "pinch"}
    return terms.CostInputs(real_actions=actions, ee_pos=ee, ee_quat=quat, context=ctx,
                            extents={"capsule": CAPSULE_EXT}, geom=types.SimpleNamespace(**GEOM))


def _term(name, tcp, radius, weight=1.0):
    return weight * float(terms.TERMS[name](_inputs(tcp, radius)).reshape(-1)[0])


# ---------------------------------------------------------------------------------------------
# 1. Grasp-extent propagation: the radius, and every radius derived from it.


def test_grasp_radius_and_derived_radii():
    """The declared feature's half-width, not the owner's keepout extent, sizes the grasp."""
    geom = types.SimpleNamespace(**GEOM)
    old = float(CAPSULE_EXT[1])          # what _grasp_frame fell back to
    new = RIM_HALF_W                     # what the declaration now propagates

    rows = []
    for label, r in (("before (whole-object)", old), ("after (declared rim)", new)):
        dead = terms.grasp_slack(geom, r)
        rows.append((label, r, dead, max(dead, GEOM["close_xy_floor"]), r + GEOM["finger_r"],
                     max(r + GEOM["aperture_margin"] - GEOM["open_half"], 0.0)))
    print("  radius     dead_zone  close_gate  straddle_keepout  aperture_floor")
    for label, r, dead, gate, keep, floor in rows:
        print(f"  {label:22s} r={r * 1e3:6.1f}mm  dead={dead * 1e3:6.1f}mm  "
              f"gate={gate * 1e3:5.1f}mm  keepout={keep * 1e3:6.1f}mm  floor={floor * 1e3:6.1f}mm")

    assert abs(rows[0][2] - 0.127) < 1e-3, f"before: dead zone {rows[0][2]:.4f}, expected 0.127"
    assert abs(rows[1][2] - 0.004) < 1e-6, f"after: dead zone {rows[1][2]:.4f}, expected 0.004"
    assert rows[0][5] > 0.24 and rows[1][5] == 0.0, "the rim must fit the aperture; the machine must not"
    assert 0.010 <= rows[1][1] <= 0.012, f"effective grasp radius {rows[1][1]} is not the rim's"


def test_grasp_term_values_before_and_after():
    """Every pinch term, at the rim and on the hover shell the failing seeds parked on."""
    old, new = float(CAPSULE_EXT[1]), RIM_HALF_W
    weights = {"aperture_region": 80.0, "center_region": 120.0, "straddle": 30.0, "close_gripper": 2.0}
    print(f"  {'term':16s} {'@rim before':>12s} {'@rim after':>12s} "
          f"{'@hover before':>14s} {'@hover after':>13s}")
    at = {}
    for name, w in weights.items():
        vals = tuple(_term(name, tcp, r, w) for tcp in (TCP_ON_RIM, TCP_HOVER) for r in (old, new))
        at[name] = vals
        print(f"  {name:16s} {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:14.4f} {vals[3]:13.4f}")

    # aperture_region: 80*(r + 0.006 - 0.04)^2 = 6.43 weighted for the machine, everywhere.
    assert abs(at["aperture_region"][0] - 6.43) < 0.05, at["aperture_region"][0]
    assert at["aperture_region"][1] == 0.0, "the rim fits the aperture: no floor may remain"
    assert at["aperture_region"][2] == at["aperture_region"][0], "the old floor was position-independent"
    # straddle: a 0.3175m keepout centred on the rim repels the fingers standing on it.
    assert at["straddle"][0] > 5.0 and at["straddle"][1] == 0.0, at["straddle"][:2]
    # center_region: 12.7cm of dead zone left the hover shell all but free (0.13 against the
    # 6.43 aperture_region floor it paid everywhere); a 4mm dead zone makes it a real gradient.
    assert at["center_region"][2] < 0.2, at["center_region"][2]
    assert at["center_region"][3] > 20 * at["center_region"][2], \
        f"after must pull off the hover shell: {at['center_region'][2]:.3f} -> {at['center_region'][3]:.3f}"
    # close_gripper: the gate must not be satisfied 16cm away from the rim (closed-on-air).
    assert at["close_gripper"][2] < 0.05, "before: the close command was satisfied on the hover shell"
    assert at["close_gripper"][3] > 1.9, "after: closing 16cm off the rim must cost the full command"
    # ... and it must still command the close AT the rim, where the fingers are on the feature.
    assert at["close_gripper"][1] == 0.0 and at["center_region"][1] == 0.0, at["close_gripper"][:2]


# ---------------------------------------------------------------------------------------------
# 2. Hold resolution at the stage's grasp point (seed 6).


def _seed6_world(points):
    """Run seed 6's evidence through the real sensor, latch and world."""
    world = SensedWorld(_Perception(), ApertureGraspSensor(), track="fk")
    world.seed("capsule", CENTROID)
    env = _Env(q=STALLED_AP, tcp=TCP_ON_RIM)
    for _ in range(20):                       # close duty 99.8%, aperture stalled: a settled close
        world.observe(env, True, candidates={"capsule"}, points=points)
    return world, env


def test_seed6_hold_resolves_against_the_declared_grasp_point():
    """TCP 1.2mm from the declared rim, fingers stalled in the held band -> the hold is seen."""
    d_rim = float(np.linalg.norm(TCP_ON_RIM - RIM))
    d_centroid = float(np.linalg.norm(TCP_ON_RIM - CENTROID))
    print(f"  TCP->rim {d_rim * 1e3:.1f}mm, TCP->centroid {d_centroid * 1e3:.0f}mm, "
          f"proximity gate {ApertureGraspSensor().proximity * 1e3:.0f}mm, aperture {STALLED_AP}")

    before, _ = _seed6_world(points=None)                       # centroid beliefs (the defect)
    after, _ = _seed6_world(points={"capsule": RIM})            # the stage's grasp point
    print(f"  hold latch: before={before.held()!r}  after={after.held()!r}")
    assert before.sensor.holding(), "fixture check: the fingers must read as stalled on something"
    assert before.held() is None, "pre-fix control: the centroid belief is out of proximity"
    assert after.held() == "capsule", "the grasp at the declared rim must be recognised"


def test_seed6_stage_one_is_advanceable():
    """The REAL _stage_reached, on seed 6's state, must advance the lid grasp."""
    from vlm_dp.bridge import VlmDpBridge

    world, env = _seed6_world(points={"capsule": RIM})
    stage = Stage(name="grasp capsule", gripper="close", grasp_obj="capsule", payload=None,
                  target=(lambda: RIM), contact="pinch")
    bridge = _bare_bridge(world, env, stage)
    print(f"  advance test = {bridge.advance_test_name(stage)!r}, "
          f"grasp_slack={bridge._grasp_slack('capsule'):.4f}m")
    assert bridge._stage_reached(stage, {}), "stage 1 must be advanceable from seed 6's state"

    # Pre-fix control: the same rule, with the latch resolved against the centroid belief.
    stale, _ = _seed6_world(points=None)
    bridge.world = stale
    assert not bridge._stage_reached(stage, {}), \
        "fixture check: the old centroid-resolved hold must NOT advance (else this proves nothing)"


def test_thin_feature_contact_is_not_confused_with_a_free_close():
    """A settled close on a 10mm handle is above the generic air threshold."""
    stage = Stage(name="grasp cover", gripper="close", grasp_obj="cover", payload=None,
                  target=(lambda: RIM), contact="pinch")
    env = _Env(q=0.69, tcp=RIM)
    world = types.SimpleNamespace(held=(lambda: None))
    bridge = _bare_bridge(world, env, stage)
    bridge.grounding.objects = [
        SceneObject(name="cover", pos=(lambda: RIM), extents=CAPSULE_EXT,
                    grasp_extent=0.005)
    ]
    bridge._obj_pos = {"cover": (lambda: RIM)}
    bridge._extents = {"cover": CAPSULE_EXT}

    # Reproduce the runtime race: the aperture enters the air band after four applied close
    # steps, eight steps before the sensor can possibly certify a settled contact.
    bridge.sensor = ApertureGraspSensor()
    for _ in range(4):
        bridge.sensor.observe(env, True)
    assert bridge.sensor.closed_on_air() and not bridge.sensor.closed()
    assert bridge._thin_feature_settling("cover"), "thin contact must keep closing until settled"
    assert not bridge._thin_feature_held("cover"), "unsettled contact must not certify early"

    def sensor_at(aperture):
        sensor = ApertureGraspSensor()
        env.q = aperture
        for _ in range(20):
            sensor.observe(env, True)
        return sensor

    bridge.sensor = sensor_at(0.69)
    assert bridge.sensor.closed_on_air(), "fixture: old fixed threshold must call this air"
    assert bridge._payload_held("cover"), "width-predicted thin-handle contact must certify"
    preflight_sensor = bridge._preflight_sensor(holding=True, stage=stage)
    expected = preflight_sensor.q_free - bridge._AP_SLOPE * 0.010
    assert abs(preflight_sensor.aperture() - expected) < 1e-6
    bridge.sensor = sensor_at(0.40)
    assert bridge.sensor.holding(), "fixture: a wide lid-edge obstruction passes the generic sensor"
    assert not bridge._payload_held("cover"), (
        "a contact visibly wider than the measured handle must not enter lift")
    bridge.sensor = sensor_at(bridge.sensor.q_free)
    assert bridge.sensor.closed_on_air(), "fixture: a free close must be in the air band"
    assert not bridge._payload_held("cover"), "a true free close must still trigger recovery"
    assert not bridge._thin_feature_settling("cover"), "a true free close must reopen immediately"

    # A noisy candidate cannot suppress recovery forever: after the bounded age it is rejected.
    bridge.sensor = ApertureGraspSensor()
    for i in range(30):
        env.q = (0.67, 0.71, 0.69)[i % 3]
        bridge.sensor.observe(env, True)
    assert bridge.sensor.closed_on_air() and not bridge.sensor.closed()
    assert bridge.sensor.close_age() > bridge._thin_feature_settle_max_steps
    assert not bridge._thin_feature_settling("cover"), "an unsteady contact must eventually reopen"

def test_confirmed_thin_grasp_debounces_lift_fluctuation():
    """A settled thin grasp must survive transient lift motion, but not a persistent free close."""
    stage = Stage(name="lift cover", gripper="hold", grasp_obj=None, payload="cover",
                  target=(lambda: RIM), contact="pinch")
    env = _Env(q=0.69, tcp=RIM)
    world = types.SimpleNamespace(held=(lambda: "cover"))
    bridge = _bare_bridge(world, env, stage)
    bridge.grounding.objects = [
        SceneObject(name="cover", pos=(lambda: RIM), extents=CAPSULE_EXT,
                    grasp_extent=0.005)
    ]
    bridge._obj_pos = {"cover": (lambda: RIM)}
    bridge._extents = {"cover": CAPSULE_EXT}
    bridge._thin_feature_loss_ratio = 0.8
    bridge._thin_feature_loss_grace = 2
    bridge._thin_loss_replans = 0
    bridge._ground_err_debug = False
    bridge.grasp_z0 = {}
    bridge.grasp_confirm = 10

    # Load-induced motion crosses the generic air threshold and breaks settling, while the
    # aperture remains consistent with the visually measured 10mm handle.
    sensor = ApertureGraspSensor()
    for q in [0.69] * 12 + [0.71]:
        env.q = q
        sensor.observe(env, True)
    bridge.sensor = sensor
    assert sensor.closed_on_air() and not sensor.closed()
    assert bridge._invariant_violated(stage, {}) is None
    assert bridge._thin_loss_replans == 0

    # A persistent free close is debounced twice, then correctly declared lost.
    sensor = ApertureGraspSensor()
    env.q = sensor.q_free
    for _ in range(20):
        sensor.observe(env, True)
    bridge.sensor = sensor
    assert bridge._invariant_violated(stage, {}) is None
    assert bridge._invariant_violated(stage, {}) is None
    assert bridge._invariant_violated(stage, {}) == "empty hand"


def _bare_bridge(world, env, stage):
    """A VlmDpBridge with only the attributes the rules under test read."""
    from vlm_dp.bridge import VlmDpBridge

    bridge = object.__new__(VlmDpBridge)
    bridge.stage_idx = 0
    bridge.stage_replans = 0
    bridge.grounding = types.SimpleNamespace(
        stages=[stage], completion=None,
        objects=[SceneObject(name="capsule", pos=(lambda: RIM), extents=CAPSULE_EXT,
                             grasp_extent=RIM_HALF_W)])
    bridge._obj_pos = {"capsule": (lambda: RIM)}
    bridge._extents = {"capsule": CAPSULE_EXT}
    bridge._plan_authoritative = True
    bridge._pred_place_transitions = True
    bridge.advance_mode = "sensed"
    bridge.flag_fallback = False
    bridge._grasp_advance_on_hold = True
    bridge._grasp_probe = np.zeros(3)
    bridge.grasp_eps = 0.02
    bridge._geom_ns = types.SimpleNamespace(**GEOM)
    bridge.geom = dict(GEOM)
    bridge.hold_authority = "latched"
    bridge._hold_world = None
    bridge._place_settle = 5
    bridge._place_seen = None
    bridge._place_since = None
    bridge._pred_transition = None
    bridge.lift_tol = 0.01
    bridge.env = env
    bridge.world = world
    bridge.sensor = getattr(world, "sensor", None)
    # Sensor construction parameters, for the preflight's real sensor.
    bridge.stall_margin, bridge.settle_eps = 0.15, 0.01
    bridge.close_steps, bridge.settle_steps = 12, 12
    bridge._last_cmd_close = True
    bridge._thin_feature_contact_ratio = 0.5
    bridge._thin_feature_width_band = 0.08
    bridge._thin_feature_loss_ratio = 0.8
    bridge._thin_feature_loss_grace = 2
    bridge._thin_loss_replans = 0
    bridge._thin_feature_settle_max_steps = 26
    bridge.legacy_sensor = False
    bridge.hold_enter = bridge.hold_exit = None
    # Reopen bounds.
    bridge._reopen, bridge._reopen_ap, bridge._reopen_cooldown_left = False, [], 0
    bridge._reopen_max, bridge._reopen_stall = 12, 4
    bridge._reopen_stall_eps, bridge._reopen_travel = 0.01, 0.02
    bridge._reopen_cooldown = 8
    return bridge


# ---------------------------------------------------------------------------------------------
# 3. The reopen recovery is bounded.


class _ApertureStub:
    q_touch = 0.05
    q_free = 0.7854
    stall_margin_exit = 0.15

    def __init__(self):
        self.q = 0.0

    def aperture(self):
        return self.q

    def closed_on_air(self):
        return self.q >= self.q_free - self.stall_margin_exit


def _reopen_run(bridge, sequence):
    """Feed apertures to the escape test; return (replan_index, reason) or None."""
    bridge._reopen_ap = []
    for i, q in enumerate(sequence):
        bridge.sensor.q = q
        why = bridge._reopen_escape()
        if why is not None:
            return i, why
    return None


def test_reopen_escapes_a_wedged_gripper():
    """Fingers wedged at 0.23-0.38 rad reach neither exit condition; the escape is bounded."""
    bridge = _bare_bridge(None, None, None)
    bridge.sensor = _ApertureStub()

    wedged = [0.62, 0.44, 0.30, 0.24, 0.235, 0.236, 0.235, 0.234, 0.235, 0.234]
    got = _reopen_run(bridge, wedged)
    assert got is not None, "a wedged gripper must not hold the open command for ever"
    print(f"  wedged: escaped at replan {got[0]} -- {got[1]}")

    frozen = [0.30] * 40                     # never moved at all: the timeout is the backstop
    got = _reopen_run(bridge, frozen)
    assert got is not None and got[0] < bridge._reopen_max, got
    print(f"  frozen: escaped at replan {got[0]} -- {got[1]}")


def test_reopen_survives_a_normal_recovery():
    """An ordinary reopen travels to open in a few replans and must never be abandoned."""
    bridge = _bare_bridge(None, None, None)
    bridge.sensor = _ApertureStub()
    normal = [0.64, 0.48, 0.31, 0.17, 0.06]   # is_open() clears it on the next replan
    assert _reopen_run(bridge, normal) is None, "normal closed-on-air recovery must be preserved"
    print(f"  normal: {normal} -> no escape (the real exit is is_open())")

    # A reopen that has barely started -- fingers still in the free-close range -- is an ordinary
    # slow recovery, not a wedge. Only the timeout may end it (observed on tea seed 3).
    slow = [0.785, 0.74, 0.73, 0.727, 0.727, 0.727, 0.727]
    got = _reopen_run(bridge, slow)
    assert got is None or "timeout" in got[1], \
        f"a still-closed-on-air gripper must not be called wedged: {got}"
    print(f"  still closed on air: {slow} -> {got and got[1] or 'no escape'}")


# ---------------------------------------------------------------------------------------------
# 4a. The grasp-geometry preflight asserts the radius the cost will use.


def _rekep(**kw):
    from vlm_dp.grounding.rekep import RekepGrounding

    return RekepGrounding(vlm="fake", task_key="capsule", place_obj="capsule", stages="vlm",
                          open_half=GEOM["open_half"], geom=GEOM, **kw)


def _preflight_args(local_ext):
    cloud = RIM + np.random.default_rng(0).normal(scale=0.05, size=(200, 3))
    return dict(metadata={"grasp_keypoints": [0], "release_keypoints": [-1]},
                keypoints=np.array([RIM]), name_for=(lambda k: "capsule"),
                obj_ext={"capsule": CAPSULE_EXT}, local_ext=local_ext,
                probe_ext={"capsule": 0.02}, usd_ext={"capsule": CAPSULE_EXT},
                clouds={"capsule": cloud}, roles={}, declared_grasp={"capsule": 0})


def test_preflight_refuses_a_whole_object_fallback_on_a_declared_grasp():
    """The would-have-failed proof for fix 1: pre-fix grounding is refused, post-fix passes."""
    ground = _rekep()
    try:
        ground._preflight(**_preflight_args(local_ext={}))          # pre-fix: no extent propagated
    except (SystemExit, ValueError) as exc:
        print(f"  pre-fix grounding refused: {str(exc).splitlines()[-1].strip()[:120]}")
    else:
        raise AssertionError("the preflight accepted a declared grasp sized by the whole object")

    ground._preflight(**_preflight_args(local_ext={"capsule": RIM_HALF_W}))   # post-fix: accepted
    radius, src = ground._grasp_geometry("capsule", {"capsule": CAPSULE_EXT},
                                         {"capsule": RIM_HALF_W})
    assert (radius, src) == (RIM_HALF_W, "local grasp extent")
    fallback, src = ground._grasp_geometry("capsule", {"capsule": CAPSULE_EXT}, {})
    assert fallback == CAPSULE_EXT[1] and "WHOLE-OBJECT" in src


def test_preflight_radii_match_the_cost_terms_exactly():
    """The preflight must report the SAME numbers the terms compute, not a re-derivation."""
    ground = _rekep()
    for radius in (RIM_HALF_W, CAPSULE_EXT[1], 0.05):
        reported = ground._grasp_radii(radius)
        assert reported["dead_zone"] == terms.grasp_slack(types.SimpleNamespace(**GEOM), radius)
    print(f"  radii agree with terms.grasp_slack at r in "
          f"{ {RIM_HALF_W, CAPSULE_EXT[1], 0.05} }")


# ---------------------------------------------------------------------------------------------
# 4b. The advanceability preflight runs the real hold path.


def test_advance_preflight_uses_the_real_sensor_and_latch():
    """A physically valid grasp at the declared keypoint must fire the REAL completion logic."""
    from vlm_dp.grounding.predicates import HistoryBuffer

    world, env = _seed6_world(points={"capsule": RIM})
    stage = Stage(name="grasp capsule", gripper="close", grasp_obj="capsule", payload=None,
                  target=(lambda: RIM), contact="pinch")
    bridge = _bare_bridge(world, env, stage)
    bridge._pred_hist = HistoryBuffer(maxlen=8, dt=1.0 / 15.0)
    synth = {"kp": np.array([RIM]), "eef": RIM, "moved": ()}

    sensor = bridge._preflight_sensor(holding=True)
    assert isinstance(sensor, ApertureGraspSensor), "the preflight must drive the REAL sensor class"
    assert sensor.holding() and not sensor.closed_on_air(), \
        f"the synthesized carry state must certify: aperture {sensor.aperture():.3f}"

    fired, test = bridge.preflight_probe(0, stage, synth)
    print(f"  post-fix: {test!r} -> {'fires' if fired else 'NEVER FIRES'}")
    assert fired, "a valid grasp at the declared point must fire the real advance test"

    # Pre-fix proof: resolve the hold against the owner's centroid belief, as the runtime did.
    bridge._hold_points = lambda names: {"capsule": CENTROID}
    fired_old, _ = bridge.preflight_probe(0, stage, synth)
    print(f"  pre-fix (hold resolved at the centroid): {'fires' if fired_old else 'NEVER FIRES'}")
    assert not fired_old, ("fixture check: the pre-fix hold resolution must fail this preflight, "
                          "or the preflight is not testing the real path")


# ---------------------------------------------------------------------------------------------


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n== {fn.__name__}")
        fn()
    print(f"\nOK: {len(tests)} tests")


if __name__ == "__main__":
    main()
