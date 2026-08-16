"""CPU tests pinning WHO owns hold state and how long a belief may stay un-correctable.

Three defects lived here, all of the same shape: a stateful decision was made once, correctly, and
then either re-derived statelessly by its consumer or silently retracted by an unrelated event.

  1. Two hold authorities. SensedWorld.observe latches the hold with hysteresis; the bridge threw
     that away and re-derived sensor.held_object() per call. The re-derived form asks a stricter
     question -- "would a hold certify from scratch at this instant?" -- true only ~4% of steps.
     Measured: the pear was physically in the hand by median step 17 but the stage did not certify
     until 159 (against 55 on the last good run). Both consumers starved, and because one of them is
     the gripper latch, the loop is self-reinforcing: no certification -> latch disarmed -> hand
     opens -> still no certification.
  2. A shared, unconditional vision clock. A correction rejected for exceeding the jump bound still
     advanced the clock, so the bound collapsed to one step's budget and rejected the same
     displacement for ever. An object knocked during a failed grasp froze at its pre-knock belief.
  3. A contact anchor retracted by one open COMMAND, while the close gate explicitly tolerates 25%
     open commands inside its window.

Run: ``python -m vlm_dp.tests.test_hold_authority``.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.grasp_sensor import ApertureGraspSensor
from vlm_dp.world import SensedWorld

PEAR, OPEN = 0.258, 0.0          # finger angles from the sensor's calibration docstring


class _FK:
    """Gripper pose from 'joint encoders'. The stub reports whatever the test placed the hand at."""

    def __init__(self, owner):
        self.owner = owner

    def grasp_point(self, _q, _offset):
        import torch
        p = torch.tensor(self.owner.tcp_xyz, dtype=torch.float64).reshape(1, 3)
        return p, torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)


class _Env:
    """Minimal env: a finger angle and a hand position, both driven by the test."""

    def __init__(self, q=0.0, tcp=(0.0, 0.0, 0.0)):
        self.q = q
        self.tcp_xyz = list(tcp)
        self.fk = _FK(self)

    def gripper_q(self):
        return self.q

    def tcp(self):
        return np.asarray(self.tcp_xyz, dtype=np.float64)

    def q0(self):
        import torch
        return torch.zeros(7, dtype=torch.float64)


class _Perception:
    """Returns whatever the test says the camera saw."""

    def __init__(self):
        self.seen = {}
        self.distrust = set()

    def observe(self, _env):
        return dict(self.seen)


def _world(track="fk"):
    return SensedWorld(_Perception(), ApertureGraspSensor(), track=track)


def _settle_onto_pear(world, env, steps=None):
    """Drive a clean, fully-commanded close onto the pear until the hold latches."""
    s = world.sensor
    n = steps if steps is not None else s.close_steps + s.settle_steps + 2
    env.q = PEAR
    for _ in range(n):
        world.observe(env, commanded_close=True, candidates={"pear"})
    return world


def test_a_hold_latches_at_all():
    """Baseline: without this the other tests could pass vacuously."""
    world = _world()
    env = _Env(tcp=(0.0, 0.0, 0.0))
    world.seed("pear", (0.0, 0.0, 0.0))
    _settle_onto_pear(world, env)
    assert world.held() == "pear", (
        f"a settled, fully-commanded close on a nearby object must latch a hold, got {world.held()!r}")


def test_the_latch_survives_flicker_that_the_instantaneous_predicate_does_not():
    """THE regression. Four opens inside a 12-command window drop the duty below close_duty, so the
    stateless predicate stops certifying -- while the fingers never move off the pear. The latched
    authority must keep the hold, because the object never left the hand."""
    world = _world()
    env = _Env(tcp=(0.0, 0.0, 0.0))
    world.seed("pear", (0.0, 0.0, 0.0))
    _settle_onto_pear(world, env)
    s = world.sensor

    # Fingers stay stalled on the pear; only the COMMAND flickers, 4 opens in the last 12.
    for i in range(12):
        world.observe(env, commanded_close=(i % 3 != 0), candidates={"pear"})

    instantaneous = s.held_object({"pear": np.zeros(3)}, env.tcp())
    assert instantaneous is None, (
        "fixture is not exercising the gap: the stateless predicate still certifies, so this "
        "sequence does not distinguish the two authorities")
    assert world.held() == "pear", (
        "the latched hold was lost to command flicker alone. The fingers never left the pear "
        "(angle held at the acquired grip), so the object is still in the hand; this is exactly the "
        "case the grip-signature hysteresis exists to ride out")


def test_the_latch_still_releases_when_the_fingers_actually_open():
    """Hysteresis must not become a rubber stamp: a real release has to be seen."""
    world = _world()
    env = _Env(tcp=(0.0, 0.0, 0.0))
    world.seed("pear", (0.0, 0.0, 0.0))
    _settle_onto_pear(world, env)
    assert world.held() == "pear"
    env.q = OPEN                                    # fingers physically open
    for _ in range(4):
        world.observe(env, commanded_close=False, candidates={"pear"})
    assert world.held() is None, (
        "an open hand must drop the latched hold, or a drop can never be detected and backtracking "
        "never fires")


def test_a_tolerated_open_command_does_not_erase_the_contact_anchor():
    """_pending is the hand pose at contact onset, used to correct certification lag. It was cleared
    by a single commanded_close=False -- a transient the close gate is explicitly built to tolerate.
    The anchor is a claim about the fingers, so only the fingers may retract it."""
    world = _world()
    env = _Env(tcp=(0.0, 0.0, 0.0))
    world.seed("pear", (0.0, 0.0, 0.0))
    env.q = PEAR
    world.observe(env, commanded_close=True, candidates={"pear"})
    anchor = world._pending
    assert anchor is not None, "fixture: contact onset should have been anchored"

    world.observe(env, commanded_close=False, candidates={"pear"})   # one tolerated open command
    assert world._pending is not None, (
        "one open COMMAND erased the contact anchor while the fingers stayed closed on the object. "
        "The close gate tolerates 25% opens in its window, so this retraction contradicts it, and "
        "the hold is later re-anchored to a hand pose that has moved on")
    assert np.allclose(world._pending[0], anchor[0]), "the anchor must not silently re-seat either"


def _held_via(world, env, authority, payload="pear"):
    """The bridge's decision, called the way the bridge calls it.

    vlm_dp.hold is imported rather than vlm_dp.bridge on purpose: the bridge pulls in Isaac, so a
    test that imported it would ERROR identically with and without the defect and pin nothing.
    """
    from vlm_dp.hold import payload_held
    return payload_held(payload, authority, world, world.sensor, env.tcp(), world._pos[payload])


def test_the_bridge_reads_the_latched_hold_not_a_re_derived_one():
    """The defect as the system actually saw it. _payload_held gates BOTH the stage advance and the
    gripper latch, so when it re-derives the stateless predicate the failure closes a loop: no
    certification -> latch disarmed -> the hand reopens -> still no certification. Fixing the
    sensor's SIGNAL cannot escape that, which is why three separate sensor fixes never moved it."""
    world = _world()
    env = _Env(tcp=(0.0, 0.0, 0.0))
    world.seed("pear", (0.0, 0.0, 0.0))
    _settle_onto_pear(world, env)
    for i in range(12):                             # command flicker, fingers never leave the pear
        world.observe(env, commanded_close=(i % 3 != 0), candidates={"pear"})

    assert _held_via(world, env, "latched"), (
        "the bridge did not see a hold that the world has latched. This is the regression: the "
        "grasp stage cannot certify, so it never advances and the gripper latch never arms")
    assert not _held_via(world, env, "instant"), (
        "fixture check: hold_authority='instant' must reproduce the OLD behaviour, or the A/B knob "
        "is not actually restoring what the old runs did")


def test_a_displaced_object_is_eventually_re_acquired_not_frozen():
    """The shared unconditional clock made rejection permanent. With a per-object clock keyed to the
    last ACCEPTED correction, a 10 cm displacement is rejected at first (it outruns one step's
    budget) and admitted once enough un-corrected time has accrued."""
    world = _world(track="reperceive")
    world.reperceive_every = 1                      # correct every step, so the test is short
    env = _Env(tcp=(1.0, 1.0, 1.0))                 # hand far away: nothing is held
    world.seed("apple", (0.0, 0.0, 0.0))
    world.perception.seen = {"apple": np.array([0.10, 0.0, 0.0])}   # knocked 10 cm

    accepted_at = None
    for step in range(1, 40):
        world.observe(env, commanded_close=False, candidates=None)
        if float(np.linalg.norm(world._pos["apple"] - np.array([0.10, 0.0, 0.0]))) < 1e-9:
            accepted_at = step
            break

    assert accepted_at is not None, (
        "the belief never caught up to a 10 cm displacement in 40 steps. The rate clock is being "
        "advanced on REJECTED corrections too, so the budget never grows and this object is frozen "
        "at its stale position permanently -- the controller then reaches at empty table")
    assert accepted_at > 1, (
        f"accepted immediately (step {accepted_at}); the jump guard is not rejecting at all, which "
        "would let a tracker that latched onto the gripper drag the belief off the object")


def test_the_rate_budget_is_per_object():
    """An occluded object's budget must accrue while OTHER objects are being corrected.

    The discriminating case, and the one the weight task actually hits: the apple is out of view
    (behind the arm) for a stretch while the pear is corrected every step. A shared clock is
    advanced by the pear's corrections, so when the apple finally reappears -- displaced, because a
    failed grasp knocked it -- its budget is one step wide and the correction is rejected. It is
    then rejected every subsequent step for the same reason, and the belief is frozen for good.
    """
    world = _world(track="reperceive")
    world.reperceive_every = 1
    env = _Env(tcp=(1.0, 1.0, 1.0))                 # hand far away: nothing is held
    world.seed("apple", (0.0, 0.0, 0.0))
    world.seed("pear", (0.5, 0.0, 0.0))

    for step in range(1, 21):                       # apple occluded; only the pear is seen
        world.perception.seen = {"pear": np.array([0.5 + 0.001 * step, 0.0, 0.0])}
        world.observe(env, commanded_close=False, candidates=None)

    knocked = np.array([0.20, 0.0, 0.0])            # apple reappears, moved 20 cm by a failed grasp
    world.perception.seen = {"pear": np.array([0.52, 0.0, 0.0]), "apple": knocked}
    world.observe(env, commanded_close=False, candidates=None)

    assert float(np.linalg.norm(world._pos["apple"] - knocked)) < 1e-9, (
        "the apple's correction was rejected after 20 steps out of view. Its jump budget was "
        "consumed by the PEAR's corrections, so the clock is still shared; the apple stays frozen "
        "at a stale position and the controller reaches at empty table for the rest of the episode")


_TESTS = [test_a_hold_latches_at_all,
          test_the_latch_survives_flicker_that_the_instantaneous_predicate_does_not,
          test_the_latch_still_releases_when_the_fingers_actually_open,
          test_a_tolerated_open_command_does_not_erase_the_contact_anchor,
          test_the_bridge_reads_the_latched_hold_not_a_re_derived_one,
          test_a_displaced_object_is_eventually_re_acquired_not_frozen,
          test_the_rate_budget_is_per_object]


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
