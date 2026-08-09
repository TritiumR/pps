"""Regression tests for articulated press-stage advancement."""

from __future__ import annotations

import sys
import types

import numpy as np

# bridge only needs this name for annotations in these CPU-only tests. Stub it
# before import so Isaac Sim's pxr runtime is not required.
_droid = types.ModuleType("sim_common.envs.droid")
_droid.DroidEnv = object
sys.modules.setdefault("sim_common.envs.droid", _droid)

from vlm_dp.bridge import VlmDpBridge
from vlm_dp.grounding import Stage


def _bridge():
    bridge = VlmDpBridge.__new__(VlmDpBridge)
    bridge.advance_mode = "sensed"
    bridge.lift_tol = 0.01
    bridge.env = types.SimpleNamespace(tcp=lambda: np.zeros(3))
    bridge._pos = lambda _name: np.array([0.0, 0.0, 1.0])
    bridge.sensor = types.SimpleNamespace(released=lambda: False)
    bridge._place_seen = None
    bridge._place_since = None
    bridge._place_settle = 2
    bridge.stage_replans = 0
    bridge.flag_fallback = False
    return bridge


def test_press_hold_waits_for_tcp_waypoint_not_payload_height():
    reached = [False]
    stage = Stage(
        name="seat lid",
        target=lambda: np.array([0.0, 0.0, 0.2]),
        gripper="hold",
        payload="lid",
        contact="press",
        done=lambda: reached[0],
    )
    bridge = _bridge()

    # The articulated lip starts above the seat target. A payload-height gate
    # would advance immediately even though the TCP has not reached its waypoint.
    assert not bridge._stage_reached(stage, {})
    reached[0] = True
    assert bridge._stage_reached(stage, {})


def test_press_place_does_not_wait_for_a_pinch_release_sensor():
    reached = [False]
    stage = Stage(
        name="open lid",
        target=lambda: np.zeros(3),
        gripper="place",
        payload="lid",
        contact="press",
        done=lambda: reached[0],
    )
    bridge = _bridge()

    assert not bridge._stage_reached(stage, {})
    reached[0] = True
    assert bridge._stage_reached(stage, {})


def test_hold_can_advance_on_its_subgoal_instead_of_keypoint_height():
    reached = [False]
    stage = Stage(
        name="lift pod",
        target=lambda: np.array([0.0, 0.0, 1.03]),
        gripper="hold",
        payload="pod",
        done=lambda: reached[0],
        advance_on_done=True,
    )
    bridge = _bridge()

    # A visual keypoint can sit above the object's root forever. The explicit
    # ReKep lift constraint is the meaningful completion signal in that case.
    assert not bridge._stage_reached(stage, {})
    reached[0] = True
    assert bridge._stage_reached(stage, {})


def test_visual_rise_accepts_corrected_visual_belief():
    bridge = _bridge()
    bridge.track = "visual"
    bridge.world = types.SimpleNamespace(
        visual_position=lambda _name: np.array([0.0, 0.0, 1.0]))
    bridge._grasp_visual_z0 = 1.0
    bridge.grasp_z0 = {"egg": 1.0}
    bridge._pos = lambda _name: np.array([0.0, 0.0, 1.02])
    assert bridge._thin_visual_rose("egg", rise=0.015)
    bridge.track = "fk"
    assert not bridge._thin_visual_rose("egg", rise=0.015)


def test_latched_grasp_stays_closed_until_place_subgoal():
    reached = [False]
    stage = Stage(
        name="place egg",
        target=lambda: np.zeros(3),
        gripper="place",
        payload="egg",
        done=lambda: reached[0],
    )
    bridge = _bridge()
    bridge.grounding = types.SimpleNamespace(stages=[stage])
    bridge.stage_idx = 0
    bridge._grasp_close_latched = True
    bridge._carry_release_latched = False
    bridge._grip_debounce = 0
    actions, _ = bridge.filter_plan(np.zeros((2, 8)), 2)
    assert np.all(actions[:, 7] == 1.0)
    reached[0] = True
    actions, _ = bridge.filter_plan(np.ones((2, 8)), 2)
    assert np.all(actions[:, 7] == 0.0)


_TESTS = [
    test_press_hold_waits_for_tcp_waypoint_not_payload_height,
    test_press_place_does_not_wait_for_a_pinch_release_sensor,
    test_hold_can_advance_on_its_subgoal_instead_of_keypoint_height,
    test_visual_rise_accepts_corrected_visual_belief,
    test_latched_grasp_stays_closed_until_place_subgoal,
]


def main():
    for test in _TESTS:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\nALL PASS ({len(_TESTS)} tests)")


if __name__ == "__main__":
    main()
