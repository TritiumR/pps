"""A latched hold must clear when the object leaves the fingers.

Measured on stack (`results/stack/_sgm_rekep*/101.jsonl`): the gripper grasps cubeA at closure
0.037, the cube slips, the fingers travel on to 0.076 -- free close -- and `held` stays latched
for the remaining 200 steps, so the stage ladder ping-pongs 0<->1 and the episode never ends.

Two guards should have caught it and neither was live:
  * `hold_lost` gates `closed_on_air` behind `hysteresis`, which is set only when a config passes
    hold_enter/hold_exit -- only churn.yaml does. Its sibling `released` uses the same signal
    ungated, so the gate is the anomaly.
  * `HoldLatch._SLIP_MARGIN` is 0.18 on the Isaac radian scale; MuJoCo closure spans 0-0.08, so
    "closure < grip + 0.18" is unconditionally true and the slip guard is dead.

Both fixes are opt-in (`advance.hold_free_close`, `advance.slip_margin`); the defaults reproduce
the ghost hold, which is what the first case pins.
"""

from __future__ import annotations

from vlm_dp.grasp_sensor import ApertureGraspSensor
from vlm_dp.hold import HoldLatch

# runner._SENSOR: the MuJoCo metre-scale convention.
_MG = dict(q_free=0.078, stall_margin=0.012, q_touch=0.008, settle_eps=0.004,
           settle_steps=12, close_steps=12, proximity=0.10)
_GRASP, _AIR = 0.037, 0.076         # closure holding a 40 mm cube; closure on air


class _Env:
    """Minimal env stand-in: the sensor reads gripper_q() only."""

    def __init__(self):
        self.closure = 0.0

    def gripper_q(self):
        return self.closure


def _latch_then(closure_after, *, lost_on_free_close=None, slip_margin=None, steps=20):
    """Return (held after a settled grasp, held after `closure_after` persists)."""
    sensor = ApertureGraspSensor(**_MG, lost_on_free_close=lost_on_free_close)
    latch, env = HoldLatch(sensor, slip_margin=slip_margin), _Env()
    for closure in (_GRASP, closure_after):
        for _ in range(steps):
            env.closure = closure
            sensor.observe(env, True)
            latch.update({"cubeA": (0.0, 0.0, 0.0)}, (0.0, 0.0, 0.0))
        if closure == _GRASP:
            grasped = latch.held()
    return grasped, latch.held()


def test_default_ghost_holds():
    """The shipped defaults reproduce the bug: free close does not clear the latch."""
    assert _latch_then(_AIR) == ("cubeA", "cubeA")


def test_free_close_release():
    """hold_free_close makes the fingers meeting on air clear the hold."""
    assert _latch_then(_AIR, lost_on_free_close=True) == ("cubeA", None)


def test_slip_margin_release():
    """A closure-scale slip margin clears the hold without touching the sensor."""
    assert _latch_then(_AIR, slip_margin=0.02) == ("cubeA", None)


def test_real_hold_survives():
    """Neither guard may drop a hold that is still closed on the object."""
    assert _latch_then(0.040, lost_on_free_close=True) == ("cubeA", "cubeA")
    assert _latch_then(0.040, slip_margin=0.02) == ("cubeA", "cubeA")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all ghost-hold tests passed")
