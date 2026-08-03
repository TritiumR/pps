"""Gripper latch (debounce_gripper) and descent-stall detector (descent_stalled): pure-fn tests."""

import numpy as np

from vlm_dp.grasp_recovery import debounce_gripper, descent_stalled


def test_stall_detector():
    # active descent at the measured ~5.4mm/replan: never stalled
    zs = [0.50 - 0.0054 * i for i in range(6)]
    assert not descent_stalled(zs, 3, 0.003)
    # flat hover with NO prior descent (transit at carry altitude): NOT contact --
    # this exact case released a payload from 16cm in the v3 probe
    assert not descent_stalled([0.30] * 6, 3, 0.003)
    # ascent (post-grasp lift): not contact
    assert not descent_stalled([0.30, 0.31, 0.32, 0.33, 0.34], 3, 0.003)
    # descent then hard stop (contact): >=2cm dropped within lookback, then flat
    zs = [0.50, 0.474, 0.454, 0.444, 0.4438, 0.4437, 0.4436]
    assert not descent_stalled(zs[:5], 3, 0.003)       # still settling into the stop
    assert descent_stalled(zs, 3, 0.003)
    # window not yet full
    assert not descent_stalled([0.46, 0.4438, 0.4437], 3, 0.003)


def _chunk(grips):
    a = np.zeros((len(grips), 8), dtype=np.float32)
    a[:, 7] = grips
    return a


def test_single_step_spike_suppressed():
    a = _chunk([0.62, 0.01, 0.60, 0.61])
    sup, run, close = debounce_gripper(a, 4, 2, held=True, open_run=0, close_val=1.0)
    assert [i for i, _ in sup] == [1] and np.isclose(sup[0][1], 0.01)
    assert np.isclose(a[1, 7], 0.62)       # replaced with the last executed close
    assert run == 0 and np.isclose(close, 0.61)   # counter reset by the trailing closes


def test_sustained_release_passes_from_step_n():
    a = _chunk([0.01, 0.01, 0.01, 0.01])
    sup, run, _ = debounce_gripper(a, 4, 2, held=True, open_run=0, close_val=0.62)
    assert [i for i, _ in sup] == [0]      # only the first open is delayed
    assert np.isclose(a[0, 7], 0.62) and np.isclose(a[1, 7], 0.01)
    assert run == 4


def test_release_continues_across_chunks():
    a1 = _chunk([0.62, 0.01])
    sup1, run, close = debounce_gripper(a1, 2, 2, held=True, open_run=0, close_val=1.0)
    assert [i for i, _ in sup1] == [1]
    a2 = _chunk([0.01, 0.01])
    sup2, run, _ = debounce_gripper(a2, 2, 2, held=True, open_run=run, close_val=close)
    assert sup2 == []                      # run carried over: no re-delay
    assert run == 3


def test_not_held_passes_through():
    a = _chunk([0.62, 0.01, 0.62])
    sup, _, _ = debounce_gripper(a, 3, 2, held=False, open_run=0, close_val=1.0)
    assert sup == [] and np.isclose(a[1, 7], 0.01)


def test_arm_channels_untouched():
    a = _chunk([0.62, 0.01])
    a[:, :7] = 0.5
    debounce_gripper(a, 2, 2, held=True, open_run=0, close_val=1.0)
    assert np.all(a[:, :7] == 0.5)


def test_beyond_executed_slice_untouched():
    a = _chunk([0.62, 0.01, 0.01, 0.01])
    sup, _, _ = debounce_gripper(a, 2, 2, held=True, open_run=0, close_val=1.0)
    assert [i for i, _ in sup] == [1]
    assert np.isclose(a[2, 7], 0.01)       # not executed this replan, left as planned


def test_disabled_is_identity():
    a = _chunk([0.62, 0.01])
    raw = a.copy()
    # hold_n=1 means an open passes immediately (run 1 >= 1): behaviourally off
    sup, _, _ = debounce_gripper(a, 2, 1, held=True, open_run=0, close_val=1.0)
    assert sup == [] and np.array_equal(a, raw)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
