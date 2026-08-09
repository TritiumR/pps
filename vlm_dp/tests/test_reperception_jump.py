"""Regression tests for rejecting impossible stale-object re-perception jumps."""
from __future__ import annotations

import numpy as np

from vlm_dp.world import SensedWorld


class _Perception:
    def __init__(self, seen):
        self.seen = seen
        self.distrust = set()

    def observe(self, env):
        return self.seen


class _Tracker:
    device = "cpu"

    def __init__(self):
        self.rebased = []

    def rebase(self, positions):
        self.rebased.append(dict(positions))


def _world(seen):
    world = SensedWorld.__new__(SensedWorld)
    world.perception = _Perception(seen)
    world.visual = _Tracker()
    world._held = None
    world._stale = {"egg"}
    world._pos = {"egg": np.array([1.0, 2.0, 3.0])}
    world._rot = {"egg": np.eye(3)}
    world._last_visual = {}
    world.names = ["egg"]
    world.relax_identity_when_stale = True
    return world


def test_impossible_jump_is_rejected_and_remains_stale():
    world = _world({"egg": np.array([1.5, 2.0, 3.0])})
    world.refresh(None)
    assert np.allclose(world._pos["egg"], [1.0, 2.0, 3.0])
    assert world.stale() == {"egg"}
    assert world.visual_position("egg") is None
    assert world.visual.rebased == []


def test_local_displacement_is_accepted_and_clears_stale():
    moved = np.array([1.04, 1.97, 3.01])
    world = _world({"egg": moved})
    world._reprime_tracker = lambda env: None
    world.refresh(None)
    assert np.allclose(world._pos["egg"], moved)
    assert world.stale() == set()
    assert np.allclose(world.visual_position("egg"), moved)
    assert len(world.visual.rebased) == 1


_TESTS = [
    test_impossible_jump_is_rejected_and_remains_stale,
    test_local_displacement_is_accepted_and_clears_stale,
]


def main():
    for test in _TESTS:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\nALL PASS ({len(_TESTS)} tests)")


if __name__ == "__main__":
    main()
