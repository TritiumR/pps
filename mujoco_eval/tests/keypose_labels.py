"""Regression test for the keypose labeler.

Locks the property the training targets depend on: every step in a phase points at that phase's
LAST step, and the phase boundaries fall exactly on the commanded-gripper edges.
"""

from __future__ import annotations

import numpy as np

from openpi.training import keypose_labels as kp


def _demo(command):
    """Build a [T, 8] action table whose joints encode the step index."""
    length = len(command)
    actions = np.zeros((length, 8), dtype=np.float32)
    actions[:, :7] = np.arange(length, dtype=np.float32)[:, None]
    actions[:, kp.GRIPPER_DIM] = np.asarray(command, dtype=np.float32)
    return actions


def test_edges_are_the_command_transitions():
    actions = _demo([0, 0, 0, 1, 1, 1, 1, 0, 0])
    assert kp.gripper_edges(actions).tolist() == [3, 7]


def test_every_step_points_at_its_phase_end():
    actions = _demo([0, 0, 0, 1, 1, 1, 1, 0, 0])
    # Phases are [0,3) -> ends at 2, [3,7) -> ends at 6, [7,9) -> ends at 8.
    assert kp.phase_end_indices(actions).tolist() == [2, 2, 2, 6, 6, 6, 6, 8, 8]

    targets = kp.keypose_targets(actions)
    assert targets[:, 0].tolist() == [2, 2, 2, 6, 6, 6, 6, 8, 8]
    assert targets.shape == actions.shape


def test_keypose_at_matches_the_table():
    actions = _demo([0, 0, 1, 1, 0])
    table = kp.keypose_targets(actions)
    for t in range(len(actions)):
        assert np.array_equal(kp.keypose_at(actions, t), table[t])


def test_no_edges_points_everything_at_the_last_frame():
    actions = _demo([0, 0, 0, 0])
    assert kp.phase_end_indices(actions).tolist() == [3, 3, 3, 3]


def test_empty_is_not_a_crash():
    assert kp.phase_end_indices(np.zeros((0, 8), dtype=np.float32)).tolist() == []


def test_keypose_is_further_away_than_the_action_chunk():
    """The point of the token: it is a distant goal, not the next few actions."""
    actions = _demo([0] * 40 + [1] * 40)
    horizon = 15
    step = 0
    chunk_reach = float(actions[step + horizon, 0] - actions[step, 0])
    keypose_reach = float(kp.keypose_at(actions, step)[0] - actions[step, 0])
    assert keypose_reach > chunk_reach, "keypose must reach past the action chunk"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all keypose labeler tests passed")
