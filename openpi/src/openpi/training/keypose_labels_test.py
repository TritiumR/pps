from __future__ import annotations

import numpy as np

from openpi.training import awe_waypoints
from openpi.training import keypose_labels


def _actions(command: list[float]) -> np.ndarray:
    actions = np.zeros((len(command), 8), dtype=np.float32)
    actions[:, :7] = np.arange(len(command), dtype=np.float32)[:, None]
    actions[:, 7] = command
    return actions


def test_binary_edges_preserve_legacy_behavior_exactly() -> None:
    actions = _actions([0, 0, 1, 1, 0, 0, 1])
    legacy = np.flatnonzero(np.diff(actions[:, 7]) != 0) + 1
    np.testing.assert_array_equal(keypose_labels.gripper_edges(actions), legacy)
    np.testing.assert_array_equal(keypose_labels.phase_end_indices(actions), [1, 1, 3, 3, 5, 5, 6])


def test_continuous_edges_backtrack_to_ramp_start() -> None:
    actions = _actions([0, 0, 0.1, 0.2, 0.2, 0.4, 0.6, 0.8, 1, 1, 0.9, 0.7, 0.4, 0.1, 0, 0])
    np.testing.assert_array_equal(keypose_labels.gripper_edges(actions), [2, 10])
    ends = keypose_labels.phase_end_indices(actions)
    assert ends[0] == 1
    assert ends[2] == 9
    assert ends[10] == 15


def test_partial_loosening_above_threshold_is_not_a_phase() -> None:
    actions = _actions([0, 0.2, 0.6, 0.9, 0.7, 0.55, 0.8, 0.4, 0.1, 0])
    np.testing.assert_array_equal(keypose_labels.gripper_edges(actions), [1, 7])


def test_one_frame_threshold_chatter_is_rejected() -> None:
    actions = _actions([0, 0.1, 0.51, 0.49, 0.2, 0, 0, 0.2, 0.6, 0.8, 1, 1])
    np.testing.assert_array_equal(keypose_labels.gripper_edges(actions), [7])


def test_waypoints_consume_corrected_phase_ends() -> None:
    actions = _actions([0, 0.2, 0.6, 0.9, 1, 1, 0.8, 0.4, 0.1, 0])
    indices = awe_waypoints.waypoint_target_indices(actions, k=2, stride=1)
    assert indices.shape == (10, 3)
    np.testing.assert_array_equal(indices[:, -1], keypose_labels.phase_end_indices(actions))
    assert np.all(np.diff(indices, axis=1) >= 0)
