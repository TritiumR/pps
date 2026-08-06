"""Regression tests for the AWE waypoint port.

The property that matters: waypoints land where linear interpolation FAILS, not at even spacing.
That is the whole reason the block is worth steering, and the reason a naive straight-line action
pull degraded monotonically on square.
"""

from __future__ import annotations

import numpy as np

from openpi.training import awe_waypoints as awe


def _trajectory(points, per_leg=10, dim=8):
    """Piecewise-linear path through `points` in dim-0, zeros elsewhere."""
    legs = []
    for start, end in zip(points[:-1], points[1:]):
        legs.append(np.linspace(start, end, per_leg, endpoint=False))
    legs.append(np.asarray([points[-1]]))
    line = np.concatenate(legs)
    out = np.zeros((len(line), dim), dtype=np.float32)
    out[:, 0] = line
    return out


def test_edge_error_is_zero_on_a_straight_line():
    states = _trajectory([0.0, 1.0], per_leg=20)
    assert awe.edge_error(states, 0, len(states) - 1) < 1e-6


def test_edge_error_detects_a_corner():
    states = _trajectory([0.0, 1.0, 0.0], per_leg=20)
    assert awe.edge_error(states, 0, len(states) - 1) > 0.1


def test_waypoints_land_on_corners_not_even_spacing():
    """A path with one sharp corner: the first waypoint should sit at the corner."""
    states = _trajectory([0.0, 1.0, 0.0], per_leg=20)
    errors = awe.edge_error_matrix(states)
    interior, worst = awe.fixed_k_interior(errors, 0, len(states) - 1, k=1)
    corner = 20
    assert abs(int(interior[0]) - corner) <= 2, f"waypoint at {interior[0]}, corner at {corner}"
    assert worst < awe.edge_error(states, 0, len(states) - 1), "must reduce the worst error"


def test_more_waypoints_never_increase_the_worst_error():
    states = _trajectory([0.0, 1.0, 0.0, 0.8], per_leg=12)
    errors = awe.edge_error_matrix(states)
    last = np.inf
    for k in (1, 2, 3, 4):
        _, worst = awe.fixed_k_interior(errors, 0, len(states) - 1, k=k)
        assert worst <= last + 1e-9, f"k={k} worsened the bottleneck"
        last = worst


def test_short_horizon_pads_with_the_endpoint():
    states = _trajectory([0.0, 1.0], per_leg=2)
    errors = awe.edge_error_matrix(states)
    interior, _ = awe.fixed_k_interior(errors, 0, len(states) - 1, k=5)
    assert len(interior) == 5, "block width must be fixed"
    assert int(interior[-1]) == len(states) - 1


def test_balanced_error_does_not_let_the_arm_drown_the_gripper():
    delta = np.zeros((4, 8))
    delta[:, 7] = 1.0                       # gripper only
    gripper_only = awe.balanced_error(delta)
    delta = np.zeros((4, 8))
    delta[:, 0] = 1.0                       # one arm joint only
    arm_only = awe.balanced_error(delta)
    # 1 gripper dim carries the same half-weight as all 7 arm dims, so it must score HIGHER
    # than a single arm joint moving by the same amount.
    assert float(gripper_only.mean()) > float(arm_only.mean())


def test_targets_have_fixed_shape_and_end_at_the_phase_end():
    actions = np.zeros((40, 8), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 1.0, 40)
    actions[20:, 7] = 1.0                   # gripper closes at 20 -> phase boundary
    targets = awe.waypoint_targets(actions, k=3, stride=2)
    assert targets.shape == (40, 4, 8)
    # A step inside the first phase must end on that phase's last row, not the episode's.
    assert np.allclose(targets[5, -1], actions[19]), "endpoint must be the PHASE end"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all AWE waypoint tests passed")
