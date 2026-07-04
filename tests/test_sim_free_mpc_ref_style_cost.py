from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_ref_style import RefStyleCostWeights, RefStyleStateCost  # noqa: E402


def test_ref_style_cost_uses_requested_mean_and_terminal_terms():
    weights = RefStyleCostWeights(
        reach=2.0,
        terminal=3.0,
        orient=4.0,
        floor=5.0,
        smooth=6.0,
        local=7.0,
        yaw=0.0,
        straddle=0.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = RefStyleStateCost("custom", weights)
    real_actions = torch.zeros((1, 2, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 0.1], [1.0, 0.0, 0.3]]], dtype=torch.float32)
    tcp_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    context = {
        "eef_pos": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "z_floor": 0.2,
        "a_local": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "target_axis": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    }

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    dist_sq = torch.tensor([1.01, 0.09], dtype=torch.float32)
    expected = 2.0 * dist_sq.mean() + 3.0 * dist_sq[-1]
    expected = expected + 5.0 * torch.tensor([0.01, 0.0], dtype=torch.float32).mean()
    assert torch.allclose(cost, expected.view(1))


def test_ref_style_pick_cost_aligns_tip_height_to_object_center():
    weights = RefStyleCostWeights(
        reach=0.0,
        terminal=0.0,
        orient=0.0,
        floor=0.0,
        smooth=0.0,
        local=0.0,
        yaw=0.0,
        straddle=0.0,
        tip_z=10.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = RefStyleStateCost("weight", weights)
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]], [[0.0, 0.0, 1.05]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)}},
        "subtasks": {"grasp_pear": torch.tensor(False)},
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "a_local": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        "tcp_to_tip_z": 0.03,
    }

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]
