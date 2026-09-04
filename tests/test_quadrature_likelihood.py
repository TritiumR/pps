import math

import torch

from sim_free_mpc.quadrature_likelihood import (
    action_log_likelihood,
    estimate_log_partition_sobol,
    map_unit_to_action_chunks,
)


def test_unit_map_has_expected_constant_jacobian_away_from_limits():
    unit = torch.tensor([[[0.0] * 8, [1.0] * 8]], dtype=torch.float64)
    actions, log_jacobian = map_unit_to_action_chunks(
        unit,
        torch.zeros(7, dtype=torch.float64),
        joint_delta=0.1,
        joint_limits=torch.tensor([[-10.0, 10.0]] * 7, dtype=torch.float64),
    )
    assert torch.allclose(actions[0, 0, :7], torch.full((7,), -0.1, dtype=torch.float64))
    assert torch.allclose(actions[0, 1, :7], torch.zeros(7, dtype=torch.float64))
    assert torch.allclose(log_jacobian, torch.tensor([14 * math.log(0.2)], dtype=torch.float64))


def test_constant_energy_recovers_action_domain_volume():
    horizon = 2
    delta = 0.1
    constant = 3.25

    def cost(actions):
        return actions.new_full((actions.shape[0],), constant)

    estimate = estimate_log_partition_sobol(
        cost,
        torch.zeros(7),
        horizon=horizon,
        joint_delta=delta,
        temperature=0.5,
        num_points=256,
        num_scrambles=2,
        batch_size=64,
        joint_limits=torch.tensor([[-10.0, 10.0]] * 7),
    )
    expected = 7 * horizon * math.log(2 * delta) - constant / 0.5
    assert abs(estimate.log_partition - expected) < 1e-6
    assert abs(action_log_likelihood(constant, estimate.log_partition, 0.5) + 7 * horizon * math.log(2 * delta)) < 1e-6
