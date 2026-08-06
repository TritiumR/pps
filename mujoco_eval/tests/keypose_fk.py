"""Regression tests for keypose FK steering.

Locks the four properties the steering loop depends on: the KL cap matches the stated formula
(and disables cleanly at 0), the guided keypose is a genuine cost-weighted mean, the action pull
lands on the straight joint path, and the proposal cloud stays inside its bounds.
"""

from __future__ import annotations

import math

import torch

from mujoco_eval.steering import keypose_fk as kfk


def test_kl_cap_matches_the_formula():
    delta = torch.tensor([3.0, 4.0])          # ||delta|| = 5
    sigma = 1.0
    raw = 0.5 * 25.0                          # 12.5
    capped, applied = kfk.kl_capped(delta, sigma, max_kl=2.0)
    scale = math.sqrt(2.0 / raw)
    assert torch.allclose(capped, scale * delta, atol=1e-6)
    assert abs(applied - 2.0) < 1e-6, "applied KL must saturate exactly at the limit"


def test_kl_cap_is_a_noop_below_the_limit():
    delta = torch.tensor([0.1, 0.0])
    capped, applied = kfk.kl_capped(delta, sigma=1.0, max_kl=100.0)
    assert torch.allclose(capped, delta)
    assert applied < 0.01


def test_zero_kl_disables_guidance():
    delta = torch.tensor([5.0, -2.0])
    capped, applied = kfk.kl_capped(delta, sigma=1.0, max_kl=0.0)
    assert torch.all(capped == 0.0) and applied == 0.0


def test_guided_keypose_is_a_cost_weighted_mean():
    proposals = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # One proposal far cheaper than the rest -> the mean must sit essentially on it.
    guided, potential = kfk.guided_from_costs(proposals, [10.0, 0.0, 10.0], temperature=0.1)
    assert torch.allclose(guided, proposals[1], atol=1e-3)
    assert potential < 0.0, "log-potential of positive costs is negative"

    # Equal costs -> the plain average, and the potential equals -cost/T.
    flat, pot = kfk.guided_from_costs(proposals, [1.0, 1.0, 1.0], temperature=0.5)
    assert torch.allclose(flat, proposals.mean(dim=0), atol=1e-6)
    assert abs(pot - (-1.0 / 0.5)) < 1e-5


def test_action_pull_lands_on_the_straight_path():
    rows, keypose_row = 5, 4
    chunk = torch.zeros(rows + 1, 8)
    chunk[keypose_row, :7] = 1.0                     # goal one unit away in every arm joint
    q0 = torch.zeros(8)
    full = kfk.action_l1_pull(chunk, keypose_row, q0, step_size=1.0)
    # step_size 1.0 replaces the rows outright with the interpolated path.
    expected = torch.linspace(1.0 / keypose_row, 1.0, keypose_row)
    assert torch.allclose(full[:keypose_row, 0], expected, atol=1e-6)
    assert torch.allclose(full[keypose_row], chunk[keypose_row]), "keypose row must not move"


def test_action_pull_off_is_identity():
    chunk = torch.randn(6, 8)
    assert torch.equal(kfk.action_l1_pull(chunk.clone(), 4, torch.zeros(8), 0.0), chunk)


def test_proposals_respect_bounds_and_centre():
    gen = torch.Generator().manual_seed(0)
    centre = torch.zeros(8)
    cloud = kfk.sample_proposals(centre, scale=1.0, count=512, generator=gen,
                                 lower=-0.5, upper=0.5)
    assert cloud.shape == (512, 8)
    assert float(cloud.min()) >= -0.5 and float(cloud.max()) <= 0.5
    assert abs(float(cloud.mean())) < 0.05, "cloud must stay centred on the policy keypose"


def test_resample_follows_the_potential():
    gen = torch.Generator().manual_seed(0)
    idx = kfk.resample_parents([-1e4, 0.0, -1e4, -1e4], count=4, generator=gen)
    assert idx.shape[0] == 4 and torch.all(idx == 1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all keypose_fk tests passed")
