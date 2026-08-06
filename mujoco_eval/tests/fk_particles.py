"""Regression test for the Feynman-Kac particle helpers.

Checks the three properties the steering loop depends on: ESS is exact at the uniform and
degenerate ends, systematic resampling is unbiased and never lands out of range, and the
potential ranks a particle nearer the proxy target above one further away.
"""

from __future__ import annotations

import torch

from mujoco_eval.steering import fk


def test_ess_endpoints():
    assert abs(fk.ess(torch.zeros(6)) - 6.0) < 1e-9
    spike = torch.full((6,), -1e4)
    spike[2] = 0.0
    assert abs(fk.ess(spike) - 1.0) < 1e-6


def test_resample_in_range_and_follows_weight():
    gen = torch.Generator().manual_seed(0)
    log_w = torch.tensor([-1e4, -1e4, 0.0, -1e4, -1e4, -1e4])
    idx = fk.systematic_resample(log_w, gen)
    assert idx.shape == (6,)
    assert int(idx.min()) >= 0 and int(idx.max()) <= 5
    assert torch.all(idx == 2), "all mass on particle 2 must produce 6 copies of it"

    # A 3:1 split should reproduce the ratio to within one slot (systematic, not multinomial).
    log_w = torch.log(torch.tensor([0.75, 0.25]))
    counts = torch.zeros(2)
    for _ in range(200):
        for j in fk.systematic_resample(log_w, gen).tolist():
            counts[j] += 1
    assert abs(counts[0] / counts.sum() - 0.75) < 0.05


def test_normalize_is_scale_free():
    g = torch.tensor([-1e6, -2e6, -3e6, -4e6])
    n = fk.normalize_logits(g)
    assert abs(float(n.mean())) < 1e-4 and abs(float(n.std()) - 1.0) < 1e-4
    # Ordering must survive: normalization is affine and the slope is positive.
    assert torch.all(n.argsort() == g.argsort())
    assert torch.all(fk.normalize_logits(torch.full((4,), 5.0)) == 0.0)


def test_potential_prefers_the_nearer_particle():
    target = torch.zeros(4, 8)
    x = torch.zeros(3, 4, 8)
    x[1, :, :7] = 0.1
    x[2, :, :7] = 1.0
    g = fk.log_potential(x, target, weight=1.0, dims=7)
    assert g[0] > g[1] > g[2], "log potential must decrease with distance from the proxy chunk"
    assert abs(float(g[0])) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all fk particle tests passed")
