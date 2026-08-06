"""Prove the planner<->proxy space bridge round-trips under both action normalizations.

`--steer policy_base` runs the PROXY's own reverse chain, so every denoise level crosses the bridge
in both directions -- unlike `additive`, which only maps one way for the addend. A bridge that is
not an exact inverse therefore accumulates error over the chain instead of appearing once.

    python -m mujoco_eval.tests.space_bridge
"""
from __future__ import annotations

import sys
import types

import numpy as np

from ..steering.proxy import AdditiveScoreSteering

_TOL = 1e-4          # float32 round-trip noise is ~1e-6; this leaves headroom without hiding a bug


class _Stats:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std


def _build(action_norm: str, horizon: int = 15, dim: int = 8):
    """An AdditiveScoreSteering wired to synthetic stats of the given normalization."""
    rng = np.random.default_rng(0)
    if action_norm == "demo_delta":
        std_rows = np.abs(rng.normal(0.05, 0.02, (horizon, dim))).astype(np.float32) + 1e-3
        mean_rows = np.zeros((horizon, dim), np.float32)
        mean_rows[:, 7] = 0.5
        info = dict(action_horizon=horizon, action_dim=dim, action_norm="demo_delta",
                    action_mean_rows=mean_rows.tolist(), action_std_rows=std_rows.tolist())
    else:
        info = dict(action_horizon=horizon, action_dim=dim, action_norm="droid_quantile",
                    action_q01=(-np.ones(dim)).tolist(), action_q99=np.ones(dim).tolist())
    policy = types.SimpleNamespace(_metadata={
        "output_norm_stats": {"actions": _Stats(
            np.zeros(dim, np.float32),
            np.abs(rng.normal(0.08, 0.02, dim)).astype(np.float32) + 1e-3)},
        "use_quantile_norm": False})
    return AdditiveScoreSteering(types.SimpleNamespace(ready_info=info),
                                 gamma=0.4, policy=policy, horizon=horizon)


def main() -> int:
    failed = 0
    x = np.random.default_rng(1).standard_normal((7, 15, 8)).astype(np.float32)
    for norm in ("demo_delta", "droid_quantile"):
        bridge = _build(norm)
        err = float(np.abs(bridge._to_planner_space(bridge._to_proxy_space(x)) - x).max())
        ok = err < _TOL
        failed += not ok
        print(f"  {norm:16s} round-trip max abs err = {err:.3e}   {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
