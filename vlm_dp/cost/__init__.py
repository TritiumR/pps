"""The VLM cost: CompositeCost (base_cost) over config-selected terms (terms)."""
import torch


def guard_cost(cost):
    """Wrap a cost so NaN/inf map to a large finite value before the sampler's softmax."""

    class _Guarded:
        def __init__(self, c):
            self._c = c

        def __call__(self, **kw):
            return torch.nan_to_num(self._c(**kw), nan=1e12, posinf=1e12, neginf=1e12)

        def target(self, *a, **k):
            return self._c.target(*a, **k)

    return _Guarded(cost)
