"""numpy->torch shim for evaluating ReKep constraint code batched over sampler candidates.

ReKep's ``ConstraintGenerator`` emits numpy functions ``fn(end_effector(3,), keypoints(K,3)) -> cost``
that a loader ``exec``s in a sandbox whose only injected name is ``np``. The sampler needs the same
function over a batch of candidate TCPs ``[K,H,3]``. ``TorchNumpyShim`` is a torch-backed stand-in for
that ``np`` surface; its reductions default to ``axis=-1``, so a constraint written for a ``(3,)`` vector
returns a scalar on ``(3,)`` and ``[K,H]`` on ``[K,H,3]`` unchanged. Prompts and generated code untouched.
"""
import os

import torch


class _Linalg:
    def __init__(self, dev):
        self._dev = dev

    def norm(self, x, axis=-1, keepdims=False):
        # Coerce to float: a constraint's integer-literal vector (e.g. np.array([0,0,1])) comes through
        # as a Long tensor, which torch's vector_norm rejects.
        if torch.is_tensor(x) and not x.is_floating_point():
            x = x.to(torch.float32)
        return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims)


class TorchNumpyShim:
    """A torch-backed stand-in for ``numpy`` covering the ops ReKep constraints use.

    Reductions default to ``axis=-1`` so a constraint written for a ``(3,)`` vector also
    evaluates correctly on a batched ``[...,3]`` tensor. Unknown attributes raise, so an
    unsupported op surfaces as a clear error to extend (rather than a silent miss).
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self.linalg = _Linalg(device)
        self.pi = torch.pi

    # --- construction ---
    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, torch.float32)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def array(self, obj, dtype=None):
        if isinstance(obj, (list, tuple)):
            # All-int list -> long tensor (numpy infers int64), so it can index keypoints.
            if all(isinstance(o, int) and not isinstance(o, bool) for o in obj):
                return torch.tensor(list(obj), device=self.device, dtype=torch.long)
            elems = [self.array(o) for o in obj]
            if any(isinstance(e, torch.Tensor) and e.ndim > 0 for e in elems):
                return torch.stack([self._t(e) for e in elems])
            return torch.stack([self._t(e).reshape(()) for e in elems])
        return self._t(obj)

    asarray = array

    def stack(self, seq, axis=0):
        return torch.stack([self._t(s) for s in seq], dim=axis)

    def concatenate(self, seq, axis=0):
        return torch.cat([self._t(s) for s in seq], dim=axis)

    # --- vector algebra (last-axis by default) ---
    def dot(self, a, b):
        return (self._t(a) * self._t(b)).sum(dim=-1)

    def cross(self, a, b):
        return torch.linalg.cross(self._t(a), self._t(b), dim=-1)

    # --- elementwise / reductions ---
    def abs(self, x):
        return torch.abs(self._t(x))

    def sqrt(self, x):
        return torch.sqrt(self._t(x))

    def sign(self, x):
        return torch.sign(self._t(x))

    def arccos(self, x):
        return torch.arccos(torch.clamp(self._t(x), -1.0, 1.0))

    def arctan2(self, y, x):
        return torch.arctan2(self._t(y), self._t(x))

    def maximum(self, a, b):
        return torch.maximum(self._t(a), self._t(b))

    def minimum(self, a, b):
        return torch.minimum(self._t(a), self._t(b))

    def clip(self, x, lo, hi):
        return torch.clamp(self._t(x), float(lo), float(hi))

    def mean(self, x, axis=None):
        x = self._t(x)
        return x.mean() if axis is None else x.mean(dim=axis)

    def sum(self, x, axis=None):
        x = self._t(x)
        return x.sum() if axis is None else x.sum(dim=axis)


def load_torch_constraints(txt_path, get_grasping_cost_fn, shim):
    """Like ``rekep.utils.load_functions_from_txt`` but injects ``shim`` as ``np``.

    Same sandbox contract as upstream (only ``np`` and ``get_grasping_cost_by_keypoint_idx``
    are available to the constraint code); the sole change is the numpy backend. Returns the
    list of constraint callables in file order, or ``[]`` if the file is missing.
    """
    if txt_path is None or not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        functions_text = f.read()
    gvars = {"np": shim, "get_grasping_cost_by_keypoint_idx": get_grasping_cost_fn}
    lvars = {}
    exec(functions_text, gvars, lvars)  # noqa: S102 - sandboxed VLM constraint code
    return list(lvars.values())


def make_torch_constraint(callables):
    """Sum a stage's loaded constraint callables into ``constraint_fn(pos[K,H,3], kp[N,3])->[K,H]``.

    Each callable is a GPT ``fn(end_effector, keypoints)``; with the shim it returns ``[K,H]``
    (or a scalar for keypoint-only / grasping-cost terms, broadcast to ``[K,H]``). The sum is
    the stage cost the DIAL sampler minimizes (lower = closer to satisfying the sub-goal).
    """

    def constraint_fn(pos, keypoints):
        total = None
        for fn in callables:
            c = fn(pos, keypoints)
            if not isinstance(c, torch.Tensor):
                c = torch.as_tensor(float(c), device=pos.device, dtype=pos.dtype)
            if c.ndim == 0:
                c = c.expand(pos.shape[0], pos.shape[1])
            total = c if total is None else total + c
        if total is None:
            return torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
        return total

    return constraint_fn
