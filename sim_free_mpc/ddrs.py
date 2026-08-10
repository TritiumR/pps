"""Direct Density-Ratio Steering: one learned scalar that tilts the MBD candidate weights.

The quantity is r_phi(a, o) ~= log p_expert(a | o) - log q_base(a | o), and the whole method rests
on WHICH q_base that is. The denominator is the base policy the robot actually runs -- MBD's
final clean candidates under their own softmax weights -- not the unweighted proposal cloud those
candidates were drawn from. A classifier trained against the cloud would spend its capacity
re-deriving the base cost it is supposed to be correcting; trained against the weighted
distribution it can only express what the base gets WRONG.

That is also why the tilt is applied at the FINAL clean level only: that is the one level whose
candidate distribution matches the negatives the classifier saw. Applying it earlier would score
a distribution the ratio was never fit against.

Inference is one forward pass on [K, ...] and adds nothing to the sampler's structure:

    log w_new = log w_base + gamma * r_phi,     w_new = softmax(log w_new)

gamma = 0 is exactly softmax(log w_base) = w_base, so the default-off path is the base bit-for-bit
(verified end-to-end, not argued: see agent_tests/_ddr_bitexact.py).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch import nn


class RatioNet(nn.Module):
    """Geometric context + flattened action chunk -> one logit.

    One architecture, one budget, no sweeps: 2 x 256 SiLU. The context is the same geometric
    state the planner's own cost reads (object poses + q0), so the ratio is conditioned on the
    observation in the representation the controller already trusts, rather than on pixels the
    rest of the stack never sees.
    """

    def __init__(self, ctx_dim: int, chunk_rows: int, chunk_dims: int = 8, width: int = 256):
        super().__init__()
        self.ctx_dim, self.chunk_rows, self.chunk_dims = ctx_dim, chunk_rows, chunk_dims
        d = ctx_dim + chunk_rows * chunk_dims
        self.net = nn.Sequential(nn.Linear(d, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 1))

    def forward(self, ctx, chunk):
        """ctx [N, C], chunk [N, R, D] -> [N] logit."""
        x = torch.cat([ctx, chunk.reshape(chunk.shape[0], -1)], dim=-1)
        return self.net(x).squeeze(-1)


def context_vector(context, q0, keys=("can", "bin")):
    """The planner's geometric context as a flat vector: object positions then q0.

    Deliberately the poses the cost already uses. Anything richer would make the ratio depend on
    information the controller cannot condition on at inference.
    """
    objects = context.get("objects", {}) or {}
    parts = []
    for name in keys:
        pos = (objects.get(name) or {}).get("pos")
        parts.append(np.zeros(3) if pos is None else np.asarray(pos, dtype=np.float64).reshape(3))
    parts.append(np.asarray(q0, dtype=np.float64).reshape(-1)[:7])
    return np.concatenate(parts).astype(np.float32)


class DDRS:
    """Inference-side wrapper the planner holds: normalisation, the net, and gamma."""

    def __init__(self, checkpoint, gamma=1.0, device="cpu"):
        blob = torch.load(str(checkpoint), map_location=device, weights_only=False)
        meta = blob["meta"]
        self.net = RatioNet(meta["ctx_dim"], meta["chunk_rows"], meta["chunk_dims"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval().to(device)
        self.gamma = float(gamma)
        self.device = device
        self.meta = meta
        self.ctx_mean = torch.as_tensor(meta["ctx_mean"], dtype=torch.float32, device=device)
        self.ctx_std = torch.as_tensor(meta["ctx_std"], dtype=torch.float32, device=device)
        self.chunk_mean = torch.as_tensor(meta["chunk_mean"], dtype=torch.float32, device=device)
        self.chunk_std = torch.as_tensor(meta["chunk_std"], dtype=torch.float32, device=device)
        self.last = None

    @staticmethod
    def meta_from(ctx, chunks):
        """Normalisation statistics, stored with the weights so inference cannot drift."""
        c = np.asarray(ctx, dtype=np.float64)
        a = np.asarray(chunks, dtype=np.float64)
        return {"ctx_dim": int(c.shape[1]), "chunk_rows": int(a.shape[1]),
                "chunk_dims": int(a.shape[2]),
                "ctx_mean": c.mean(0).tolist(), "ctx_std": (c.std(0) + 1e-6).tolist(),
                "chunk_mean": a.reshape(-1, a.shape[-1]).mean(0).tolist(),
                "chunk_std": (a.reshape(-1, a.shape[-1]).std(0) + 1e-6).tolist()}

    def log_ratio(self, real_chunks, context, q0):
        """[K] log-ratio for K decoded real action chunks at one observation."""
        chunk = torch.as_tensor(np.asarray(real_chunks), dtype=torch.float32, device=self.device)
        rows = self.meta["chunk_rows"]
        if chunk.shape[1] < rows:                      # never silently pad a short chunk
            raise ValueError(f"DDRS expects >= {rows} chunk rows, got {chunk.shape[1]}")
        chunk = chunk[:, :rows, : self.meta["chunk_dims"]]
        ctx = torch.as_tensor(context_vector(context, q0), dtype=torch.float32,
                              device=self.device).unsqueeze(0).expand(chunk.shape[0], -1)
        with torch.no_grad():
            r = self.net((ctx - self.ctx_mean) / self.ctx_std,
                         (chunk - self.chunk_mean) / self.chunk_std)
        return r.to(torch.float32)


def tilt_weights(weights, log_ratio, gamma):
    """log w_new = log w_base + gamma * r, renormalised. gamma=0 returns w_base unchanged."""
    if float(gamma) == 0.0:
        return weights
    logw = torch.log(torch.clamp(weights, min=1e-30)) + float(gamma) * log_ratio.to(weights.dtype)
    return torch.softmax(logw, dim=0)


def ess(w):
    return float(1.0 / torch.clamp(w.pow(2).sum(), min=1e-12))


def save(path, net, meta, extra=None):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "meta": meta, "extra": extra or {}}, str(path))
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {k: v for k, v in meta.items() if not isinstance(v, list)},
                   "extra": extra or {}}, f, indent=1)
    return path
