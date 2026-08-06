"""VLS-style guidance: steer a LEARNED policy with a differentiable geometric objective.

The inverse of everything else in this package. Elsewhere a learned proxy steers the MBD base;
here the proxy IS the base and the geometric objective is the steering signal, applied as a
normalised gradient on the clean prediction at every denoising level.

Three choices, each of which removes a failure we measured:

  * NORMALISED gradient. `g / (|g| + eps)` decouples guidance magnitude from the objective's
    scale. Our additive arms had |s_task|/|s_base| running 0.10 -> 1.46 across levels and a
    cosine of -0.687; with a unit-norm gradient there is no second field to be commensurate with.

  * SIGMOID gate on task PROGRESS, not on a fixed gamma. strength = sigmoid(-k(p - x0)) where
    p = 1 - reward/reward_at_stage_start. Full strength when far, ~0 once past x0 of the way.
    A single global gamma was measured flat and then harmful (square 0.4 -> 39, 0.7 -> 34); the
    gate is the state-dependent coefficient that analysis said was needed.

  * OBJECTIVE ONLY, xyz only, executable rows only. The policy owns feasibility, collision and
    the gripper. Our own cost mixes objective with 15+ regularisers, and the 26-term version
    anti-aligns at -0.687 where the 1-term ReKep constraint reaches -0.208.

Not a product of experts and not claimed to sample any particular distribution: it is gradient
guidance on the policy's own clean prediction, which is what VLS does.
"""
from __future__ import annotations

import math

import torch


class VLSGuidance:
    """Normalised-gradient guidance with a progress gate, over a differentiable objective."""

    def __init__(self, objective, *, scale=1.0, sigmoid_k=12.0, sigmoid_x0=0.7,
                 action_rows=None, start_frac=1.0 / 3.0):
        self.objective = objective          # (ee_pos [B,T,3]) -> reward, higher is better
        self.scale = float(scale)
        self.sigmoid_k = float(sigmoid_k)
        self.sigmoid_x0 = float(sigmoid_x0)
        self.action_rows = action_rows
        self.start_frac = float(start_frac)
        self._stage_init = None
        self.target = None          # set per replan from the stage context
        self.trace = []

    def reset_stage(self):
        """Forget the stage's reference reward; progress is measured relative to it."""
        self._stage_init = None

    def _strength(self, reward):
        """Sigmoid gate on fractional progress toward the stage goal."""
        if self._stage_init is None or self._stage_init > -1e-9:
            self._stage_init = float(reward)
            return 1.0, 0.0
        p = 1.0 - (float(reward) / self._stage_init)
        p = max(0.0, min(1.2, p))
        return 1.0 / (1.0 + math.exp(self.sigmoid_k * (p - self.sigmoid_x0))), p

    def gradient(self, x0_chunk, fk):
        """Return (unit-norm gradient wrt the chunk, reward, gate, progress).

        x0_chunk: [B, H, D] clean joint chunk in REAL units. Only the arm columns carry gradient;
        the near-binary gripper column is left alone -- it dominates an L1 pull and, thresholded,
        is where the Isaac run's grasp stalled.
        """
        x = x0_chunk.detach().clone().requires_grad_(True)
        rows = self.action_rows or x.shape[1]
        with torch.enable_grad():
            ee = fk.forward(x[:, :rows, :7]).ee_pos
            reward = self.objective(ee)
            r = reward.sum() if reward.ndim > 0 else reward
            g = torch.autograd.grad(r, x, retain_graph=False, create_graph=False)[0]
        n = torch.linalg.vector_norm(g)
        g = g / (n + 1e-8) if float(n) > 1e-8 else g
        g[:, :, 7:] = 0.0
        strength, progress = self._strength(float(r.detach()))
        self.trace.append({"reward": round(float(r.detach()), 5),
                           "progress": round(progress, 4),
                           "gate": round(strength, 4),
                           "grad_norm": round(float(n), 5)})
        return g, float(r.detach()), strength, progress

    def apply(self, x0_chunk, fk, alpha_bar=None):
        """Return the guided clean chunk: ascend the reward, gated by progress."""
        g, _r, strength, _p = self.gradient(x0_chunk, fk)
        s = self.scale * strength
        if alpha_bar is not None:                    # DDPM score->epsilon conversion, as in VLS
            s *= math.sqrt(max(1.0 - float(alpha_bar), 0.0))
        return x0_chunk + s * g                      # objective is a REWARD: ascend
