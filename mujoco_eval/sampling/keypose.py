"""Sample joint action chunks with auxiliary end-effector waypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .. import paths
paths.ensure_repo_on_path()

from sim_free_mpc.planner import SimFreeMPC
from vlm_dp.cost.terms import TERMS, CostInputs


@dataclass(frozen=True)
class KPConfig:
    """Configure keypose sampling and cost weights."""

    k: int = 5
    w_scale: float = 0.05
    align: float = 10.0
    subgoal_lift: float = 60.0
    subgoal_place: float = 60.0
    smooth: float = 20.0
    clear: float = 40.0
    warm_start: bool = True


def shift_polyline(start, pts, end, k):
    """Shift and resample waypoints along a receding-horizon path."""
    path = np.concatenate([np.asarray(start, dtype=np.float64).reshape(1, 3),
                           np.asarray(pts, dtype=np.float64).reshape(-1, 3)], axis=0)
    path[-1] = np.asarray(end, dtype=np.float64).reshape(3)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(path[-1:], k, axis=0)
    q = np.linspace(0.0, s[-1], k + 1)[1:]
    out = np.stack([np.interp(q, s, path[:, i]) for i in range(3)], axis=1)
    out[-1] = path[-1]
    return out


class KPCost:
    """Add keypose terms to an attached planner cost."""

    def __init__(self, inner, cfg: KPConfig):
        self._inner = inner
        self.kp_cfg = cfg
        self._w = None

    def set_waypoints(self, w_world):
        """Set candidate world-frame waypoints for the next cost call."""
        self._w = w_world

    def target(self, *a, **k):
        return self._inner.target(*a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *, real_actions, ee_pos, ee_quat=None, context):
        total = self._inner(real_actions=real_actions, ee_pos=ee_pos, ee_quat=ee_quat,
                            context=context)
        terms = dict(self._inner.last_terms)
        feas = self._inner.last_cost_feasibility
        task = self._inner.last_cost_task
        prior = self._inner.last_cost_prior

        ctx = context
        payload = ctx.get("payload")
        placing = payload is not None and ctx.get("place_target") is not None
        lifting = payload is not None and not placing
        w = self._w
        if w is not None and (lifting or placing):
            if w.shape[0] != ee_pos.shape[0]:
                raise ValueError(f"waypoint batch {w.shape[0]} != candidate batch {ee_pos.shape[0]}"
                                 " (kp_begin_replan / set_waypoints contract)")
            cfg = self.kp_cfg
            objects = ctx.get("objects", {})
            extents = {n: o["extents"] for n, o in objects.items() if "extents" in o}
            I_w = CostInputs(real_actions, w, None, ctx, extents, self._inner.geom)
            sub_fn = TERMS["lift_terminal"] if lifting else TERMS["place_terminal"]
            sub_w = cfg.subgoal_lift if lifting else cfg.subgoal_place
            kp_subgoal = sub_w * sub_fn(I_w)
            d2 = w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]
            kp_smooth = cfg.smooth * d2.pow(2).sum(dim=(1, 2))
            kp_clear = cfg.clear * TERMS["carry_clear"](I_w)
            kp_align = cfg.align * (ee_pos[:, -1] - w[:, 0]).pow(2).sum(dim=-1)
            for name, v in (("kp_subgoal", kp_subgoal), ("kp_smooth", kp_smooth),
                            ("kp_clear", kp_clear), ("kp_align", kp_align)):
                terms[name] = v.detach()
            total = total + kp_subgoal + kp_smooth + kp_clear + kp_align
            task = task + kp_subgoal.detach()
            prior = prior + (kp_smooth + kp_align).detach()
            feas = feas + kp_clear.detach()

        self.last_terms = terms
        self.last_cost_feasibility = feas
        self.last_cost_task = task
        self.last_cost_prior = prior
        self.last_stage = self._inner.last_stage
        return total


class KPPlanner(SimFreeMPC):
    """Extend SimFreeMPC with auxiliary waypoint rows."""

    def __init__(self, policy, config, kp: KPConfig):
        super().__init__(policy, config)
        self.kp = kp
        self._kp_line = None
        self._kp_warm = None
        self._kp_stage_key = None

    def reset_episode(self):
        super().reset_episode()
        self._kp_line = None
        self._kp_warm = None
        self._kp_stage_key = None

    def kp_begin_replan(self, context, stage_key):
        """Set the waypoint reference line and return an optional warm start."""
        eef = np.asarray(context["eef_pos"], dtype=np.float64).reshape(3)
        target = np.asarray(context["target"], dtype=np.float64).reshape(3)
        end = target.copy()
        payload = context.get("payload")
        objects = context.get("objects", {})
        if payload is not None and payload in objects:
            off = np.asarray(objects[payload]["pos"], dtype=np.float64).reshape(3) - eef
            end = target - off
        ks = np.arange(1, self.kp.k + 1, dtype=np.float64)[:, None] / float(self.kp.k)
        line = eef[None] + ks * (end[None] - eef[None])
        self._kp_line = torch.as_tensor(line, dtype=torch.float32)
        if stage_key != self._kp_stage_key:
            self._kp_stage_key = stage_key
            self._kp_warm = None
        if not self.kp.warm_start or self._kp_warm is None:
            return None
        shifted = shift_polyline(eef, self._kp_warm, end, self.kp.k)
        return torch.as_tensor((shifted - line) / self.kp.w_scale, dtype=torch.float32)

    def kp_extract_w(self, x_t):
        """Decode world-frame waypoints from the decision tensor."""
        w_model = x_t[0, -self.kp.k:, :3].detach().cpu()
        return (self._kp_line + w_model * self.kp.w_scale).numpy()

    def kp_finish_replan(self, x_t):
        """Store and return the final waypoint plan."""
        w = self.kp_extract_w(x_t)
        if self.kp.warm_start:
            self._kp_warm = w
        return w

    def _cost_active_samples(self, samples, x_template, active_dims, policy_inputs, context):
        if self._kp_line is None:
            raise RuntimeError("KPPlanner: kp_begin_replan was not called before this replan")
        k = self.kp.k
        w_world = (self._kp_line.to(device=samples.device, dtype=samples.dtype).unsqueeze(0)
                   + samples[:, -k:, :3] * self.kp.w_scale)
        self.cost.set_waypoints(w_world)
        return super()._cost_active_samples(samples[:, :-k, :], x_template[:, :-k, :],
                                            active_dims, policy_inputs, context)