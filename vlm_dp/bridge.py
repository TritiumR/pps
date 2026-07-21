"""Attaches the VLM grounding and its CompositeCost to eval_steering's sim_free_mpc planner."""
from __future__ import annotations

import torch

from sim_common.envs.droid import DroidEnv
from vlm_dp.grounding import get_source
from vlm_dp.world import GTWorld
from vlm_dp.cost.base_cost import CompositeCost
from vlm_dp.stage import _capture_held, _should_advance
from vlm_dp.cost import guard_cost
from vlm_dp.context import build_context

# Styles the planner calls with tcp_pos=; CompositeCost only accepts ee_pos= (style "priority").
_TCP_STYLES = ("ref_style", "explore", "grasp_flow", "capsule_flow")


class VlmDpBridge:
    """Swaps the planner's cost and feeds it grounding-derived context each replan."""

    def __init__(self, ground, roles, cost_cfg, *, task_key="weight", device="cuda:0",
                 advance_persist=2, commit_hold=5, regress_persist=2, lift_tol=0.03):
        """Args:
          ground: grounding source -- ``gt`` | ``rekep_fake`` | ``rekep_real``.
          roles: ``{grasp_obj, place_obj, grasp_objs}`` from the task_prompts.json entry.
          cost_cfg: parsed cost YAML (``cost.terms`` / ``cost.geometry``).
          task_key: short task key for the rekep sources ("weight").
          advance_persist: replans the advance signal must hold before the stage bumps.
          commit_hold: forwarded to ``_should_advance``.
          regress_persist: replans the grasp flag must be gone before regressing.
          lift_tol: metres below the lift target that still counts as lifted.
        """
        self.ground_name = ground
        self.roles = dict(roles)
        self.terms = cost_cfg["cost"]["terms"]
        self.geom = cost_cfg["cost"]["geometry"]
        self.task_key = task_key
        self.device = device
        self.advance_persist = int(advance_persist)
        self.commit_hold = int(commit_hold)
        self.regress_persist = int(regress_persist)
        self.lift_tol = float(lift_tol)
        # Per-episode state, set in reset():
        self.env = None
        self.world = None
        self.grounding = None
        self.stage_idx = 0
        self.advance_streak = 0
        self.lost_streak = 0
        self.held_offset = None
        self.plan_ref = None

    def attach_cost(self, mpc):
        """Replace the planner's cost. Call once; the cost is stateless across episodes."""
        style = mpc.config.cost_style
        if style != "priority" or style in _TCP_STYLES:
            raise ValueError(
                f"--vlm_cost requires --mpc_cost priority; got cost_style={style!r}. CompositeCost "
                f"has no tcp_pos= parameter, and sim_free_mpc/planner.py calls {_TCP_STYLES} "
                f"with tcp_pos=.")
        mpc.cost = guard_cost(CompositeCost(self.terms, self.geom))

    def reset(self, raw_env):
        """Re-bind and re-ground. Must run after every ``env.reset()``."""
        self.env = DroidEnv.attach(raw_env, device=self.device)
        self.world = GTWorld(raw_env)                      # raw env; grounding needs the wrapper
        src = get_source(self.ground_name, task_key=self.task_key, perception=None, **self.roles)
        self.grounding = src.ground(self.env, self.world)
        self.stage_idx = 0
        self.advance_streak = 0
        self.lost_streak = 0
        self.held_offset = _capture_held(self.env, self.grounding,
                                         self.grounding.stages[0].held_idx)
        self.plan_ref = None

    def stage(self):
        return self.grounding.stages[self.stage_idx]

    def context(self, raw_env, obs):
        """World-frame cost context for the current stage (one per replan)."""
        return build_context(raw_env, obs, self.grounding, self.stage(),
                             plan_ref=self.plan_ref, held_offset=self.held_offset)

    def observe_plan(self, actions, executed_steps):
        """Record the decoded chunk for the next replan's consistency term, shifted by what executes."""
        joints = torch.as_tensor(actions, dtype=torch.float32, device=self.device)[..., :7]
        k = min(int(executed_steps), joints.shape[0])
        self.plan_ref = torch.cat([joints[k:], joints[-1:].expand(k, 7)], dim=0)

    def advance(self, flags):
        """Advance the stage on its progress signal; regress to the grasp stage on payload loss."""
        stage = self.stage()
        placed = stage.gripper == "place" and bool(stage.done())
        lost = (stage.payload is not None and not placed
                and not flags.get(f"grasp_{stage.payload}", False))
        self.lost_streak = self.lost_streak + 1 if lost else 0
        if self.lost_streak >= self.regress_persist:
            back = self._grasp_stage_idx(stage.payload)
            if back is not None and back != self.stage_idx:
                print(f"[vlm_dp] REGRESS -> {back}: {self.grounding.stages[back].name} "
                      f"(lost {stage.payload})", flush=True)
                self.stage_idx = back
                self.advance_streak = 0
                self.lost_streak = 0
                self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
                return
        if stage.gripper == "hold" and stage.payload is not None:
            # Lift stages advance at the full lift height, not the grounding's small rise confirm.
            z = float(self.world.object_pose(stage.payload)[0][2])
            signal = z >= float(stage.target()[2]) - self.lift_tol
        else:
            signal = bool(_should_advance(stage, flags, 0, self.commit_hold))
        self.advance_streak = self.advance_streak + 1 if signal else 0
        if self.advance_streak >= self.advance_persist and self.stage_idx + 1 < len(self.grounding.stages):
            self.stage_idx += 1
            self.advance_streak = 0
            self.lost_streak = 0
            self.held_offset = _capture_held(self.env, self.grounding, self.stage().held_idx)
            print(f"[vlm_dp] stage -> {self.stage_idx}: {self.stage().name}", flush=True)

    def _grasp_stage_idx(self, payload):
        """Index of the ``close`` stage that grasps ``payload``, or None."""
        for i, s in enumerate(self.grounding.stages):
            if s.gripper == "close" and s.grasp_obj == payload:
                return i
        return None
