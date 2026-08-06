"""Expert proposes, base verifies -- the composition read from the other direction.

The expert is the stronger policy here (11/50 against the base's 5/50), so the better-posed
combination starts from it and lets the base act as a constraint rather than a judge.

Ranking is on the TERM_ROLES *feasibility* bucket only (collision, floor, keepout) -- the base
answers "is this physically admissible", never "is this the right thing to do". That deliberately
excludes `not_hold`, a task term whose sigma reads demonstration-scale motion as near-stasis and
charges a demonstration 32x what it charges the base's own candidates.
"""
from __future__ import annotations

import numpy as np
import torch


def _rank_cost_of(planner, real_chunk, inputs, ctx, rank="feasibility"):
    """Ranking cost of one REAL joint chunk, or None if the cost has no role split.

    rank picks the bucket the base judges on:
      feasibility  keepout only (clear, floor, carry_clear, release_rise_first). Measured to have
                   no discrimination -- expert-minus-base median 0.0000 over 820 chunks, so no
                   gate ever fires and the arm degenerates to the expert alone.
      task         the phase attractors (reach, close_gripper, place_reach, ...). These vary with
                   plan quality, and WHICH of them is live is what makes the hand-off implicitly
                   phase-adaptive: grasp terms early, place terms late, no hand-placed boundary.
      task_no_nh   task minus not_hold. The sampler needs not_hold (deleting it gives base 0/25),
                   but its sigma charges demonstration-scale motion ~32x what it charges the
                   base's own candidates, so as a JUDGE of the expert it penalises the right
                   behaviour. Keep it in the sampler, drop it from the veto.
    """
    chunk = torch.as_tensor(np.asarray(real_chunk), dtype=torch.float32).unsqueeze(0)
    joints = chunk[..., :7]
    fk = planner.fk.forward(joints)
    ee_pos, ee_quat = fk.ee_pos, fk.ee_quat
    root_pos, root_quat = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if root_pos is not None and root_quat is not None:
        from sim_free_mpc.fk import transform_points_wxyz
        root_pos = torch.as_tensor(root_pos, dtype=ee_pos.dtype)
        root_quat = torch.as_tensor(root_quat, dtype=ee_pos.dtype)
        ee_pos = transform_points_wxyz(root_pos, root_quat, ee_pos)
    planner.cost(real_actions=chunk, ee_pos=ee_pos, ee_quat=ee_quat, context=ctx)
    bucket = "feasibility" if rank == "feasibility" else "task"
    value = getattr(planner.cost, f"last_cost_{bucket}", None)
    if value is None:
        return None
    out = float(torch.as_tensor(value).reshape(-1)[0])
    terms = {
        k: round(float(torch.as_tensor(v).reshape(-1)[0]), 4)
        for k, v in (getattr(planner.cost, "last_terms", None) or {}).items()
    }
    if rank == "task_no_nh" and "not_hold" in terms:
        out -= terms["not_hold"]
    if rank == "task_no_ch" and "carry_hold" in terms:
        # E4 measured carry_hold as ~98% of the judge inversion (+22 median at stage 2): its
        # quasi-static at-the-seat release model charges the expert's working insertion. Keep it
        # in the sampler, drop it from the veto.
        out -= terms["carry_hold"]
    return out, terms


def verify_chunk(planner, env, ctx, args, base_plan, steer, inputs):
    """Take the expert's chunk unless the base scores it worse than its own by > --verify_gate.

    Returns (plan, record). The gate is a slack, not a tie-break: a small deficit is accepted,
    because the expert is the better task policy (19/50 against the base's 7/50) and the base is
    only there to catch the cases it handles better -- which under --verify_rank task is the grasp
    phase, where the base clears stage 0 on 96% of episodes against the expert's 70%.
    """
    rank = getattr(args, "verify_rank", "feasibility")
    expert = steer.expert_chunk()
    r_exp = _rank_cost_of(planner, expert, inputs, ctx, rank)
    r_base = _rank_cost_of(planner, base_plan, inputs, ctx, rank)
    if r_exp is None or r_base is None:            # cost has no role split -> expert drives
        return expert, {"verify": "expert", "verify_split": False}
    (f_exp, t_exp), (f_base, t_base) = r_exp, r_base
    take_expert = f_exp <= f_base + float(args.verify_gate)
    return (expert if take_expert else base_plan), {
        "verify": "expert" if take_expert else "base",
        "verify_feas_expert": round(f_exp, 4),
        "verify_feas_base": round(f_base, 4),
        # Per-term breakdown of BOTH chunks: names which term carries the deficit (E4).
        "verify_terms_expert": t_exp,
        "verify_terms_base": t_base,
    }
