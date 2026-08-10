"""Score a batch of candidate action chunks under the planner's cost.

`verify._rank_cost_of` already does this for ONE chunk (FK -> planner.cost -> role bucket).
`CompositeCost` is batched -- it returns [K] for [K, H, D] input -- so the batch version needs
only the FK and the role selection, not a loop.

Buckets mirror --verify_rank. `task_no_ch` exists because E4 measured `carry_hold` carrying ~98%
of the base cost's inversion against working behaviour (+22 median at the insert stage, every
other term <= 0.4); it is needed IN the sampler but is a bad judge, so a steering signal that
ranks proposals should drop it.
"""

from __future__ import annotations

import types

import numpy as np
import torch


def keypose_world_pose(planner, real_chunks, ctx, keypose_row=None):
    """World-frame TCP pose at the keypose row: (ee_pos [K,3], ee_quat [K,4]).

    Same FK path `chunk_costs` uses for the planner ranker, sliced to the one row that the
    proposal cloud perturbs.
    """
    chunks = torch.as_tensor(np.asarray(real_chunks), dtype=torch.float32)
    if chunks.ndim == 2:
        chunks = chunks.unsqueeze(0)
    row = int(chunks.shape[1]) - 1 if keypose_row is None else int(keypose_row)
    fk = planner.fk.forward(chunks[..., :7])
    ee_pos, ee_quat = fk.ee_pos[:, row], fk.ee_quat[:, row]
    root_pos, root_quat = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if root_pos is not None and root_quat is not None:
        from sim_free_mpc.fk import transform_points_wxyz
        ee_pos = transform_points_wxyz(
            torch.as_tensor(root_pos, dtype=ee_pos.dtype),
            torch.as_tensor(root_quat, dtype=ee_pos.dtype), ee_pos.unsqueeze(1)).squeeze(1)
    return ee_pos, ee_quat


def rekep_subgoal_costs(planner, real_chunks, ctx, keypose_row=None):
    """Return [K] ranking costs = the plan's stage sub-goal, evaluated at the keypose.

    The program's single-authority rule: the controller must not invent task semantics. The
    VLM/ReKep plan DECLARES the stage sub-goal; a proposal ranker may only evaluate it. So this
    is Cory's C_subgoal(w^K) verbatim -- the compiled stage constraint read at the terminal
    keypose, with the held keypoints riding the candidate gripper pose through the in-hand offset
    grounded at grasp.

    It carries no hold, contact or tracking term: those are controller semantics, and the
    composite that shipped them measured as a weak judge (E4's carry_hold inversion). Evaluation
    is delegated to the registered `rekep_keypose` term rather than re-derived here, so the
    ranker and the cost term cannot drift apart.

    Returns None when the stage declares no sub-goal, so the caller falls back rather than
    ranking every proposal on a constant.
    """
    if ctx.get("constraint") is None:
        return None
    from vlm_dp.cost.terms import TERMS

    ee_pos, ee_quat = keypose_world_pose(planner, real_chunks, ctx, keypose_row=keypose_row)
    inputs = types.SimpleNamespace(
        real_actions=None, ee_pos=ee_pos.unsqueeze(1), ee_quat=ee_quat.unsqueeze(1),
        context=ctx, extents={}, geom=getattr(planner.cost, "geom", None))
    return TERMS["rekep_keypose"](inputs).reshape(-1)


def chunk_costs(planner, real_chunks, ctx, bucket="task_no_ch", ranker="planner",
                keypose_row=None, action_rows=None):
    """Return [K] costs for [K, H, D] real joint chunks.

    bucket: 'total' | 'task' | 'feasibility' | 'task_no_ch' | 'task_no_nh'.
    ranker: 'planner' (the CompositeCost, default) | 'keypose' (surface-contact geometry on the
    keypose row -- see steering/keypose_cost.py for why a different KIND of signal, not fewer
    terms, is the surviving hypothesis) | 'rekep_subgoal' (the plan's OWN stage sub-goal on the
    keypose row -- see rekep_subgoal_costs). Falls back to the planner when the stage offers no
    target to measure against, so a missing target never silently ranks on a constant.
    """
    if ranker == "rekep_subgoal":
        out = rekep_subgoal_costs(planner, real_chunks, ctx, keypose_row=keypose_row)
        if out is not None:
            return out
    if ranker == "keypose":
        from .keypose_cost import proposal_costs

        row = (int(keypose_row) if keypose_row is not None
               else int(np.asarray(real_chunks).shape[1]) - 1)
        out = proposal_costs(planner, real_chunks, ctx, row, action_rows=action_rows)
        if out is not None:
            return out
    chunks = torch.as_tensor(np.asarray(real_chunks), dtype=torch.float32)
    if chunks.ndim == 2:
        chunks = chunks.unsqueeze(0)
    fk = planner.fk.forward(chunks[..., :7])
    ee_pos, ee_quat = fk.ee_pos, fk.ee_quat
    root_pos, root_quat = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if root_pos is not None and root_quat is not None:
        from sim_free_mpc.fk import transform_points_wxyz
        ee_pos = transform_points_wxyz(
            torch.as_tensor(root_pos, dtype=ee_pos.dtype),
            torch.as_tensor(root_quat, dtype=ee_pos.dtype), ee_pos)

    total = planner.cost(real_actions=chunks, ee_pos=ee_pos, ee_quat=ee_quat, context=ctx)
    if bucket == "total":
        return torch.as_tensor(total).reshape(-1)

    role = "feasibility" if bucket == "feasibility" else "task"
    value = getattr(planner.cost, f"last_cost_{role}", None)
    if value is None:                                  # cost has no role split
        return torch.as_tensor(total).reshape(-1)
    out = torch.as_tensor(value).reshape(-1).clone()

    drop = {"task_no_ch": "carry_hold", "task_no_nh": "not_hold"}.get(bucket)
    if drop is not None:
        term = (getattr(planner.cost, "last_terms", None) or {}).get(drop)
        if term is not None:
            out = out - torch.as_tensor(term).reshape(-1)
    return out
