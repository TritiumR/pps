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

import numpy as np
import torch


def chunk_costs(planner, real_chunks, ctx, bucket="task_no_ch", ranker="planner",
                keypose_row=None, action_rows=None):
    """Return [K] costs for [K, H, D] real joint chunks.

    bucket: 'total' | 'task' | 'feasibility' | 'task_no_ch' | 'task_no_nh'.
    ranker: 'planner' (the CompositeCost, default) | 'keypose' (surface-contact geometry on the
    keypose row -- see steering/keypose_cost.py for why a different KIND of signal, not fewer
    terms, is the surviving hypothesis). Falls back to the planner when the stage offers no
    target to measure against, so a missing target never silently ranks on a constant.
    """
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
