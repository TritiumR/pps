"""Rank keypose proposals by contact geometry, in the shape Cory's PickBallCost uses.

Why this exists, measured rather than assumed:

`chunk_costs` ranks proposals with the planner's 27-term CompositeCost, and the resulting spread
across a 128-proposal cloud is 4.2% relative at the default proposal width -- a near-uniform
Feynman-Kac softmax, i.e. guidance in name only. The obvious explanation, that 27 terms dilute a
one-row perturbation, was TESTED AND REFUTED: under `--ground rekep`, where every stage carries a
live constraint, the keypose term alone separated 5.8% against the full composite's 19.7% at
matched width. Fewer terms is not the answer.

What survives is a difference in KIND, not count. `rekep_keypose` is point-to-point --
||TCP - keypoint||. Cory's task cost is surface-to-surface: sample ~24 points per hand geom, keep
those whose normals face the palm, and measure their distance to the ball's surface. A point
distance is flat in gripper orientation and finger opening; a surface distance is not, and those
are exactly the degrees of freedom a keypose proposal perturbs.

So this scores a proposal by where the FINGER SURFACES end up relative to the object's surface,
evaluated on the keypose row only. It is a ranking signal, never the sampler's objective: the base
cost is untouched, so every existing base and additive number stays comparable.
"""

from __future__ import annotations

import numpy as np
import torch

# Cory's weights (PickBallPhaseKeyposeCost): the task term dominates, tracking is a light
# regularizer that keeps the executable rows heading for the keypose it scored.
_TASK_WEIGHT = 30.0
_TRACK_WEIGHT = 1.0


def _axis(quat_wxyz, index):
    """Return a rotation-matrix column as a unit vector, from a wxyz quaternion."""
    w, x, y, z = quat_wxyz.unbind(-1)
    cols = (
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1),
        torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1),
        torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1),
    )
    v = cols[index]
    return v / torch.clamp(torch.linalg.vector_norm(v, dim=-1, keepdim=True), min=1e-8)


def finger_samples(ee_pos, ee_quat, geom, per_finger=6, half=None):
    """Return [K, 2 * per_finger, 3] sample points on the two inner finger faces.

    The inner faces are what can touch the object, so only those are sampled -- the analogue of
    Cory's inward-facing-normal filter, which his mesh geometry needs and our two-plate gripper
    gets for free from the closing axis.
    """
    approach, closing = _axis(ee_quat, 2), _axis(ee_quat, 1)
    # F7. The candidate's own gripper channel decides how far apart the fingers are. Using a
    # fixed open_half made an open and a closed key pose with identical arm joints score the same,
    # so grasp feasibility was invisible to the ranker.
    half = float(getattr(geom, "open_half", 0.04)) if half is None else half
    tip = float(getattr(geom, "tcp_to_tip", 0.0))
    depth = float(getattr(geom, "finger_r", 0.012))
    # Along each finger, from the knuckle to the tip.
    along = torch.linspace(0.0, 1.0, per_finger, dtype=ee_pos.dtype, device=ee_pos.device)
    offsets = (tip + 2.0 * depth * along).view(1, per_finger, 1)
    base = ee_pos.unsqueeze(-2) + offsets * approach.unsqueeze(-2)
    inner = half * closing.unsqueeze(-2)
    return torch.cat([base + inner, base - inner], dim=-2)


def keypose_contact_cost(ee_pos, ee_quat, target, radius, geom, per_finger=6, half=None):
    """Mean distance from the inner finger samples to the target object's SURFACE.

    ee_pos/ee_quat are the keypose row only: [K, 3] and [K, 4]. `radius` is the object's local
    half-width, so subtracting it turns a centre distance into a surface distance -- the whole
    point, since a proposal that closes the fingers around the object and one that hovers at the
    same centre distance score identically under a centre metric.
    """
    samples = finger_samples(ee_pos, ee_quat, geom, per_finger=per_finger)
    delta = samples - torch.as_tensor(target, dtype=samples.dtype).view(1, 1, 3)
    surface = torch.linalg.vector_norm(delta, dim=-1) - float(radius)
    return surface.abs().mean(dim=-1)


def track_cost(real_chunk, keypose_row, action_rows=None):
    """L1 between the executable rows and the nearest goal they are meant to reach.

    Cory's tracking term. Without it a proposal can score well on contact geometry while the rows
    that actually execute head somewhere else.

    `action_rows` marks where the executable rows end. When goal rows sit between them and the
    keypose (the AWE waypoint block), the actions are pulled toward W1 -- the nearest waypoint --
    rather than straight at the keypose. That is the entire point of the waypoints: the straight
    pull measurably degrades (37/16/0% as its strength rose), because AWE places waypoints
    exactly where a straight line stops describing the motion.
    """
    chunk = torch.as_tensor(np.asarray(real_chunk), dtype=torch.float32)
    end = keypose_row if action_rows is None else int(action_rows)
    goal = chunk[:, end : end + 1, :7] if end < keypose_row else \
        chunk[:, keypose_row : keypose_row + 1, :7]
    return (chunk[:, :end, :7] - goal).abs().mean(dim=(1, 2))


def proposal_costs(planner, real_chunks, ctx, keypose_row, *, action_rows=None, per_finger=6,
                   task_weight=_TASK_WEIGHT, track_weight=_TRACK_WEIGHT):
    """Return [K] ranking costs for [K, H, D] real joint chunks.

    Mirrors `chunk_cost.chunk_costs`' signature and return so the two are interchangeable behind
    one flag. Returns None when the stage has no target to measure against, so the caller can fall
    back rather than silently rank on a constant.
    """
    # H3. This used to read ctx["place_target_pos"], which NOTHING in the tree ever writes -- so
    # the place branch was always None and every payload stage silently fell through to the grasp
    # object's position (or to the caller's fallback ranker). context.py publishes the stage's
    # canonical target as ctx["target"]; use it, and only for payload stages.
    target = ctx.get("target") if ctx.get("payload") is not None else None
    if target is None:
        objects = ctx.get("objects", {})
        name = ctx.get("grasp_obj")
        target = objects.get(name, {}).get("pos") if name in objects else None
    if target is None:
        return None

    chunks = torch.as_tensor(np.asarray(real_chunks), dtype=torch.float32)
    fk = planner.fk.forward(chunks[..., :7])
    ee_pos, ee_quat = fk.ee_pos[:, keypose_row], fk.ee_quat[:, keypose_row]
    root_pos, root_quat = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if root_pos is not None and root_quat is not None:
        from sim_free_mpc.fk import transform_points_wxyz
        ee_pos = transform_points_wxyz(
            torch.as_tensor(root_pos, dtype=ee_pos.dtype),
            torch.as_tensor(root_quat, dtype=ee_pos.dtype), ee_pos.unsqueeze(1)).squeeze(1)

    # H4. extents live at ctx["objects"][name]["extents"], not at the top level, so the old
    # ctx.get("extents") always missed and EVERY object was ranked with the 0.05 default radius --
    # the contact cost was object-independent. Fall back only when the object is genuinely absent.
    name = ctx.get("grasp_obj")
    _ext = (ctx.get("objects", {}).get(name) or {}).get("extents")
    radius = float(_ext[1]) if _ext is not None and len(_ext) > 1 else 0.05
    if _ext is None and name:
        print(f"[keypose_cost] no extents for {name!r}; contact radius falls back to 0.05",
              flush=True)
    # F7. Map the candidate's own gripper command onto a finger half-separation: channel 7 is the
    # normalised close command (1 = closed), so half = open_half * (1 - g).
    _open_half = float(getattr(planner.cost.geom, "open_half", 0.04))
    _g = torch.clamp(chunks[:, keypose_row, 7], 0.0, 1.0)
    _half = (_open_half * (1.0 - _g)).clamp(min=0.002)
    task = keypose_contact_cost(ee_pos, ee_quat, target, radius, planner.cost.geom,
                                per_finger=per_finger, half=_half)
    return (task_weight * task
            + track_weight * track_cost(chunks, keypose_row, action_rows=action_rows))
