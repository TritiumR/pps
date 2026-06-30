"""Geometric cost terms for the sampling-MPC controller (PyTorch).

Each factory builds and returns a ``cost_fn(q_traj, q_cur, ctx) -> costs`` closure that the DIAL
sampler (`sampler.make_accel_sampler`) evaluates on a batch of candidate joint trajectories. Costs
are read purely off forward kinematics of the candidate configs; there is no dynamics rollout. The
terms mirror hydrax's ``grasp_cost``.

Tensor shapes used throughout:
    q_traj: ``[K, H, 7]`` -- K candidate trajectories, horizon H, 7 arm joints.
    q_cur:  ``[7]``       -- current joint config (the locality reference).
    ctx:                  -- per-cost context: a target, a tuple, or keypoints (see each factory).
    return: ``[K]``       -- one scalar cost per candidate.

Feasibility terms shared by the grasp costs (TCP = the FK grasp point at ``grasp_offset``):
    orient = sum_H 1 - (R @ a_local) . [0,0,-1]   approach axis points down (yaw-free).
    floor  = sum_H max(z_floor - tcp_z, 0)^2      keep the TCP above the table/board surface.
    local  = ||q_traj - q_cur||^2                 stay near the current config (smoothness).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

# ctx is intentionally polymorphic across costs (a target[3], a (target, center) tuple, or
# keypoints[N,3]), so it is typed as ``object`` and unpacked by each cost_fn.
CostFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]

_DOWN = (0.0, 0.0, -1.0)  # world "down": the desired approach direction for a top grasp.


def make_reach_cost(
    fk,
    w_reach: float = 8.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    z_floor: float = 0.0,
) -> CostFn:
    """Builds a reach-only cost (step 2: no orientation or grasp terms).

    Uses the raw FK point (``fk.fk``), not the offset grasp point.

    Args:
        fk: Forward-kinematics object exposing ``fk(q[B,7]) -> (pos[B,3], rot[B,3,3])``.
        w_reach: Weight on the linear distance-to-target term.
        w_floor: Weight on the quadratic floor-penetration penalty.
        w_local: Weight on the stay-near-current-config term.
        z_floor: World-z below which the floor penalty activates.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], target[3]) -> costs[K]``.
    """

    def cost_fn(q_traj, q_cur, target):
        k, h, _ = q_traj.shape
        pos, _ = fk.fk(q_traj.reshape(k * h, 7))
        pos = pos.reshape(k, h, 3)
        reach = torch.linalg.norm(pos - target, dim=-1).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return w_reach * reach + w_floor * floor + w_local * local

    return cost_fn


def make_grasp_cost(
    fk,
    a_local: Sequence[float],
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.107),
    w_reach: float = 8.0,
    w_orient: float = 8.0,
    w_yaw: float = 5.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    w_coll: float = 30.0,
    z_floor: float = 0.0,
    r_cube: float = 0.02,
    finger_r: float = 0.012,
    open_half: float = 0.04,
    device: str = "cuda:0",
) -> CostFn:
    """Builds a top-grasp cost for a compact (cube-like) object. Mirrors hydrax's grasp_cost.

    On top of reach/orient/floor/local, adds two grasp-specific terms:
      yaw  = 1 - max(|fy_x|, |fy_y|)   align the closing axis (EE y) to the nearest world face.
      coll = straddle penalty: each open fingertip (tcp +/- open_half * fy) is pushed off the
             object so the fingers flank it rather than penetrate it.

    Args:
        fk: FK object exposing ``grasp_point(q[B,7], offset) -> (pos[B,3], rot[B,3,3])``.
        a_local: EE-frame approach axis that points down at the grasp orientation (R_down.T @ down).
        grasp_offset: TCP offset in the hand frame (default = panda hand -> fingertip, 0.107 m).
        w_reach, w_orient, w_yaw, w_floor, w_local, w_coll: Term weights.
        z_floor: World-z below which the floor penalty activates.
        r_cube: Object half-extent used by the straddle term.
        finger_r: Effective fingertip radius.
        open_half: Half the open-gripper finger span (fingertip offset along the closing axis).
        device: Torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=(target[3], cube_center[3])) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)
    thr = r_cube + finger_r  # min fingertip-to-center distance before the straddle penalty kicks in.

    def cost_fn(q_traj, q_cur, ctx):
        target, cube_c = ctx
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        reach = torch.linalg.norm(pos - target, dim=-1).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)  # R @ a_local (the approach axis in world).
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        fy = rot[..., 1]  # EE y-axis = the gripper closing axis.
        yaw = (1.0 - torch.maximum(fy[..., 0].abs(), fy[..., 1].abs())).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        # Both open fingertips kept off the object center -> the object ends up between the fingers.
        d_l = torch.linalg.norm(pos + open_half * fy - cube_c, dim=-1)
        d_r = torch.linalg.norm(pos - open_half * fy - cube_c, dim=-1)
        coll = (torch.clamp(thr - d_l, min=0.0).pow(2)
                + torch.clamp(thr - d_r, min=0.0).pow(2)).sum(dim=1)
        return (w_reach * reach + w_orient * orient + w_yaw * yaw + w_floor * floor
                + w_local * local + w_coll * coll)

    return cost_fn


def make_lift_cost(
    fk,
    a_local: Sequence[float],
    carry_offset: Sequence[float],
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.107),
    w_reach: float = 20.0,
    w_orient: float = 8.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    z_floor: float = 0.0,
    device: str = "cuda:0",
) -> CostFn:
    """Builds a carry/lift cost (hydrax's ``_above`` on the grasped keypoint).

    The grasped object is assumed to ride rigidly with the TCP, so its predicted position is
    ``grasp_point(cfg) + carry_offset``. That predicted object position is driven to ``lift_target``
    with the gripper held down (a tilt could drop a real grasp). There are no yaw or collision terms:
    once the object is held, the open-finger straddle no longer applies.

    Args:
        fk: FK object exposing ``grasp_point``.
        a_local: EE-frame approach axis that points down (as in `make_grasp_cost`).
        carry_offset: ``object_pos - TCP`` captured at the moment of grasp.
        grasp_offset: TCP offset in the hand frame.
        w_reach, w_orient, w_floor, w_local: Term weights.
        z_floor: World-z below which the floor penalty activates.
        device: Torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=lift_target[3]) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)
    carry = torch.as_tensor(carry_offset, device=device, dtype=torch.float32)

    def cost_fn(q_traj, q_cur, ctx):
        target = ctx  # lift_target [3]
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        cube_pred = pos + carry
        reach = torch.linalg.norm(cube_pred - target, dim=-1).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return w_reach * reach + w_orient * orient + w_floor * floor + w_local * local

    return cost_fn


def make_rekep_cost(
    fk,
    constraint_fn,
    a_local: Sequence[float],
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.107),
    w_task: float = 8.0,
    w_orient: float = 8.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    z_floor: float = 0.0,
    device: str = "cuda:0",
) -> CostFn:
    """Builds the cost->DIAL bridge for a ReKep relational constraint (J_task) plus fixed J_feas.

    ``J_task`` is a ReKep constraint evaluated batch-wise on the TCP against the keypoints; ``J_feas``
    is the shared orient + floor + local feasibility. For rung 2 the constraint is hand-written; for
    rung 3 it is the VLM-generated numpy constraint run through the torch-numpy shim (`np_shim`).

    Args:
        fk: FK object exposing ``grasp_point``.
        constraint_fn: Batch-ready ReKep constraint ``constraint_fn(tcp[K,H,3], keypoints[N,3]) ->
            [K,H]`` (e.g. grasp: ``torch.linalg.norm(tcp - keypoints[i], dim=-1)``).
        a_local: EE-frame approach axis that points down.
        grasp_offset: TCP offset in the hand frame.
        w_task, w_orient, w_floor, w_local: Term weights.
        z_floor: World-z below which the floor penalty activates.
        device: Torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=keypoints[N,3]) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)

    def cost_fn(q_traj, q_cur, ctx):
        keypoints = ctx  # [N,3] world (torch, on device)
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        task = constraint_fn(pos, keypoints).sum(dim=1)  # [K]
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return w_task * task + w_orient * orient + w_floor * floor + w_local * local

    return cost_fn


def make_rekep_grasp_cost(
    fk,
    constraint_fn,
    a_local: Sequence[float],
    grasp_idx: int,
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.107),
    w_task: float = 8.0,
    w_orient: float = 8.0,
    w_yaw: float = 5.0,
    w_straddle: float = 30.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    z_floor: float = 0.0,
    finger_r: float = 0.012,
    open_half: float = 0.04,
    bar_r: float = 0.008,
    r_min: float = 0.01,
    device: str = "cuda:0",
) -> CostFn:
    """Builds `make_rekep_cost` plus two grasp-feasibility terms for enclosing a thin handle bar.

    Same ``J_task`` (the verbatim VLM constraint) and orient/floor/local feasibility as
    `make_rekep_cost`, plus two terms parameterized by the VLM-selected ``grasp_idx`` (which keypoint
    is the handle):
      yaw      = align the closing axis (EE y) RADIALLY, along the horizontal (handle - center)
                 direction, so one finger goes through the handle hole and one outside. The bar is
                 ~vertical, so "perpendicular to the bar" is degenerate; radial is the well-posed
                 target. Skipped when the radial direction is shorter than ``r_min``.
      straddle = keep both fingertips (tcp +/- open_half * fy) off the handle keypoint; with reach
                 centering the TCP on the bar, the bar ends up between the fingers.

    The keypoint centroid is the mug-body proxy (keypoint-driven, no GT). The VLM decides WHICH
    keypoint is the handle; these fixed terms decide HOW to grasp it. Generalizes the cube
    `make_grasp_cost` (yaw + collision).

    Args:
        fk: FK object exposing ``grasp_point``.
        constraint_fn: Batch-ready ReKep constraint (see `make_rekep_cost`).
        a_local: EE-frame approach axis that points down.
        grasp_idx: Index of the handle keypoint within ``ctx`` (VLM-selected).
        grasp_offset: TCP offset in the hand frame.
        w_task, w_orient, w_yaw, w_straddle, w_floor, w_local: Term weights.
        z_floor: World-z below which the floor penalty activates.
        finger_r: Effective fingertip radius.
        open_half: Half the open-gripper finger span.
        bar_r: Handle-bar radius (sets the straddle threshold with ``finger_r``).
        r_min: Minimum radial length below which the yaw term is skipped (degenerate direction).
        device: Torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=keypoints[N,3]) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)
    thr = bar_r + finger_r

    def cost_fn(q_traj, q_cur, ctx):
        keypoints = ctx  # [N,3] world (torch, on device)
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        handle = keypoints[grasp_idx]    # [3]
        center = keypoints.mean(dim=0)   # [3] mug-body proxy
        task = constraint_fn(pos, keypoints).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        fy = rot[..., 1]  # EE y-axis = the gripper closing axis.
        # Horizontal radial direction from the mug center to the handle keypoint.
        r = handle - center
        r = r * torch.tensor([1.0, 1.0, 0.0], device=device)
        r_norm = torch.linalg.norm(r)
        if float(r_norm) < r_min:  # degenerate radial direction -> skip the yaw term.
            yaw = torch.zeros(k, device=device)
        else:
            r = r / r_norm
            yaw = (1.0 - (fy[..., 0] * r[0] + fy[..., 1] * r[1]).abs()).sum(dim=1)
        d_l = torch.linalg.norm(pos + open_half * fy - handle, dim=-1)
        d_r = torch.linalg.norm(pos - open_half * fy - handle, dim=-1)
        straddle = (torch.clamp(thr - d_l, min=0.0).pow(2)
                    + torch.clamp(thr - d_r, min=0.0).pow(2)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return (w_task * task + w_orient * orient + w_yaw * yaw + w_straddle * straddle
                + w_floor * floor + w_local * local)

    return cost_fn
