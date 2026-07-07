"""Geometric cost terms for the sampling-MPC controller (PyTorch).

Each function here builds a cost: given a batch of candidate joint trajectories, it returns one score
per candidate (lower = better). The DIAL sampler (`sampler.make_accel_sampler`) calls these to rank its
samples. Scores come purely from the forward kinematics of the candidate joints -- there is no
simulation rollout.

Shapes (used throughout):
    q_traj: [K, H, 7]  -- K candidate trajectories, H steps each, 7 arm joints
    q_cur:  [7]        -- the current joint positions (reference for the "stay close" term)
    ctx:               -- per-cost goal info: a target point, a (target, center) pair, or keypoints
    return: [K]        -- one score per candidate

Terms that recur across the costs below (TCP = the gripper point: FK of the joints + grasp_offset):
    orient -- the gripper's approach axis should point down (a top-down grasp)
    floor  -- penalty for dropping the gripper below the table/board height
    local  -- keep the trajectory close to the current joints (small, smooth moves)
    smooth -- penalize change between consecutive horizon steps (damps flat-basin jitter)
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

# ctx differs per cost (a target[3], a (target, center) tuple, or keypoints[N,3]), so it's typed as
# ``object`` and unpacked inside each cost_fn.
CostFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]

_DOWN = (0.0, 0.0, -1.0)  # world "down" -- the approach direction we want for a top grasp.


def fixed_reach(tcp, keypoints):
    """Trivial relational constraint: distance from the TCP to the first keypoint ("go to this point").

    The single-target reach passed to `make_rekep_cost` when the goal is one point (a hover/lift target),
    not a multi-keypoint relation. ``tcp`` [...,3], ``keypoints`` [N,3] -> [...].
    """
    return torch.linalg.norm(tcp - keypoints[0], dim=-1)


def _clearance(pos, obstacles, r, margin):
    """Penalty for the gripper point coming within ``r + margin`` of any obstacle point.

    ``pos`` [K,H,3], ``obstacles`` [M,3] -> [K]. Pushes the path to route around obstacles; together
    with the floor term (which keeps it off the table), up-and-over emerges instead of a straight sweep.
    """
    d = torch.linalg.norm(pos[:, :, None, :] - obstacles[None, None, :, :], dim=-1)  # [K,H,M]
    return torch.clamp((r + margin) - d, min=0.0).pow(2).sum(dim=(1, 2))  # [K]


def _transit_clearance(pos, target_xy, z_clear, descend_r):
    """Keep the gripper above ``z_clear`` while horizontally far from ``target_xy``, lifting that
    requirement as it comes over the target (within ``descend_r``).

    ``pos`` [K,H,3], ``target_xy`` [2] -> [K]. Makes lift -> transit-high -> descend emerge from the
    optimizer (no scripted heights): far over the workspace it must stay high; over the target it may
    come down. (ReKep's path solver uses a table/transit clearance cost of this kind.)
    """
    d_xy = torch.linalg.norm(pos[..., :2] - target_xy, dim=-1)       # [K,H] horizontal dist to target
    gate = torch.clamp(d_xy / descend_r, 0.0, 1.0)                   # 1 far, ramps to 0 over the target
    # linear in the height deficit (constant, strong pull-up) so it isn't dwarfed by the reach term.
    return (torch.clamp(z_clear - pos[..., 2], min=0.0) * gate).sum(dim=1)  # [K]


def make_reach_cost(
    fk,
    w_reach: float = 8.0,
    w_floor: float = 50.0,
    w_local: float = 0.05,
    z_floor: float = 0.0,
) -> CostFn:
    """Reach only: pull the gripper point to a target, stay above the floor, stay near the current pose.

    The simplest cost -- no orientation or grasp shaping. Uses the raw FK point (`fk.fk`), not the
    offset gripper point.

    Args:
        fk: forward-kinematics object; ``fk(q[B,7]) -> (pos[B,3], rot[B,3,3])``.
        w_reach: weight on distance to the target.
        w_floor: weight on the below-floor penalty.
        w_local: weight on staying near the current joints.
        z_floor: world height below which the floor penalty turns on.

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
    w_smooth: float = 0.0,
    w_clear: float = 0.0,
    z_floor: float = 0.0,
    r_cube: float = 0.02,
    finger_r: float = 0.012,
    open_half: float = 0.04,
    obstacles: Sequence[Sequence[float]] | None = None,
    obstacle_r: float = 0.04,
    clear_margin: float = 0.03,
    transit_target: Sequence[float] | None = None,
    z_clear: float = 0.0,
    descend_r: float = 0.06,
    w_transit: float = 0.0,
    device: str = "cuda:0",
) -> CostFn:
    """Top-down grasp of a small, compact object (e.g. a cube).

    On top of reach/orient/floor/local, adds two grasp-shaping terms:
      yaw  -- turn the gripper so its fingers close along a world axis (square up to the object).
      coll -- push both open fingertips off the object center, so the object ends up *between* the
              fingers rather than being struck by them.

    Args:
        fk: FK object; ``grasp_point(q[B,7], offset) -> (pos[B,3], rot[B,3,3])``.
        a_local: the gripper-frame axis that ends up pointing down at the grasp orientation.
        grasp_offset: gripper-point offset in the hand frame (default = panda hand -> fingertips, 0.107 m).
        w_reach, w_orient, w_yaw, w_floor, w_local, w_coll: term weights.
        w_smooth: weight on the smoothness term (``sum ||q[t] - q[t-1]||^2``); 0 disables it.
        w_clear: weight on the obstacle-clearance term (keep the gripper away from `obstacles`); 0 off.
        z_floor: world height below which the floor penalty turns on.
        r_cube: object half-size, used to place the fingers around it.
        finger_r: effective fingertip radius.
        open_half: half the open-gripper width (fingertip offset along the closing axis).
        obstacles: ``[M,3]`` points to keep clear of (other objects' centers); None to disable.
        obstacle_r: obstacle radius; clearance kicks in within ``obstacle_r + clear_margin``.
        clear_margin: extra gap kept beyond ``obstacle_r``.
        transit_target: ``[3]`` point to descend over; when set with ``w_transit>0``, the gripper stays
            above ``z_clear`` until it's within ``descend_r`` (horizontally) of this point. None disables.
        z_clear: transit clearance height (derive from the scene -- obstacle tops + a margin).
        descend_r: horizontal radius around ``transit_target`` within which descent is allowed.
        w_transit: weight on the transit-clearance term; 0 disables it.
        device: torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=(target[3], object_center[3])) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)
    thr = r_cube + finger_r  # how close a fingertip may get to the center before it's penalized.
    obs_t = (torch.as_tensor(obstacles, device=device, dtype=torch.float32)
             if obstacles is not None and len(obstacles) else None)
    transit_xy = (torch.as_tensor(transit_target[:2], device=device, dtype=torch.float32)
                  if transit_target is not None else None)

    def cost_fn(q_traj, q_cur, ctx):
        target, cube_c = ctx
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        reach = torch.linalg.norm(pos - target, dim=-1).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)  # the approach axis, in world coordinates.
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        fy = rot[..., 1]  # the gripper's closing axis (its y-axis).
        yaw = (1.0 - torch.maximum(fy[..., 0].abs(), fy[..., 1].abs())).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        # The two fingertips (tcp +/- open_half along the closing axis) are pushed off the center.
        d_l = torch.linalg.norm(pos + open_half * fy - cube_c, dim=-1)
        d_r = torch.linalg.norm(pos - open_half * fy - cube_c, dim=-1)
        coll = (torch.clamp(thr - d_l, min=0.0).pow(2)
                + torch.clamp(thr - d_r, min=0.0).pow(2)).sum(dim=1)
        smooth = (q_traj[:, 1:] - q_traj[:, :-1]).pow(2).sum(dim=(1, 2))  # consecutive-step change
        clear = _clearance(pos, obs_t, obstacle_r, clear_margin) if obs_t is not None else 0.0
        transit = (_transit_clearance(pos, transit_xy, z_clear, descend_r)
                   if transit_xy is not None else 0.0)
        return (w_reach * reach + w_orient * orient + w_yaw * yaw + w_floor * floor
                + w_local * local + w_coll * coll + w_smooth * smooth + w_clear * clear
                + w_transit * transit)

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
    """Carry a grasped object to a target (lift, or move while holding).

    Assumes the object moves rigidly with the gripper, so its position is ``gripper_point +
    carry_offset`` -- where ``carry_offset`` is where the object sat relative to the gripper at the
    moment of grasp. That predicted object position is driven to the target, gripper kept pointing
    down (a tilt could drop a real grasp). No yaw or finger terms: once the object is held, finger
    placement no longer matters.

    Args:
        fk: FK object exposing ``grasp_point``.
        a_local: gripper-frame axis that points down (as in `make_grasp_cost`).
        carry_offset: ``object_position - gripper_point``, measured at the moment of grasp.
        grasp_offset: gripper-point offset in the hand frame.
        w_reach, w_orient, w_floor, w_local: term weights.
        z_floor: world height below which the floor penalty turns on.
        device: torch device.

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
    w_smooth: float = 0.0,
    w_clear: float = 0.0,
    z_floor: float = 0.0,
    held_idx: Sequence[int] = (),
    held_offset: Sequence[Sequence[float]] | None = None,
    obstacles: Sequence[Sequence[float]] | None = None,
    obstacle_r: float = 0.04,
    clear_margin: float = 0.03,
    transit_target: Sequence[float] | None = None,
    z_clear: float = 0.0,
    descend_r: float = 0.06,
    w_transit: float = 0.0,
    device: str = "cuda:0",
) -> CostFn:
    """Reach cost whose objective is a ReKep relational constraint instead of a fixed target.

    The objective is ``constraint_fn`` -- a function of the gripper point and the keypoints (e.g.
    "gripper at keypoint 3", or "keypoint A on keypoint B"). It can be hand-written, or generated by a
    VLM and run through the numpy->torch shim (`constraints`). The rest (orient/floor/local) is the same
    shared feasibility as the grasp costs.

    Movable (held) keypoints: when an object is grasped, its keypoints ride with the gripper. Pass
    their indices in ``held_idx`` and their positions relative to the gripper point (captured at grasp)
    in ``held_offset``; the cost then predicts each held keypoint as ``candidate_TCP + offset`` before
    evaluating the constraint, so a "place" constraint on a held keypoint actually depends on the
    candidate motion. Without this a held-keypoint constraint is constant (zero gradient).

    Args:
        fk: FK object exposing ``grasp_point``.
        constraint_fn: scores the gripper trajectory against the keypoints;
            ``constraint_fn(tcp[K,H,3], keypoints[...,N,3]) -> [K,H]``. Lower = closer to satisfying it.
        a_local: gripper-frame axis that points down.
        grasp_offset: gripper-point offset in the hand frame.
        w_task, w_orient, w_floor, w_local: term weights.
        w_smooth: weight on the smoothness term (penalizes change between consecutive horizon steps,
            ``sum ||q[t] - q[t-1]||^2``); 0 disables it. Damps the flat-basin jitter.
        w_clear: weight on the obstacle-clearance term (keep the gripper away from `obstacles`); 0 off.
        z_floor: world height below which the floor penalty turns on.
        held_idx: indices of currently-held keypoints (move with the gripper); empty if nothing held.
        held_offset: ``[len(held_idx), 3]`` held-keypoint positions relative to the gripper point.
        obstacles: ``[M,3]`` points to keep clear of (other objects' centers); None to disable.
        obstacle_r: obstacle radius; clearance kicks in within ``obstacle_r + clear_margin``.
        clear_margin: extra gap kept beyond ``obstacle_r``.
        transit_target: ``[3]`` point to descend over (e.g. the resolved place target); with ``w_transit>0``
            the gripper stays above ``z_clear`` until within ``descend_r`` (horizontally) of it. Makes
            the carry lift -> transit-high -> descend. None disables.
        z_clear: transit clearance height (derive from the scene).
        descend_r: horizontal radius around ``transit_target`` within which descent is allowed.
        w_transit: weight on the transit-clearance term; 0 disables it.
        device: torch device.

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=keypoints[N,3]) -> costs[K]``.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor(_DOWN, device=device, dtype=torch.float32)
    held_idx = list(held_idx)
    held_off = (torch.as_tensor(held_offset, device=device, dtype=torch.float32)
                if held_idx else None)  # [len(held_idx), 3]
    obs_t = (torch.as_tensor(obstacles, device=device, dtype=torch.float32)
             if obstacles is not None and len(obstacles) else None)
    transit_xy = (torch.as_tensor(transit_target[:2], device=device, dtype=torch.float32)
                  if transit_target is not None else None)

    def cost_fn(q_traj, q_cur, ctx):
        keypoints = ctx  # [N,3] world (torch, on device)
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        if held_idx:
            # Held keypoints ride with the candidate gripper point. Build per-candidate keypoints with
            # the KEYPOINT dim first -- [N,K,H,3] -- so a constraint's ``keypoints[i]`` still indexes a
            # keypoint (-> [K,H,3]); overwrite the held entries with candidate_TCP + offset.
            kp = keypoints[:, None, None, :].expand(-1, k, h, -1).clone()  # [N,K,H,3]
            for j, i in enumerate(held_idx):
                kp[i] = pos + held_off[j]
            task = constraint_fn(pos, kp).sum(dim=1)  # [K]
        else:
            task = constraint_fn(pos, keypoints).sum(dim=1)  # [K]
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        smooth = (q_traj[:, 1:] - q_traj[:, :-1]).pow(2).sum(dim=(1, 2))  # consecutive-step change
        clear = _clearance(pos, obs_t, obstacle_r, clear_margin) if obs_t is not None else 0.0
        transit = (_transit_clearance(pos, transit_xy, z_clear, descend_r)
                   if transit_xy is not None else 0.0)
        return (w_task * task + w_orient * orient + w_floor * floor + w_local * local
                + w_smooth * smooth + w_clear * clear + w_transit * transit)

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
    w_smooth: float = 0.0,
    z_floor: float = 0.0,
    finger_r: float = 0.012,
    open_half: float = 0.04,
    bar_r: float = 0.008,
    r_min: float = 0.01,
    device: str = "cuda:0",
) -> CostFn:
    """`make_rekep_cost` plus grasp shaping for closing around a thin vertical bar (e.g. a mug handle).

    Same constraint objective + orient/floor/local, plus two terms aimed at the handle keypoint
    (``grasp_idx``, which the VLM picks):
      yaw      -- turn the gripper so its fingers close along the line from the object center out to
                  the handle (one finger through the hole, one outside). The bar is ~vertical, so
                  "across the bar" is ambiguous; aligning radially is well-defined. Skipped when the
                  handle sits almost on top of the center (that direction is too short to be reliable).
      straddle -- keep both fingertips off the handle, so the bar ends up between the fingers.

    The object center is the average of all keypoints. The VLM decides WHICH keypoint is the handle;
    these terms decide HOW to grasp it.

    Args:
        fk: FK object exposing ``grasp_point``.
        constraint_fn: the ReKep constraint (see `make_rekep_cost`).
        a_local: gripper-frame axis that points down.
        grasp_idx: index of the handle keypoint in ``ctx``.
        grasp_offset: gripper-point offset in the hand frame.
        w_task, w_orient, w_yaw, w_straddle, w_floor, w_local: term weights.
        w_smooth: weight on the smoothness term (``sum ||q[t] - q[t-1]||^2``); 0 disables it.
        z_floor: world height below which the floor penalty turns on.
        finger_r: effective fingertip radius.
        open_half: half the open-gripper width.
        bar_r: handle-bar radius (with finger_r, sets how close a finger may get to the bar).
        r_min: if the center->handle direction is shorter than this, skip the yaw term.
        device: torch device.

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
        center = keypoints.mean(dim=0)   # [3] stands in for the object body
        task = constraint_fn(pos, keypoints).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        fy = rot[..., 1]  # the gripper's closing axis (its y-axis).
        # Horizontal direction from the object center out to the handle (the line to close across).
        r = handle - center
        r = r * torch.tensor([1.0, 1.0, 0.0], device=device)
        r_norm = torch.linalg.norm(r)
        if float(r_norm) < r_min:  # handle ~on top of the center: direction unreliable, skip yaw.
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
        smooth = (q_traj[:, 1:] - q_traj[:, :-1]).pow(2).sum(dim=(1, 2))  # consecutive-step change
        return (w_task * task + w_orient * orient + w_yaw * yaw + w_straddle * straddle
                + w_floor * floor + w_local * local + w_smooth * smooth)

    return cost_fn


def make_task_cost(
    fk,
    a_local: Sequence[float],
    target_pos: Sequence[float],
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.107),
    target_axis: Sequence[float] = (0.0, 0.0, -1.0),
    grasp_center: Sequence[float] | None = None,
    # weights (collaborator sim_free_mpc defaults)
    w_reach: float = 25.0,
    w_terminal: float = 40.0,
    w_smooth: float = 0.08,
    w_local: float = 0.005,
    w_orient: float = 0.25,
    w_yaw: float = 0.35,
    w_floor: float = 20.0,
    w_straddle: float = 30.0,
    w_clear: float = 25.0,
    w_transit: float = 0.0,
    w_path: float = 200.0,
    # gripper (in-cost proximity; applied only when the sampler passes g_traj)
    w_gripper: float = 0.1,
    gripper_thresh: float = 0.07,
    gripper_close_when_near: bool = True,
    # geometry
    r_obj: float = 0.05,
    open_half: float = 0.040,
    finger_r: float = 0.012,
    ee_r: float = 0.035,
    clear_margin: float = 0.020,
    z_floor: float = 0.0,
    z_clear: float = 0.0,
    descend_r: float = 0.06,
    obstacles: Sequence[Sequence[float]] | None = None,
    obstacle_r: float = 0.05,
    transit: bool = False,
    # running path constraints (relational; held keypoints ride the candidate gripper)
    path_fns: Sequence[Callable] = (),
    held_idx: Sequence[int] = (),
    held_offset: Sequence[Sequence[float]] | None = None,
    device: str = "cuda:0",
) -> CostFn:
    """One unified per-stage cost for the ReKep+DIAL Droid tasks (same terms + weights for every task).

    Reconciled with the collaborator's ``sim_free_mpc`` cost -- reach (mean + terminal, L2), smoothness,
    joint-delta (trust region), in-cost proximity gripper, downward orientation, grasp yaw/floor/straddle,
    non-target collision -- plus our transit-clearance and a 6-DoF-resolved orientation target.

    ``J_task`` = reach the resolved subgoal pose + the running path constraints; the rest is the fixed
    feasibility set. Grasp-shaping (yaw + straddle) is context-gated on ``grasp_center`` (grasp stages only).

    Args:
        target_pos: ``[3]`` resolved subgoal position (world).
        target_axis: ``[3]`` world axis the tool approach should align to -- down for grasp, the 6-DoF
            resolved approach axis for place/pour.
        grasp_center: ``[3]`` object center; enables yaw + finger-straddle. None on non-grasp stages.
        transit: enable the transit-clearance carry term toward ``target_pos``'s xy.
        path_fns: running relational constraints (torch callables ``(tcp[K,H,3], kp[N,K,H,3]) -> [K,H]``).
        held_idx / held_offset: currently-held keypoints and their offsets from the gripper (they ride it).

    Returns:
        ``cost_fn(q_traj[K,H,7], q_cur[7], ctx=keypoints[N,3], g_traj=None) -> costs[K]``. ``g_traj`` is the
        optional gripper-command channel from the sampler.
    """
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    tgt = torch.as_tensor(target_pos, device=device, dtype=torch.float32)
    axis = torch.as_tensor(target_axis, device=device, dtype=torch.float32)
    gc = (torch.as_tensor(grasp_center, device=device, dtype=torch.float32)
          if grasp_center is not None else None)
    obs_t = (torch.as_tensor(obstacles, device=device, dtype=torch.float32)
             if obstacles is not None and len(obstacles) else None)
    transit_xy = tgt[:2] if transit else None
    held_idx = list(held_idx)
    held_off = torch.as_tensor(held_offset, device=device, dtype=torch.float32) if held_idx else None
    thr = r_obj + finger_r

    def cost_fn(q_traj, q_cur, ctx, g_traj=None):
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        # reach: mean over the horizon + a terminal weight (collaborator's form), squared distance.
        d2 = ((pos - tgt) ** 2).sum(dim=-1)  # [k,h]
        cost = w_reach * d2.mean(dim=1) + w_terminal * d2[:, -1]
        # orient: align the tool approach axis to the target axis (down for grasp, resolved for place).
        appr = torch.einsum("khij,j->khi", rot, a_local)
        cost = cost + w_orient * (1.0 - (appr * axis).sum(dim=-1)).mean(dim=1)
        # fixed feasibility: floor, smoothness, trust region.
        cost = cost + w_floor * torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).mean(dim=1)
        cost = cost + w_smooth * (q_traj[:, 1:] - q_traj[:, :-1]).pow(2).sum(dim=(1, 2))
        cost = cost + w_local * (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        # grasp-shaping (yaw + finger-straddle), only when grasping a known object center.
        if gc is not None:
            fy = rot[..., 1]  # gripper closing axis
            cost = cost + w_yaw * (1.0 - torch.maximum(fy[..., 0].abs(), fy[..., 1].abs())).mean(dim=1)
            d_l = torch.linalg.norm(pos + open_half * fy - gc, dim=-1)
            d_r = torch.linalg.norm(pos - open_half * fy - gc, dim=-1)
            straddle = torch.clamp(thr - d_l, min=0.0).pow(2) + torch.clamp(thr - d_r, min=0.0).pow(2)
            cost = cost + w_straddle * straddle.mean(dim=1)
        # non-target collision: soft keepout around the other objects.
        if obs_t is not None:
            keepout = obstacle_r + ee_r + clear_margin
            dd = torch.linalg.norm(pos[:, :, None, :] - obs_t[None, None, :, :], dim=-1)  # [k,h,M]
            cost = cost + w_clear * torch.clamp(keepout - dd, min=0.0).pow(2).sum(dim=2).mean(dim=1)
        # transit-clearance carry (lift -> transit-high -> descend), emergent from the scene.
        if transit_xy is not None:
            cost = cost + w_transit * _transit_clearance(pos, transit_xy, z_clear, descend_r)
        # gripper (in-cost proximity), only when the sampler supplies a gripper channel.
        if g_traj is not None:
            near = (torch.linalg.norm(pos - tgt, dim=-1) < gripper_thresh).float()  # [k,h]
            desired = near if gripper_close_when_near else (1.0 - near)
            cost = cost + w_gripper * (g_traj[..., 0] - desired).pow(2).mean(dim=1)
        # running path constraints: held keypoints ride the candidate gripper (like make_rekep_cost).
        # Only per-step GEOMETRIC constraints (return [k,h], e.g. "keep upright") enter the running cost;
        # CONTACT/bookkeeping ones (e.g. "still grasping", a scalar) are enforced at the stage transition.
        if path_fns:
            kp = ctx[:, None, None, :].expand(-1, k, h, -1).clone()  # [N,k,h,3]
            for j, i in enumerate(held_idx):
                # rigid ride: held keypoint = candidate TCP + candidate_R @ local_offset (rot + trans),
                # so a geometric path constraint (e.g. "keep upright") actually sees the gripper's tilt.
                kp[i] = pos + torch.einsum("khij,j->khi", rot, held_off[j])
            for pf in path_fns:
                v = pf(pos, kp)
                if torch.is_tensor(v) and v.ndim == 2:
                    cost = cost + w_path * torch.clamp(v, min=0.0).sum(dim=1)
        return cost

    return cost_fn
