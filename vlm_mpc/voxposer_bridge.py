"""VoxPoser affordance map -> DIAL-MPC cost (the VoxPoser analog of vlm_mpc.np_shim/costs).

VoxPoser's LMP composes a dense `affordance_map` (+ optional `avoidance_map`); its planner turns
those into a smooth scalar **cost field** (rekep/voxposer planners.py:37-43):

    target   = normalize( distance_transform_edt(1 - affordance) )   # 0 inside the region, grows out
    obstacle = normalize( gaussian_filter(avoidance, sigma) )
    costmap  = normalize( target*target_w + obstacle*obstacle_w )

We reuse that *exact* construction (`build_costmap`) and then let **DIAL sample the costmap at the
candidate TCPs** (`make_voxposer_cost`) instead of greedy-descending it -- the planner is the one
piece GOAL.md does not borrow. World->voxel is the affine twin of voxposer.interfaces.pc2voxel.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _normalize(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-9)


def build_costmap(affordance_map, avoidance_map=None, target_w=2.0, obstacle_w=1.0, sigma=2.0):
    """Affordance(+avoidance) voxel grids -> normalized cost field (low at the affordance region).

    Faithful to voxposer/planners.py:37-43. Returns a float64 numpy array, same shape as the maps.
    """
    aff = np.asarray(affordance_map, dtype=np.float64)
    target = _normalize(distance_transform_edt(1.0 - aff))
    if avoidance_map is not None and np.any(avoidance_map):
        obstacle = _normalize(gaussian_filter(np.asarray(avoidance_map, dtype=np.float64), sigma=sigma))
    else:
        obstacle = np.zeros_like(target)
    return _normalize(target * target_w + obstacle * obstacle_w)


def sample_costmap(costmap, pos, bounds_min, bounds_max):
    """Trilinear-sample `costmap` [Dx,Dy,Dz] at world positions `pos` [...,3] -> cost [...].

    Manual trilinear with explicit [x,y,z] indexing (NOT grid_sample -- avoids its axis-order trap).
    Voxel coords match pc2voxel: v = (pos - min)/(max - min) * (D-1), per axis.
    """
    dev = costmap.device
    Dt = torch.tensor([s - 1 for s in costmap.shape], device=dev, dtype=torch.float32)  # [3]
    bmin = torch.as_tensor(bounds_min, device=dev, dtype=torch.float32)
    bmax = torch.as_tensor(bounds_max, device=dev, dtype=torch.float32)
    v = (pos - bmin) / (bmax - bmin) * Dt
    v = torch.clamp(v, torch.zeros(3, device=dev), Dt)
    v0 = torch.clamp(torch.floor(v), torch.zeros(3, device=dev), Dt).long()
    v1 = torch.clamp(v0 + 1, max=Dt.long())
    f = v - v0.float()
    x0, y0, z0 = v0[..., 0], v0[..., 1], v0[..., 2]
    x1, y1, z1 = v1[..., 0], v1[..., 1], v1[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = costmap[x0, y0, z0] * (1 - fx) + costmap[x1, y0, z0] * fx
    c10 = costmap[x0, y1, z0] * (1 - fx) + costmap[x1, y1, z0] * fx
    c01 = costmap[x0, y0, z1] * (1 - fx) + costmap[x1, y0, z1] * fx
    c11 = costmap[x0, y1, z1] * (1 - fx) + costmap[x1, y1, z1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def make_voxposer_cost(fk, costmap, bounds_min, bounds_max, a_local, grasp_offset=(0.0, 0.0, 0.107),
                       w_task=8.0, w_orient=8.0, w_floor=50.0, w_local=0.05, z_floor=0.0,
                       device="cuda:0"):
    """VoxPoser cost field as the DIAL J_task + the same fixed J_feas as make_rekep_cost.

    cost_fn(q_traj[K,H,7], q_cur[7], ctx) -> costs[K]. `ctx` is ignored (the costmap is static
    pre-grasp; closed over). J_task = costmap sampled at the grasp TCP, summed over H.
    """
    cm = torch.as_tensor(costmap, device=device, dtype=torch.float32)
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)

    def cost_fn(q_traj, q_cur, ctx):
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        task = sample_costmap(cm, pos, bounds_min, bounds_max).sum(dim=1)   # [K]
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return w_task * task + w_orient * orient + w_floor * floor + w_local * local

    return cost_fn


def make_voxposer_grasp_cost(fk, costmap, bounds_min, bounds_max, a_local, target,
                             grasp_offset=(0.0, 0.0, 0.107), w_task=8.0, w_orient=8.0, w_yaw=5.0,
                             w_straddle=30.0, w_floor=50.0, w_local=0.05, z_floor=0.0,
                             finger_r=0.012, open_half=0.04, obj_r=0.012, device="cuda:0"):
    """make_voxposer_cost + the two fixed grasp-feasibility terms so the gripper ENCLOSES the object.

    Same costmap reach (J_task) + orient/floor/local as make_voxposer_cost, plus, referencing the
    affordance centroid ``target`` as the object center:
      yaw      = align the closing axis (EE y) to the nearest WORLD axis (the axis-aligned-object
                 default, as in the rung-1 cube make_grasp_cost; no object-orientation is known from a
                 scalar affordance field).
      straddle = keep both fingertips (tcp +/- open_half*fy) off ``target`` so the object sits between
                 them (reach centers the TCP on the affordance -> object enclosed).
    """
    cm = torch.as_tensor(costmap, device=device, dtype=torch.float32)
    a_local = torch.as_tensor(a_local, device=device, dtype=torch.float32)
    down = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
    target = torch.as_tensor(target, device=device, dtype=torch.float32)
    thr = obj_r + finger_r

    def cost_fn(q_traj, q_cur, ctx):
        k, h, _ = q_traj.shape
        pos, rot = fk.grasp_point(q_traj.reshape(k * h, 7), grasp_offset)
        pos = pos.reshape(k, h, 3)
        rot = rot.reshape(k, h, 3, 3)
        task = sample_costmap(cm, pos, bounds_min, bounds_max).sum(dim=1)
        appr = torch.einsum("khij,j->khi", rot, a_local)
        orient = (1.0 - (appr * down).sum(-1)).sum(dim=1)
        fy = rot[..., 1]                                                    # EE y-axis = closing axis
        yaw = (1.0 - torch.maximum(fy[..., 0].abs(), fy[..., 1].abs())).sum(dim=1)
        d_l = torch.linalg.norm(pos + open_half * fy - target, dim=-1)
        d_r = torch.linalg.norm(pos - open_half * fy - target, dim=-1)
        straddle = (torch.clamp(thr - d_l, min=0.0).pow(2)
                    + torch.clamp(thr - d_r, min=0.0).pow(2)).sum(dim=1)
        floor = torch.clamp(z_floor - pos[..., 2], min=0.0).pow(2).sum(dim=1)
        local = (q_traj - q_cur).pow(2).sum(dim=(1, 2))
        return (w_task * task + w_orient * orient + w_yaw * yaw + w_straddle * straddle
                + w_floor * floor + w_local * local)

    return cost_fn


def affordance_region(center_world, radius_m, bounds_min, bounds_max, map_size):
    """Hand-built affordance grid: 1 within `radius_m` (a ball) of `center_world`, else 0.

    The V1 'fake' analog of VoxPoser's set_voxel_by_radius -- a region on the GT handle. Returns the
    grid [D,D,D] and the world-frame coords of the region voxels (for the side-panel viz).
    """
    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)
    lin = [np.linspace(bmin[i], bmax[i], map_size) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    d = np.sqrt((gx - center_world[0]) ** 2 + (gy - center_world[1]) ** 2 + (gz - center_world[2]) ** 2)
    grid = (d <= radius_m).astype(np.float64)
    if grid.sum() == 0:  # radius smaller than a voxel -> mark the nearest voxel
        idx = np.unravel_index(np.argmin(d), d.shape)
        grid[idx] = 1.0
    pts = np.stack([gx[grid > 0], gy[grid > 0], gz[grid > 0]], axis=1)
    return grid, pts
