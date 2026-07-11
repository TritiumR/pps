"""Unit check for vlm_mpc.voxposer_bridge: validates world->voxel->trilinear indexing (the one
fiddly part) and the costmap-from-affordance construction. No Isaac boot.

    docker compose exec -T pps /isaac-sim/python.sh agent_tests/_voxposer_cost_check.py
"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dial_mpc.voxposer_bridge import affordance_region, build_costmap, sample_costmap  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BMIN = [0.0, 0.0, 0.0]
BMAX = [1.0, 2.0, 0.5]   # deliberately non-cubic bounds to catch per-axis scaling bugs
D = 50


def main():
    ok = True

    # (A) axis-order + scaling: costmap[i,j,k] = i/(D-1) (encodes the X voxel index, normalized).
    cm = np.zeros((D, D, D))
    cm[:] = (np.arange(D) / (D - 1))[:, None, None]
    cm_t = torch.as_tensor(cm, device=DEV, dtype=torch.float32)
    # query world x sweeps; the sampled cost must track x (world (x-BMIN)/(BMAX-BMIN)), and be
    # invariant to y,z. Voxel x = wx/1.0*(D-1); cost = x_index/(D-1) = wx.
    for wx in [0.1, 0.37, 0.5, 0.83]:
        for wy, wz in [(0.4, 0.1), (1.6, 0.45)]:
            pos = torch.tensor([wx * 1.0, wy, wz], device=DEV)
            c = float(sample_costmap(cm_t, pos, BMIN, BMAX))
            if abs(c - wx) > 0.02:
                print(f"[vox-cost] AXIS FAIL: world x={wx} -> cost {c:.3f} (expected ~{wx})")
                ok = False
    print(f"[vox-cost] (A) axis-order + per-axis scaling: {'OK' if ok else 'FAIL'}")

    # (B) batched shape: [K,H,3] in -> [K,H] out.
    pos_b = torch.rand(8, 4, 3, device=DEV) * torch.tensor([1.0, 2.0, 0.5], device=DEV)
    cb = sample_costmap(cm_t, pos_b, BMIN, BMAX)
    shape_ok = tuple(cb.shape) == (8, 4)
    print(f"[vox-cost] (B) batched [K,H] shape: {tuple(cb.shape)} {'OK' if shape_ok else 'FAIL'}")
    ok = ok and shape_ok

    # (C) affordance region -> costmap: cost is ~min at the region center, higher away, monotonic.
    center = np.array([0.5, 1.0, 0.25])
    grid, pts = affordance_region(center, radius_m=0.05, bounds_min=BMIN, bounds_max=BMAX, map_size=D)
    costmap = torch.as_tensor(build_costmap(grid), device=DEV, dtype=torch.float32)
    cc = float(sample_costmap(costmap, torch.tensor(center, device=DEV, dtype=torch.float32), BMIN, BMAX))
    # walk away from the center along +x; cost should rise monotonically
    ray = [float(sample_costmap(costmap, torch.tensor([0.5 + dx, 1.0, 0.25], device=DEV), BMIN, BMAX))
           for dx in [0.0, 0.1, 0.2, 0.3, 0.4]]
    mono = all(ray[i + 1] >= ray[i] - 1e-4 for i in range(len(ray) - 1))
    center_low = cc <= ray[-1] - 0.1 and cc < 0.05
    print(f"[vox-cost] (C) region center cost={cc:.3f} ray={[round(r, 2) for r in ray]} "
          f"monotonic={mono} center_low={center_low} (region voxels={int(grid.sum())})")
    ok = ok and mono and center_low

    print(f"[vox-cost] RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
