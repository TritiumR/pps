"""Shape/finiteness smoke test for make_rekep_grasp_cost (no Isaac boot; uses the real FrankaFK).

    docker compose exec -T pps /isaac-sim/python.sh agent_tests/_grasp_cost_check.py
"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim_common.fk import FrankaFK  # noqa: E402
from dial_mpc.costs import make_rekep_grasp_cost  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def gpt_like(tcp, keypoints):
    return torch.linalg.norm(tcp - keypoints[0], dim=-1)   # reach to keypoint 0


def main():
    fk = FrankaFK(device=DEV)
    a_local = np.array([0.0, 0.0, -1.0])
    K, H = 16, 8
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.rand(K, H, 7, generator=g, device=DEV) * 0.5            # plausible joint values
    keypoints = torch.tensor([[0.52, -0.06, 0.06], [0.55, 0.0, 0.09], [0.58, -0.04, 0.07],
                              [0.50, 0.02, 0.05]], device=DEV)

    cost = make_rekep_grasp_cost(fk, gpt_like, a_local, grasp_idx=0, device=DEV)
    c = cost(q, q[0, 0], keypoints)
    ok_shape = tuple(c.shape) == (K,)
    ok_finite = bool(torch.isfinite(c).all())
    print(f"[grasp-cost] normal: shape={tuple(c.shape)} finite={ok_finite} "
          f"min={float(c.min()):.2f} max={float(c.max()):.2f}")

    # degenerate: handle == mug center (r_norm ~ 0) -> yaw term must be skipped, still finite
    kp_deg = torch.zeros(4, 3, device=DEV) + torch.tensor([0.55, 0.0, 0.06], device=DEV)
    c2 = cost(q, q[0, 0], kp_deg)
    ok_deg = tuple(c2.shape) == (K,) and bool(torch.isfinite(c2).all())
    print(f"[grasp-cost] degenerate(handle==center): shape={tuple(c2.shape)} "
          f"finite={bool(torch.isfinite(c2).all())}")

    ok = ok_shape and ok_finite and ok_deg
    print(f"[grasp-cost] RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
