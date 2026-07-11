"""Unit check for vlm_mpc.np_shim: a GPT-style ReKep constraint must (1) load & run under the
torch shim, (2) give the SAME value batched [K,H,3] as numpy does point-by-point, and (3) reduce
last-axis so a (3,) input yields a scalar and [K,H,3] yields [K,H]. No Isaac boot needed.

    docker compose exec -T pps /isaac-sim/python.sh agent_tests/_shim_check.py
"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rekep.utils import get_callable_grasping_cost_fn, load_functions_from_txt  # noqa: E402
from sim_common.constraints import TorchNumpyShim, load_torch_constraints, make_torch_constraint  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# A representative GPT grasp-stage constraint (+ a relational one exercising np.array/mean/dot).
CONSTRAINT_TXT = '''
def stage1_subgoal_constraint1(end_effector, keypoints):
    """Align the end-effector with the handle (keypoint 3)."""
    return np.linalg.norm(end_effector - keypoints[3])

def stage1_subgoal_constraint2(end_effector, keypoints):
    """End-effector 2cm above the midpoint of keypoints 1 and 2 (relational)."""
    offsetted_point = np.mean(keypoints[np.array([1, 2])], axis=0) + np.array([0.0, 0.0, 0.02])
    return np.linalg.norm(end_effector - offsetted_point)
'''


def main():
    tmp = os.path.join(_REPO, "agent_tests", "_shim_stage1.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(CONSTRAINT_TXT)

    grasp_fn = get_callable_grasping_cost_fn([3])
    shim = TorchNumpyShim(device=DEV)

    # numpy reference (upstream loader) and torch (shim loader)
    np_funcs = load_functions_from_txt(tmp, grasp_fn)
    th_funcs = load_torch_constraints(tmp, grasp_fn, shim)
    th_cost = make_torch_constraint(th_funcs)

    rng = np.random.default_rng(0)
    keypoints_np = rng.uniform(0.3, 0.7, size=(5, 3))
    keypoints_th = torch.as_tensor(keypoints_np, device=DEV, dtype=torch.float32)

    # (A) scalar parity: single (3,) end-effector, numpy vs shim, per constraint.
    ee_np = rng.uniform(0.3, 0.7, size=(3,))
    ee_th = torch.as_tensor(ee_np, device=DEV, dtype=torch.float32)
    max_scalar_err = 0.0
    for nf, tf in zip(np_funcs, th_funcs):
        v_np = float(nf(ee_np, keypoints_np))
        v_th = tf(ee_th, keypoints_th)
        assert v_th.ndim == 0, f"expected scalar on (3,), got shape {tuple(v_th.shape)}"
        max_scalar_err = max(max_scalar_err, abs(v_np - float(v_th)))
    print(f"[shim] (A) scalar np-vs-torch max abs err = {max_scalar_err:.2e}")

    # (B) batched shape + parity: [K,H,3] candidates vs numpy looped point-by-point.
    K, H = 4, 2
    pos_np = rng.uniform(0.3, 0.7, size=(K, H, 3))
    pos_th = torch.as_tensor(pos_np, device=DEV, dtype=torch.float32)
    batched = th_cost(pos_th, keypoints_th)
    assert tuple(batched.shape) == (K, H), f"batched cost shape {tuple(batched.shape)} != (K,H)"
    ref = np.zeros((K, H))
    for i in range(K):
        for j in range(H):
            ref[i, j] = sum(float(nf(pos_np[i, j], keypoints_np)) for nf in np_funcs)
    max_batch_err = float(np.abs(ref - batched.detach().cpu().numpy()).max())
    print(f"[shim] (B) batched [K,H] shape OK; np-vs-torch max abs err = {max_batch_err:.2e}")

    os.remove(tmp)
    ok = max_scalar_err < 1e-5 and max_batch_err < 1e-4
    print(f"[shim] RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
