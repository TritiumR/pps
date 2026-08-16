"""Geometry and constraint-evaluation helpers used by vlm_dp."""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

# TCP offset from panda_hand to the Robotiq grasp point.
ROBOTIQ_GRASP_OFFSET = (0.0, 0.0, 0.1716)

# Fallback half-extents: grasp radius, keepout radius, and half-height.
DEFAULT_EXTENT = (0.05, 0.05, 0.05)


def quat_wxyz_to_R(q_wxyz):
    """Convert an IsaacLab wxyz quaternion to a rotation matrix."""
    q = np.asarray(q_wxyz)
    return _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def center_from_points(pts, support_top=None):
    """Estimate an object's grasp center from its depth points."""
    top = float(np.percentile(pts[:, 2], 95))

    if support_top is not None:
        zc = 0.5 * (support_top + top)
        band = pts[pts[:, 2] >= np.percentile(pts[:, 2], 85)]
        xy = (band[:, :2].min(axis=0) + band[:, :2].max(axis=0)) / 2.0
        return np.array([xy[0], xy[1], zc], dtype=np.float64)

    xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
    return np.array(
        [
            xy[0],
            xy[1],
            float((top + np.percentile(pts[:, 2], 5)) / 2.0),
        ],
        dtype=np.float64,
    )


def usd_extents(E, scene_objects):
    """Read object half-extents from USD bounding boxes."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
    except Exception as exc:
        print(
            f"[sim_helpers] USD extents unavailable ({exc}), using defaults",
            flush=True,
        )
        return {}

    extents = {}

    for name in scene_objects:
        prim_path = None
        getters = (
            lambda: str(E.env.scene[name].root_physx_view.prim_paths[0]),
            lambda: str(E.env.scene[name].cfg.prim_path).replace(
                "{ENV_REGEX_NS}",
                "/World/envs/env_0",
            ),
        )

        for getter in getters:
            try:
                prim_path = getter()
                break
            except Exception:
                continue

        if prim_path is None:
            continue

        try:
            bounds = bbox_cache.ComputeWorldBound(
                stage.GetPrimAtPath(prim_path)
            ).ComputeAlignedRange()
            half = np.abs(
                0.5
                * (
                    np.array(bounds.GetMax(), dtype=float)
                    - np.array(bounds.GetMin(), dtype=float)
                )
            )
            if np.all(np.isfinite(half)) and 0 < half.max() < 5.0:
                extents[name] = (
                    float(min(half[0], half[1])),
                    float(max(half[0], half[1])),
                    float(half[2]),
                )
        except Exception:
            continue

    return extents


class _Linalg:
    def __init__(self, dev):
        self._dev = dev

    def norm(self, x, axis=-1, keepdims=False):
        if torch.is_tensor(x) and not x.is_floating_point():
            x = x.to(torch.float32)
        return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims)


class TorchNumpyShim:
    """Provide the NumPy operations used by generated constraints via PyTorch."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.linalg = _Linalg(device)
        self.pi = torch.pi

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, torch.float32)
        return torch.as_tensor(
            x,
            device=self.device,
            dtype=torch.float32,
        )

    def array(self, obj, dtype=None):
        if isinstance(obj, (list, tuple)):
            if all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in obj
            ):
                return torch.tensor(
                    list(obj),
                    device=self.device,
                    dtype=torch.long,
                )

            elems = [self.array(item) for item in obj]
            if any(
                isinstance(elem, torch.Tensor) and elem.ndim > 0
                for elem in elems
            ):
                return torch.stack([self._t(elem) for elem in elems])

            return torch.stack(
                [self._t(elem).reshape(()) for elem in elems]
            )

        return self._t(obj)

    asarray = array

    def stack(self, seq, axis=0):
        return torch.stack(
            [self._t(item) for item in seq],
            dim=axis,
        )

    def concatenate(self, seq, axis=0):
        return torch.cat(
            [self._t(item) for item in seq],
            dim=axis,
        )

    def dot(self, a, b):
        return (self._t(a) * self._t(b)).sum(dim=-1)

    def cross(self, a, b):
        return torch.linalg.cross(
            self._t(a),
            self._t(b),
            dim=-1,
        )

    def abs(self, x):
        return torch.abs(self._t(x))

    def sqrt(self, x):
        return torch.sqrt(self._t(x))

    def sign(self, x):
        return torch.sign(self._t(x))

    def arccos(self, x):
        return torch.arccos(
            torch.clamp(self._t(x), -1.0, 1.0)
        )

    def arctan2(self, y, x):
        return torch.arctan2(
            self._t(y),
            self._t(x),
        )

    def maximum(self, a, b):
        return torch.maximum(
            self._t(a),
            self._t(b),
        )

    def minimum(self, a, b):
        return torch.minimum(
            self._t(a),
            self._t(b),
        )

    def clip(self, x, lo, hi):
        return torch.clamp(
            self._t(x),
            float(lo),
            float(hi),
        )

    def mean(self, x, axis=None):
        x = self._t(x)
        return x.mean() if axis is None else x.mean(dim=axis)

    def sum(self, x, axis=None):
        x = self._t(x)
        return x.sum() if axis is None else x.sum(dim=axis)


def load_torch_constraints(txt_path, get_grasping_cost_fn, shim):
    """Load generated constraint functions with the PyTorch NumPy shim."""
    if txt_path is None or not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8") as file:
        functions_text = file.read()

    global_vars = {
        "np": shim,
        "get_grasping_cost_by_keypoint_idx": get_grasping_cost_fn,
    }
    local_vars = {}

    exec(functions_text, global_vars, local_vars)  # noqa: S102
    return list(local_vars.values())


def make_torch_constraint(callables):
    """Combine stage constraints into one batched cost function."""

    def constraint_fn(pos, keypoints):
        total = None

        for fn in callables:
            cost = fn(pos, keypoints)

            if not isinstance(cost, torch.Tensor):
                cost = torch.as_tensor(
                    float(cost),
                    device=pos.device,
                    dtype=pos.dtype,
                )

            if cost.ndim == 0:
                cost = cost.expand(
                    pos.shape[0],
                    pos.shape[1],
                )

            total = cost if total is None else total + cost

        if total is None:
            return torch.zeros(
                pos.shape[0],
                pos.shape[1],
                device=pos.device,
                dtype=pos.dtype,
            )

        return total

    return constraint_fn