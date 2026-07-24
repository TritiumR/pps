"""Self-contained copies of the sim_common helpers vlm_dp needs, so the package does not import
sim_common for geometry or constraint evaluation.

Vendored from sim_common/geometry.py and sim_common/constraints.py, kept in sync by hand. The Isaac
DroidEnv is not vendored here: it is the simulation environment, not a helper, and stays a
sim_common import.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

# TCP offset (panda_hand to Robotiq grasp point), calibrated. Mirrors sim_common.envs.droid.
ROBOTIQ_GRASP_OFFSET = (0.0, 0.0, 0.1716)

# Fallback half-extents (grip, keepout, half_height) for an object with no measured USD extent.
DEFAULT_EXTENT = (0.05, 0.05, 0.05)


# --------------------------------------------------------------------------------------- geometry

def quat_wxyz_to_R(q_wxyz):
    """IsaacLab (w,x,y,z) quaternion to 3x3 rotation matrix."""
    q = np.asarray(q_wxyz)
    return _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def center_from_points(pts, support_top=None):
    """Grasp centre from a depth cloud, at the object's geometric mid-height (approx centre of mass).

    support_top (the surface the object rests on) sets the mid-height (support_top + top) / 2, with xy
    from the top cap. The widest visible band sits about 1 to 2 cm too high because the top is better
    seen than the occluded bottom, which pushes the grasp onto a tapered object's weak upper third. The
    mid-height lands near the centre of mass. Without support_top the plain visible-extent midpoint is
    used. Returns a float64 [3] point.
    """
    top = float(np.percentile(pts[:, 2], 95))
    if support_top is not None:
        zc = 0.5 * (support_top + top)
        # xy from the top cap: its small cross-section is least depth-biased. The equator's near-half
        # bbox centre would shift about one radius toward the camera.
        band = pts[pts[:, 2] >= np.percentile(pts[:, 2], 85)]
        xy = (band[:, :2].min(axis=0) + band[:, :2].max(axis=0)) / 2.0
        return np.array([xy[0], xy[1], zc], dtype=np.float64)
    xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
    return np.array([xy[0], xy[1], float((top + np.percentile(pts[:, 2], 5)) / 2.0)],
                    dtype=np.float64)


def usd_extents(E, scene_objects):
    """Per-object (grip, keepout, half_height) from the sim's USD bounding boxes, keyed by name.

    Reads geometry straight from the simulator so no per-object radii need hand-specifying. The narrow
    horizontal half-extent is used for grasping and the wide one for collision.
    """
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                       useExtentsHint=True)
    except Exception as exc:
        print(f"[sim_helpers] USD extents unavailable ({exc}), using defaults", flush=True)
        return {}
    extents = {}
    for name in scene_objects:
        prim_path = None
        for getter in (lambda: str(E.env.scene[name].root_physx_view.prim_paths[0]),
                       lambda: str(E.env.scene[name].cfg.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")):
            try:
                prim_path = getter()
                break
            except Exception:
                continue
        if prim_path is None:
            continue
        try:
            bounds = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
            half = np.abs(0.5 * (np.array(bounds.GetMax(), dtype=float) - np.array(bounds.GetMin(), dtype=float)))
            if np.all(np.isfinite(half)) and 0 < half.max() < 5.0:
                extents[name] = (float(min(half[0], half[1])), float(max(half[0], half[1])), float(half[2]))
        except Exception:
            continue
    return extents


# ------------------------------------------------------------------------ ReKep constraint shim

class _Linalg:
    def __init__(self, dev):
        self._dev = dev

    def norm(self, x, axis=-1, keepdims=False):
        # Coerce to float. A constraint's integer-literal vector such as np.array([0,0,1]) arrives as a
        # Long tensor, which torch's vector_norm rejects.
        if torch.is_tensor(x) and not x.is_floating_point():
            x = x.to(torch.float32)
        return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims)


class TorchNumpyShim:
    """Torch-backed stand-in for numpy covering the ops ReKep constraints use.

    Reductions default to axis=-1 so a constraint written for a (3,) vector also evaluates correctly on
    a batched [..., 3] tensor. Unknown attributes raise, so an unsupported op surfaces as a clear error
    to extend rather than a silent miss.
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self.linalg = _Linalg(device)
        self.pi = torch.pi

    # Construction.
    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, torch.float32)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def array(self, obj, dtype=None):
        if isinstance(obj, (list, tuple)):
            # All-int list maps to a long tensor (numpy infers int64) so it can index keypoints.
            if all(isinstance(o, int) and not isinstance(o, bool) for o in obj):
                return torch.tensor(list(obj), device=self.device, dtype=torch.long)
            elems = [self.array(o) for o in obj]
            if any(isinstance(e, torch.Tensor) and e.ndim > 0 for e in elems):
                return torch.stack([self._t(e) for e in elems])
            return torch.stack([self._t(e).reshape(()) for e in elems])
        return self._t(obj)

    asarray = array

    def stack(self, seq, axis=0):
        return torch.stack([self._t(s) for s in seq], dim=axis)

    def concatenate(self, seq, axis=0):
        return torch.cat([self._t(s) for s in seq], dim=axis)

    # Vector algebra, last-axis by default.
    def dot(self, a, b):
        return (self._t(a) * self._t(b)).sum(dim=-1)

    def cross(self, a, b):
        return torch.linalg.cross(self._t(a), self._t(b), dim=-1)

    # Elementwise and reductions.
    def abs(self, x):
        return torch.abs(self._t(x))

    def sqrt(self, x):
        return torch.sqrt(self._t(x))

    def sign(self, x):
        return torch.sign(self._t(x))

    def arccos(self, x):
        return torch.arccos(torch.clamp(self._t(x), -1.0, 1.0))

    def arctan2(self, y, x):
        return torch.arctan2(self._t(y), self._t(x))

    def maximum(self, a, b):
        return torch.maximum(self._t(a), self._t(b))

    def minimum(self, a, b):
        return torch.minimum(self._t(a), self._t(b))

    def clip(self, x, lo, hi):
        return torch.clamp(self._t(x), float(lo), float(hi))

    def mean(self, x, axis=None):
        x = self._t(x)
        return x.mean() if axis is None else x.mean(dim=axis)

    def sum(self, x, axis=None):
        x = self._t(x)
        return x.sum() if axis is None else x.sum(dim=axis)


def load_torch_constraints(txt_path, get_grasping_cost_fn, shim):
    """Like rekep.utils.load_functions_from_txt but injects shim as np.

    Same sandbox contract as upstream: only np and get_grasping_cost_by_keypoint_idx are available to
    the constraint code, and the sole change is the numpy backend. Returns the constraint callables in
    file order, or [] if the file is missing.
    """
    if txt_path is None or not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        functions_text = f.read()
    gvars = {"np": shim, "get_grasping_cost_by_keypoint_idx": get_grasping_cost_fn}
    lvars = {}
    exec(functions_text, gvars, lvars)  # noqa: S102 - sandboxed VLM constraint code
    return list(lvars.values())


def make_torch_constraint(callables):
    """Sum a stage's constraint callables into constraint_fn(pos[K,H,3], kp[N,3]) -> [K,H].

    Each callable is a GPT fn(end_effector, keypoints). With the shim it returns [K,H], or a scalar for
    keypoint-only and grasping-cost terms, broadcast to [K,H]. The sum is the stage cost the DIAL
    sampler minimizes, where lower is closer to satisfying the sub-goal.
    """

    def constraint_fn(pos, keypoints):
        total = None
        for fn in callables:
            c = fn(pos, keypoints)
            if not isinstance(c, torch.Tensor):
                c = torch.as_tensor(float(c), device=pos.device, dtype=pos.dtype)
            if c.ndim == 0:
                c = c.expand(pos.shape[0], pos.shape[1])
            total = c if total is None else total + c
        if total is None:
            return torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
        return total

    return constraint_fn
