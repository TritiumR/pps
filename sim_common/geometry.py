"""Small scene/pose geometry helpers shared across sim_common.

``quat_wxyz_to_R`` converts an IsaacLab (w,x,y,z) quaternion to a rotation matrix; ``usd_extents`` reads
per-object half-extents from the simulator's USD bounding boxes, falling back to ``DEFAULT_EXTENT``.
"""
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# Fallback half-extents (grip, keepout, half_height) for an object with no measured USD extent.
DEFAULT_EXTENT = (0.05, 0.05, 0.05)


def quat_wxyz_to_R(q_wxyz):
    """IsaacLab (w,x,y,z) quaternion -> 3x3 rotation matrix."""
    q = np.asarray(q_wxyz)
    return _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def center_from_points(pts, support_top=None):
    """Object centre from a depth cloud: the middle of the visible extent on every axis.

    One rule for xy and z; no object-model frame assumption, so it holds for lying objects.
    ``support_top``: the surface the object RESTS ON. A settled convex object's centre is midway
    between its visible top and that surface; without it the visible silhouette bottom stands in,
    which reads ~1-2cm high on resting fruit (the bulge occludes the true bottom) and the pinch
    lands above the equator and ejects the object.
    """
    top = np.percentile(pts[:, 2], 95)
    if support_top is not None:
        # Resting convex object: its top cap sits directly above its centre, and the cap is the
        # least view-biased part of the cloud (the full silhouette's extent centre shifts ~1cm
        # toward the camera and the off-centre pinch loses the carry).
        band = pts[pts[:, 2] >= np.percentile(pts[:, 2], 85)]
        xy = (band[:, :2].min(axis=0) + band[:, :2].max(axis=0)) / 2.0
        return np.array([xy[0], xy[1], float((top + support_top) / 2.0)], dtype=np.float64)
    xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
    return np.array([xy[0], xy[1], float((top + np.percentile(pts[:, 2], 5)) / 2.0)],
                    dtype=np.float64)


def usd_extents(E, scene_objects):
    """Per-object ``(grip, keepout, half_height)`` from the sim's USD bounding boxes, keyed by name.

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
        print(f"[geometry] USD extents unavailable ({exc}); using defaults", flush=True)
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
