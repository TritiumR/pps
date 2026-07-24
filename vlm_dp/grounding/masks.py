"""Join keypoints/object names to their depth points, whichever segmenter produced the frame."""
import re

import numpy as np


def _masked_points(grounded, env, name):
    """World points belonging to ``name`` in this frame, or None if it has none."""
    points_of = grounded.get("points_of")
    if points_of is not None:            # a real segmenter already knows its objects by name
        return points_of(name)
    rel = re.sub(r"^/World/envs/env_[^/]*/", "", env.scene[name].cfg.prim_path)
    ids = [i for i, prim in grounded["id_to_prim"].items() if rel and rel in prim]
    sel = np.isin(grounded["masks"], ids) & np.isfinite(grounded["points"]).all(axis=-1)
    return grounded["points"][sel] if int(sel.sum()) else None


def _nearest_kp(keypoints, point):
    return int(np.argmin(np.linalg.norm(keypoints - point, axis=1)))


def narrow_axis(pts, min_aspect=1.3):
    """Unit horizontal direction of an object's NARROW axis (PCA of the xy cloud), or None if the
    object is roughly round (long/short spread < ``min_aspect``, so no axis is worth grasping across).
    Grasp across the narrow axis = the gripper closes along this direction (the object fits)."""
    if pts is None or len(pts) < 8:
        return None
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(xy.T @ xy)          # ascending eigenvalues
    if evals[0] <= 1e-9 or (evals[1] / evals[0]) ** 0.5 < min_aspect:
        return None                                    # ~round: narrow axis ill-defined, don't constrain
    n = evecs[:, 0]                                    # smallest-spread direction = the narrow axis
    n = n / (np.linalg.norm(n) + 1e-9)
    return (float(n[0]), float(n[1]), 0.0)


def object_for_keypoint(grounded, env, point, names):
    """Name the object whose masked depth points are nearest this keypoint (or None)."""
    point = np.asarray(point)
    best, best_d = None, np.inf
    for name in names:
        pts = _masked_points(grounded, env, name)
        if pts is None:
            continue
        d = float(np.linalg.norm(pts - point, axis=1).min())
        if d < best_d:
            best, best_d = name, d
    return best


def local_centroid(grounded, env, name, near=None, radius=0.06):
    """Grasp center = centroid of `name`'s masked depth points, restricted to those within `radius` of
    `near` when given (the local graspable region, e.g. a handle). Single-camera -> visible-surface bias.
    """
    pts = _masked_points(grounded, env, name)
    if pts is None:
        return None
    if near is not None:
        local = pts[np.linalg.norm(pts - np.asarray(near), axis=1) < radius]
        if len(local):
            pts = local
    return pts.mean(axis=0)
