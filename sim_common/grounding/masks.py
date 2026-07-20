"""Map a keypoint or an object name to the depth points belonging to that object.

Shared by the grounding sources and the fake-VLM stub -- turns a segmented frame plus back-projected depth
into per-object world points, nearest-keypoint lookups, and centroids.

Whichever segmenter produced the frame, the join is the same question: which pixels are this object? When a
real segmenter ran it answers that directly, by name, and ``propose_keypoints`` hands its accessor down on
the grounded frame. Otherwise the simulator's instance ids are joined to names through their prim paths.
"""
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


def object_for_keypoint(grounded, env, point, names):
    """Which of ``names`` a keypoint sits on, by GT-mask grounding (pass the scene's rigid-object names).

    The real VLM picks keypoint *indices*; this maps a chosen keypoint's world position back to the
    object whose masked depth points it is nearest to. Lets the grasp target that object's local
    centroid (the perception grasp center) in live mode, where the fake VLM's role table is gone.
    Returns the object name, or None if no candidate has masked points.
    """
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
