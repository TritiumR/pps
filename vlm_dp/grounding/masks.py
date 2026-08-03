"""Match object names and keypoints to segmented depth points."""

import re

import numpy as np


def _masked_points(grounded, env, name):
    """Return world points assigned to an object, or None."""
    points_of = grounded.get("points_of")
    if points_of is not None:
        return points_of(name)

    rel = re.sub(
        r"^/World/envs/env_[^/]*/",
        "",
        env.scene[name].cfg.prim_path,
    )
    ids = [
        obj_id
        for obj_id, prim in grounded["id_to_prim"].items()
        if rel and rel in prim
    ]
    selected = (
        np.isin(grounded["masks"], ids)
        & np.isfinite(grounded["points"]).all(axis=-1)
    )
    return grounded["points"][selected] if int(selected.sum()) else None


def _nearest_kp(keypoints, point):
    return int(np.argmin(np.linalg.norm(keypoints - point, axis=1)))


def narrow_axis(pts, min_aspect=1.3):
    """Return the object's horizontal narrow axis, or None if undefined."""
    if pts is None or len(pts) < 8:
        return None

    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(xy.T @ xy)

    if (
        eigenvalues[0] <= 1e-9
        or (eigenvalues[1] / eigenvalues[0]) ** 0.5 < min_aspect
    ):
        return None

    axis = eigenvectors[:, 0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return float(axis[0]), float(axis[1]), 0.0


def object_for_keypoint(grounded, env, point, names):
    """Return the object nearest to a keypoint, or None."""
    point = np.asarray(point)
    best_name = None
    best_distance = np.inf

    for name in names:
        pts = _masked_points(grounded, env, name)
        if pts is None:
            continue

        distance = float(np.linalg.norm(pts - point, axis=1).min())
        if distance < best_distance:
            best_name = name
            best_distance = distance

    return best_name


def local_centroid(grounded, env, name, near=None, radius=0.06):
    """Return the centroid of an object's full or local point cloud."""
    pts = _masked_points(grounded, env, name)
    if pts is None:
        return None

    if near is not None:
        local = pts[
            np.linalg.norm(pts - np.asarray(near), axis=1) < radius
        ]
        if len(local):
            pts = local

    return pts.mean(axis=0)