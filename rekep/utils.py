"""Small shared helpers for the rekep package: config loading, geometry (bounds filter,
quaternion -> matrix), and the sandboxed VLM-constraint loader + grasping-cost factory.
"""

import os

import numpy as np
import yaml


def get_config(config_path):
    """Load a YAML config file into a dict."""
    assert config_path and os.path.exists(config_path), f"config not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config_path() -> str:
    """Path to the packaged default config (``rekep/configs/default.yaml``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml")


def load_default_config() -> dict:
    """Load rekep's packaged default config."""
    return get_config(default_config_path())


def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """Boolean mask of points inside the workspace box.

    ``strict=False`` pads the xy / lower-z bounds by 10% so points just outside are kept.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion (IsaacLab convention)."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def get_callable_grasping_cost_fn(grasping_keypoint_indices):
    """Constraint-sandbox helper ``(i) -> 0.0 if keypoint i is held else 1.0``.

    Injected so a path constraint can express "still grasping keypoint i".
    """
    held = set(int(i) for i in grasping_keypoint_indices)

    def get_grasping_cost_by_keypoint_idx(i):
        return 0.0 if int(i) in held else 1.0

    return get_grasping_cost_by_keypoint_idx


def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    """Exec a stage's VLM constraint file -> list of ``(end_effector, keypoints) -> float``.

    Sandboxed: the constraint code only sees ``np`` and ``get_grasping_cost_by_keypoint_idx``.
    Returns [] if the file is missing.
    """
    if txt_path is None or not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        functions_text = f.read()
    gvars = {
        "np": np,
        "get_grasping_cost_by_keypoint_idx": get_grasping_cost_fn,
    }
    lvars = {}
    exec(functions_text, gvars, lvars)  # noqa: S102 - sandboxed VLM constraint code
    return list(lvars.values())
