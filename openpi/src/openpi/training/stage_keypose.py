"""Task-space STAGE KEYPOSE labels: the 5-D object a k-conditioned policy is commanded with.

The keypose experiments so far made the keypose a PREDICTED OUTPUT -- p(a, k_hat | o), a row the
model denoises alongside the actions. This module supplies the other direction, p(a | o, k): k is
an INPUT, and the question is whether the representation was ever the problem or only its passive
role.

k is deliberately NOT the joint-space keypose `keypose_labels` produces. A commanded keypose has
to be something a planner can emit from scene geometry, and a ReKep stage constraint speaks about
where the gripper must BE, not about seven joint angles. So k is task space and 5-D:

    [x, y, z, yaw, grip]      EE position, gripper yaw about world z, commanded gripper state

Frames and conventions, all fixed here so the trainer and the evaluator cannot disagree:

  xyz   world frame, metres, read from ``obs/eef_pos``. Normalised by a workspace box, the same
        scheme the tray goal-conditioned policy uses: ``(x - centre) / half_extent``.
  yaw   atan2(R[1, 0], R[0, 0]) of the EE rotation, i.e. the heading of the gripper's own x axis
        in the world xy plane, WRAPPED INTO [-pi/2, pi/2). The wrap is not cosmetic: a parallel
        gripper is symmetric under a half turn, so yaw and yaw + pi are the same physical pose
        and an unwrapped label would ask the policy to fit a discontinuity that carries no
        information. Normalised by pi/2.
  grip  the COMMANDED gripper channel ``obs/joint_actions[:, GRIPPER_DIM]``, which on this data
        takes only {0, 1}. Carried as -1 (open) / +1 (closed). It is a BINARY COMMAND, never a
        continuous search dimension -- see the evaluator, which fills it from stage semantics.

Segmentation is reused, not reinvented: segment ends come from `awe_waypoints`, whose waypoints
are placed where linear interpolation fails worst and which never span a grasp or a release
because its segments are `keypose_labels`' commanded-gripper phases. For step t the label is the
pose at the NEXT segment end at or after t -- "the terminal event of the stage t is in".

Nothing here is square-specific.
"""

from __future__ import annotations

import numpy as np

from openpi.training import awe_waypoints, keypose_labels

# k = [x, y, z, yaw, grip].
KEYPOSE_DIM = 5
GRIPPER_DIM = keypose_labels.GRIPPER_DIM
YAW_PERIOD = np.pi  # parallel-jaw symmetry: yaw and yaw + pi are the same pose


def wrap_yaw(yaw):
    """Wrap an angle into [-pi/2, pi/2), the parallel-jaw gripper's fundamental domain."""
    y = np.asarray(yaw, dtype=np.float64)
    return (y + 0.5 * YAW_PERIOD) % YAW_PERIOD - 0.5 * YAW_PERIOD


# `obs/eef_quat` in the mimicgen-converted datasets is WXYZ, not the xyzw robosuite exposes
# live. Measured, not assumed: read as wxyz the end-effector's approach axis at the grasp frame is
# (0.033, 0.004, -0.997) with per-axis std (0.049, 0.056, 0.002) over twelve demos -- tool down,
# as a top grasp must be. Read as xyzw the same axis has std 0.685 on y and no consistent
# direction, i.e. noise. Getting this backwards silently turns the yaw dimension into a constant.
QUAT_CONVENTION = "wxyz"


def rotation_from_quat(quat, convention: str = QUAT_CONVENTION):
    """[N, 3, 3] rotation matrices from a quaternion array."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if convention == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif convention == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"unknown quaternion convention {convention!r}")
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def yaw_from_quat(quat, convention: str = QUAT_CONVENTION):
    """Gripper yaw about world z: the heading of the tool's own x axis, wrapped to jaw symmetry.

    Taken from a rotation column rather than from an Euler decomposition: with the tool pointing
    down, roll/pitch/yaw decompositions sit at a gimbal singularity and the yaw term becomes
    numerically unstable, while a horizontal column's heading stays well conditioned.
    """
    r = rotation_from_quat(quat, convention)
    return wrap_yaw(np.arctan2(r[:, 1, 0], r[:, 0, 0]))


def awe_boundaries(joint_actions, k_awe: int, stride: int = 2) -> np.ndarray:
    """FIXED segment boundaries: each gripper phase's own AWE waypoints, plus the phase end.

    `awe_waypoints.waypoint_target_indices` recomputes the waypoint block from every anchor, which
    is right for the row-appending use it was written for -- each predicted row wants its own
    receding block -- but wrong for SEGMENTATION: the "next waypoint" then slides forward almost
    every frame and the label degenerates into a short-horizon target rather than a stage's
    terminal event. Measured on square: recomputing per anchor gives 24-40 segments per demo and a
    median 0.6 s lookahead; solving each phase ONCE gives k_awe + 1 segments per phase and a
    lookahead of the right order. So the DP is run from each phase's start and the chosen
    waypoints are frozen as boundaries for the whole phase.
    """
    actions = np.asarray(joint_actions, dtype=np.float32)
    ends = keypose_labels.phase_end_indices(actions)
    if len(actions) == 0:
        return np.zeros(0, dtype=np.int64)
    std = actions.std(axis=0)
    normalized = (actions - actions.mean(axis=0)) / np.where(std < 1e-6, 1.0, std)
    bounds, start = [], 0
    for boundary in np.flatnonzero(np.r_[ends[1:] != ends[:-1], True]):
        phase_end = int(ends[start])
        if phase_end > start and int(k_awe) > 0:
            lattice = np.arange(start, phase_end + 1, max(stride, 1), dtype=np.int64)
            if lattice[-1] != phase_end:
                lattice = np.concatenate([lattice, [phase_end]])
            errors = awe_waypoints.edge_error_matrix(normalized[lattice])
            interior, _ = awe_waypoints.fixed_k_interior(errors, 0, len(lattice) - 1, int(k_awe))
            bounds.extend(int(v) for v in lattice[interior])
        bounds.append(phase_end)
        start = int(boundary) + 1
    return np.unique(np.asarray(sorted(set(bounds)), dtype=np.int64))


def segment_end_indices(joint_actions, k_awe: int = 1, stride: int = 2) -> np.ndarray:
    """end[t]: the frame index of the terminal event of the segment containing t.

    ``k_awe = 0`` uses the commanded-gripper phases themselves (`keypose_labels`) -- three
    segments on square, whose terminal events are the grasp, the release and the episode end.
    ``k_awe >= 1`` subdivides each phase with that many FIXED AWE waypoints (see
    `awe_boundaries`), so mid-transport stages get terminal events too.
    """
    actions = np.asarray(joint_actions, dtype=np.float32)
    if int(k_awe) <= 0:
        return keypose_labels.phase_end_indices(actions)
    bounds = awe_boundaries(actions, k_awe=int(k_awe), stride=stride)
    idx = np.searchsorted(bounds, np.arange(len(actions)), side="left")
    return bounds[np.clip(idx, 0, len(bounds) - 1)]


def stage_keypose_targets(eef_pos, eef_quat, joint_actions, k_awe: int = 1,
                          stride: int = 2) -> np.ndarray:
    """[T, 5] RAW (unnormalised) stage keypose per timestep: xyz, wrapped yaw, +-1 gripper."""
    pos = np.asarray(eef_pos, dtype=np.float64).reshape(-1, 3)
    yaw = yaw_from_quat(eef_quat)
    grip = np.where(np.asarray(joint_actions, dtype=np.float64)[:, GRIPPER_DIM] > 0.5, 1.0, -1.0)
    ends = segment_end_indices(joint_actions, k_awe=k_awe, stride=stride)
    return np.stack([pos[ends, 0], pos[ends, 1], pos[ends, 2], yaw[ends], grip[ends]],
                    axis=1).astype(np.float32)


def normalize(k, norm) -> np.ndarray:
    """Raw k -> the model's input space. Inverse of `denormalize`.

    xyz rides a workspace box, yaw rides its half-period, and the gripper is already +-1 and is
    passed through: rescaling a binary command would only blur the one dimension that must stay
    exactly two-valued.
    """
    k = np.asarray(k, dtype=np.float64).reshape(-1, KEYPOSE_DIM)
    centre = np.asarray(norm["centre"], dtype=np.float64)
    half = np.asarray(norm["half_extent"], dtype=np.float64)
    out = np.empty_like(k)
    out[:, :3] = (k[:, :3] - centre) / half
    out[:, 3] = wrap_yaw(k[:, 3]) / (0.5 * YAW_PERIOD)
    out[:, 4] = np.where(k[:, 4] > 0.0, 1.0, -1.0)
    return out.astype(np.float32)


def denormalize(k_norm, norm) -> np.ndarray:
    """Model input space -> raw k. Used by anything that has to report a commanded pose."""
    k = np.asarray(k_norm, dtype=np.float64).reshape(-1, KEYPOSE_DIM)
    centre = np.asarray(norm["centre"], dtype=np.float64)
    half = np.asarray(norm["half_extent"], dtype=np.float64)
    out = np.empty_like(k)
    out[:, :3] = k[:, :3] * half + centre
    out[:, 3] = wrap_yaw(k[:, 3] * (0.5 * YAW_PERIOD))
    out[:, 4] = np.where(k[:, 4] > 0.0, 1.0, -1.0)
    return out


def workspace_box(hdf5_path, demo_names=None, margin: float = 0.05) -> dict:
    """Fit the xyz normalisation box to the reachable EE range, with margin.

    Frozen into the run's metadata so evaluation normalises exactly as training did, and so a
    commanded k outside the demonstrated range still maps to a finite, ordered input rather than
    being clipped into it.
    """
    import h5py

    lo, hi = None, None
    with h5py.File(hdf5_path, "r") as f:
        names = demo_names if demo_names is not None else sorted(f["data"].keys())
        for name in names:
            pos = np.asarray(f["data"][name]["obs/eef_pos"], dtype=np.float64)
            lo = pos.min(axis=0) if lo is None else np.minimum(lo, pos.min(axis=0))
            hi = pos.max(axis=0) if hi is None else np.maximum(hi, pos.max(axis=0))
    lo, hi = lo - margin, hi + margin
    centre = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), 1e-3)
    return {
        "kind": "workspace_box",
        "centre": [float(v) for v in centre],
        "half_extent": [float(v) for v in half],
        "yaw_period": float(YAW_PERIOD),
        "gripper_values": [-1.0, 1.0],
        "note": ("k_norm = [(xyz - centre) / half_extent, wrap(yaw) / (pi/2), grip]; "
                 "world frame, metres; yaw wrapped into [-pi/2, pi/2) for jaw symmetry; "
                 "gripper is a binary -1/+1 command and is never rescaled"),
    }
