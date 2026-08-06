"""AWE waypoint targets: the K poses where linear interpolation fails worst.

A port of Cory's `add_sort_ball_awe_trajectory_targets.py` to single-arm data.

The point of AWE is *not* "K evenly spaced poses". Waypoints are chosen by dynamic programming to
minimise the WORST linear-interpolation error along the trajectory (a minimax objective over
segments). So they land exactly where a straight line between poses stops describing the motion --
which is the property that makes a waypoint block worth steering. Our first attempt at coupling
actions to a distant keypose used plain linear interpolation and degraded monotonically
(l1 0.0/0.3/0.6 -> 37%/16%/0% on square); AWE is the principled answer to why.

Segments are the phases from `keypose_labels` (commanded-gripper edges), so waypoints never span a
grasp or a release. Targets are absolute joint targets, the same space as the action and keypose
rows, so they append to a chunk and flow through the existing normalization.

`stride` subsamples the lattice the DP runs on: it bounds the O(L^2) edge-error matrix without
changing which poses can be selected at the resolution that matters.
"""

from __future__ import annotations

import numpy as np

from openpi.training import keypose_labels

ARM_DIMS = 7


def balanced_error(delta, arm_dims: int = ARM_DIMS) -> np.ndarray:
    """RMS error weighting arm and gripper channels equally.

    Without the balance the 7 arm joints drown the single gripper channel, and waypoints stop
    being placed at grasp/release transitions -- the ones that matter most.
    """
    squared = np.asarray(delta, dtype=np.float64) ** 2
    arm = squared[..., :arm_dims].mean(axis=-1)
    rest = squared[..., arm_dims:]
    if rest.shape[-1] == 0:
        return np.sqrt(arm)
    return np.sqrt(0.5 * arm + 0.5 * rest.mean(axis=-1))


def edge_error(states, start: int, end: int) -> float:
    """Worst deviation of the true path from the straight line between two lattice poses."""
    if end <= start + 1:
        return 0.0
    alpha = np.linspace(0.0, 1.0, end - start + 1)[:, None]
    line = (1.0 - alpha) * states[start] + alpha * states[end]
    return float(balanced_error(states[start : end + 1] - line).max())


def edge_error_matrix(states) -> np.ndarray:
    """All-pairs forward edge errors; inf below the diagonal (waypoints are ordered)."""
    count = len(states)
    out = np.full((count, count), np.inf, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    for start in range(count - 1):
        for end in range(start + 1, count):
            out[start, end] = edge_error(states, start, end)
    return out


def fixed_k_interior(edge_errors, start: int, end: int, k: int):
    """K ordered interior lattice slots minimising the worst segment error.

    Bottleneck shortest path with exactly k+1 edges. Short horizons pad with the endpoint, so the
    target block always has a fixed width.
    """
    if start == end:
        return np.full(k, end, dtype=np.int64), 0.0
    interior_count = min(int(k), max(0, end - start - 1))
    count = edge_errors.shape[0]
    previous = np.full(count, np.inf, dtype=np.float64)
    previous[start] = 0.0
    parents = []
    forward = np.triu(np.ones((count, count), dtype=bool), k=1)
    columns = np.arange(count)
    for _ in range(interior_count + 1):
        candidates = np.maximum(previous[:, None], edge_errors)
        candidates[~forward] = np.inf
        parent = np.argmin(candidates, axis=0).astype(np.int64)
        previous = candidates[parent, columns]
        parents.append(parent)
    if not np.isfinite(previous[end]):
        return np.full(k, end, dtype=np.int64), float("inf")
    path, cursor = [int(end)], int(end)
    for parent in reversed(parents):
        cursor = int(parent[cursor])
        path.append(cursor)
    path.reverse()
    interior = np.asarray(path[1:-1], dtype=np.int64)
    if len(interior) < k:
        interior = np.concatenate([interior, np.full(k - len(interior), end, dtype=np.int64)])
    return interior, float(previous[end])


def waypoint_targets(joint_actions, k: int = 5, stride: int = 2) -> np.ndarray:
    """Return [T, k+1, D] absolute targets: k AWE waypoints then the phase endpoint.

    Normalisation is per-dimension over the whole demo, so the interpolation error metric is not
    dominated by whichever joint happens to have the largest range.
    """
    actions = np.asarray(joint_actions, dtype=np.float32)
    length, dim = actions.shape
    out = np.repeat(actions[:, None, :], k + 1, axis=1)
    if length == 0:
        return out

    std = actions.std(axis=0)
    normalized = (actions - actions.mean(axis=0)) / np.where(std < 1e-6, 1.0, std)

    ends = keypose_labels.phase_end_indices(actions)
    boundaries = np.flatnonzero(np.r_[ends[1:] != ends[:-1], True])
    segment_start = 0
    for boundary in boundaries:
        segment_end = int(ends[segment_start])
        if segment_end <= segment_start:
            segment_start = int(boundary) + 1
            continue
        lattice = np.arange(segment_start, segment_end + 1, max(stride, 1), dtype=np.int64)
        if lattice[-1] != segment_end:
            lattice = np.concatenate([lattice, [segment_end]])
        errors = edge_error_matrix(normalized[lattice])
        endpoint = len(lattice) - 1
        for position, anchor in enumerate(lattice[:-1]):
            interior, _ = fixed_k_interior(errors, position, endpoint, k)
            rows = lattice[interior]
            block = np.concatenate([actions[rows], actions[segment_end][None]], axis=0)
            stop = int(lattice[position + 1]) if position + 1 < len(lattice) else segment_end + 1
            out[int(anchor) : stop] = block
        segment_start = int(boundary) + 1
    return out


def audit(hdf5_path: str, k: int = 5, stride: int = 2, limit: int | None = None) -> dict:
    """Report waypoint spacing and residual interpolation error across demos."""
    import h5py

    spans, residuals, demos = [], [], 0
    with h5py.File(hdf5_path, "r") as f:
        for name in sorted(f["data"].keys())[:limit]:
            actions = np.asarray(f["data"][name]["obs/joint_actions"], dtype=np.float32)
            targets = waypoint_targets(actions, k=k, stride=stride)
            demos += 1
            spans.append(float(np.abs(targets[:, -1, :ARM_DIMS] - actions[:, :ARM_DIMS]).mean()))
            residuals.append(float(np.abs(targets[:, 0, :ARM_DIMS] - actions[:, :ARM_DIMS]).mean()))
    return {
        "demos": demos,
        "k": k,
        "stride": stride,
        "mean_endpoint_reach_rad": round(float(np.mean(spans)), 4),
        "mean_first_waypoint_reach_rad": round(float(np.mean(residuals)), 4),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.hdf5_path, args.k, args.stride, args.limit), indent=2))


if __name__ == "__main__":
    main()
