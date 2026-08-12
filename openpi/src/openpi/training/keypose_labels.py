"""Per-timestep keypose targets for demonstration data.

The keypose for step t is the joint target that ENDS the phase t belongs to -- a distant goal
rather than the next 0.75 s of motion, which is what makes it a different object from the action
chunk and worth its own token.

Phase boundaries come from the COMMANDED gripper channel, ``obs/joint_actions[:, GRIPPER_DIM]``.
Binary command traces retain their literal change-point semantics. Continuous command traces are
converted to the same open/closed state using the robot execution convention ``closed > 0.5``;
the boundary is then backtracked to the beginning of the monotonic command ramp. This makes the
preceding phase end on its last stable command rather than in the middle of a close/release ramp.
Nothing here is task-specific: any number of semantic edges yields that many + 1 phases.

Keyposes are returned in the SAME space as the action rows (absolute joint targets read from
``obs/joint_actions``), so a keypose row appends to a chunk and flows through the existing
normalization unchanged.
"""

from __future__ import annotations

import numpy as np

GRIPPER_DIM = 7
DEFAULT_CLOSED_THRESHOLD = 0.5
DEFAULT_MIN_STATE_STEPS = 2
DEFAULT_MAX_PLATEAU_STEPS = 4
DEFAULT_MONOTONIC_TOLERANCE = 1e-6


def _is_literal_binary(command: np.ndarray) -> bool:
    """Whether the trace has the legacy, literal ``{0, 1}`` representation."""
    return bool(np.all((command == 0) | (command == 1)))


def _stable_semantic_crossings(
    command: np.ndarray,
    *,
    closed_threshold: float,
    min_state_steps: int,
) -> np.ndarray:
    """Find threshold crossings whose new state persists long enough to be semantic."""
    if len(command) < 2:
        return np.zeros(0, dtype=np.int64)
    state = command > float(closed_threshold)
    current = bool(state[0])
    crossings: list[int] = []
    cursor = 1
    required = max(1, int(min_state_steps))
    while cursor < len(state):
        if bool(state[cursor]) == current:
            cursor += 1
            continue
        run_end = cursor + 1
        while run_end < len(state) and bool(state[run_end]) == bool(state[cursor]):
            run_end += 1
        if run_end - cursor >= required or run_end == len(state):
            crossings.append(cursor)
            current = bool(state[cursor])
        cursor = run_end
    return np.asarray(crossings, dtype=np.int64)


def _ramp_start(
    command: np.ndarray,
    crossing: int,
    *,
    max_plateau_steps: int,
    monotonic_tolerance: float,
) -> int:
    """Backtrack a threshold crossing to the first sample of its monotonic ramp."""
    direction = 1.0 if command[crossing] > command[crossing - 1] else -1.0
    candidate = int(crossing)
    cursor = int(crossing)
    plateau_steps = 0
    tolerance = max(0.0, float(monotonic_tolerance))
    plateau_limit = max(0, int(max_plateau_steps))
    while cursor > 0:
        progress = direction * float(command[cursor] - command[cursor - 1])
        if progress > tolerance:
            candidate = cursor
            plateau_steps = 0
            cursor -= 1
            continue
        if progress >= -tolerance and plateau_steps < plateau_limit:
            plateau_steps += 1
            cursor -= 1
            continue
        break
    return candidate


def gripper_edges(
    joint_actions,
    *,
    closed_threshold: float = DEFAULT_CLOSED_THRESHOLD,
    min_state_steps: int = DEFAULT_MIN_STATE_STEPS,
    max_plateau_steps: int = DEFAULT_MAX_PLATEAU_STEPS,
    monotonic_tolerance: float = DEFAULT_MONOTONIC_TOLERANCE,
) -> np.ndarray:
    """Return semantic commanded-gripper phase starts.

    Literal binary traces deliberately retain the original ``diff != 0`` path.
    Continuous traces use thresholded semantics and the start of each command ramp.
    """
    actions = np.asarray(joint_actions)
    if actions.ndim != 2 or actions.shape[1] <= GRIPPER_DIM:
        raise ValueError(f"joint_actions must have shape [T, D>{GRIPPER_DIM}], got {actions.shape}")
    command = actions[:, GRIPPER_DIM]
    if not np.all(np.isfinite(command)):
        raise ValueError("commanded gripper trace contains non-finite values")
    if _is_literal_binary(command):
        return np.flatnonzero(np.diff(command) != 0) + 1
    crossings = _stable_semantic_crossings(
        command,
        closed_threshold=closed_threshold,
        min_state_steps=min_state_steps,
    )
    return np.asarray(
        [
            _ramp_start(
                command,
                int(crossing),
                max_plateau_steps=max_plateau_steps,
                monotonic_tolerance=monotonic_tolerance,
            )
            for crossing in crossings
        ],
        dtype=np.int64,
    )


def phase_end_indices(joint_actions) -> np.ndarray:
    """Return end[t]: the last index of the phase containing t, for every t.

    Edges delimit phases, so a phase runs up to the step before the next edge. The final phase
    ends at the last frame.
    """
    length = len(joint_actions)
    if length == 0:
        return np.zeros(0, dtype=np.int64)

    ends = np.full(length, length - 1, dtype=np.int64)
    start = 0
    for edge in gripper_edges(joint_actions):
        ends[start:edge] = edge - 1
        start = edge
    return ends


def keypose_targets(joint_actions) -> np.ndarray:
    """Return [T, D] absolute keypose targets, one per timestep."""
    actions = np.asarray(joint_actions, dtype=np.float32)
    if actions.size == 0:
        return actions.reshape(0, 0)
    return actions[phase_end_indices(actions)]


def keypose_at(joint_actions, step_idx: int) -> np.ndarray:
    """Return the single keypose row for one timestep, without building the full table."""
    actions = np.asarray(joint_actions, dtype=np.float32)
    return actions[phase_end_indices(actions)[step_idx]]


def audit(hdf5_path: str, limit: int | None = None) -> dict:
    """Summarise the edge pattern across demos.

    The labeler is only trustworthy if the pattern is uniform, so this reports the distribution
    rather than asserting one shape. Run it before training on a new dataset.
    """
    import collections

    import h5py

    patterns: collections.Counter = collections.Counter()
    values: set[float] = set()
    positions: list[tuple[float, ...]] = []
    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f["data"].keys())[:limit]
        for name in names:
            actions = np.asarray(f["data"][name]["obs/joint_actions"])
            command = actions[:, GRIPPER_DIM]
            values.update(np.unique(command).tolist())
            edges = gripper_edges(actions)
            patterns[(len(edges), tuple(int(command[i]) for i in edges))] += 1
            positions.append(tuple(float(e) / len(command) for e in edges))
    return {
        "demos": len(names),
        "command_values": sorted(values),
        "edge_patterns": {str(k): v for k, v in patterns.most_common()},
        "edge_positions_mean": np.mean(positions, axis=0).round(3).tolist()
        if positions and len({len(p) for p in positions}) == 1
        else None,
        "uniform": len(patterns) == 1,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.hdf5_path, args.limit), indent=2))


if __name__ == "__main__":
    main()
