"""Per-timestep keypose targets for demonstration data.

The keypose for step t is the joint target that ENDS the phase t belongs to -- a distant goal
rather than the next 0.75 s of motion, which is what makes it a different object from the action
chunk and worth its own token.

Phase boundaries come from the COMMANDED gripper channel, ``obs/joint_actions[:, GRIPPER_DIM]``.
On square it takes only {0.0, 1.0}: the 0->1 edge closes on the nut handle, the 1->0 edge releases
it on the peg. Audited over all 200 demos of demo_224.hdf5, every one has exactly those two edges,
at 41% +- 6% and 93% +- 1% of the episode. So no threshold is tuned and no subtask labels are
needed -- the demo hdf5 carries none. Nothing here is square-specific: any number of edges yields
that many + 1 phases.

Keyposes are returned in the SAME space as the action rows (absolute joint targets read from
``obs/joint_actions``), so a keypose row appends to a chunk and flows through the existing
normalization unchanged.
"""

from __future__ import annotations

import numpy as np

GRIPPER_DIM = 7


def gripper_edges(joint_actions) -> np.ndarray:
    """Return the indices at which the commanded gripper changes state."""
    command = np.asarray(joint_actions)[:, GRIPPER_DIM]
    return np.flatnonzero(np.diff(command) != 0) + 1


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
