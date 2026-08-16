"""Compute gripper-event baselines from demonstration datasets.

The extracted statistics normalize failure-gate thresholds by the demonstration
distribution. Segmentation uses only commanded gripper transitions, so it does
not depend on task-specific state.

Run once per dataset:
  python -m vlm_dp.gate_stats /path/to/generated_dataset.hdf5 /path/to/gate_stats.json
"""

import json
import sys

import numpy as np

# Values above this threshold command the gripper to close.
CLOSE_THRESHOLD = 0.5


def _events(gripper_cmd):
    """Return gripper transitions as (step, kind) pairs."""
    closed = gripper_cmd > CLOSE_THRESHOLD
    events = []
    for t in range(1, len(closed)):
        if closed[t] and not closed[t - 1]:
            events.append((t, "close"))
        elif not closed[t] and closed[t - 1]:
            events.append((t, "open"))
    return events


def extract(hdf5_path):
    """Extract phase-duration and gripper-event statistics."""
    import h5py

    to_first_close, holds, gaps, reopens_in_approach = [], [], [], []
    with h5py.File(hdf5_path, "r") as f:
        for demo_name in sorted(f["data"].keys()):
            demo = f["data"][demo_name]
            cmd = np.asarray(demo["obs/joint_actions"])[:, 7]
            ev = _events(cmd)
            closes = [t for t, k in ev if k == "close"]
            opens = [t for t, k in ev if k == "open"]
            if not closes:
                continue

            to_first_close.append(closes[0])

            # Count reopen events before the first close.
            reopens_in_approach.append(sum(1 for t in opens if t < closes[0]))

            # Measure close-to-open holds and open-to-close gaps.
            for i, c in enumerate(closes):
                nxt = next((t for t in opens if t > c), None)
                if nxt is not None:
                    holds.append(nxt - c)
            for o in opens:
                nxt = next((t for t in closes if t > o), None)
                if nxt is not None:
                    gaps.append(nxt - o)

    def stats(xs):
        if not xs:
            return None
        a = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(a.size),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    return {
        "source": str(hdf5_path),
        "num_demos": len(to_first_close),
        "grasp_phase_steps": stats(to_first_close),
        "hold_phase_steps": stats(holds),
        "regrasp_gap_steps": stats(gaps),
        "approach_reopens": stats(reopens_in_approach),
    }


def main():
    hdf5_path, out_path = sys.argv[1], sys.argv[2]
    result = extract(hdf5_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()