"""Shared stage-advance helpers for vlm_dp."""

from __future__ import annotations

import numpy as np

from vlm_dp.sim_helpers import ROBOTIQ_GRASP_OFFSET


def _capture_held(env, grounding, held_idx):
    """Capture held keypoint offsets in the gripper frame."""
    if not held_idx or grounding.keypoints is None:
        return None

    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    pos, rot = env.fk.grasp_point(
        env.q0().unsqueeze(0),
        ROBOTIQ_GRASP_OFFSET,
    )
    tcp = pos[0].detach().cpu().numpy()
    rmat = rot[0].detach().cpu().numpy()

    return np.stack([rmat.T @ (kps[i] - tcp) for i in held_idx])


def _should_advance(stage, flags, hold, commit_hold):
    """Return whether the current stage should advance."""
    if stage.done_flag is not None and stage.done_flag in flags:
        return flags[stage.done_flag]

    return (
        stage.gripper == "close"
        and hold >= commit_hold
    ) or stage.done()