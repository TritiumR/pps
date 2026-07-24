"""Stage-advance predicates shared by the vlm_dp bridge and the vlm_base driver."""
from __future__ import annotations

import numpy as np

from vlm_dp.sim_helpers import ROBOTIQ_GRASP_OFFSET


def _capture_held(env, grounding, held_idx):
    """Gripper-local offsets of held keypoints at stage entry, so they ride the gripper rigidly."""
    if not held_idx or grounding.keypoints is None:
        return None
    kps = np.asarray(grounding.keypoints(), dtype=np.float64)
    pos, rot = env.fk.grasp_point(env.q0().unsqueeze(0), ROBOTIQ_GRASP_OFFSET)
    tcp, rmat = pos[0].detach().cpu().numpy(), rot[0].detach().cpu().numpy()
    return np.stack([rmat.T @ (kps[i] - tcp) for i in held_idx])


def _should_advance(stage, flags, hold, commit_hold):
    """Advance on the stage's task-progress flag when it names one, else the held-grasp fallback."""
    if stage.done_flag is not None and stage.done_flag in flags:
        return flags[stage.done_flag]
    return (stage.gripper == "close" and hold >= commit_hold) or stage.done()
