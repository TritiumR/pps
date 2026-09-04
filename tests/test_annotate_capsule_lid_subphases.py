from __future__ import annotations

import numpy as np

from tools.annotate_capsule_lid_subphases_wandb import (
    GRASP_POD,
    LID_APPROACH,
    LID_PULL,
    LID_RELEASE,
    PLACE_POD,
    infer_capsule_subphases,
)


def test_infer_capsule_subphases_splits_motion_open_and_release() -> None:
    frame_count = 15
    capsule_qpos = np.zeros((frame_count, 3), dtype=np.float32)
    capsule_qpos[4:, 1] = np.asarray(
        [-0.03, -0.1, -0.2, -0.3, -0.4, -0.51, -0.6, -0.6, -0.6, -0.6, -0.6],
        dtype=np.float32,
    )
    gripper = np.zeros((frame_count, 2), dtype=np.float32)
    gripper[:9] = np.asarray([0.4, -0.4], dtype=np.float32)
    gripper[12:14] = np.asarray([0.4, -0.4], dtype=np.float32)
    eef = np.zeros((frame_count, 3), dtype=np.float32)
    can_pose = np.ones((frame_count, 7), dtype=np.float32)
    can_pose[:, 3] = 1.0
    can_pose[12:, :3] = 0.05

    phase, metadata = infer_capsule_subphases(
        capsule_joint_position=capsule_qpos,
        gripper_position=gripper,
        end_effector_position=eef,
        can_root_pose=can_pose,
        lid_joint_index=1,
    )

    assert phase[3] == LID_APPROACH
    assert phase[4] == LID_PULL
    assert phase[9] == LID_RELEASE
    assert phase[12] == GRASP_POD
    assert phase[14] == PLACE_POD
    assert metadata["lid_approach_frames"] == 4
    assert metadata["lid_pull_frames"] > 0
    assert metadata["lid_release_frames"] > 0
