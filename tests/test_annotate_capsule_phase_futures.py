from __future__ import annotations

import numpy as np

from tools.annotate_capsule_phase_futures_wandb import (
    PHASE_GRASP_POD,
    PHASE_OPEN_LID,
    PHASE_PLACE_POD,
    infer_capsule_phase,
    phase_future_annotations,
)


def test_infer_capsule_phase_uses_first_cumulative_achievements() -> None:
    frame_count = 12
    capsule_qpos = np.zeros((frame_count, 3), dtype=np.float32)
    capsule_qpos[3:, 1] = -0.6
    gripper = np.zeros((frame_count, 2), dtype=np.float32)
    gripper[5:9] = np.asarray([0.4, -0.4], dtype=np.float32)
    eef = np.zeros((frame_count, 3), dtype=np.float32)
    can_pose = np.ones((frame_count, 7), dtype=np.float32)
    can_pose[:, 3] = 1.0
    can_pose[6:, :3] = 0.05

    phase, metadata = infer_capsule_phase(
        capsule_joint_position=capsule_qpos,
        gripper_position=gripper,
        end_effector_position=eef,
        can_root_pose=can_pose,
        lid_joint_index=1,
    )

    np.testing.assert_array_equal(
        phase,
        np.asarray(
            [
                PHASE_OPEN_LID,
                PHASE_OPEN_LID,
                PHASE_OPEN_LID,
                PHASE_OPEN_LID,
                PHASE_GRASP_POD,
                PHASE_GRASP_POD,
                PHASE_GRASP_POD,
                PHASE_PLACE_POD,
                PHASE_PLACE_POD,
                PHASE_PLACE_POD,
                PHASE_PLACE_POD,
                PHASE_PLACE_POD,
            ],
            dtype=np.uint8,
        ),
    )
    assert metadata["open_lid_end"] == 3
    assert metadata["grasp_pod_end"] == 6


def test_phase_future_annotations_share_terminal_tail_within_each_phase() -> None:
    phase = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.uint8)
    qpos = np.arange(12, dtype=np.float32).reshape(6, 2)

    annotations = phase_future_annotations(
        phase,
        qpos,
        tail_frames=2,
        global_offset=10,
    )

    np.testing.assert_array_equal(
        annotations["future_qpos_indices"],
        np.asarray(
            [
                [11, 12],
                [11, 12],
                [11, 12],
                [13, 14],
                [13, 14],
                [15, -1],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        annotations["future_qpos"][0, :2],
        qpos[[1, 2]],
    )
    assert annotations["future_qpos_mask"][5].tolist() == [True, False]
