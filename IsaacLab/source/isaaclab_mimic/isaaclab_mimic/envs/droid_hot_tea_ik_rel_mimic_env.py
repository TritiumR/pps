# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from .droid_tea_ik_rel_mimic_env import DroidTeaIKRelMimicEnv


class DroidHotTeaIKRelMimicEnv(DroidTeaIKRelMimicEnv):
    """Mimic wrapper for hot tea that aliases the hot teapot as the source demo teapot."""

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        object_pose_matrix = super().get_object_poses(env_ids=env_ids)
        if "hot_teapot" in object_pose_matrix:
            object_pose_matrix["teapot"] = object_pose_matrix["hot_teapot"]
        return object_pose_matrix
