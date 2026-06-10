# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import tea_ik_rel_visuomotor_env_cfg, tea_joint_pos_pointcloud_env_cfg


@configclass
class DroidTeaIkRelPointCloudEnvCfg(
    tea_ik_rel_visuomotor_env_cfg.DroidTeaIkRelVisuomotorEnvCfg
):
    """Configuration for the tea task with Droid robot using IK control and point cloud observations."""

    observations: tea_joint_pos_pointcloud_env_cfg.ObservationsCfg = (
        tea_joint_pos_pointcloud_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        tea_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(self)

