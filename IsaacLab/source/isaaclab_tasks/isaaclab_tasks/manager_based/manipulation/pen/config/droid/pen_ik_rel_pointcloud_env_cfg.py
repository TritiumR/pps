# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import pen_ik_rel_visuomotor_env_cfg, pen_joint_pos_pointcloud_env_cfg


@configclass
class DroidPenIkRelPointCloudEnvCfg(
    pen_ik_rel_visuomotor_env_cfg.DroidPenIkRelVisuomotorEnvCfg
):
    """Configuration for the pen task with Droid robot using IK control and point cloud observations."""

    observations: pen_joint_pos_pointcloud_env_cfg.ObservationsCfg = (
        pen_joint_pos_pointcloud_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        pen_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(self)
        pen_joint_pos_pointcloud_env_cfg._configure_pointcloud_pen_assets(self)
