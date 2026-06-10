# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import drink_ik_rel_visuomotor_env_cfg, drink_joint_pos_pointcloud_env_cfg


@configclass
class DroidDrinkIkRelPointCloudEnvCfg(
    drink_ik_rel_visuomotor_env_cfg.DroidDrinkIkRelVisuomotorEnvCfg
):
    """Configuration for the drink task with Droid robot using IK control and point cloud observations."""

    observations: drink_joint_pos_pointcloud_env_cfg.ObservationsCfg = (
        drink_joint_pos_pointcloud_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        drink_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(self)
