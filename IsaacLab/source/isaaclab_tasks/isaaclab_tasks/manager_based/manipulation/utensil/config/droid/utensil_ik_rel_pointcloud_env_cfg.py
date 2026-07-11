# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import utensil_ik_rel_visuomotor_env_cfg, utensil_joint_pos_pointcloud_env_cfg


@configclass
class DroidUtensilIkRelPointCloudEnvCfg(
    utensil_ik_rel_visuomotor_env_cfg.DroidUtensilIkRelVisuomotorEnvCfg
):
    """Configuration for the utensil insertion task with Droid robot using IK control and point cloud observations."""

    observations: utensil_joint_pos_pointcloud_env_cfg.ObservationsCfg = (
        utensil_joint_pos_pointcloud_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        utensil_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(self)
