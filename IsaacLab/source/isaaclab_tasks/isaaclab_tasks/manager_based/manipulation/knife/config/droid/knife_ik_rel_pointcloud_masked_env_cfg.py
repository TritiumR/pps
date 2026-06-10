# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import knife_ik_rel_visuomotor_env_cfg, knife_joint_pos_pointcloud_masked_env_cfg


@configclass
class DroidKnifeIkRelPointCloudMaskedEnvCfg(
    knife_ik_rel_visuomotor_env_cfg.DroidKnifeIkRelVisuomotorEnvCfg
):
    """Configuration for the knife task with Droid robot using IK control and masked point cloud observations."""

    observations: knife_joint_pos_pointcloud_masked_env_cfg.ObservationsCfg = (
        knife_joint_pos_pointcloud_masked_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        knife_joint_pos_pointcloud_masked_env_cfg._configure_masked_pointcloud_table_cameras(self)
