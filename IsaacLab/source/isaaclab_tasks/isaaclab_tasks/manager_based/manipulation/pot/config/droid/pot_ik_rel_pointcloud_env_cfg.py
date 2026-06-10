# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import pot_ik_rel_visuomotor_env_cfg, pot_joint_pos_pointcloud_env_cfg


@configclass
class DroidPotIkRelPointCloudEnvCfg(
    pot_ik_rel_visuomotor_env_cfg.DroidPotIkRelVisuomotorEnvCfg
):
    """Configuration for the pot task with Droid robot using IK control and point cloud observations."""

    observations: pot_joint_pos_pointcloud_env_cfg.ObservationsCfg = (
        pot_joint_pos_pointcloud_env_cfg.ObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        pot_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(self)
        pot_joint_pos_pointcloud_env_cfg._configure_pointcloud_pot_stove_materials(self)
