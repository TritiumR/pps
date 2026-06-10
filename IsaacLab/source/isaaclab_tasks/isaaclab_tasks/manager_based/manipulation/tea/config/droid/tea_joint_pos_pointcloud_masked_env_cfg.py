# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.tea import mdp

from . import tea_joint_pos_pointcloud_env_cfg


_POINTCLOUD_MASK_DATA_TYPE = "instance_id_segmentation_fast"
_POINTCLOUD_MASK_ROOTS = ("Robotiq_2F_85", "E_teapot_5", "E_teacup005_20")


def _configure_masked_pointcloud_table_cameras(env_cfg):
    """Enable instance-id segmentation on the point-cloud cameras for masking."""
    tea_joint_pos_pointcloud_env_cfg._configure_pointcloud_table_cameras(env_cfg)

    env_cfg.scene.table_cam.data_types = [
        "rgb",
        "distance_to_image_plane",
        _POINTCLOUD_MASK_DATA_TYPE,
    ]
    env_cfg.scene.table_cam.colorize_instance_id_segmentation = False

    env_cfg.scene.table_cam_mirror.data_types = [
        "rgb",
        "distance_to_image_plane",
        _POINTCLOUD_MASK_DATA_TYPE,
    ]
    env_cfg.scene.table_cam_mirror.colorize_instance_id_segmentation = False


@configclass
class ObservationsCfg(tea_joint_pos_pointcloud_env_cfg.ObservationsCfg):
    """Observation specifications for the masked point-cloud tea MDP."""

    @configclass
    class PolicyCfg(tea_joint_pos_pointcloud_env_cfg.ObservationsCfg.PolicyCfg):
        point_positions = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_positions,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 1024,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )
        point_color = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_color,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 1024,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidTeaJointPosPointCloudMaskedEnvCfg(
    tea_joint_pos_pointcloud_env_cfg.DroidTeaJointPosPointCloudEnvCfg
):
    """Point-cloud tea task with per-camera segmentation masking for relevant scene objects."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_masked_pointcloud_table_cameras(self)

