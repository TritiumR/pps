# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.knife import mdp

from . import knife_joint_pos_visuomotor_env_cfg


def _configure_pointcloud_table_cameras(env_cfg):
    """Augment the original table camera with depth and add a mirrored capture camera."""
    table_cam = env_cfg.scene.table_cam
    table_cam.data_types = ["rgb", "distance_to_image_plane"]

    pos = table_cam.offset.pos
    rot = table_cam.offset.rot
    env_cfg.scene.table_cam_mirror = CameraCfg(
        prim_path=str(table_cam.prim_path).replace("table_cam", "table_cam_mirror"),
        height=table_cam.height,
        width=table_cam.width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=table_cam.spawn,
        offset=CameraCfg.OffsetCfg(
            pos=(pos[0], -pos[1], pos[2]),
            rot=(-rot[0], rot[1], -rot[2], rot[3]),
            convention=table_cam.offset.convention,
        ),
    )


@configclass
class ObservationsCfg(knife_joint_pos_visuomotor_env_cfg.ObservationsCfg):
    """Observation specifications for the point-cloud knife MDP."""

    @configclass
    class PolicyCfg(knife_joint_pos_visuomotor_env_cfg.ObservationsCfg.PolicyCfg):
        point_positions = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_positions,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 8192,
                "normalize_color": False,
            },
        )
        point_color = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_color,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 8192,
                "normalize_color": False,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidKnifeJointPosPointCloudEnvCfg(
    knife_joint_pos_visuomotor_env_cfg.DroidKnifeJointPosVisuomotorEnvCfg
):
    """Configuration for the knife task with Droid robot using joint position control and point cloud observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_pointcloud_table_cameras(self)
