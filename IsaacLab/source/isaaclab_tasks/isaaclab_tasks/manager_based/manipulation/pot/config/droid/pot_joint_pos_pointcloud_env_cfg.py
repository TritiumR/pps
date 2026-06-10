# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pot import mdp
from isaaclab_tasks.manager_based.manipulation.pot.mdp import pot_events

from . import pot_joint_pos_visuomotor_env_cfg


_KITCHEN_ROOT = "{ENV_REGEX_NS}/interactive_kitchen_with_parlor"
_TABLE_MATERIAL_PATH = f"{_KITCHEN_ROOT}/model_largecabinet/materials/mat_1DB3671726E88062"
_POT_AND_STOVE_MESH_PATHS = [
    f"{_KITCHEN_ROOT}/model_kitchenware006/E_pot1_1",
    f"{_KITCHEN_ROOT}/model_kitchenware006/E_cover_2",
    f"{_KITCHEN_ROOT}/model_largecabinet/E_body_14/E_parts04_17",
    f"{_KITCHEN_ROOT}/model_largecabinet/E_body_14/E_parts05_18",
    f"{_KITCHEN_ROOT}/model_largecabinet/E_knob_01_89",
    f"{_KITCHEN_ROOT}/model_largecabinet/E_knob_02_90",
]

_POINTCLOUD_MASK_DATA_TYPE = "instance_id_segmentation_fast"
_POINTCLOUD_MASK_ROOTS = ("E_pot1_1", "E_cover_2", "egg", "Robotiq_2F_85")


def _configure_pointcloud_table_cameras(env_cfg):
    """Augment the original table camera with depth + instance segmentation, and add a mirrored capture camera."""
    table_cam = env_cfg.scene.table_cam
    table_cam.data_types = ["rgb", "distance_to_image_plane", _POINTCLOUD_MASK_DATA_TYPE]
    table_cam.colorize_instance_id_segmentation = False

    pos = table_cam.offset.pos
    rot = table_cam.offset.rot
    env_cfg.scene.table_cam_mirror = CameraCfg(
        prim_path=str(table_cam.prim_path).replace("table_cam", "table_cam_mirror"),
        height=table_cam.height,
        width=table_cam.width,
        data_types=["rgb", "distance_to_image_plane", _POINTCLOUD_MASK_DATA_TYPE],
        colorize_instance_id_segmentation=False,
        spawn=table_cam.spawn,
        offset=CameraCfg.OffsetCfg(
            pos=(pos[0], -pos[1], pos[2]),
            rot=(-rot[0], rot[1], -rot[2], rot[3]),
            convention=table_cam.offset.convention,
        ),
    )


def _configure_pointcloud_pot_stove_materials(env_cfg):
    """Rebind pot and stove visuals to share the kitchen counter (table) material."""
    env_cfg.events.retexture_pot_and_stove = EventTerm(
        func=pot_events.rebind_mesh_material,
        mode="prestartup",
        params={
            "target_mesh_prim_paths": list(_POT_AND_STOVE_MESH_PATHS),
            "source_material_prim_path": _TABLE_MATERIAL_PATH,
        },
    )


@configclass
class ObservationsCfg(pot_joint_pos_visuomotor_env_cfg.ObservationsCfg):
    """Observation specifications for the point-cloud pot MDP."""

    @configclass
    class PolicyCfg(pot_joint_pos_visuomotor_env_cfg.ObservationsCfg.PolicyCfg):
        point_positions = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_positions,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 512,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )
        point_color = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_color,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 512,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidPotJointPosPointCloudEnvCfg(
    pot_joint_pos_visuomotor_env_cfg.DroidPotJointPosVisuomotorEnvCfg
):
    """Configuration for the pot task with Droid robot using joint position control and point cloud observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_pointcloud_table_cameras(self)
        _configure_pointcloud_pot_stove_materials(self)
