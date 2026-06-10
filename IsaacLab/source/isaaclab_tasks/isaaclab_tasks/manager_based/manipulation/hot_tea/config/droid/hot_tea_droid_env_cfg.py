# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.hot_tea import hot_tea_env_cfg
from isaaclab_tasks.manager_based.manipulation.tea import mdp
from isaaclab_tasks.manager_based.manipulation.tea.config.droid import (
    tea_ik_rel_pointcloud_env_cfg,
    tea_ik_rel_pointcloud_masked_env_cfg,
    tea_ik_rel_visuomotor_env_cfg,
    tea_joint_pos_pointcloud_env_cfg,
    tea_joint_pos_pointcloud_masked_env_cfg,
    tea_joint_pos_visuomotor_env_cfg,
)
from isaaclab_tasks.manager_based.manipulation.tea.mdp import tea_events
from isaaclab_tasks.manager_based.manipulation.tea.tea_env_cfg import (
    TEAPOT_GRASP_DIFF_THRESHOLD,
    TEAPOT_MAX_RELATIVE_ROLL_RAD,
    TEAPOT_MOUTH_LOCAL_OFFSET,
    TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
    TEAPOT_ROLL_THRESHOLD_RAD,
)

HOT_TEA_OBJECT_MIN_SEPARATION = 0.18
_POINTCLOUD_MASK_DATA_TYPE = "instance_id_segmentation_fast"
_POINTCLOUD_MASK_ROOTS = (
    "Robotiq_2F_85",
    "E_teapot_5",
    hot_tea_env_cfg.HOT_TEAPOT_PRIM_NAME,
    "E_teacup005_20",
)


@configclass
class EventCfg(hot_tea_env_cfg.EventCfg):
    """Droid hot tea events with three randomized tea objects."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_tea_objects = EventTerm(
        func=tea_events.randomize_object_pose,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg("teapot"),
                SceneEntityCfg("teacup"),
                SceneEntityCfg("hot_teapot"),
            ],
            "pose_range": {
                "x": (
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[0]
                    - tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_X_RANGE / 2.0,
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[0]
                    + tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_X_RANGE / 2.0,
                ),
                "y": (
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[1]
                    - tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_Y_RANGE / 2.0,
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[1]
                    + tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_Y_RANGE / 2.0,
                ),
                "z": (
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[2],
                    tea_joint_pos_visuomotor_env_cfg.TEA_OBJECT_RESET_CENTER[2],
                ),
                "roll": (np.pi / 2.0, np.pi / 2.0),
                "pitch": (0.0, 0.0),
                "yaw": (np.pi / 2.0 - 0.5, np.pi / 2.0 + 0.5),
            },
            "min_separation": HOT_TEA_OBJECT_MIN_SEPARATION,
        },
    )

    init_franka_arm_pose = EventTerm(
        func=tea_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0,
                -1 / 5 * np.pi,
                0.0,
                -4 / 5 * np.pi,
                0.0,
                3 / 5 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=tea_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ObservationsCfg(tea_joint_pos_visuomotor_env_cfg.ObservationsCfg):
    """Hot tea visuomotor observations targeting the hot teapot."""

    @configclass
    class SubtaskCfg(tea_joint_pos_visuomotor_env_cfg.ObservationsCfg.SubtaskCfg):
        grasp_teapot = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("hot_teapot"),
                "diff_threshold": TEAPOT_GRASP_DIFF_THRESHOLD,
            },
        )

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PointCloudObservationsCfg(tea_joint_pos_pointcloud_env_cfg.ObservationsCfg):
    """Hot tea point-cloud observations targeting the hot teapot."""

    @configclass
    class SubtaskCfg(ObservationsCfg.SubtaskCfg):
        pass

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Hot tea terminations targeting the hot teapot."""

    teapot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("hot_teapot"),
        },
    )

    teacup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("teacup"),
        },
    )

    teapot_over_rolled = DoneTerm(
        func=mdp.teapot_relative_roll_exceeds_max,
        params={
            "teapot_cfg": SceneEntityCfg("hot_teapot"),
            "max_relative_roll_rad": TEAPOT_MAX_RELATIVE_ROLL_RAD,
        },
    )

    success = DoneTerm(
        func=mdp.task_done_tea,
        params={
            "teapot_cfg": SceneEntityCfg("hot_teapot"),
            "teacup_cfg": SceneEntityCfg("teacup"),
            "mouth_offset": TEAPOT_MOUTH_LOCAL_OFFSET,
            "xy_threshold": TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
            "min_roll_rad": TEAPOT_ROLL_THRESHOLD_RAD,
        },
    )


@configclass
class MaskedPointCloudObservationsCfg(tea_joint_pos_pointcloud_masked_env_cfg.ObservationsCfg):
    """Masked point-cloud observations including the extra teapot."""

    @configclass
    class PolicyCfg(tea_joint_pos_pointcloud_masked_env_cfg.ObservationsCfg.PolicyCfg):
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

    @configclass
    class SubtaskCfg(ObservationsCfg.SubtaskCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


def _use_hot_tea_events(env_cfg):
    """Replace the parent tea event config with the hot tea variant."""
    env_cfg.events = EventCfg()


@configclass
class DroidHotTeaJointPosVisuomotorEnvCfg(
    tea_joint_pos_visuomotor_env_cfg.DroidTeaJointPosVisuomotorEnvCfg
):
    """Hot tea task with Droid robot using joint position control."""

    observations: ObservationsCfg = ObservationsCfg()
    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)


@configclass
class DroidHotTeaIkRelVisuomotorEnvCfg(
    tea_ik_rel_visuomotor_env_cfg.DroidTeaIkRelVisuomotorEnvCfg
):
    """Hot tea task with Droid robot using relative IK control."""

    observations: ObservationsCfg = ObservationsCfg()
    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)


@configclass
class DroidHotTeaJointPosPointCloudEnvCfg(
    tea_joint_pos_pointcloud_env_cfg.DroidTeaJointPosPointCloudEnvCfg
):
    """Hot tea task with Droid robot using joint position control and point clouds."""

    observations: PointCloudObservationsCfg = PointCloudObservationsCfg()
    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)


@configclass
class DroidHotTeaIkRelPointCloudEnvCfg(
    tea_ik_rel_pointcloud_env_cfg.DroidTeaIkRelPointCloudEnvCfg
):
    """Hot tea task with Droid robot using relative IK control and point clouds."""

    observations: PointCloudObservationsCfg = PointCloudObservationsCfg()
    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)


@configclass
class DroidHotTeaJointPosPointCloudMaskedEnvCfg(
    tea_joint_pos_pointcloud_masked_env_cfg.DroidTeaJointPosPointCloudMaskedEnvCfg
):
    """Hot tea task with Droid robot using joint position control and masked point clouds."""

    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    observations: MaskedPointCloudObservationsCfg = MaskedPointCloudObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)


@configclass
class DroidHotTeaIkRelPointCloudMaskedEnvCfg(
    tea_ik_rel_pointcloud_masked_env_cfg.DroidTeaIkRelPointCloudMaskedEnvCfg
):
    """Hot tea task with Droid robot using relative IK control and masked point clouds."""

    scene: hot_tea_env_cfg.HotTeaSceneCfg = hot_tea_env_cfg.HotTeaSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    events: EventCfg = EventCfg()
    observations: MaskedPointCloudObservationsCfg = MaskedPointCloudObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_hot_tea_events(self)
