# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.capsule import mdp

from . import capsule_joint_pos_visuomotor_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values and point clouds."""

        actions = ObsTerm(func=mdp.last_action)
        joint_actions = ObsTerm(func=mdp.last_droid_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        point_positions = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_positions,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 2048,
                "normalize_color": False,
            },
        )
        point_color = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_color,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 2048,
                "normalize_color": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        open_coffee_lid = ObsTerm(
            func=mdp.coffee_lid_opened,
            params={
                "capsule_cfg": SceneEntityCfg("capsule"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        grasp_pod = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("can"),
                "diff_threshold": 0.1,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class DroidCapsuleJointPosPointCloudEnvCfg(
    capsule_joint_pos_visuomotor_env_cfg.DroidCapsuleJointPosVisuomotorEnvCfg
):
    """Configuration for capsule task with Droid robot using joint position control and point clouds."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.table_cam.data_types = ["rgb", "distance_to_image_plane"]
        self.scene.wrist_cam = None

        self.scene.table_cam_mirror = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0/table_cam_mirror",
            height=720,
            width=1280,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0476,
                horizontal_aperture=2.5452,
                vertical_aperture=1.4721,
                clipping_range=(1e-4, 5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0046203368, 0.5388594867, 0.4540183681),
                rot=(0.5078392969, 0.7575422903, 0.3175587775, 0.2595868830),
                convention="ros",
            ),
        )

        self.scene.table_cam.height = 720
        self.scene.table_cam.width = 1280
        self.scene.table_cam_mirror.height = 720
        self.scene.table_cam_mirror.width = 1280

        self.image_obs_list = ["table_cam", "table_cam_mirror"]
