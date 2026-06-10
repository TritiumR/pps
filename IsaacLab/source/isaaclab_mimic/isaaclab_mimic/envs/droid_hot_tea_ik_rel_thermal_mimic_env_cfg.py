# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.hot_tea.config.droid.hot_tea_droid_thermal_env_cfg import (
    DroidHotTeaIkRelThermalEnvCfg,
    ThermalObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.tea import mdp
from isaaclab_tasks.manager_based.manipulation.tea.tea_env_cfg import (
    TEAPOT_GRASP_DIFF_THRESHOLD,
    TEAPOT_MAX_RELATIVE_ROLL_RAD,
    TEAPOT_MOUTH_LOCAL_OFFSET,
    TEAPOT_MOUTH_TEACUP_XY_THRESHOLD,
    TEAPOT_ROLL_THRESHOLD_RAD,
    TerminationsCfg as TeaTerminationsCfg,
)


@configclass
class HotTeaThermalMimicObservationsCfg(ThermalObservationsCfg):
    """Thermal hot tea observations with subtask terms targeting the hot teapot."""

    @configclass
    class SubtaskCfg(ThermalObservationsCfg.SubtaskCfg):
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
class HotTeaThermalMimicTerminationsCfg(TeaTerminationsCfg):
    """Hot tea terminations with success and safety checks targeting the hot teapot."""

    teapot_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("hot_teapot"),
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
class DroidHotTeaIKRelThermalMimicEnvCfg(DroidHotTeaIkRelThermalEnvCfg, MimicEnvCfg):
    """Mimic config for Droid hot tea IK Rel thermal env."""

    observations: HotTeaThermalMimicObservationsCfg = HotTeaThermalMimicObservationsCfg()
    terminations: HotTeaThermalMimicTerminationsCfg = HotTeaThermalMimicTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "isaac_lab_droid_hot_tea_ik_rel_thermal_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        action_noise = 0.008

        self.subtask_configs["franka"] = [
            SubTaskConfig(
                object_ref="teapot",
                subtask_term_signal="grasp_teapot",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=action_noise,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="teacup",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=action_noise,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
