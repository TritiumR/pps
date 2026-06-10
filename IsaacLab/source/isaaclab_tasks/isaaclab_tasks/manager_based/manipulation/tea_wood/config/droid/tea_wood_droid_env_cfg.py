# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.tea.config.droid import (
    tea_ik_rel_pointcloud_env_cfg,
    tea_ik_rel_pointcloud_masked_env_cfg,
    tea_ik_rel_visuomotor_env_cfg,
    tea_joint_pos_pointcloud_env_cfg,
    tea_joint_pos_pointcloud_masked_env_cfg,
    tea_joint_pos_visuomotor_env_cfg,
)
from isaaclab_tasks.manager_based.manipulation.tea.mdp import tea_events
from isaaclab_tasks.manager_based.manipulation.tea_wood.tea_wood_env_cfg import (
    TEA_BOARD_MATERIAL_PATH,
    TEACUP_MESH_PATH,
    TEAPOT_BODY_MESH_PATH,
    TEAPOT_LID_MESH_PATH,
)


@configclass
class EventCfg(tea_joint_pos_visuomotor_env_cfg.EventCfg):
    """Droid tea events with teapot and cup bound to the tea-board material."""

    teapot_body_wood_material = EventTerm(
        func=tea_events.bind_existing_visual_material,
        mode="prestartup",
        params={
            "prim_path_regex": TEAPOT_BODY_MESH_PATH,
            "material_path_regex": TEA_BOARD_MATERIAL_PATH,
        },
    )

    teapot_lid_wood_material = EventTerm(
        func=tea_events.bind_existing_visual_material,
        mode="prestartup",
        params={
            "prim_path_regex": TEAPOT_LID_MESH_PATH,
            "material_path_regex": TEA_BOARD_MATERIAL_PATH,
        },
    )

    teacup_wood_material = EventTerm(
        func=tea_events.bind_existing_visual_material,
        mode="prestartup",
        params={
            "prim_path_regex": TEACUP_MESH_PATH,
            "material_path_regex": TEA_BOARD_MATERIAL_PATH,
        },
    )


def _use_tea_wood_events(env_cfg):
    """Replace the parent tea event config with the wood-textured variant."""
    env_cfg.events = EventCfg()


@configclass
class DroidTeaWoodJointPosVisuomotorEnvCfg(
    tea_joint_pos_visuomotor_env_cfg.DroidTeaJointPosVisuomotorEnvCfg
):
    """Wood-textured tea task with Droid robot using joint position control."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)


@configclass
class DroidTeaWoodIkRelVisuomotorEnvCfg(
    tea_ik_rel_visuomotor_env_cfg.DroidTeaIkRelVisuomotorEnvCfg
):
    """Wood-textured tea task with Droid robot using relative IK control."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)


@configclass
class DroidTeaWoodJointPosPointCloudEnvCfg(
    tea_joint_pos_pointcloud_env_cfg.DroidTeaJointPosPointCloudEnvCfg
):
    """Wood-textured tea task with Droid robot using joint position control and point clouds."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)


@configclass
class DroidTeaWoodIkRelPointCloudEnvCfg(
    tea_ik_rel_pointcloud_env_cfg.DroidTeaIkRelPointCloudEnvCfg
):
    """Wood-textured tea task with Droid robot using relative IK control and point clouds."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)


@configclass
class DroidTeaWoodJointPosPointCloudMaskedEnvCfg(
    tea_joint_pos_pointcloud_masked_env_cfg.DroidTeaJointPosPointCloudMaskedEnvCfg
):
    """Wood-textured tea task with Droid robot using joint position control and masked point clouds."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)


@configclass
class DroidTeaWoodIkRelPointCloudMaskedEnvCfg(
    tea_ik_rel_pointcloud_masked_env_cfg.DroidTeaIkRelPointCloudMaskedEnvCfg
):
    """Wood-textured tea task with Droid robot using relative IK control and masked point clouds."""

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_tea_wood_events(self)

