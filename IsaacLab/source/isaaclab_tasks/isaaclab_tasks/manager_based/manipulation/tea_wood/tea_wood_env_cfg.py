# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.tea.mdp import tea_events
from isaaclab_tasks.manager_based.manipulation.tea.tea_env_cfg import EventCfg as TeaEventCfg
from isaaclab_tasks.manager_based.manipulation.tea.tea_env_cfg import TeaEnvCfg

TEA_BOARD_MATERIAL_PATH = "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/materials/mat_E3C99DBD88780441"
TEAPOT_BODY_MESH_PATH = "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teapot_5/P_c680523765f8dbb9"
TEAPOT_LID_MESH_PATH = "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teapot_5/E_Tealid_6/P_26d56f6b8359dbb9"
TEACUP_MESH_PATH = "{ENV_REGEX_NS}/interactive_diningroom/model_TeaTable/E_teacup005_20/P_d10277eee33f9aca"


@configclass
class EventCfg(TeaEventCfg):
    """Tea task events with teapot and cup bound to the tea-board material."""

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


@configclass
class TeaWoodEnvCfg(TeaEnvCfg):
    """Tea task variant with teapot and cup using the tea-board material."""

    events: EventCfg = EventCfg()

