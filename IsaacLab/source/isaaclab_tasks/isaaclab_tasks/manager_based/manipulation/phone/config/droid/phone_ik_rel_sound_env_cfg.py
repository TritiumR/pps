# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import phone_ik_rel_visuomotor_env_cfg, phone_joint_pos_sound_env_cfg


@configclass
class DroidPhoneIkRelSoundEnvCfg(
    phone_ik_rel_visuomotor_env_cfg.DroidPhoneIkRelVisuomotorEnvCfg
):
    """Configuration for phone task with Droid robot, IK control, and sound observations."""

    observations: phone_joint_pos_sound_env_cfg.ObservationsCfg = phone_joint_pos_sound_env_cfg.ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.phone_2.spawn.visual_material = None
        self.scene.phone_2.spawn.visual_material_path = "material"
