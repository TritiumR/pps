# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.phone import mdp

from . import phone_joint_pos_visuomotor_env_cfg


PHONE_SOUND_PARAMS = {
    "phone_cfg": SceneEntityCfg("phone_1"),
    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    "mic_spacing": 0.25,
    "sample_rate": 48_000,
    "window_seconds": 2.0,
    "attenuation_power": 3.0,
    "n_mels": 80,
    "stft_window_ms": 25.0,
    "stft_hop_ms": 10.0,
    "n_fft": 2048,
}


@configclass
class ObservationsCfg(phone_joint_pos_visuomotor_env_cfg.ObservationsCfg):
    """Observation specifications for the phone task with virtual microphone sound."""

    @configclass
    class PolicyCfg(phone_joint_pos_visuomotor_env_cfg.ObservationsCfg.PolicyCfg):
        mic1_log_mel = ObsTerm(
            func=mdp.phone_mic1_log_mel,
            params=PHONE_SOUND_PARAMS,
        )
        mic2_log_mel = ObsTerm(
            func=mdp.phone_mic2_log_mel,
            params=PHONE_SOUND_PARAMS,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidPhoneJointPosSoundEnvCfg(
    phone_joint_pos_visuomotor_env_cfg.DroidPhoneJointPosVisuomotorEnvCfg
):
    """Configuration for phone task with Droid robot, joint position control, and sound observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.phone_2.spawn.visual_material = None
        self.scene.phone_2.spawn.visual_material_path = "material"
