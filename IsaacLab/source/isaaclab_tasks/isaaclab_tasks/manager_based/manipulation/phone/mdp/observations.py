# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the phone task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_PHONE_AUDIO_CONFIG = {
    "sample_rate": 48_000,
    "window_seconds": 2.0,
    "speed_of_sound": 343.0,
    "attenuation_power": 1.0,
    "reference_distance": 1.0,
    "min_distance": 1e-3,
    "stft_window_ms": 25.0,
    "stft_hop_ms": 10.0,
    "n_fft": 2048,
    "n_mels": 80,
    "f_min": 50.0,
    "f_max": None,
    "eps": 1e-8,
}


def _default_phone_ringtone_path() -> str:
    """Return the default ringtone path in the phone task package."""
    return str(Path(__file__).resolve().parents[1] / "ringtone.wav")


def _load_looped_phone_audio(wav_path: str, sample_rate: int) -> np.ndarray:
    """Load a mono WAV, normalize it, and resample if needed."""
    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(wav_path, always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak
    if len(audio) == 0:
        raise ValueError(f"Phone ringtone WAV is empty: {wav_path}")

    return audio


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _build_phone_mel_filterbank(config: dict) -> np.ndarray:
    """Build the same triangular Mel filterbank used by scripts/tools/generate_spectrogram.py."""
    sample_rate = config["sample_rate"]
    n_fft = config["n_fft"]
    f_max = config["f_max"] if config["f_max"] is not None else sample_rate / 2

    mel_min = _hz_to_mel(config["f_min"])
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, config["n_mels"] + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    mel_fb = np.zeros((config["n_mels"], n_fft // 2 + 1), dtype=np.float64)
    for mel_idx in range(config["n_mels"]):
        left = bin_indices[mel_idx]
        center = bin_indices[mel_idx + 1]
        right = bin_indices[mel_idx + 2]
        if center > left:
            mel_fb[mel_idx, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            mel_fb[mel_idx, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return mel_fb


def _compute_phone_base_log_mel(source_audio: np.ndarray, policy_time_seconds: float, config: dict) -> np.ndarray:
    """Compute one reference-distance log-Mel window using the spectrogram tool's STFT settings."""
    from scipy import signal

    sample_rate = config["sample_rate"]
    window_samples = int(round(config["window_seconds"] * sample_rate))
    end_sample = int(round(policy_time_seconds * sample_rate))
    start_sample = end_sample - window_samples

    # Treat the ringtone as an infinite loop. This lets timestep 0 observe the
    # window [-2s, 0s] instead of a zero-padded pre-episode window.
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    window = source_audio[np.mod(sample_indices, len(source_audio))]

    win_length = int(round(config["stft_window_ms"] / 1000.0 * sample_rate))
    hop_length = int(round(config["stft_hop_ms"] / 1000.0 * sample_rate))
    _, _, zxx = signal.stft(
        window,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=config["n_fft"],
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    mel = _build_phone_mel_filterbank(config) @ power
    return np.log(mel + config["eps"]).astype(np.float32)


def _get_phone_base_log_mel_for_steps(
    env: ManagerBasedRLEnv,
    episode_steps: torch.Tensor,
    wav_path: str | None,
    config: dict,
) -> torch.Tensor:
    """Return cached reference-distance log-Mel windows for each environment episode step."""
    resolved_wav_path = wav_path if wav_path is not None else _default_phone_ringtone_path()
    cache_key = (
        resolved_wav_path,
        config["sample_rate"],
        config["window_seconds"],
        config["stft_window_ms"],
        config["stft_hop_ms"],
        config["n_fft"],
        config["n_mels"],
        config["f_min"],
        config["f_max"],
        env.cfg.episode_length_s,
        env.step_dt,
    )
    cache = getattr(env, "_phone_audio_log_mel_cache", None)
    if cache is None or cache.get("key") != cache_key:
        cache = {
            "key": cache_key,
            "source_audio": _load_looped_phone_audio(resolved_wav_path, config["sample_rate"]),
            "step_log_mel": {},
        }
        setattr(env, "_phone_audio_log_mel_cache", cache)

    unique_steps, inverse = torch.unique(episode_steps.detach().to("cpu"), sorted=False, return_inverse=True)
    base_windows = []
    for step in unique_steps.tolist():
        step = int(step)
        if step not in cache["step_log_mel"]:
            cache["step_log_mel"][step] = _compute_phone_base_log_mel(
                source_audio=cache["source_audio"],
                policy_time_seconds=step * env.step_dt,
                config=config,
            )
        base_windows.append(torch.from_numpy(cache["step_log_mel"][step]))

    base_stack = torch.stack(base_windows, dim=0).to(device=env.device)
    return base_stack[inverse.to(device=env.device)]


def _phone_microphone_log_mels(
    env: ManagerBasedRLEnv,
    phone_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    wav_path: str | None = None,
    mic_spacing: float = 0.25,
    mic_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    sample_rate: int = 48_000,
    window_seconds: float = 2.0,
    speed_of_sound: float = 343.0,
    attenuation_power: float = 1.0,
    reference_distance: float = 1.0,
    min_distance: float = 1e-3,
    stft_window_ms: float = 25.0,
    stft_hop_ms: float = 10.0,
    n_fft: int = 2048,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Generate two gripper-mounted virtual microphone log-Mel observations."""
    config = {
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "speed_of_sound": speed_of_sound,
        "attenuation_power": attenuation_power,
        "reference_distance": reference_distance,
        "min_distance": min_distance,
        "stft_window_ms": stft_window_ms,
        "stft_hop_ms": stft_hop_ms,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "f_min": f_min,
        "f_max": f_max,
        "eps": eps,
    }
    episode_steps_key = tuple(int(step) for step in env.episode_length_buf.detach().to("cpu").tolist())
    cache_key = (
        env.common_step_counter,
        episode_steps_key,
        phone_cfg.name,
        ee_frame_cfg.name,
        wav_path,
        mic_spacing,
        mic_axis,
        tuple(sorted(config.items(), key=lambda item: item[0])),
    )
    cached = getattr(env, "_phone_microphone_log_mels_step_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["value"]

    phone: RigidObject = env.scene[phone_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    axis_local = torch.tensor(mic_axis, dtype=ee_pos_w.dtype, device=env.device).unsqueeze(0)
    axis_local = axis_local / torch.linalg.vector_norm(axis_local, dim=1, keepdim=True).clamp_min(1e-6)
    axis_w = math_utils.quat_apply(ee_quat_w, axis_local.repeat(ee_pos_w.shape[0], 1))

    mic_offset_w = 0.5 * mic_spacing * axis_w
    mic1_pos_w = ee_pos_w + mic_offset_w
    mic2_pos_w = ee_pos_w - mic_offset_w

    phone_pos_w = phone.data.root_pos_w
    distance_mic1 = torch.linalg.vector_norm(mic1_pos_w - phone_pos_w, dim=1).clamp_min(config["min_distance"])
    distance_mic2 = torch.linalg.vector_norm(mic2_pos_w - phone_pos_w, dim=1).clamp_min(config["min_distance"])

    base_log_mel = _get_phone_base_log_mel_for_steps(
        env=env,
        episode_steps=env.episode_length_buf,
        wav_path=wav_path,
        config=config,
    )

    attenuation_power = config["attenuation_power"]
    reference_distance = config["reference_distance"]
    mic1_log_gain_power = 2.0 * attenuation_power * torch.log(reference_distance / distance_mic1)
    mic2_log_gain_power = 2.0 * attenuation_power * torch.log(reference_distance / distance_mic2)

    value = {
        "mic1_log_mel": base_log_mel + mic1_log_gain_power.view(-1, 1, 1),
        "mic2_log_mel": base_log_mel + mic2_log_gain_power.view(-1, 1, 1),
    }
    setattr(env, "_phone_microphone_log_mels_step_cache", {"key": cache_key, "value": value})
    return value


def phone_mic1_log_mel(
    env: ManagerBasedRLEnv,
    phone_cfg: SceneEntityCfg = SceneEntityCfg("phone_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wav_path: str | None = None,
    mic_spacing: float = 0.25,
    mic_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    sample_rate: int = 48_000,
    window_seconds: float = 2.0,
    speed_of_sound: float = 343.0,
    attenuation_power: float = 1.0,
    reference_distance: float = 1.0,
    min_distance: float = 1e-3,
    stft_window_ms: float = 25.0,
    stft_hop_ms: float = 10.0,
    n_fft: int = 2048,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-Mel spectrogram for virtual microphone 1 on the gripper."""
    return _phone_microphone_log_mels(
        env,
        phone_cfg=phone_cfg,
        ee_frame_cfg=ee_frame_cfg,
        wav_path=wav_path,
        mic_spacing=mic_spacing,
        mic_axis=mic_axis,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        speed_of_sound=speed_of_sound,
        attenuation_power=attenuation_power,
        reference_distance=reference_distance,
        min_distance=min_distance,
        stft_window_ms=stft_window_ms,
        stft_hop_ms=stft_hop_ms,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        eps=eps,
    )["mic1_log_mel"]


def phone_mic2_log_mel(
    env: ManagerBasedRLEnv,
    phone_cfg: SceneEntityCfg = SceneEntityCfg("phone_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wav_path: str | None = None,
    mic_spacing: float = 0.25,
    mic_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    sample_rate: int = 48_000,
    window_seconds: float = 2.0,
    speed_of_sound: float = 343.0,
    attenuation_power: float = 1.0,
    reference_distance: float = 1.0,
    min_distance: float = 1e-3,
    stft_window_ms: float = 25.0,
    stft_hop_ms: float = 10.0,
    n_fft: int = 2048,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-Mel spectrogram for virtual microphone 2 on the gripper."""
    return _phone_microphone_log_mels(
        env,
        phone_cfg=phone_cfg,
        ee_frame_cfg=ee_frame_cfg,
        wav_path=wav_path,
        mic_spacing=mic_spacing,
        mic_axis=mic_axis,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        speed_of_sound=speed_of_sound,
        attenuation_power=attenuation_power,
        reference_distance=reference_distance,
        min_distance=min_distance,
        stft_window_ms=stft_window_ms,
        stft_hop_ms=stft_hop_ms,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        eps=eps,
    )["mic2_log_mel"]


def object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The position of the object in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_pos_w


def object_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The orientation of the object in the world frame."""
    rigid_object: RigidObject = env.scene[object_cfg.name]

    return rigid_object.data.root_quat_w


def ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert (
                len(gripper_joint_ids) == 2
            ), "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = (
                robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            )
            finger_joint_2 = -1 * robot.data.joint_pos[
                :, gripper_joint_ids[1]
            ].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError(
                "[Error] Cannot find gripper_joint_names in the environment config"
            )


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    # print(f"pose_diff: {pose_diff}")

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(
            -1, 1
        )  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert (
                len(gripper_joint_ids) == 2
            ), "Observations only support parallel gripper for now"

            gripper_1 = torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                    env.device
                )
            )
            gripper_2 = torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                    env.device
                )
            )

            # print(f"gripper_1: {gripper_1}")
            # print(f"gripper_2: {gripper_2}")
            # print(f"env.cfg.gripper_threshold: {env.cfg.gripper_threshold}")

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    )
                )
                > env.cfg.gripper_threshold,
            )
            # print(f"grasped: {grasped}")
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(
                        env.device
                    )
                )
                > env.cfg.gripper_threshold,
            )
            # print(f"grasped: {grasped}")

    return grasped


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    elif return_key is None:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
