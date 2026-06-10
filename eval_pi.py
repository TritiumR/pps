import os
import sys
import subprocess

# Use the IsaacLab copy bundled in this repository (self-contained), not an
# external checkout. Prepend its source packages so `import isaaclab*` resolves
# from here regardless of the current working directory. Because tasks load from
# this in-repo copy, their `os.path.dirname(__file__)`-relative asset paths also
# resolve to <repo>/IsaacLab/assets.
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _isaaclab_pkg in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_mimic",
):
    _isaaclab_pkg_src = os.path.join(_ISAACLAB_DIR, "source", _isaaclab_pkg)
    if _isaaclab_pkg_src not in sys.path:
        sys.path.insert(0, _isaaclab_pkg_src)

# Let PhysX and the policy share the GPU without JAX grabbing most of it up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from isaaclab.app import AppLauncher
import pinocchio

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# from openpi.policies import libero_policy

import time
from multiprocessing.managers import SharedMemoryManager
import cv2
import numpy as np
import torch
import dill
import hydra
import pandas as pd
import argparse
import copy
from tqdm import tqdm
from botocore.exceptions import NoCredentialsError

# from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict_droid,
)

import scipy.spatial.transform as R


_SOUND_VIDEO_SCALE = None
_SOUND_VIDEO_MAX_DISTANCE_M = 0.05
_SOUND_AUDIO_CACHE = None
_SOUND_AUDIO_SAMPLE_RATE = 48_000
_SOUND_AUDIO_ATTENUATION_POWER = 2.0
_SOUND_AUDIO_REFERENCE_DISTANCE = 1.0
_SOUND_AUDIO_MIN_DISTANCE = 1e-3


def _phone_ringtone_path():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav",
        )
    )


def _to_numpy_unbatched(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _model_type_value(train_config):
    model_type = getattr(getattr(train_config, "model", None), "model_type", None)
    return getattr(model_type, "value", str(model_type))


def _policy_needs_sound(train_config):
    model = getattr(train_config, "model", None)
    return bool(getattr(model, "use_sound_prefix", False)) or _model_type_value(train_config) == "proxy_sound"


def _policy_needs_pointcloud(train_config):
    model = getattr(train_config, "model", None)
    return bool(getattr(model, "use_pointcloud_prefix", False)) or _model_type_value(train_config) in {
        "proxy_pointcloud",
        "proxy_dp3",
    }


def _policy_uses_thermal_overlay(train_config):
    data_name = type(getattr(train_config, "data", None)).__name__.lower()
    config_name = str(getattr(train_config, "name", "")).lower()
    return "thermal" in data_name or "thermal" in config_name


def _first_existing(env_obs_dict, keys):
    for key in keys:
        if key in env_obs_dict:
            return env_obs_dict[key]
    return None


def _add_sound_observation(obs, env_obs_dict):
    mic1 = _first_existing(env_obs_dict, ("mic1_log_mel", "sound_mic1_log_mel"))
    mic2 = _first_existing(env_obs_dict, ("mic2_log_mel", "sound_mic2_log_mel"))
    sound = _first_existing(env_obs_dict, ("sound",))
    if sound is not None:
        obs["observation/sound"] = _to_numpy_unbatched(sound).astype(np.float32)
    elif mic1 is not None and mic2 is not None:
        obs["observation/mic1_log_mel"] = _to_numpy_unbatched(mic1).astype(np.float32)
        obs["observation/mic2_log_mel"] = _to_numpy_unbatched(mic2).astype(np.float32)


def _add_pointcloud_observation(obs, env_obs_dict):
    pointcloud = _first_existing(env_obs_dict, ("pointcloud",))
    if pointcloud is not None:
        obs["observation/pointcloud"] = _to_numpy_unbatched(pointcloud).astype(np.float32)
        return

    coord = _first_existing(
        env_obs_dict,
        ("point_positions", "point_position", "pointcloud_coord", "pointcloud_positions"),
    )
    color = _first_existing(env_obs_dict, ("point_color", "pointcloud_color"))
    if coord is not None and color is not None:
        obs["observation/pointcloud_coord"] = _to_numpy_unbatched(coord).astype(np.float32)
        obs["observation/pointcloud_color"] = _to_numpy_unbatched(color).astype(np.float32)


def get_pi_observation(env_obs_dict, train_config=None):
    obs = dict()
    use_thermal_overlay = train_config is not None and _policy_uses_thermal_overlay(train_config)
    # print(env_obs_dict['table_cam'].cpu().numpy()[0].shape)
    table_cam = _to_numpy_unbatched(env_obs_dict["table_cam"])
    wrist_cam = _to_numpy_unbatched(env_obs_dict["wrist_cam"])
    if use_thermal_overlay and "thermal_table_cam" in env_obs_dict:
        table_cam = _overlay_thermal_on_rgb(table_cam, env_obs_dict["thermal_table_cam"])
    if use_thermal_overlay and "thermal_wrist_cam" in env_obs_dict:
        wrist_cam = _overlay_thermal_on_rgb(wrist_cam, env_obs_dict["thermal_wrist_cam"])
    obs["observation/exterior_image_1_left"] = table_cam
    obs["observation/wrist_image_left"] = wrist_cam
    # print(env_obs_dict['joint_pos'].cpu().numpy()[0])
    joint_pos = _to_numpy_unbatched(env_obs_dict["joint_pos"])
    obs["observation/joint_position"] = joint_pos[:7]
    obs["observation/gripper_position"] = (
        joint_pos[7:8]
    )
    if train_config is not None and _policy_needs_sound(train_config):
        _add_sound_observation(obs, env_obs_dict)
    if train_config is not None and _policy_needs_pointcloud(train_config):
        _add_pointcloud_observation(obs, env_obs_dict)

    return obs


def _to_numpy_image(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _to_uint8_rgb(image):
    image = _to_numpy_image(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.size > 0 and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _get_camera_rgb(env, camera_name):
    try:
        camera = env.scene[camera_name]
    except KeyError:
        return None

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return None
    return _to_uint8_rgb(rgb)


def _overlay_thermal_on_rgb(rgb_image, thermal_image, alpha=0.45):
    rgb_image = _to_uint8_rgb(rgb_image)
    thermal_image = _to_uint8_rgb(thermal_image)

    if rgb_image.shape[:2] != thermal_image.shape[:2]:
        thermal_image = cv2.resize(
            thermal_image,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.addWeighted(rgb_image, 1.0 - alpha, thermal_image, alpha, 0.0)


def _has_sound_observation(env_policy_obs):
    if env_policy_obs is None:
        return False
    return "mic1_log_mel" in env_policy_obs and "mic2_log_mel" in env_policy_obs


def _to_numpy_spectrogram(spectrogram):
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.detach().cpu().numpy()
    spectrogram = np.asarray(spectrogram)
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]
    while spectrogram.ndim > 2 and spectrogram.shape[0] == 1:
        spectrogram = spectrogram[0]
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected spectrogram shape [F, T], got {spectrogram.shape}.")
    return spectrogram.astype(np.float32)


def _build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    mel_fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for mel_idx in range(n_mels):
        left = bin_indices[mel_idx]
        center = bin_indices[mel_idx + 1]
        right = bin_indices[mel_idx + 2]
        if center > left:
            mel_fb[mel_idx, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            mel_fb[mel_idx, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return mel_fb


def _get_sound_video_scale():
    global _SOUND_VIDEO_SCALE
    if _SOUND_VIDEO_SCALE is not None:
        return _SOUND_VIDEO_SCALE

    sample_rate = 48_000
    n_fft = 2048
    n_mels = 80
    f_min = 50.0
    eps = 1e-8
    reference_distance = 1.0
    attenuation_power = _SOUND_AUDIO_ATTENUATION_POWER
    ringtone_path = _phone_ringtone_path()

    try:
        import soundfile as sf
        from scipy import signal

        audio, sr = sf.read(ringtone_path, always_2d=True)
        audio = audio.mean(axis=1).astype(np.float64)
        if sr != sample_rate:
            gcd = np.gcd(sr, sample_rate)
            audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak

        win_length = int(round(25.0 / 1000.0 * sample_rate))
        hop_length = int(round(10.0 / 1000.0 * sample_rate))
        _, _, zxx = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        mel_fb = _build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=sample_rate / 2,
        )
        base_log_mel = np.log(mel_fb @ (np.abs(zxx) ** 2) + eps)
        gain_at_max = 2.0 * attenuation_power * np.log(reference_distance / _SOUND_VIDEO_MAX_DISTANCE_M)
        vmax = float(np.percentile(base_log_mel, 99) + gain_at_max)
    except Exception as exc:
        print(f"[WARN] Failed to compute global sound video scale: {exc}")
        vmax = 2.0

    vmin = float(np.log(eps))
    if vmax <= vmin:
        vmax = vmin + 1.0
    _SOUND_VIDEO_SCALE = (vmin, vmax)
    return _SOUND_VIDEO_SCALE


def _load_phone_ringtone_audio(sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    global _SOUND_AUDIO_CACHE
    if _SOUND_AUDIO_CACHE is not None and _SOUND_AUDIO_CACHE["sample_rate"] == sample_rate:
        return _SOUND_AUDIO_CACHE["audio"]

    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(_phone_ringtone_path(), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak
    if len(audio) == 0:
        raise ValueError("Phone ringtone WAV is empty.")

    _SOUND_AUDIO_CACHE = {"sample_rate": sample_rate, "audio": audio.astype(np.float32)}
    return _SOUND_AUDIO_CACHE["audio"]


def _quat_apply_wxyz(quat, vec):
    quat_xyz = quat[..., 1:]
    quat_w = quat[..., :1]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def _get_gripper_mic_distances(env, mic_spacing=0.25, mic_axis=(0.0, 1.0, 0.0)):
    try:
        ee_frame = env.scene["ee_frame"]
        phone = env.scene["phone_1"]
    except KeyError:
        return None

    ee_pos_w = ee_frame.data.target_pos_w[:1, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:1, 0, :]
    axis_local = torch.tensor(mic_axis, dtype=ee_pos_w.dtype, device=ee_pos_w.device).view(1, 3)
    axis_local = axis_local / torch.linalg.vector_norm(axis_local, dim=1, keepdim=True).clamp_min(1e-6)
    axis_w = _quat_apply_wxyz(ee_quat_w, axis_local)

    mic_offset_w = 0.5 * mic_spacing * axis_w
    mic1_pos_w = ee_pos_w + mic_offset_w
    mic2_pos_w = ee_pos_w - mic_offset_w
    phone_pos_w = phone.data.root_pos_w[:1]

    distance_mic1 = torch.linalg.vector_norm(mic1_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    distance_mic2 = torch.linalg.vector_norm(mic2_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    return float(distance_mic1.item()), float(distance_mic2.item())


def _build_stereo_sound_audio_frame(
    env,
    frame_index,
    fps=15,
    sample_rate=_SOUND_AUDIO_SAMPLE_RATE,
):
    distances = _get_gripper_mic_distances(env)
    if distances is None:
        return None

    source_audio = _load_phone_ringtone_audio(sample_rate=sample_rate)
    start_sample = int(round(frame_index * sample_rate / fps))
    end_sample = int(round((frame_index + 1) * sample_rate / fps))
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    mono = source_audio[np.mod(sample_indices, len(source_audio))]

    distance_mic1, distance_mic2 = distances
    gain_mic1 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic1) ** _SOUND_AUDIO_ATTENUATION_POWER
    gain_mic2 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic2) ** _SOUND_AUDIO_ATTENUATION_POWER
    return np.stack((mono * gain_mic1, mono * gain_mic2), axis=1).astype(np.float32)


def _write_stereo_sound_audio(audio_path, audio_frames, sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    if not audio_frames:
        return None

    import soundfile as sf

    audio = np.concatenate(audio_frames, axis=0)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def _visualize_log_mel_spectrogram(
    spectrogram,
    target_height,
    target_width,
    label,
    vmin=None,
    vmax=None,
):
    spectrogram = _to_numpy_spectrogram(spectrogram)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None:
        vmin = float(np.percentile(spectrogram, 1))
    if vmax is None:
        vmax = float(np.percentile(spectrogram, 99))
    if vmax <= vmin:
        normalized = np.zeros_like(spectrogram, dtype=np.float32)
    else:
        normalized = (spectrogram - vmin) / (vmax - vmin)

    image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    image = np.flipud(image)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    cv2.putText(
        image,
        label,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _build_visualization_frame(obs, env, use_thermal_overlay, env_policy_obs=None):
    table_image = _to_uint8_rgb(obs["observation/exterior_image_1_left"])
    wrist_image = _to_uint8_rgb(obs["observation/wrist_image_left"])

    if use_thermal_overlay:
        table_rgb = _get_camera_rgb(env, "table_cam")
        wrist_rgb = _get_camera_rgb(env, "wrist_cam")
        if table_rgb is not None:
            table_image = _overlay_thermal_on_rgb(table_rgb, table_image)
        if wrist_rgb is not None:
            wrist_image = _overlay_thermal_on_rgb(wrist_rgb, wrist_image)

    rgb_frame = np.concatenate((table_image, wrist_image), axis=1)

    if _has_sound_observation(env_policy_obs):
        spectrogram_width = max(1, rgb_frame.shape[1] // 2)
        mic1_spectrogram = _to_numpy_spectrogram(env_policy_obs["mic1_log_mel"])
        mic2_spectrogram = _to_numpy_spectrogram(env_policy_obs["mic2_log_mel"])
        mic1_spectrogram = np.nan_to_num(mic1_spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
        mic2_spectrogram = np.nan_to_num(mic2_spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
        mic_vmin, mic_vmax = _get_sound_video_scale()
        mic1_image = _visualize_log_mel_spectrogram(
            mic1_spectrogram,
            target_height=rgb_frame.shape[0],
            target_width=spectrogram_width,
            label="mic1_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        mic2_image = _visualize_log_mel_spectrogram(
            mic2_spectrogram,
            target_height=rgb_frame.shape[0],
            target_width=spectrogram_width,
            label="mic2_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        return np.concatenate((mic1_image, rgb_frame, mic2_image), axis=1)

    return rgb_frame


def _write_rgb_video_frame(video_writer, video_path, frame_rgb, fps=15):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
        )
    video_writer.write(frame_bgr)
    return video_writer


def _episode_video_name(seed, success, suffix=""):
    video_name = f"{seed}{suffix}"
    if success is None:
        return video_name + "_invalid.mp4"
    if success:
        return video_name + "_success.mp4"
    return video_name + "_fail.mp4"


def _finalize_video(tmp_video_path, final_video_path, video_writer, label, audio_path=None):
    if video_writer is not None and os.path.exists(tmp_video_path):
        if audio_path is not None and os.path.exists(audio_path):
            mux_cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                tmp_video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                final_video_path,
            ]
            try:
                subprocess.run(mux_cmd, check=True)
                os.remove(tmp_video_path)
                os.remove(audio_path)
                print(f"{label} video saved with stereo sound")
                return
            except Exception as exc:
                print(f"[WARN] Failed to mux {label} audio into video: {exc}")
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        os.replace(tmp_video_path, final_video_path)
        print(f"{label} video saved")
    else:
        print(f"no {label} frames recorded; skipping video save")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the model on the real droid robot."
    )
    parser.add_argument("--model_name", type=str, default="pi05_droid")
    parser.add_argument("--task", type=str, required=True)
    # parser.add_argument("--prompt", type=str, default="pick up the red cube and put it on the blue cube, then pick up the green cube and put it on the red cube")
    parser.add_argument(
        "--prompt", type=str, default="stack the red cube on the blue cube"
    )
    parser.add_argument("--name", type=str, default="demo")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to the output directory."
    )
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=21)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory. If not provided, will try to download from S3 or use default location.",
    )
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--ood_mode", action="store_true")
    parser.add_argument("--ood_mode_light", type=str, default="none") # "light_intensity", "light_color", "light_texture", "all" or "none"
    parser.add_argument("--ood_mode_camera", type=str, default="none") # "camera_position", "camera_orientation", "all" or "none"
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

# output path
output_path = os.path.join("results", f"{args.task}/{args.name}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Make the robot env
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import asyncio
import gymnasium as gym
import inspect
import random

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# Setup output paths and get env name
output_dir = os.path.join("results", f"{args.task}/{args.name}")
output_file_name = "eval.hdf5"
task_name = args.task
if task_name:
    task_name = args.task.split(":")[-1]
env_name = task_name

print(f"Environment name: {env_name}")
use_thermal_overlay_video = "thermal" in env_name.lower()
if use_thermal_overlay_video:
    print("Thermal task detected; saving RGB/thermal overlay rollout videos.")

# Configure environment
env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)

env_cfg.env_name = env_name

# For ood evaluation
env_cfg.eval_mode = args.ood_mode
env_cfg.light_eval_type = args.ood_mode_light
env_cfg.camera_eval_type = args.ood_mode_camera

# Extract success checking function
success_term = None
if hasattr(env_cfg.terminations, "success"):
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
else:
    raise NotImplementedError(
        "No success termination term was found in the environment."
    )

# # Configure for data generation
# env_cfg.terminations = None
# env_cfg.observations.policy.concatenate_terms = False

# Create environment
env = gym.make(env_name, cfg=env_cfg).unwrapped

# load checkpoint
config = _config.get_config(args.model_name)
checkpoint_dir = args.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    # checkpoint_dir = download.maybe_download(
    #     f"gs://openpi-assets/checkpoints/{args.model_name}"
    # )
    checkpoint_dir = download.maybe_download(
        f"gs://openpi-assets-simeval/{args.model_name}"
    )

# Create a trained policy.
vla_policy = policy_config.create_trained_policy(config, checkpoint_dir)

steps_per_inference = 8
CONTROL_FREQUENCY = 15
max_steps = args.max_steps

# example = libero_policy.make_libero_example()

# print(example)

# Warmup inference once
env_obs_dict, _ = env.reset()
obs = get_pi_observation(env_obs_dict["policy"], config)
obs["prompt"] = args.prompt

initial_obs = copy.deepcopy(obs)

print("Warming up policy inference")
with torch.no_grad():
    actions = vla_policy.infer(copy.deepcopy(obs))["actions"]

print("Ready!")
for seed in range(args.seed_start, args.seed_end):
    success = None

    # Set seed for generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reset before starting
    env_obs_dict, _ = env.reset()
    # Execute an initial neutral action to reset the action buffer
    # Use current joint positions with open gripper (gripper value 1.0 = open)
    initial_joint_pos = env_obs_dict["policy"]["joint_pos"].cpu().numpy()[0]
    initial_action = np.concatenate([initial_joint_pos[:7], [0.0]])  # Open gripper
    env_obs_dict, _, _, _, _ = env.step(
        torch.tensor(initial_action[None], dtype=torch.float32)
    )

    obs = get_pi_observation(env_obs_dict["policy"], config)
    obs["prompt"] = args.prompt

    tmp_video_path = os.path.join(output_path, f"{seed}_recording.mp4")
    tmp_audio_path = os.path.join(output_path, f"{seed}_recording.wav")
    video_writer = None
    sound_audio_frames = []
    tmp_thermal_video_paths = {}
    thermal_video_writers = {}
    if use_thermal_overlay_video:
        tmp_thermal_video_paths = {
            "table": os.path.join(output_path, f"{seed}_thermal_table_recording.mp4"),
            "wrist": os.path.join(output_path, f"{seed}_thermal_wrist_recording.mp4"),
        }
        thermal_video_writers = {"table": None, "wrist": None}

    # hack_action = np.ones(8) * -0.001
    # ========== policy control loop ==============
    step_idx = 0
    success = False
    for step_idx in tqdm(range(max_steps), desc="Policy Control Loop"):
        try:
            s = time.time()

            if step_idx % steps_per_inference == 0:
                # print('predict_action')
                # run inference
                with torch.no_grad():
                    actions = vla_policy.infer(copy.deepcopy(obs))["actions"]
                    # print('actions: ', actions.shape)
                    # print("Inference latency:", time.time() - s)

                # execute actions
                start_idx = 0
                end_idx = start_idx + steps_per_inference
                actions = actions[start_idx:end_idx]

            action_step = actions[step_idx % steps_per_inference]

            # velocity to delta action
            # action_step[:7] = action_step[:7] / CONTROL_FREQUENCY
            # action_step[-1] = action_step[-1] - 0.01

            # delta action to absolute action
            # action_step[:7] += obs["observation/joint_position"]
            # action_step[-1] = action_step[-1] * 2 - 1
            # action_step[:7] += initial_obs['observation/joint_position']

            # print("action_step: ", action_step)
            # print("gripper_action_step: ", action_step[-1])

            # perform step
            env_obs_dict, rewards, terminated, truncated, extras = env.step(
                torch.tensor(action_step[None], dtype=torch.float32)
            )

            # if step_idx % 20 == 0:
            #     # Reset
            #     env_obs_dict, _ = env.reset()

            obs = get_pi_observation(env_obs_dict["policy"], config)
            obs["prompt"] = args.prompt

            # Check for task success using success_term
            task_success = bool(success_term.func(env, **success_term.params)[0])

            # save visualization
            vis_image = _build_visualization_frame(
                obs,
                env,
                use_thermal_overlay=use_thermal_overlay_video,
                env_policy_obs=env_obs_dict["policy"],
            )
            video_writer = _write_rgb_video_frame(video_writer, tmp_video_path, vis_image)
            if _has_sound_observation(env_obs_dict["policy"]):
                sound_audio_frame = _build_stereo_sound_audio_frame(
                    env,
                    frame_index=len(sound_audio_frames),
                    fps=CONTROL_FREQUENCY,
                )
                if sound_audio_frame is not None:
                    sound_audio_frames.append(sound_audio_frame)

            if use_thermal_overlay_video:
                thermal_frames = {
                    "table": _to_uint8_rgb(obs["observation/exterior_image_1_left"]),
                    "wrist": _to_uint8_rgb(obs["observation/wrist_image_left"]),
                }
                for camera_name, thermal_frame in thermal_frames.items():
                    thermal_video_writers[camera_name] = _write_rgb_video_frame(
                        thermal_video_writers[camera_name],
                        tmp_thermal_video_paths[camera_name],
                        thermal_frame,
                    )

            # obs_list.pop(0)
            # obs_list.append(copy.deepcopy(obs))
            # update buffers

            step_idx += 1

            if terminated or truncated or task_success:
                print("terminated or truncated or task completed")
                success = task_success
                break

        except KeyboardInterrupt:
            print("Interrupted!")
            break

    if video_writer is not None:
        video_writer.release()
    for thermal_video_writer in thermal_video_writers.values():
        if thermal_video_writer is not None:
            thermal_video_writer.release()
    audio_path = _write_stereo_sound_audio(tmp_audio_path, sound_audio_frames)

    # save video
    if success is None:
        print("invalid")
    elif success:
        print("success")
    else:
        print("fail")

    video_name = _episode_video_name(seed, success)
    video_path = os.path.join(output_path, video_name)
    _finalize_video(tmp_video_path, video_path, video_writer, "overlay", audio_path=audio_path)

    if use_thermal_overlay_video:
        for camera_name, tmp_thermal_video_path in tmp_thermal_video_paths.items():
            thermal_video_name = _episode_video_name(
                seed,
                success,
                suffix=f"_thermal_{camera_name}",
            )
            thermal_video_path = os.path.join(output_path, thermal_video_name)
            _finalize_video(
                tmp_thermal_video_path,
                thermal_video_path,
                thermal_video_writers[camera_name],
                f"thermal {camera_name}",
            )

env.close()

simulation_app.close()
