"""Visualize phone RGB and sound observations from an HDF5 dataset.

The output frame layout matches the phone sound rollout visualization in eval_pi.py:

    mic1_log_mel | table_cam + wrist_cam | mic2_log_mel

Example:
    python scripts/tools/visualize_phone_sound_dataset.py \
        --input_file ../diffusion_policy/data/phone/generated_dataset_sound.hdf5 \
        --output_dir ../diffusion_policy/data/phone/generated_dataset_sound_videos \
        --fps 15
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import cv2
import h5py
import numpy as np


SOUND_VIDEO_MAX_DISTANCE_M = 0.05
SOUND_AUDIO_ATTENUATION_POWER = 2.0
SOUND_AUDIO_SAMPLE_RATE = 48_000
SOUND_AUDIO_EPS = 1e-8


def phone_ringtone_path() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav",
        )
    )


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.size > 0 and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def to_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = np.asarray(spectrogram)
    while spectrogram.ndim > 2 and spectrogram.shape[0] == 1:
        spectrogram = spectrogram[0]
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected spectrogram shape [F, T], got {spectrogram.shape}.")
    return spectrogram.astype(np.float32)


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
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


def eval_pi_sound_video_scale() -> tuple[float, float]:
    sample_rate = 48_000
    n_fft = 2048
    n_mels = 80
    eps = 1e-8
    reference_distance = 1.0
    ringtone_path = phone_ringtone_path()

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
        mel_fb = build_mel_filterbank(sample_rate, n_fft, n_mels, 50.0, sample_rate / 2)
        base_log_mel = np.log(mel_fb @ (np.abs(zxx) ** 2) + eps)
        gain_at_max = 2.0 * SOUND_AUDIO_ATTENUATION_POWER * np.log(reference_distance / SOUND_VIDEO_MAX_DISTANCE_M)
        vmax = float(np.percentile(base_log_mel, 99) + gain_at_max)
    except Exception as exc:
        print(f"[WARN] Failed to compute eval_pi sound scale from ringtone: {exc}")
        vmax = 2.0

    vmin = float(np.log(eps))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_phone_ringtone_audio(sample_rate: int = SOUND_AUDIO_SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(phone_ringtone_path(), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak
    if len(audio) == 0:
        raise ValueError("Phone ringtone WAV is empty.")
    return audio.astype(np.float32)


def compute_base_log_mel(
    source_audio: np.ndarray,
    frame_idx: int,
    fps: int,
    mel_fb: np.ndarray,
    sample_rate: int = SOUND_AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    from scipy import signal

    window_seconds = 2.0
    win_length = int(round(25.0 / 1000.0 * sample_rate))
    hop_length = int(round(10.0 / 1000.0 * sample_rate))
    n_fft = 2048

    window_samples = int(round(window_seconds * sample_rate))
    end_sample = int(round(frame_idx * sample_rate / fps))
    start_sample = end_sample - window_samples
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    window = source_audio[np.mod(sample_indices, len(source_audio))]

    _, _, zxx = signal.stft(
        window,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return np.log(mel_fb @ (np.abs(zxx) ** 2) + SOUND_AUDIO_EPS).astype(np.float32)


def estimate_audio_gain(mic_log_mel: np.ndarray, base_log_mel: np.ndarray) -> float:
    mic_log_mel = to_spectrogram(mic_log_mel)
    if mic_log_mel.shape != base_log_mel.shape:
        raise ValueError(f"Mic/base spectrogram shape mismatch: {mic_log_mel.shape} vs {base_log_mel.shape}")

    delta = mic_log_mel - base_log_mel
    finite = np.isfinite(delta)
    if not np.any(finite):
        return 0.0

    log_power_gain = float(np.median(delta[finite]))
    gain = float(np.exp(0.5 * log_power_gain))
    return max(gain, 0.0)


def build_audio_chunk(
    source_audio: np.ndarray,
    mic1_log_mel: np.ndarray,
    mic2_log_mel: np.ndarray,
    frame_idx: int,
    fps: int,
    mel_fb: np.ndarray,
    sample_rate: int = SOUND_AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    base_log_mel = compute_base_log_mel(source_audio, frame_idx, fps, mel_fb, sample_rate)
    gain_mic1 = estimate_audio_gain(mic1_log_mel, base_log_mel)
    gain_mic2 = estimate_audio_gain(mic2_log_mel, base_log_mel)

    start_sample = int(round(frame_idx * sample_rate / fps))
    end_sample = int(round((frame_idx + 1) * sample_rate / fps))
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    mono = source_audio[np.mod(sample_indices, len(source_audio))]
    return np.stack((mono * gain_mic1, mono * gain_mic2), axis=1).astype(np.float32)


def write_stereo_audio(audio_path: str, audio_chunks: list[np.ndarray], sample_rate: int = SOUND_AUDIO_SAMPLE_RATE):
    if not audio_chunks:
        return None

    import soundfile as sf

    audio = np.concatenate(audio_chunks, axis=0)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def mux_audio_into_video(video_path: str, audio_path: str, output_path: str):
    mux_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(mux_cmd, check=True)


def dataset_sound_video_scale(
    data_group: h5py.Group,
    demo_names: list[str],
    mic_keys: tuple[str, str],
    stride: int,
) -> tuple[float, float]:
    values = []
    for demo_name in demo_names:
        obs_group = data_group[demo_name]["obs"]
        for mic_key in mic_keys:
            if mic_key not in obs_group:
                continue
            values.append(np.asarray(obs_group[mic_key][::stride]).reshape(-1))

    if not values:
        raise KeyError(f"Could not find sound keys {mic_keys} in selected demos.")

    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        return float(np.log(1e-8)), 2.0

    vmin = float(np.percentile(all_values, 1))
    vmax = float(np.percentile(all_values, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def visualize_log_mel_spectrogram(
    spectrogram: np.ndarray,
    target_height: int,
    target_width: int,
    label: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    spectrogram = to_spectrogram(spectrogram)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)

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


def build_visualization_frame(
    table_image: np.ndarray,
    wrist_image: np.ndarray,
    mic1_log_mel: np.ndarray,
    mic2_log_mel: np.ndarray,
    sound_vmin: float,
    sound_vmax: float,
) -> np.ndarray:
    table_image = to_uint8_rgb(table_image)
    wrist_image = to_uint8_rgb(wrist_image)

    if table_image.shape[:2] != wrist_image.shape[:2]:
        wrist_image = cv2.resize(
            wrist_image,
            (table_image.shape[1], table_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    rgb_frame = np.concatenate((table_image, wrist_image), axis=1)
    spectrogram_width = max(1, rgb_frame.shape[1] // 2)
    mic1_image = visualize_log_mel_spectrogram(
        mic1_log_mel,
        target_height=rgb_frame.shape[0],
        target_width=spectrogram_width,
        label="mic1_log_mel",
        vmin=sound_vmin,
        vmax=sound_vmax,
    )
    mic2_image = visualize_log_mel_spectrogram(
        mic2_log_mel,
        target_height=rgb_frame.shape[0],
        target_width=spectrogram_width,
        label="mic2_log_mel",
        vmin=sound_vmin,
        vmax=sound_vmax,
    )
    return np.concatenate((mic1_image, rgb_frame, mic2_image), axis=1)


def open_video_writer(path: str, frame_shape: tuple[int, int, int], fps: int) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def parse_episode_filter(episodes: str | None) -> set[int] | None:
    if episodes is None:
        return None
    return {int(item.strip()) for item in episodes.split(",") if item.strip()}


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def export_demo_video(
    demo_group: h5py.Group,
    output_path: str,
    fps: int,
    camera_keys: tuple[str, str],
    mic_keys: tuple[str, str],
    sound_vmin: float,
    sound_vmax: float,
    write_audio: bool,
):
    obs_group = demo_group["obs"]
    table_key, wrist_key = camera_keys
    mic1_key, mic2_key = mic_keys

    for key in (table_key, wrist_key, mic1_key, mic2_key):
        if key not in obs_group:
            raise KeyError(f"Missing obs/{key}. Available obs keys: {list(obs_group.keys())}")

    num_frames = min(
        obs_group[table_key].shape[0],
        obs_group[wrist_key].shape[0],
        obs_group[mic1_key].shape[0],
        obs_group[mic2_key].shape[0],
    )
    if num_frames <= 0:
        print(f"  Skipping {output_path}: no frames")
        return

    temp_video_path = output_path
    temp_audio_path = None
    if write_audio:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        temp_video = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="phone_sound_video_", dir=output_dir, delete=False
        )
        temp_video_path = temp_video.name
        temp_video.close()

    audio_chunks = []
    source_audio = None
    mel_fb = None
    if write_audio:
        try:
            source_audio = load_phone_ringtone_audio()
            mel_fb = build_mel_filterbank(SOUND_AUDIO_SAMPLE_RATE, 2048, 80, 50.0, SOUND_AUDIO_SAMPLE_RATE / 2)
        except Exception as exc:
            print(f"  [WARN] Failed to prepare audio track; writing silent video only: {exc}")
            write_audio = False

    first_frame = build_visualization_frame(
        obs_group[table_key][0],
        obs_group[wrist_key][0],
        obs_group[mic1_key][0],
        obs_group[mic2_key][0],
        sound_vmin,
        sound_vmax,
    )
    writer = open_video_writer(temp_video_path, first_frame.shape, fps)
    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    if write_audio:
        audio_chunks.append(
            build_audio_chunk(source_audio, obs_group[mic1_key][0], obs_group[mic2_key][0], 0, fps, mel_fb)
        )

    for frame_idx in range(1, num_frames):
        frame = build_visualization_frame(
            obs_group[table_key][frame_idx],
            obs_group[wrist_key][frame_idx],
            obs_group[mic1_key][frame_idx],
            obs_group[mic2_key][frame_idx],
            sound_vmin,
            sound_vmax,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if write_audio:
            audio_chunks.append(
                build_audio_chunk(
                    source_audio,
                    obs_group[mic1_key][frame_idx],
                    obs_group[mic2_key][frame_idx],
                    frame_idx,
                    fps,
                    mel_fb,
                )
            )

    writer.release()
    if write_audio:
        temp_audio = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="phone_sound_audio_", dir=os.path.dirname(output_path), delete=False
        )
        temp_audio_path = temp_audio.name
        temp_audio.close()
        try:
            write_stereo_audio(temp_audio_path, audio_chunks)
            mux_audio_into_video(temp_video_path, temp_audio_path, output_path)
            os.remove(temp_video_path)
            os.remove(temp_audio_path)
            print(f"  Saved: {output_path} ({num_frames} frames, stereo audio)")
            return
        except Exception as exc:
            print(f"  [WARN] Failed to mux audio; keeping video-only output: {exc}")
            if temp_audio_path is not None and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            os.replace(temp_video_path, output_path)

    print(f"  Saved: {output_path} ({num_frames} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize phone RGB and sound observations from HDF5 demos.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to '<input_file>_phone_sound_videos'.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS.")
    parser.add_argument("--episodes", type=str, default=None, help="Comma-separated demo indices, e.g. '0,1,5'.")
    parser.add_argument("--table_key", type=str, default="table_cam", help="Table camera observation key.")
    parser.add_argument("--wrist_key", type=str, default="wrist_cam", help="Wrist camera observation key.")
    parser.add_argument("--mic1_key", type=str, default="mic1_log_mel", help="Mic 1 log-Mel observation key.")
    parser.add_argument("--mic2_key", type=str, default="mic2_log_mel", help="Mic 2 log-Mel observation key.")
    parser.add_argument("--no_audio", action="store_true", help="Do not mux reconstructed stereo audio into the MP4.")
    parser.add_argument(
        "--normalization",
        choices=("eval_pi", "dataset"),
        default="eval_pi",
        help="Sound color scale. 'eval_pi' uses the same ringtone/0.05m global scale as eval_pi.py.",
    )
    parser.add_argument(
        "--scale_stride",
        type=int,
        default=4,
        help="Frame stride used only for --normalization dataset.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.splitext(args.input_file)[0] + "_phone_sound_videos"

    requested_episodes = parse_episode_filter(args.episodes)

    with h5py.File(args.input_file, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"{args.input_file} does not contain a 'data' group.")

        data_group = h5_file["data"]
        demo_names = sorted_demo_names(data_group)
        if requested_episodes is not None:
            demo_names = [name for name in demo_names if int(name.split("_")[-1]) in requested_episodes]
        if not demo_names:
            raise ValueError("No selected demos found.")

        if args.normalization == "dataset":
            sound_vmin, sound_vmax = dataset_sound_video_scale(
                data_group,
                demo_names,
                (args.mic1_key, args.mic2_key),
                max(args.scale_stride, 1),
            )
        else:
            sound_vmin, sound_vmax = eval_pi_sound_video_scale()

        print(f"Found {len(demo_names)} selected episodes in {args.input_file}")
        print(f"Sound visualization scale: vmin={sound_vmin:.4f}, vmax={sound_vmax:.4f}")

        for demo_name in demo_names:
            demo_idx = int(demo_name.split("_")[-1])
            demo_group = data_group[demo_name]
            success = demo_group.attrs.get("success", None)
            status = "unknown" if success is None else ("success" if bool(success) else "fail")
            output_path = os.path.join(args.output_dir, f"{demo_name}_{status}_phone_sound.mp4")
            print(f"[{demo_name}] success={success}")
            export_demo_video(
                demo_group,
                output_path,
                fps=args.fps,
                camera_keys=(args.table_key, args.wrist_key),
                mic_keys=(args.mic1_key, args.mic2_key),
                sound_vmin=sound_vmin,
                sound_vmax=sound_vmax,
                write_audio=not args.no_audio,
            )

    print(f"Done. Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
