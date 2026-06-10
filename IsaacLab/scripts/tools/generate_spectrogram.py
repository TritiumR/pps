from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioWindowConfig:
    sample_rate: int = 48_000
    window_seconds: float = 2.0

    speed_of_sound: float = 343.0
    attenuation_power: float = 1.0
    reference_distance: float = 1.0
    min_distance: float = 1e-3

    # STFT parameters
    stft_window_ms: float = 25.0
    stft_hop_ms: float = 10.0
    n_fft: int = 2048

    # Mel parameters
    n_mels: int = 80
    f_min: float = 50.0
    f_max: float | None = None

    eps: float = 1e-8
    noise_std: float = 0.0


def load_mono_wav(path: str | Path, target_sr: int) -> np.ndarray:
    """
    Load a WAV file, convert to mono, resample if needed, and normalize.
    """
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)

    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = signal.resample_poly(audio, up=up, down=down)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


def repeat_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Repeat source audio until it reaches target_length.
    Useful when the ringtone WAV is shorter than the episode.
    """
    if len(audio) == 0:
        raise ValueError("Input audio is empty.")

    num_repeats = int(np.ceil(target_length / len(audio)))
    repeated = np.tile(audio, num_repeats)
    return repeated[:target_length]


def fractional_delay(
    audio: np.ndarray,
    delay_samples: float,
    output_length: int,
) -> np.ndarray:
    """
    Apply positive fractional delay.

    y[n] = x[n - delay_samples]
    """
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative.")

    source_index = np.arange(output_length, dtype=np.float64) - delay_samples
    original_index = np.arange(len(audio), dtype=np.float64)

    return np.interp(
        source_index,
        original_index,
        audio,
        left=0.0,
        right=0.0,
    )


def simulate_two_mic_audio(
    source_audio: np.ndarray,
    distance_mic1: float,
    distance_mic2: float,
    config: AudioWindowConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulate two microphone waveforms from a mono ringtone.

    Args:
        source_audio:
            Mono source waveform, shape [T].
        distance_mic1:
            Distance from ringing phone to microphone 1, in meters.
        distance_mic2:
            Distance from ringing phone to microphone 2, in meters.
        config:
            Audio simulation config.

    Returns:
        mic_audio:
            Simulated two-channel waveform, shape [2, T_out].
        metadata:
            Gain and delay information.
    """
    distances = np.array([distance_mic1, distance_mic2], dtype=np.float64)
    distances = np.maximum(distances, config.min_distance)

    absolute_delays_sec = distances / config.speed_of_sound

    # Only relative delay matters for the policy.
    relative_delays_sec = absolute_delays_sec - absolute_delays_sec.min()
    relative_delays_samples = relative_delays_sec * config.sample_rate

    gains = (config.reference_distance / distances) ** config.attenuation_power

    output_length = len(source_audio) + int(np.ceil(relative_delays_samples.max())) + 1
    mic_audio = np.zeros((2, output_length), dtype=np.float64)

    for mic_idx in range(2):
        delayed = fractional_delay(
            audio=source_audio,
            delay_samples=relative_delays_samples[mic_idx],
            output_length=output_length,
        )
        mic_audio[mic_idx] = gains[mic_idx] * delayed

    if config.noise_std > 0:
        mic_audio += np.random.normal(0.0, config.noise_std, size=mic_audio.shape)

    # Global normalization preserves mic1/mic2 relative loudness.
    peak = np.max(np.abs(mic_audio))
    if peak > 1.0:
        mic_audio = mic_audio / peak

    metadata = {
        "distance_mic1": float(distances[0]),
        "distance_mic2": float(distances[1]),
        "gain_mic1": float(gains[0]),
        "gain_mic2": float(gains[1]),
        "relative_delay_mic1_samples": float(relative_delays_samples[0]),
        "relative_delay_mic2_samples": float(relative_delays_samples[1]),
        "inter_mic_delay_samples": float(
            (absolute_delays_sec[1] - absolute_delays_sec[0]) * config.sample_rate
        ),
        "inter_mic_delay_seconds": float(absolute_delays_sec[1] - absolute_delays_sec[0]),
    }

    return mic_audio, metadata


def crop_past_window(
    mic_audio: np.ndarray,
    policy_time_seconds: float,
    config: AudioWindowConfig,
) -> np.ndarray:
    """
    Crop the past fixed-length audio window ending at policy_time_seconds.

    Returns:
        window_audio:
            Shape [2, window_samples].
    """
    if mic_audio.ndim != 2 or mic_audio.shape[0] != 2:
        raise ValueError(f"Expected mic_audio shape [2, T], got {mic_audio.shape}.")

    window_samples = int(round(config.window_seconds * config.sample_rate))
    end_sample = int(round(policy_time_seconds * config.sample_rate))
    start_sample = end_sample - window_samples

    window = np.zeros((2, window_samples), dtype=np.float64)

    src_start = max(start_sample, 0)
    src_end = min(end_sample, mic_audio.shape[1])

    if src_end > src_start:
        dst_start = src_start - start_sample
        dst_end = dst_start + (src_end - src_start)
        window[:, dst_start:dst_end] = mic_audio[:, src_start:src_end]

    return window


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(config: AudioWindowConfig) -> np.ndarray:
    """
    Build a triangular Mel filterbank.

    Returns:
        mel_fb:
            Shape [n_mels, n_fft // 2 + 1].
    """
    sr = config.sample_rate
    f_max = config.f_max if config.f_max is not None else sr / 2

    mel_min = hz_to_mel(config.f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_indices = np.floor((config.n_fft + 1) * hz_points / sr).astype(int)
    bin_indices = np.clip(bin_indices, 0, config.n_fft // 2)

    mel_fb = np.zeros((config.n_mels, config.n_fft // 2 + 1), dtype=np.float64)

    for mel_idx in range(config.n_mels):
        left = bin_indices[mel_idx]
        center = bin_indices[mel_idx + 1]
        right = bin_indices[mel_idx + 2]

        if center > left:
            mel_fb[mel_idx, left:center] = (
                np.arange(left, center) - left
            ) / max(center - left, 1)

        if right > center:
            mel_fb[mel_idx, center:right] = (
                right - np.arange(center, right)
            ) / max(right - center, 1)

    return mel_fb


def compute_past_2s_spectrogram_features(
    window_audio: np.ndarray,
    config: AudioWindowConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute fixed-size spectrogram features from a two-mic past-window waveform.

    Returns:
        features:
            log_mel_mic1: [n_mels, T]
            log_mel_mic2: [n_mels, T]
            log_mel_avg:  [n_mels, T]
            log_mel_diff: [n_mels, T]
            policy_input: [2, n_mels, T], containing [avg, diff]
    """
    win_length = int(round(config.stft_window_ms / 1000.0 * config.sample_rate))
    hop_length = int(round(config.stft_hop_ms / 1000.0 * config.sample_rate))

    stft_values = []

    for mic_idx in range(2):
        _, _, zxx = signal.stft(
            window_audio[mic_idx],
            fs=config.sample_rate,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=config.n_fft,
            boundary=None,
            padded=False,
        )
        stft_values.append(zxx)

    # Shape [2, F, T]
    stft_pair = np.stack(stft_values, axis=0)

    power_mic1 = np.abs(stft_pair[0]) ** 2
    power_mic2 = np.abs(stft_pair[1]) ** 2

    mel_fb = build_mel_filterbank(config)

    mel_mic1 = mel_fb @ power_mic1
    mel_mic2 = mel_fb @ power_mic2

    log_mel_mic1 = np.log(mel_mic1 + config.eps)
    log_mel_mic2 = np.log(mel_mic2 + config.eps)

    log_mel_avg = 0.5 * (log_mel_mic1 + log_mel_mic2)
    log_mel_diff = log_mel_mic1 - log_mel_mic2

    # Main policy input: [C, F, T]
    policy_input = np.stack(
        [
            log_mel_avg,
            log_mel_diff,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "stft_pair": stft_pair,
        "log_mel_mic1": log_mel_mic1.astype(np.float32),
        "log_mel_mic2": log_mel_mic2.astype(np.float32),
        "log_mel_avg": log_mel_avg.astype(np.float32),
        "log_mel_diff": log_mel_diff.astype(np.float32),
        "policy_input": policy_input,
    }


def generate_past_2s_policy_audio_input(
    wav_path: str | Path,
    distance_mic1: float,
    distance_mic2: float,
    policy_time_seconds: float,
    episode_duration_seconds: float,
    config: AudioWindowConfig = AudioWindowConfig(),
) -> Dict[str, np.ndarray | float]:
    """
    Full pipeline:

        WAV ringtone
        -> repeat to episode length
        -> simulate two-mic waveform by distance
        -> crop past 2 seconds
        -> compute fixed-size spectrogram features

    Args:
        wav_path:
            Clean mono or stereo ringtone WAV. Stereo is converted to mono.
        distance_mic1:
            Source-to-mic1 distance in meters.
        distance_mic2:
            Source-to-mic2 distance in meters.
        policy_time_seconds:
            Current policy time. The returned spectrogram uses [t-2s, t].
        episode_duration_seconds:
            Length of simulated episode. Source ringtone is repeated to this length.
        config:
            Audio simulation and spectrogram config.

    Returns:
        Dictionary containing policy_input and metadata.
    """
    source = load_mono_wav(wav_path, target_sr=config.sample_rate)

    required_samples = int(round(episode_duration_seconds * config.sample_rate))
    source = repeat_to_length(source, required_samples)

    mic_audio, metadata = simulate_two_mic_audio(
        source_audio=source,
        distance_mic1=distance_mic1,
        distance_mic2=distance_mic2,
        config=config,
    )

    window_audio = crop_past_window(
        mic_audio=mic_audio,
        policy_time_seconds=policy_time_seconds,
        config=config,
    )

    features = compute_past_2s_spectrogram_features(
        window_audio=window_audio,
        config=config,
    )

    return {
        **features,
        **metadata,
        "window_audio": window_audio,
        "mic_audio": mic_audio,
        "policy_time_seconds": policy_time_seconds,
    }

r1 = 0.5
r2 = 0.4
current_time = 0.0
episode_duration = 10.0
config = AudioWindowConfig()

result = generate_past_2s_policy_audio_input(
    wav_path="source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav",
    distance_mic1=r1,
    distance_mic2=r2,
    policy_time_seconds=current_time,
    episode_duration_seconds=episode_duration,
    config=config,
)
