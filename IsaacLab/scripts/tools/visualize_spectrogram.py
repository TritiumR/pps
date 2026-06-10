from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from generate_spectrogram import AudioWindowConfig, hz_to_mel, mel_to_hz

def get_mel_center_frequencies(config: AudioWindowConfig) -> np.ndarray:
    """
    Approximate Mel-bin center frequencies for plotting.
    """
    f_max = config.f_max if config.f_max is not None else config.sample_rate / 2

    mel_min = hz_to_mel(config.f_min)
    mel_max = hz_to_mel(f_max)

    # Use center points, excluding the two edge points used for triangular filters.
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)[1:-1]
    return np.asarray(mel_to_hz(mel_points), dtype=np.float64)


def plot_single_spectrogram(
    spec: np.ndarray,
    title: str,
    config: AudioWindowConfig,
    save_path: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Plot one log-Mel spectrogram.

    Args:
        spec:
            Spectrogram array with shape [n_mels, time_frames].
        title:
            Figure title.
        config:
            Audio config.
        save_path:
            Optional path to save the figure.
        vmin, vmax:
            Optional fixed color limits.
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected spectrogram shape [F, T], got {spec.shape}.")

    hop_seconds = config.stft_hop_ms / 1000.0
    duration_seconds = spec.shape[1] * hop_seconds

    mel_freqs = get_mel_center_frequencies(config)

    plt.figure(figsize=(8, 4.5))
    plt.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration_seconds, mel_freqs[0], mel_freqs[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="log power")
    plt.xlabel("Time in past 2s window (s)")
    plt.ylabel("Frequency (Hz, Mel bins)")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def visualize_past_2s_audio_features(
    result: Mapping[str, np.ndarray],
    config: AudioWindowConfig,
    save_dir: str | Path | None = None,
) -> None:
    """
    Visualize simulated two-microphone spectrogram features.

    Required result keys:
        log_mel_mic1
        log_mel_mic2
        log_mel_avg
        log_mel_diff
    """
    log_mel_mic1 = result["log_mel_mic1"]
    log_mel_mic2 = result["log_mel_mic2"]
    log_mel_avg = result["log_mel_avg"]
    log_mel_diff = result["log_mel_diff"]

    # Shared color scale for mic1/mic2 so loudness difference is visually meaningful.
    mic_vmin = float(min(np.percentile(log_mel_mic1, 1), np.percentile(log_mel_mic2, 1)))
    mic_vmax = float(max(np.percentile(log_mel_mic1, 99), np.percentile(log_mel_mic2, 99)))

    # Difference should use symmetric color limits around zero.
    diff_abs = float(np.percentile(np.abs(log_mel_diff), 99))
    diff_vmin = -diff_abs
    diff_vmax = diff_abs

    save_dir_path = Path(save_dir) if save_dir is not None else None

    def maybe_path(name: str) -> Path | None:
        if save_dir_path is None:
            return None
        return save_dir_path / name

    plot_single_spectrogram(
        log_mel_mic1,
        title="Mic 1 log-Mel spectrogram",
        config=config,
        save_path=maybe_path("mic1_log_mel.png"),
        vmin=mic_vmin,
        vmax=mic_vmax,
    )

    plot_single_spectrogram(
        log_mel_mic2,
        title="Mic 2 log-Mel spectrogram",
        config=config,
        save_path=maybe_path("mic2_log_mel.png"),
        vmin=mic_vmin,
        vmax=mic_vmax,
    )

    plot_single_spectrogram(
        log_mel_avg,
        title="Average log-Mel spectrogram",
        config=config,
        save_path=maybe_path("avg_log_mel.png"),
    )

    plot_single_spectrogram(
        log_mel_diff,
        title="Difference log-Mel spectrogram: mic1 - mic2",
        config=config,
        save_path=maybe_path("diff_log_mel.png"),
        vmin=diff_vmin,
        vmax=diff_vmax,
    )