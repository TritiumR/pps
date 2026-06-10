from generate_spectrogram import AudioWindowConfig, generate_past_2s_policy_audio_input
from visualize_spectrogram import visualize_past_2s_audio_features

RINGTONE_PATH = "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav"

config = AudioWindowConfig(
    sample_rate=48_000,
    window_seconds=2.0,
    n_mels=80,
    stft_window_ms=25.0,
    stft_hop_ms=10.0,
    n_fft=2048,
    noise_std=0.001,
)

result = generate_past_2s_policy_audio_input(
    wav_path=RINGTONE_PATH,
    distance_mic1=0.10,
    distance_mic2=0.40,
    policy_time_seconds=3.5,
    episode_duration_seconds=10.0,
    config=config,
)

visualize_past_2s_audio_features(
    result=result,
    config=config,
    save_dir="spectrogram_visualization",
)
