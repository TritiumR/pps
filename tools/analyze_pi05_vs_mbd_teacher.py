#!/usr/bin/env python3
"""Matched-observation consistency comparison for Pi05 and the cached MBD teacher."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pathlib

os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpi.models import model as model_api
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as training_config
from sim_common.fk import FrankaFK


STAGES = (
    "approach pear",
    "lift pear",
    "carry pear",
    "place pear",
    "approach apple",
    "lift apple",
    "carry apple",
    "place apple",
)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def row_rms(values: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, values.ndim))
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=axes))


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)


def repeat_metrics(labels: np.ndarray) -> dict[str, float]:
    mean = labels.mean(axis=0)
    noise = labels - mean
    mean_rms = float(np.sqrt(np.mean(np.square(mean, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    cosines = []
    rmses = []
    for left, right in itertools.combinations(range(len(labels)), 2):
        cosines.append(float(row_cosine(labels[left : left + 1], labels[right : right + 1])[0]))
        rmses.append(float(row_rms((labels[left] - labels[right])[None])[0]))
    norms = row_rms(labels)
    return {
        "mean_label_rms": mean_rms,
        "repeat_noise_rms": noise_rms,
        "snr": float(mean_rms / max(noise_rms, 1e-12)),
        "pairwise_cosine_mean": float(np.mean(cosines)),
        "pairwise_cosine_min": float(np.min(cosines)),
        "pairwise_rmse_mean": float(np.mean(rmses)),
        "label_rms_cv": float(norms.std() / max(norms.mean(), 1e-12)),
    }


def raw_observation(demo: h5py.Group, step: int, prompt: str) -> dict[str, object]:
    return {
        "observation/exterior_image_1_left": np.asarray(demo["obs/table_cam"][step]),
        "observation/wrist_image_left": np.asarray(demo["obs/wrist_cam"][step]),
        "observation/joint_position": np.asarray(demo["obs/joint_pos"][step, :7], dtype=np.float32),
        "observation/gripper_position": np.asarray(demo["obs/gripper_pos"][step, :1], dtype=np.float32),
        "prompt": prompt,
    }


def repeated_model_observation(policy, raw: dict[str, object], count: int):
    _, single_inputs = policy.obs_to_input(copy.deepcopy(raw))

    def repeat(value):
        if value is None:
            return None
        repeats = (count,) + (1,) * (value.ndim - 1)
        return value.repeat(repeats)

    inputs = jax.tree.map(repeat, single_inputs)
    return model_api.Observation.from_dict(inputs), inputs


@torch.inference_mode()
def query_fixed_pi05(model, observation, states: torch.Tensor, times: torch.Tensor, levels: tuple[int, ...]):
    images, image_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    state = torch.nn.functional.pad(
        state, (0, model.config.action_dim - state.shape[1]), mode="constant", value=0
    )
    prefix, prefix_masks, prefix_attention = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks
    )
    attention = make_att_2d_masks(prefix_masks, prefix_attention)
    attention = model._prepare_attention_masks_4d(attention)
    positions = torch.cumsum(prefix_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    result = {}
    repeats = state.shape[0]
    for level in levels:
        x_t = states[0, level].unsqueeze(0).repeat(repeats, 1, 1)
        time = times[0, level].repeat(repeats)
        result[level] = model.denoise_step(
            state, prefix_masks, cache, x_t, time
        )[..., :8].float().cpu().numpy()
    return result


def decode_pi05(policy, inputs: dict, normalized_actions: torch.Tensor) -> np.ndarray:
    normalized = normalized_actions.detach().float().cpu().numpy()
    state = inputs["state"].detach().float().cpu().numpy()
    decoded = []
    for index in range(len(normalized)):
        output = policy._output_transform(
            {"actions": normalized[index], "state": state[index]}
        )
        decoded.append(np.asarray(output["actions"], dtype=np.float32))
    return np.stack(decoded)


def endpoints_from_absolute(actions: np.ndarray) -> np.ndarray:
    fk = FrankaFK(device="cpu")
    targets = actions[:, :, -1, :7]
    with torch.inference_mode():
        position, _ = fk.fk(torch.from_numpy(targets.reshape(-1, 7)).float())
    return position.cpu().numpy().reshape(*targets.shape[:2], 3)


def k_metrics(decoded_absolute: np.ndarray, current_q: np.ndarray) -> dict[str, np.ndarray]:
    observations, k = decoded_absolute.shape[:2]
    joint_delta = decoded_absolute[..., :7] - current_q[:, None, None, :]
    pair_cosine = np.zeros(observations, dtype=np.float64)
    pair_rmse = np.zeros(observations, dtype=np.float64)
    pair_count = 0
    for left, right in itertools.combinations(range(k), 2):
        pair_cosine += row_cosine(joint_delta[:, left], joint_delta[:, right])
        pair_rmse += row_rms(joint_delta[:, left] - joint_delta[:, right])
        pair_count += 1
    pair_cosine /= pair_count
    pair_rmse /= pair_count
    endpoints = endpoints_from_absolute(decoded_absolute)
    center = endpoints.mean(axis=1, keepdims=True)
    endpoint_spread = np.linalg.norm(endpoints - center, axis=-1).mean(axis=1)
    endpoint_pairwise = np.zeros(observations, dtype=np.float64)
    for left, right in itertools.combinations(range(k), 2):
        endpoint_pairwise += np.linalg.norm(endpoints[:, left] - endpoints[:, right], axis=-1)
    endpoint_pairwise /= pair_count
    final_gripper = decoded_absolute[:, :, -1, 7] > 0.5
    majority = np.maximum(final_gripper.mean(axis=1), 1.0 - final_gripper.mean(axis=1))
    return {
        "joint_delta_chunk_pairwise_cosine": pair_cosine,
        "joint_delta_chunk_pairwise_rmse": pair_rmse,
        "tcp_endpoint_spread_m": endpoint_spread,
        "tcp_endpoint_pairwise_distance_m": endpoint_pairwise,
        "gripper_majority_fraction": majority,
        "gripper_unanimous": (majority == 1.0).astype(np.float64),
    }


def summarize_k(metrics: dict[str, np.ndarray], stages: np.ndarray) -> dict[str, object]:
    return {
        "overall": {name: summary(values) for name, values in metrics.items()},
        "by_stage": {
            STAGES[stage]: {
                "num_observations": int(np.sum(stages == stage)),
                **{name: summary(values[stages == stage]) for name, values in metrics.items()},
            }
            for stage in range(len(STAGES))
        },
    }


def timestep_metrics(states: np.ndarray, labels: np.ndarray, times: np.ndarray | None = None):
    updates = [
        (states[:, :, level + 1] - states[:, :, level]).reshape(-1, *states.shape[3:])
        for level in range(states.shape[2] - 1)
    ]
    rows = []
    for level, update in enumerate(updates):
        label = labels[:, :, level].reshape(-1, *labels.shape[3:])
        row = {
            "iteration": level,
            "label_rms": summary(row_rms(label)),
            "actual_update_rms": summary(row_rms(update)),
        }
        if times is not None:
            row["time_mean"] = float(times[:, :, level].mean())
            row["delta_time_abs_mean"] = float(
                np.abs(times[:, :, level + 1] - times[:, :, level]).mean()
            )
        if level + 1 < len(updates):
            row["cosine_to_next_update"] = summary(row_cosine(update, updates[level + 1]))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--mbd-cache", type=pathlib.Path, required=True)
    parser.add_argument("--mbd-fixed-json", type=pathlib.Path, required=True)
    parser.add_argument("--mbd-action-stats", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-config", default="pi05_droid_jointpos")
    parser.add_argument("--prompt", default="put pear and apple on the scale")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = json.loads(args.mbd_fixed_json.read_text())
    selected_records = [record for record in fixed["records"] if int(record["iteration"]) == 0]
    selected = [(record["demo"], int(record["step"])) for record in selected_records]
    stages = np.asarray([int(record["stage"]) for record in selected_records], dtype=np.int64)
    if len(selected) != 128 or any(np.sum(stages == stage) != 16 for stage in range(8)):
        raise ValueError("Expected the matched 128-observation, 16-per-stage fixed-point sample.")

    with np.load(args.mbd_cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        observations = int(metadata["num_observations"])
        k = int(metadata["trajectories_per_observation"])
        levels = int(metadata["labels_per_trajectory"])
        horizon, action_dim = cache["x_t"].shape[1:]
        demos = cache["demo_name"].reshape(observations, k, levels)[:, 0, 0].astype(str)
        steps = cache["step_index"].reshape(observations, k, levels)[:, 0, 0]
        lookup = {(name, int(step)): index for index, (name, step) in enumerate(zip(demos, steps, strict=True))}
        indices = np.asarray([lookup[key] for key in selected], dtype=np.int64)
        shape = (observations, k, levels, horizon, action_dim)
        mbd_states = cache["x_t"].reshape(shape)[indices]
        mbd_labels = cache["epsilon"].reshape(shape)[indices]

    stats = json.loads(args.mbd_action_stats.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    mbd_delta = mbd_states[:, :, -1] * std + mean

    current_q = np.empty((len(selected), 7), dtype=np.float32)
    pi05_decoded = []
    pi05_states = []
    pi05_times = []
    pi05_labels = []
    fixed_records = []

    config = training_config.get_config(args.pi05_config)
    policy = policy_config.create_trained_policy(
        config,
        args.pi05_checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )
    teacher = policy._model
    teacher.eval()
    probe_levels = (0, 5, 10)

    with h5py.File(args.hdf5, "r") as source, torch.inference_mode():
        for index, ((demo_name, step), stage) in enumerate(zip(selected, stages, strict=True)):
            demo = source["data"][demo_name]
            current_q[index] = np.asarray(demo["obs/joint_pos"][step, :7], dtype=np.float32)
            raw = raw_observation(demo, step, args.prompt)
            observation, inputs = repeated_model_observation(policy, raw, args.k)
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            states, times, labels, final = teacher.forward_for_distill(
                observation, args.num_steps, teacher_flow_path_noise_std=0.0
            )
            repeated = query_fixed_pi05(teacher, observation, states, times, probe_levels)
            for level in probe_levels:
                fixed_records.append(
                    {
                        "demo": demo_name,
                        "step": step,
                        "stage": int(stage),
                        "stage_name": STAGES[int(stage)],
                        "level": level,
                        "time": float(times[0, level].item()),
                        **repeat_metrics(repeated[level]),
                    }
                )
            pi05_decoded.append(decode_pi05(policy, inputs, final)[..., :8])
            pi05_states.append(states[..., :8].float().cpu().numpy())
            pi05_times.append(times.float().cpu().numpy())
            pi05_labels.append(labels[..., :8].float().cpu().numpy())
            print(f"pi05 {index + 1}/{len(selected)} {demo_name}:{step}", flush=True)

    pi05_decoded = np.stack(pi05_decoded)
    pi05_states = np.stack(pi05_states)
    pi05_times = np.stack(pi05_times)
    pi05_labels = np.stack(pi05_labels)
    mbd_absolute = mbd_delta.copy()
    mbd_absolute[..., :7] += current_q[:, None, None, :]

    pi05_fixed_by_level = {
        str(level): {
            key: float(np.mean([record[key] for record in fixed_records if record["level"] == level]))
            for key in (
                "mean_label_rms",
                "repeat_noise_rms",
                "snr",
                "pairwise_cosine_mean",
                "pairwise_cosine_min",
                "pairwise_rmse_mean",
                "label_rms_cv",
            )
        }
        for level in probe_levels
    }
    report = {
        "settings": {
            "observations": len(selected),
            "observations_per_stage": 16,
            "k": args.k,
            "steps": args.num_steps,
            "selection": "exact same observations as MBD fixed-label experiment",
            "pi05_schedule": "native bin-sampled flow-matching rollout",
            "mbd_schedule": "native 10-update DDIM/MBD rollout",
            "raw_label_magnitude_directly_comparable": False,
        },
        "fixed_point_repeatability": {
            "mbd": fixed["aggregate_by_iteration"],
            "pi05": pi05_fixed_by_level,
        },
        "timestep_consistency": {
            "mbd": timestep_metrics(mbd_states, mbd_labels),
            "pi05": timestep_metrics(pi05_states, pi05_labels, pi05_times),
        },
        "k8_consistency": {
            "mbd": summarize_k(k_metrics(mbd_absolute, current_q), stages),
            "pi05": summarize_k(k_metrics(pi05_decoded, current_q), stages),
        },
    }
    (args.output_dir / "pi05_vs_mbd_teacher_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "pi05_matched_samples.npz",
        demo=np.asarray([item[0] for item in selected]),
        step=np.asarray([item[1] for item in selected]),
        stage=stages,
        states=pi05_states,
        times=pi05_times,
        labels=pi05_labels,
        decoded_actions=pi05_decoded,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for teacher_name, color in (("mbd", "tab:blue"), ("pi05", "tab:orange")):
        rows = report["timestep_consistency"][teacher_name]
        scale = np.asarray([row["actual_update_rms"]["mean"] for row in rows])
        axes[0].plot(scale / max(scale[0], 1e-12), marker="o", label=teacher_name, color=color)
        axes[1].plot(
            [row["cosine_to_next_update"]["mean"] for row in rows[:-1]],
            marker="o", label=teacher_name, color=color,
        )
    axes[0].set(title="Relative update scale", xlabel="native rollout step", ylabel="RMS / step-0 RMS")
    axes[1].set(title="Adjacent update direction", xlabel="native rollout step", ylabel="cosine")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.savefig(args.output_dir / "pi05_vs_mbd_timestep.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
