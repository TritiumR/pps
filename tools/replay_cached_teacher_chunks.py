#!/usr/bin/env python3
"""Replay cached K=8 teacher chunks from exact demo states and record tiled videos."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import h5py
import numpy as np
import torch

from isaaclab.app import AppLauncher


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Weight-Droid-Visuomotor-v0")
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--action-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--demo-index", type=int, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--post-steps", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    return parser


ARGS = _parser().parse_args()
APP = AppLauncher(ARGS).app

import gymnasium as gym  # noqa: E402
import isaaclab_mimic.envs  # noqa: E402,F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from sim_free_mpc.action_space import clamp_real_action_chunk  # noqa: E402
from vlm_dp.offline_context import _weight_stage, weight_episode_signals  # noqa: E402


STAGES = (
    "approach_pear",
    "lift_pear",
    "carry_pear",
    "place_pear",
    "approach_apple",
    "lift_apple",
    "carry_apple",
    "place_apple",
)


def _demo_number(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def _midpoints(demo: h5py.Group) -> list[int]:
    signals = weight_episode_signals(demo)
    length = int(demo.attrs["num_samples"])
    stages = np.asarray([min(int(_weight_stage(signals, step)), 7) for step in range(length)])
    result = []
    for stage in range(8):
        indices = np.flatnonzero(stages == stage)
        if not len(indices):
            raise ValueError(f"Missing stage {stage} in demo.")
        result.append(int(indices[len(indices) // 2]))
    return result


def _state_at(group: h5py.Group, step: int, count: int, device: str):
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            item = torch.from_numpy(np.asarray(value[step]))
            result[key] = item.unsqueeze(0).repeat(count, *([1] * item.ndim)).to(device)
        else:
            result[key] = _state_at(value, step, count, device)
    return result


def _uint8_batch(value) -> np.ndarray:
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if value.dtype == np.uint8:
        return value
    value = value.astype(np.float32)
    if float(np.nanmin(value)) < -0.01:
        value = (value + 1.0) * 127.5
    elif float(np.nanmax(value)) <= 1.01:
        value = value * 255.0
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _tile(images: np.ndarray, *, title: str, action_step: str) -> np.ndarray:
    rows = []
    for row in range(2):
        cells = []
        for column in range(4):
            index = row * 4 + column
            image = images[index].copy()
            cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"K={index}",
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(image)
        rows.append(np.concatenate(cells, axis=1))
    canvas = np.concatenate(rows, axis=0)
    header = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{title} | {action_step}",
        (8, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((header, canvas), axis=0)


def _camera(env_obs) -> np.ndarray:
    policy = env_obs["policy"]
    if "table_cam" not in policy:
        raise KeyError(f"table_cam missing from policy observation: {list(policy)}")
    return _uint8_batch(policy["table_cam"])


def main() -> None:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=1)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env = gym.make(ARGS.task, cfg=env_cfg).unwrapped
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(ARGS.action_stats.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    with h5py.File(ARGS.hdf5, "r") as source, np.load(ARGS.cache, allow_pickle=False) as cache:
        names = sorted(source["data"], key=_demo_number)
        name = names[ARGS.demo_index]
        demo = source["data"][name]
        steps = _midpoints(demo)
        cache_demo = cache["demo_name"].astype(str)
        cache_step = cache["step_index"]
        cache_iteration = cache["iteration"]
        cache_trajectory = cache["trajectory_id"]
        manifest = {"demo": name, "videos": []}

        for stage, step in enumerate(steps):
            rows = np.flatnonzero(
                (cache_demo == name) & (cache_step == step) & (cache_iteration == 10)
            )
            order = np.argsort(cache_trajectory[rows])
            rows = rows[order]
            if len(rows) != 8:
                raise ValueError(f"Expected K=8 final chunks for {name} step {step}, got {len(rows)}")
            normalized = np.asarray(cache["x_t"][rows], dtype=np.float32)
            decoded = normalized * std + mean
            current_q = np.asarray(demo["obs/joint_pos"][step, :7], dtype=np.float32)
            real = decoded.copy()
            real[:, :, :7] += current_q[None, None, :]
            real = (
                clamp_real_action_chunk(
                    torch.from_numpy(real),
                    current_joint_pos=current_q,
                    max_joint_delta=0.05,
                )
                .cpu()
                .numpy()
            )

            state = _state_at(demo["states"], step, 1, env.device)
            path = ARGS.output_dir / f"{name}_stage{stage}_{STAGES[stage]}_frame{step}.mp4"
            streams = []
            for trajectory_index in range(8):
                env_obs, _ = env.reset_to(state, env_ids=None, is_relative=True)
                stream = [_camera(env_obs)[0]] * 3
                for action_index in range(real.shape[1]):
                    env_obs, _, _, _, _ = env.step(
                        torch.as_tensor(
                            real[trajectory_index : trajectory_index + 1, action_index],
                            device=env.device,
                            dtype=torch.float32,
                        )
                    )
                    stream.append(_camera(env_obs)[0])
                final_action = torch.as_tensor(
                    real[trajectory_index : trajectory_index + 1, -1],
                    device=env.device,
                    dtype=torch.float32,
                )
                for _ in range(ARGS.post_steps):
                    env_obs, _, _, _, _ = env.step(final_action)
                    stream.append(_camera(env_obs)[0])
                streams.append(stream)

            initial = _tile(
                np.stack([stream[0] for stream in streams]),
                title=f"{name} frame={step} stage={stage} {STAGES[stage]}",
                action_step="before teacher chunk",
            )
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                ARGS.fps,
                (initial.shape[1], initial.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")
            for frame_index in range(len(streams[0])):
                if frame_index < 3:
                    action_step = "before teacher chunk"
                elif frame_index < 3 + real.shape[1]:
                    action_step = f"teacher action {frame_index - 2}/{real.shape[1]}"
                else:
                    action_step = (
                        f"post-chunk hold {frame_index - 2 - real.shape[1]}/{ARGS.post_steps}"
                    )
                frame = _tile(
                    np.stack([stream[frame_index] for stream in streams]),
                    title=f"{name} frame={step} stage={stage} {STAGES[stage]}",
                    action_step=action_step,
                )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            manifest["videos"].append({"stage": stage, "step": step, "path": str(path)})
            print(f"saved {path}", flush=True)

        (ARGS.output_dir / f"{name}_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    env.close()
    APP.close()


if __name__ == "__main__":
    main()
