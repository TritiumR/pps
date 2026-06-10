# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import types

import torch


def _load_datagen_info_pool_class():
    datasets_module = types.ModuleType("isaaclab.utils.datasets")
    datasets_module.EpisodeData = object
    datasets_module.HDF5DatasetFileHandler = object
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.datasets"] = datasets_module

    sys.modules.setdefault("isaaclab_mimic", types.ModuleType("isaaclab_mimic"))
    sys.modules.setdefault("isaaclab_mimic.datagen", types.ModuleType("isaaclab_mimic.datagen"))

    datagen_dir = Path(__file__).resolve().parents[1] / "isaaclab_mimic" / "datagen"

    datagen_info_spec = importlib.util.spec_from_file_location(
        "isaaclab_mimic.datagen.datagen_info", datagen_dir / "datagen_info.py"
    )
    datagen_info_module = importlib.util.module_from_spec(datagen_info_spec)
    assert datagen_info_spec.loader is not None
    datagen_info_spec.loader.exec_module(datagen_info_module)
    sys.modules["isaaclab_mimic.datagen.datagen_info"] = datagen_info_module

    datagen_info_pool_spec = importlib.util.spec_from_file_location(
        "isaaclab_mimic.datagen.datagen_info_pool", datagen_dir / "datagen_info_pool.py"
    )
    datagen_info_pool_module = importlib.util.module_from_spec(datagen_info_pool_spec)
    assert datagen_info_pool_spec.loader is not None
    datagen_info_pool_spec.loader.exec_module(datagen_info_pool_module)
    return datagen_info_pool_module.DataGenInfoPool


DataGenInfoPool = _load_datagen_info_pool_class()


class _DummyEnv:
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"franka": actions[:, -1:]}


class _DummySubtaskConfig:
    def __init__(self, subtask_term_signal: str, subtask_term_offset_range=(0, 0), subtask_start_offset_range=(0, 0)):
        self.subtask_term_signal = subtask_term_signal
        self.subtask_term_offset_range = subtask_term_offset_range
        self.subtask_start_offset_range = subtask_start_offset_range


def _make_episode(actions_len: int, signal_windows: dict[str, tuple[int, int]]):
    actions = torch.zeros((actions_len, 7))
    subtask_term_signals = {}
    for signal_name, (start, end) in signal_windows.items():
        signal = torch.zeros(actions_len, dtype=torch.bool)
        signal[start:end] = True
        subtask_term_signals[signal_name] = signal

    return SimpleNamespace(
        data={
            "actions": actions,
            "obs": {
                "datagen_info": {
                    "eef_pose": {"franka": torch.eye(4).repeat(actions_len, 1, 1)},
                    "object_pose": {"laptop": torch.eye(4).repeat(actions_len, 1, 1)},
                    "target_eef_pose": {"franka": torch.eye(4).repeat(actions_len, 1, 1)},
                    "subtask_term_signals": subtask_term_signals,
                }
            },
        }
    )


def _make_pool() -> DataGenInfoPool:
    env_cfg = SimpleNamespace(
        datagen_config=SimpleNamespace(use_skillgen=False),
        subtask_configs={
            "franka": [
                _DummySubtaskConfig("grasped"),
                _DummySubtaskConfig("placed"),
                _DummySubtaskConfig("closed"),
            ]
        },
    )
    return DataGenInfoPool(env=_DummyEnv(), env_cfg=env_cfg, device="cpu")


def test_non_skillgen_boundaries_follow_configured_order():
    pool = _make_pool()
    episode = _make_episode(
        actions_len=30,
        signal_windows={
            "grasped": (9, 12),
            "placed": (19, 21),
            "closed": (0, 30),
        },
    )

    pool._add_episode(episode, episode_name="demo_0")

    assert pool.subtask_boundaries["franka"][0] == [(0, 10), (10, 20), (20, 30)]


def test_non_skillgen_boundaries_fall_back_to_chronological_order():
    pool = _make_pool()
    episode = _make_episode(
        actions_len=30,
        signal_windows={
            "grasped": (19, 22),
            "placed": (9, 13),
            "closed": (0, 30),
        },
    )

    pool._add_episode(episode, episode_name="demo_1")

    assert pool.subtask_boundaries["franka"][0] == [(0, 10), (10, 20), (20, 30)]
