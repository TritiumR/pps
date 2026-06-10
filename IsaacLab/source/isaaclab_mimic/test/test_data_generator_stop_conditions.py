# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_data_generator_class():
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = types.ModuleType("isaaclab.utils.math")

    envs_module = types.ModuleType("isaaclab.envs")
    envs_module.ManagerBasedRLMimicEnv = object
    envs_module.MimicEnvCfg = object
    envs_module.SubTaskConstraintCoordinationScheme = object
    envs_module.SubTaskConstraintType = object
    sys.modules["isaaclab.envs"] = envs_module

    managers_module = types.ModuleType("isaaclab.managers")
    managers_module.TerminationTermCfg = object
    sys.modules["isaaclab.managers"] = managers_module

    sys.modules.setdefault("isaaclab_mimic", types.ModuleType("isaaclab_mimic"))
    sys.modules.setdefault("isaaclab_mimic.datagen", types.ModuleType("isaaclab_mimic.datagen"))

    datagen_info_module = types.ModuleType("isaaclab_mimic.datagen.datagen_info")
    datagen_info_module.DatagenInfo = object
    sys.modules["isaaclab_mimic.datagen.datagen_info"] = datagen_info_module

    selection_strategy_module = types.ModuleType("isaaclab_mimic.datagen.selection_strategy")
    selection_strategy_module.make_selection_strategy = lambda *args, **kwargs: None
    sys.modules["isaaclab_mimic.datagen.selection_strategy"] = selection_strategy_module

    waypoint_module = types.ModuleType("isaaclab_mimic.datagen.waypoint")
    waypoint_module.MultiWaypoint = object
    waypoint_module.Waypoint = object
    waypoint_module.WaypointSequence = object
    waypoint_module.WaypointTrajectory = object
    sys.modules["isaaclab_mimic.datagen.waypoint"] = waypoint_module

    datagen_info_pool_module = types.ModuleType("isaaclab_mimic.datagen.datagen_info_pool")
    datagen_info_pool_module.DataGenInfoPool = object
    sys.modules["isaaclab_mimic.datagen.datagen_info_pool"] = datagen_info_pool_module

    datagen_dir = Path(__file__).resolve().parents[1] / "isaaclab_mimic" / "datagen"
    spec = importlib.util.spec_from_file_location(
        "isaaclab_mimic.datagen.data_generator", datagen_dir / "data_generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DataGenerator


def _load_waypoint_classes():
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = types.ModuleType("isaaclab.utils.math")

    envs_module = types.ModuleType("isaaclab.envs")
    envs_module.ManagerBasedRLMimicEnv = object
    sys.modules["isaaclab.envs"] = envs_module

    managers_module = types.ModuleType("isaaclab.managers")
    managers_module.TerminationTermCfg = object
    sys.modules["isaaclab.managers"] = managers_module

    datagen_dir = Path(__file__).resolve().parents[1] / "isaaclab_mimic" / "datagen"
    spec = importlib.util.spec_from_file_location(
        "isaaclab_mimic.datagen.waypoint", datagen_dir / "waypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MultiWaypoint, module.Waypoint


DataGenerator = _load_data_generator_class()
MultiWaypoint, Waypoint = _load_waypoint_classes()


def test_data_generator_latches_success_without_stopping():
    generated_success, should_stop = DataGenerator._update_rollout_status(
        False, {"success": True, "terminated": False}
    )

    assert generated_success is True
    assert should_stop is False


def test_data_generator_stops_immediately_on_termination():
    generated_success, should_stop = DataGenerator._update_rollout_status(
        False, {"success": False, "terminated": True}
    )

    assert generated_success is False
    assert should_stop is True


def test_data_generator_continues_while_rollout_is_active():
    generated_success, should_stop = DataGenerator._update_rollout_status(
        False, {"success": False, "terminated": False}
    )

    assert generated_success is False
    assert should_stop is False


def test_data_generator_clears_latched_success_on_termination():
    generated_success, should_stop = DataGenerator._update_rollout_status(
        True, {"success": False, "terminated": True}
    )

    assert generated_success is False
    assert should_stop is True


def test_multi_waypoint_reports_manual_termination():
    class _DummyScene:
        def get_state(self, is_relative=True):
            return {"dummy": torch.tensor(1.0)}

    class _DummyEnv:
        def __init__(self):
            self.scene = _DummyScene()
            self.device = "cpu"
            self.obs_buf = {"policy": torch.tensor([0.0])}
            self.should_terminate = {}

        def target_eef_pose_to_action(
            self,
            target_eef_pose_dict,
            gripper_action_dict,
            action_noise_dict=None,
            env_id=0,
        ):
            return torch.zeros(7)

        def step(self, action):
            return self.obs_buf, None, None, None, None

    env = _DummyEnv()
    waypoint = Waypoint(pose=torch.eye(4), gripper_action=torch.zeros(1), noise=0.0)
    multi_waypoint = MultiWaypoint({"franka": waypoint})

    success_term = SimpleNamespace(
        func=lambda env, **kwargs: torch.tensor([True]), params={}
    )
    termination_terms = [
        SimpleNamespace(func=lambda env, **kwargs: torch.tensor([True]), params={})
    ]

    result = asyncio.run(
        multi_waypoint.execute(
            env=env,
            success_term=success_term,
            termination_terms=termination_terms,
            env_id=0,
        )
    )

    assert result["success"] is False
    assert result["terminated"] is True
