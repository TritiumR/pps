# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oculus controller device for DROID-style SE(3) teleoperation."""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from collections.abc import Callable
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase, DeviceCfg


def _vec_to_reorder_mat(vec: tuple[int, int, int, int] | list[int]) -> np.ndarray:
    matrix = np.zeros((len(vec), len(vec)))
    for i, value in enumerate(vec):
        column = int(abs(value)) - 1
        matrix[i, column] = np.sign(value)
    return matrix


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _quat_diff(target_xyzw: np.ndarray, source_xyzw: np.ndarray) -> np.ndarray:
    result = Rotation.from_quat(target_xyzw) * Rotation.from_quat(source_xyzw).inv()
    return result.as_quat()


@dataclass
class Se3OculusCfg(DeviceCfg):
    """Configuration for Oculus-based SE(3) teleoperation."""

    right_controller: bool = True
    oculus_ip_address: str | None = None
    oculus_port: int = 5555
    print_fps: bool = False
    reader_module_path: str | None = None
    controller_timeout_sec: float = 5.0
    max_lin_vel: float = 1.0
    max_rot_vel: float = 1.0
    spatial_coeff: float = 1.0
    pos_action_gain: float = 5.0
    rot_action_gain: float = 2.0
    grip_threshold: float = 0.5
    rmat_reorder: tuple[int, int, int, int] = (-2, -1, -3, 4)
    retargeters: None = None


class Se3Oculus(DeviceBase):
    """Oculus controller teleoperation device using DROID-style clutch control."""

    def __init__(self, cfg: Se3OculusCfg):
        super().__init__()
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        self._controller_timeout_sec = cfg.controller_timeout_sec
        self._controller_id = "r" if cfg.right_controller else "l"
        self._grip_button = f"{self._controller_id.upper()}G"
        self._joystick_button = f"{self._controller_id.upper()}J"
        self._primary_button = "A" if cfg.right_controller else "X"
        self._secondary_button = "B" if cfg.right_controller else "Y"
        self._trigger_axis = "rightTrig" if cfg.right_controller else "leftTrig"
        self._global_to_env_mat = _vec_to_reorder_mat(cfg.rmat_reorder)
        self._additional_callbacks: dict[str, Callable] = {}
        self._previous_buttons: dict[str, Any] = {}
        self._motion_enabled = False
        self._last_action = np.zeros(7, dtype=np.float32)
        self._gripper_mode: str | None = None

        oculus_reader_cls = self._resolve_oculus_reader_class(cfg.reader_module_path)
        self._oculus_reader = oculus_reader_cls(
            ip_address=cfg.oculus_ip_address,
            port=cfg.oculus_port,
            print_FPS=cfg.print_fps,
        )

        self.reset()

    def __del__(self):
        if hasattr(self, "_oculus_reader"):
            try:
                self._oculus_reader.stop()
            except Exception:
                pass

    def __str__(self) -> str:
        hand = "right" if self._cfg.right_controller else "left"
        return f"Oculus Controller for SE(3): {self.__class__.__name__} ({hand} hand)"

    def reset(self):
        self._state = {
            "poses": {},
            "buttons": {},
            "movement_enabled": False,
            "controller_on": False,
        }
        self._update_sensor = True
        self._reset_origin = True
        self._reset_orientation = True
        self._robot_origin: dict[str, np.ndarray] | None = None
        self._vr_origin: dict[str, np.ndarray] | None = None
        self._vr_state: dict[str, np.ndarray | float] | None = None
        self._vr_to_global_mat = np.eye(4)
        self._last_read_time = time.time()
        self._motion_enabled = False
        self._last_action.fill(0.0)
        self._gripper_mode = None

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key.upper()] = func

    def is_motion_enabled(self) -> bool:
        return self._motion_enabled and self._state["controller_on"] and self._state["poses"] != {}

    def advance(self, env: Any) -> torch.Tensor:
        if env is None:
            raise ValueError("Se3Oculus.advance() requires the current environment.")

        has_tracking = self._poll_controller_state()
        if not has_tracking:
            return self._stationary_action()

        if self._update_sensor:
            self._process_reading()
            self._update_sensor = False

        if not self.is_motion_enabled() or self._vr_state is None:
            return self._stationary_action()

        robot_state = self._get_robot_state(env)

        if self._reset_origin:
            self._robot_origin = {
                "pos": robot_state["pos"].copy(),
                "quat": robot_state["quat"].copy(),
            }
            self._vr_origin = {
                "pos": np.array(self._vr_state["pos"], copy=True),
                "quat": np.array(self._vr_state["quat"], copy=True),
            }
            self._reset_origin = False

        robot_pos_offset = robot_state["pos"] - self._robot_origin["pos"]
        target_pos_offset = np.array(self._vr_state["pos"]) - self._vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset

        robot_quat_offset = _quat_diff(robot_state["quat"], self._robot_origin["quat"])
        target_quat_offset = _quat_diff(np.array(self._vr_state["quat"]), self._vr_origin["quat"])
        quat_action = _quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = Rotation.from_quat(quat_action).as_euler("xyz")

        pos_action *= self._cfg.pos_action_gain
        euler_action *= self._cfg.rot_action_gain
        pos_action, euler_action = self._limit_velocity(pos_action, euler_action)

        gripper_close = float(self._vr_state["gripper"]) > self._cfg.grip_threshold
        gripper_action = self._compute_gripper_action(env, gripper_close)

        self._last_action = np.concatenate([pos_action, euler_action, [gripper_action]]).astype(np.float32)
        self._last_action = self._last_action.clip(-1.0, 1.0)
        return torch.tensor(self._last_action, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _resolve_oculus_reader_class(module_path: str | None):
        candidate_paths: list[str] = []
        if module_path is not None:
            candidate_paths.append(module_path)

        repo_root = Path(__file__).resolve().parents[5]
        vendored_reader_parent = repo_root / "droid" / "droid" / "oculus_reader"
        if vendored_reader_parent.exists():
            candidate_paths.append(str(vendored_reader_parent))

        errors: list[str] = []
        for path in [None, *candidate_paths]:
            if path is not None and path not in sys.path:
                sys.path.append(path)
            try:
                module = importlib.import_module("oculus_reader.reader")
                return module.OculusReader
            except ModuleNotFoundError as exc:
                errors.append(str(exc))

        raise ModuleNotFoundError(
            "Failed to import `oculus_reader.reader`. Install `droid/droid/oculus_reader` or set "
            "`Se3OculusCfg.reader_module_path` to the package parent directory."
            f" Import errors: {errors}"
        )

    def _poll_controller_state(self) -> bool:
        poses, buttons = self._oculus_reader.get_transformations_and_buttons()
        self._state["controller_on"] = (time.time() - self._last_read_time) < self._controller_timeout_sec
        if poses == {} or self._controller_id not in poses:
            self._motion_enabled = False
            self._fire_callbacks(buttons)
            self._previous_buttons = dict(buttons)
            return False

        movement_enabled = self._is_pressed(buttons.get(self._grip_button, False))
        toggled = self._state["movement_enabled"] != movement_enabled
        self._update_sensor = self._update_sensor or movement_enabled
        self._reset_orientation = self._reset_orientation or self._is_pressed(buttons.get(self._joystick_button, False))
        self._reset_origin = self._reset_origin or toggled

        self._state["poses"] = poses
        self._state["buttons"] = buttons
        self._state["movement_enabled"] = movement_enabled
        self._state["controller_on"] = True
        self._last_read_time = time.time()
        self._motion_enabled = movement_enabled

        stop_updating_orientation = self._is_pressed(buttons.get(self._joystick_button, False)) or movement_enabled
        if self._reset_orientation:
            controller_pose = np.asarray(poses[self._controller_id], dtype=np.float64)
            if stop_updating_orientation:
                self._reset_orientation = False
            try:
                self._vr_to_global_mat = np.linalg.inv(controller_pose)
            except np.linalg.LinAlgError:
                self._vr_to_global_mat = np.eye(4)
                self._reset_orientation = True

        self._fire_callbacks(buttons)
        self._previous_buttons = dict(buttons)
        return True

    def _process_reading(self):
        controller_pose = np.asarray(self._state["poses"][self._controller_id], dtype=np.float64)
        controller_pose = self._global_to_env_mat @ self._vr_to_global_mat @ controller_pose
        vr_pos = self._cfg.spatial_coeff * controller_pose[:3, 3]
        vr_quat = Rotation.from_matrix(controller_pose[:3, :3]).as_quat()
        trigger_value = self._state["buttons"].get(self._trigger_axis, (0.0,))
        if isinstance(trigger_value, tuple):
            trigger_value = trigger_value[0]
        self._vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": float(trigger_value)}

    def _get_robot_state(self, env: Any) -> dict[str, np.ndarray]:
        arm_action = env.action_manager.get_term("arm_action")
        ee_pos, ee_quat = arm_action._compute_frame_pose()
        ee_pos_np = ee_pos[0].detach().cpu().numpy()
        ee_quat_wxyz = ee_quat[0].detach().cpu().numpy()
        ee_quat_xyzw = _quat_wxyz_to_xyzw(ee_quat_wxyz)
        return {"pos": ee_pos_np, "quat": ee_quat_xyzw}

    def _compute_gripper_action(self, env: Any, gripper_close: bool) -> float:
        if self._gripper_mode is None:
            gripper_action_term = env.action_manager.get_term("gripper_action")
            class_name = type(gripper_action_term).__name__
            if "BinaryZeroOne" in class_name:
                self._gripper_mode = "zero_one"
            else:
                self._gripper_mode = "signed"

        if self._gripper_mode == "zero_one":
            return 1.0 if gripper_close else 0.0
        return -1.0 if gripper_close else 1.0

    def _stationary_action(self) -> torch.Tensor:
        stationary_action = np.zeros_like(self._last_action)
        stationary_action[-1] = self._last_action[-1]
        self._last_action = stationary_action
        return torch.tensor(self._last_action, dtype=torch.float32, device=self._sim_device)

    def _limit_velocity(self, lin_vel: np.ndarray, rot_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lin_norm = np.linalg.norm(lin_vel)
        rot_norm = np.linalg.norm(rot_vel)
        if lin_norm > self._cfg.max_lin_vel:
            lin_vel = lin_vel * self._cfg.max_lin_vel / lin_norm
        if rot_norm > self._cfg.max_rot_vel:
            rot_vel = rot_vel * self._cfg.max_rot_vel / rot_norm
        return lin_vel, rot_vel

    def _fire_callbacks(self, buttons: dict[str, Any]):
        aliases = {
            "PRIMARY": self._primary_button,
            "SECONDARY": self._secondary_button,
            "RESET": self._secondary_button,
            "GRIP": self._grip_button,
            "JOYSTICK": self._joystick_button,
        }
        for key, callback in self._additional_callbacks.items():
            button_name = aliases.get(key, key)
            pressed_now = self._is_pressed(buttons.get(button_name, False))
            pressed_before = self._is_pressed(self._previous_buttons.get(button_name, False))
            if pressed_now and not pressed_before:
                callback()

    @staticmethod
    def _is_pressed(value: Any) -> bool:
        if isinstance(value, tuple):
            return bool(value and value[0] > 0.5)
        return bool(value)
