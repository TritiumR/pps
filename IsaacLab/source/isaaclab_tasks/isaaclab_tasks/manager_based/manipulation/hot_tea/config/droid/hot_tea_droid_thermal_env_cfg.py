# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.hot_tea import hot_tea_env_cfg
from isaaclab_tasks.manager_based.manipulation.hot_tea.config.droid import hot_tea_droid_env_cfg
from isaaclab_tasks.manager_based.manipulation.tea.config.droid import tea_joint_pos_visuomotor_env_cfg

_THERMAL_SEGMENTATION_DATA_TYPE = "instance_id_segmentation_fast"
_COLD_COLOR = (0.0, 36.0, 255.0)
_TEAPOT_COLOR = (255.0, 132.0, 0.0)
_HOT_TEAPOT_COLOR = (255.0, 0.0, 0.0)
_TEAPOT_TEMPERATURE = 0.55
_HOT_TEAPOT_TEMPERATURE = 1.0
_TEAPOT_BORDER_TEMPERATURE = 0.18
_HOT_TEAPOT_BORDER_TEMPERATURE = _TEAPOT_TEMPERATURE
_EDGE_FALLOFF_POWER = 0.45
_GAUSSIAN_KERNEL_CACHE: dict[tuple[int, float, str, torch.dtype], torch.Tensor] = {}


def _prim_path_matches_root(prim_path: str, prim_root: str) -> bool:
    """Return whether the USD prim path contains the requested root as a full path segment."""
    return f"/{prim_root}/" in prim_path or prim_path.endswith(f"/{prim_root}")


def _segmentation_id_temperatures(
    camera: Camera,
    segmentation_data_type: str,
    temperature_by_root: dict[str, tuple[float, float]],
) -> list[dict[int, tuple[float, float]]]:
    """Resolve per-env segmentation ids to pseudo temperatures by prim root name."""
    temperatures_per_env = []
    for env_info in camera.data.info:
        id_to_labels = env_info.get(segmentation_data_type, {}).get("idToLabels", {})
        env_temperatures: dict[int, tuple[float, float]] = {}
        for segmentation_id, prim_path in id_to_labels.items():
            if not isinstance(prim_path, str):
                continue
            for prim_root, temperature in temperature_by_root.items():
                if _prim_path_matches_root(prim_path, prim_root):
                    env_temperatures[int(segmentation_id)] = temperature
                    break
        temperatures_per_env.append(env_temperatures)
    return temperatures_per_env


def _radial_object_temperature(
    mask: torch.Tensor,
    center_temperature: float,
    border_temperature: float,
) -> torch.Tensor | None:
    """Build a nonlinear center-hot temperature profile inside a visible object mask."""
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    if y_coords.numel() == 0:
        return None

    y_float = y_coords.to(dtype=torch.float32)
    x_float = x_coords.to(dtype=torch.float32)
    y_center = y_float.mean()
    x_center = x_float.mean()

    distances = torch.sqrt((y_float - y_center) ** 2 + (x_float - x_center) ** 2)
    max_distance = torch.clamp(distances.max(), min=1.0)
    normalized_distance = (distances / max_distance).clamp(0.0, 1.0)

    center_weight = torch.cos(normalized_distance * 1.5707963267948966).clamp(0.0, 1.0)
    center_weight = center_weight.pow(_EDGE_FALLOFF_POWER)
    values = border_temperature + (center_temperature - border_temperature) * center_weight

    temperature = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    temperature[y_coords, x_coords] = values
    return temperature


def _gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a cached normalized 1-D Gaussian kernel."""
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
    if kernel_size <= 1:
        return torch.ones((1,), device=device, dtype=dtype)

    cache_key = (kernel_size, sigma, str(device), dtype)
    cached_kernel = _GAUSSIAN_KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    radius = kernel_size // 2
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    _GAUSSIAN_KERNEL_CACHE[cache_key] = kernel
    return kernel


def _smooth_temperature(
    temperature: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a scalar temperature map."""
    if kernel_size <= 1:
        return temperature.clamp(0.0, 1.0)

    kernel = _gaussian_kernel_1d(kernel_size, sigma, temperature.device, temperature.dtype)
    radius = kernel_size // 2
    image = temperature.unsqueeze(1)

    horizontal_kernel = kernel.view(1, 1, 1, kernel_size)
    vertical_kernel = kernel.view(1, 1, kernel_size, 1)
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal_kernel)
    image = F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical_kernel)
    return image.squeeze(1).clamp(0.0, 1.0)


def _colorize_temperature(
    temperature: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Map scalar pseudo temperature to a blue-orange-red RGB heat map."""
    cold = torch.tensor(_COLD_COLOR, device=temperature.device, dtype=torch.float32)
    teapot = torch.tensor(_TEAPOT_COLOR, device=temperature.device, dtype=torch.float32)
    hot = torch.tensor(_HOT_TEAPOT_COLOR, device=temperature.device, dtype=torch.float32)

    temperature_f = temperature.to(dtype=torch.float32)
    low_alpha = (temperature_f / _TEAPOT_TEMPERATURE).clamp(0.0, 1.0)
    high_alpha = ((temperature_f - _TEAPOT_TEMPERATURE) / (1.0 - _TEAPOT_TEMPERATURE)).clamp(0.0, 1.0)
    low_rgb = cold + low_alpha.unsqueeze(-1) * (teapot - cold)
    high_rgb = teapot + high_alpha.unsqueeze(-1) * (hot - teapot)
    rgb = torch.where((temperature_f <= _TEAPOT_TEMPERATURE).unsqueeze(-1), low_rgb, high_rgb)

    if normalize:
        return rgb / 255.0
    return rgb.round().clamp(0.0, 255.0).to(dtype=torch.uint8)


def pseudo_thermal_image(
    env,
    sensor_cfg: SceneEntityCfg,
    segmentation_data_type: str = _THERMAL_SEGMENTATION_DATA_TYPE,
    kernel_size: int = 21,
    sigma: float = 6.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Pseudo thermal RGB image from instance-id segmentation masks."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    segmentation = camera.data.output[segmentation_data_type]
    if segmentation.dim() == 4 and segmentation.shape[-1] == 1:
        segmentation = segmentation.squeeze(-1)

    temperature = torch.zeros(segmentation.shape, device=segmentation.device, dtype=torch.float32)
    temperatures_per_env = _segmentation_id_temperatures(
        camera=camera,
        segmentation_data_type=segmentation_data_type,
        temperature_by_root={
            "E_teapot_5": (_TEAPOT_TEMPERATURE, _TEAPOT_BORDER_TEMPERATURE),
            hot_tea_env_cfg.HOT_TEAPOT_PRIM_NAME: (
                _HOT_TEAPOT_TEMPERATURE,
                _HOT_TEAPOT_BORDER_TEMPERATURE,
            ),
        },
    )

    for env_index, id_to_temperature in enumerate(temperatures_per_env):
        for segmentation_id, (center_temperature, border_temperature) in id_to_temperature.items():
            object_temperature = _radial_object_temperature(
                segmentation[env_index] == segmentation_id,
                center_temperature=center_temperature,
                border_temperature=border_temperature,
            )
            if object_temperature is not None:
                temperature[env_index] = torch.maximum(temperature[env_index], object_temperature)

    temperature = _smooth_temperature(temperature, kernel_size=kernel_size, sigma=sigma)
    return _colorize_temperature(temperature, normalize=normalize)


@configclass
class ThermalObservationsCfg(hot_tea_droid_env_cfg.ObservationsCfg):
    """Observation specs with RGB and pseudo thermal table and wrist camera images."""

    @configclass
    class PolicyCfg(hot_tea_droid_env_cfg.ObservationsCfg.PolicyCfg):
        thermal_table_cam = ObsTerm(
            func=pseudo_thermal_image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "normalize": False,
            },
        )
        thermal_wrist_cam = ObsTerm(
            func=pseudo_thermal_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "normalize": False,
            },
        )

    policy: PolicyCfg = PolicyCfg()


def _configure_thermal_cameras(env_cfg):
    """Enable uncolorized instance-id segmentation on thermal cameras."""
    for camera_name in ("table_cam", "wrist_cam"):
        camera_cfg = getattr(env_cfg.scene, camera_name)
        if _THERMAL_SEGMENTATION_DATA_TYPE not in camera_cfg.data_types:
            camera_cfg.data_types = [*camera_cfg.data_types, _THERMAL_SEGMENTATION_DATA_TYPE]
        camera_cfg.colorize_instance_id_segmentation = False


@configclass
class DroidHotTeaJointPosThermalEnvCfg(
    hot_tea_droid_env_cfg.DroidHotTeaJointPosVisuomotorEnvCfg
):
    """Hot tea task with Droid robot, joint position control, and pseudo thermal images."""

    observations: ThermalObservationsCfg = ThermalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_thermal_cameras(self)


@configclass
class DroidHotTeaIkRelThermalEnvCfg(
    hot_tea_droid_env_cfg.DroidHotTeaIkRelVisuomotorEnvCfg
):
    """Hot tea task with Droid robot, relative IK control, and pseudo thermal images."""

    observations: ThermalObservationsCfg = ThermalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_thermal_cameras(self)
