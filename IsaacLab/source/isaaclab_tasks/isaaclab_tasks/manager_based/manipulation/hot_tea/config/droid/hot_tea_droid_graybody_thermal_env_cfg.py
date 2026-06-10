# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.manager_based.manipulation.hot_tea import hot_tea_env_cfg
from isaaclab_tasks.manager_based.manipulation.hot_tea.config.droid import hot_tea_droid_env_cfg

_THERMAL_SEGMENTATION_DATA_TYPE = "instance_id_segmentation_fast"
_THERMAL_DEPTH_DATA_TYPE = "distance_to_image_plane"
_STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8
_ENVIRONMENT_TEMPERATURE_K = 293.15
_GLOBAL_REFLECTED_TEMPERATURE_K = 301.15
_DEFAULT_EMISSIVITY = 0.90
_NORMALIZATION_MIN_TEMPERATURE_K = 289.15
_NORMALIZATION_MAX_TEMPERATURE_K = 410.15
_COLD_COLOR = (0.0, 36.0, 255.0)
_MID_COLOR = (255.0, 196.0, 0.0)
_HOT_COLOR = (255.0, 0.0, 0.0)
_DEFAULT_GAUSSIAN_KERNEL_SIZE = 17
_DEFAULT_GAUSSIAN_SIGMA = 4.0
_TEMPERATURE_VARIATION_ENABLED = True
_EMISSIVITY_VARIATION_ENABLED = True
_HEAT_SOURCE_ENABLED = True
_GAUSSIAN_KERNEL_CACHE: dict[tuple[int, float, str, torch.dtype], torch.Tensor] = {}

_MATERIAL_SPECS: tuple[dict[str, float | str | None], ...] = (
    {
        "pattern": f"/{hot_tea_env_cfg.HOT_TEAPOT_PRIM_NAME}/",
        "temperature_kelvin": 385.15,
        "emissivity": 0.96,
        "reflected_temperature_kelvin": 304.15,
        "temperature_noise_amplitude_kelvin": 3.5,
        "temperature_radial_amplitude_kelvin": 6.0,
        "temperature_vertical_amplitude_kelvin": -10.0,
        "temperature_horizontal_amplitude_kelvin": 1.5,
        "emissivity_noise_amplitude": 0.015,
        "pattern_type": "fbm",
        "heat_source_entity_name": "hot_teapot",
        "heat_source_offset_meters": (0.0, 0.0, 0.04),
        "heat_source_amplitude_kelvin": 30.0,
        "heat_source_sigma_meters": 0.03,
    },
    {
        "pattern": "/E_teapot_5/",
        "temperature_kelvin": 348.15,
        "emissivity": 0.96,
        "reflected_temperature_kelvin": 302.15,
        "temperature_noise_amplitude_kelvin": 2.5,
        "temperature_radial_amplitude_kelvin": 4.0,
        "temperature_vertical_amplitude_kelvin": -6.0,
        "temperature_horizontal_amplitude_kelvin": 0.9,
        "emissivity_noise_amplitude": 0.015,
        "pattern_type": "fbm",
        "heat_source_entity_name": "teapot",
        "heat_source_offset_meters": (0.0, 0.0, 0.04),
        "heat_source_amplitude_kelvin": 18.0,
        "heat_source_sigma_meters": 0.03,
    },
    {
        "pattern": "/E_teacup005_20/",
        "temperature_kelvin": 294.15,
        "emissivity": 0.93,
        "reflected_temperature_kelvin": 300.15,
        "temperature_noise_amplitude_kelvin": 0.4,
        "temperature_radial_amplitude_kelvin": 0.3,
        "temperature_vertical_amplitude_kelvin": 0.2,
        "temperature_horizontal_amplitude_kelvin": 0.0,
        "emissivity_noise_amplitude": 0.020,
        "pattern_type": "fbm",
    },
    {
        "pattern": "/model_TeaTable/",
        "temperature_kelvin": 293.15,
        "emissivity": 0.72,
        "reflected_temperature_kelvin": 301.15,
        "temperature_noise_amplitude_kelvin": 3.0,
        "temperature_radial_amplitude_kelvin": 0.0,
        "temperature_vertical_amplitude_kelvin": 0.0,
        "temperature_horizontal_amplitude_kelvin": 0.0,
        "emissivity_noise_amplitude": 0.150,
        "pattern_type": "streaks",
    },
    {
        "pattern": "/Robotiq_2F_85/",
        "temperature_kelvin": 296.15,
        "emissivity": 0.60,
        "reflected_temperature_kelvin": 300.15,
        "temperature_noise_amplitude_kelvin": 1.2,
        "temperature_radial_amplitude_kelvin": 0.5,
        "temperature_vertical_amplitude_kelvin": 0.4,
        "temperature_horizontal_amplitude_kelvin": 0.2,
        "emissivity_noise_amplitude": 0.030,
        "pattern_type": "fbm",
    },
    {
        "pattern": "/Robot/",
        "temperature_kelvin": 295.15,
        "emissivity": 0.55,
        "reflected_temperature_kelvin": 300.15,
        "temperature_noise_amplitude_kelvin": 1.0,
        "temperature_radial_amplitude_kelvin": 0.2,
        "temperature_vertical_amplitude_kelvin": 0.2,
        "temperature_horizontal_amplitude_kelvin": 0.1,
        "emissivity_noise_amplitude": 0.030,
        "pattern_type": "fbm",
    },
    {
        "pattern": "/interactive_diningroom/",
        "temperature_kelvin": 293.15,
        "emissivity": 0.92,
        "reflected_temperature_kelvin": 301.15,
        "temperature_noise_amplitude_kelvin": 2.5,
        "temperature_radial_amplitude_kelvin": 0.0,
        "temperature_vertical_amplitude_kelvin": 0.0,
        "temperature_horizontal_amplitude_kelvin": 0.0,
        "emissivity_noise_amplitude": 0.130,
        "pattern_type": "blotches",
    },
)


def _path_matches_pattern(prim_path: str, pattern: str) -> bool:
    """Return whether the prim path contains a full path pattern."""
    return pattern in prim_path or prim_path.endswith(pattern.rstrip("/"))


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


def _smooth_image(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a batched single-channel image."""
    if kernel_size <= 1:
        return image

    kernel = _gaussian_kernel_1d(kernel_size, sigma, image.device, image.dtype)
    radius = kernel_size // 2
    image = image.unsqueeze(1)
    horizontal_kernel = kernel.view(1, 1, 1, kernel_size)
    vertical_kernel = kernel.view(1, 1, kernel_size, 1)
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal_kernel)
    image = F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical_kernel)
    return image.squeeze(1)


def _hash_to_unit_float(seed: int) -> float:
    """Map an integer seed deterministically to [0, 1)."""
    seed = (seed ^ 0x45D9F3B) * 0x45D9F3B
    seed = seed ^ (seed >> 16)
    return float(seed & 0xFFFFFFFF) / float(0x100000000)


def _seeded_range(seed: int, min_value: float, max_value: float) -> float:
    """Return a deterministic float in [min_value, max_value]."""
    return min_value + (max_value - min_value) * _hash_to_unit_float(seed)


def _lookup_material_properties(prim_path: str) -> dict[str, float | str | tuple[float, ...] | None]:
    """Resolve base thermal properties from the visible prim path."""
    for spec in _MATERIAL_SPECS:
        if _path_matches_pattern(prim_path, str(spec["pattern"])):
            return {
                "temperature_kelvin": float(spec["temperature_kelvin"]),
                "emissivity": float(spec["emissivity"]),
                "reflected_temperature_kelvin": float(
                    spec.get("reflected_temperature_kelvin", _GLOBAL_REFLECTED_TEMPERATURE_K)
                ),
                "temperature_noise_amplitude_kelvin": float(spec.get("temperature_noise_amplitude_kelvin", 0.0)),
                "temperature_radial_amplitude_kelvin": float(spec.get("temperature_radial_amplitude_kelvin", 0.0)),
                "temperature_vertical_amplitude_kelvin": float(
                    spec.get("temperature_vertical_amplitude_kelvin", 0.0)
                ),
                "temperature_horizontal_amplitude_kelvin": float(
                    spec.get("temperature_horizontal_amplitude_kelvin", 0.0)
                ),
                "emissivity_noise_amplitude": float(spec.get("emissivity_noise_amplitude", 0.0)),
                "pattern_type": str(spec.get("pattern_type", "fbm")),
                "heat_source_entity_name": spec.get("heat_source_entity_name"),
                "heat_source_offset_meters": tuple(
                    spec.get("heat_source_offset_meters", (0.0, 0.0, 0.0))
                ),
                "heat_source_amplitude_kelvin": float(spec.get("heat_source_amplitude_kelvin", 0.0)),
                "heat_source_sigma_meters": float(spec.get("heat_source_sigma_meters", 0.05)),
            }
    return {
        "temperature_kelvin": _ENVIRONMENT_TEMPERATURE_K,
        "emissivity": _DEFAULT_EMISSIVITY,
        "reflected_temperature_kelvin": _GLOBAL_REFLECTED_TEMPERATURE_K,
        "temperature_noise_amplitude_kelvin": 0.0,
        "temperature_radial_amplitude_kelvin": 0.0,
        "temperature_vertical_amplitude_kelvin": 0.0,
        "temperature_horizontal_amplitude_kelvin": 0.0,
        "emissivity_noise_amplitude": 0.0,
        "pattern_type": "fbm",
        "heat_source_entity_name": None,
        "heat_source_offset_meters": (0.0, 0.0, 0.0),
        "heat_source_amplitude_kelvin": 0.0,
        "heat_source_sigma_meters": 0.05,
    }


def _segmentation_id_materials(
    camera: Camera,
    segmentation_data_type: str,
) -> list[dict[int, dict[str, float]]]:
    """Resolve per-env segmentation ids to thermal material properties."""
    materials_per_env: list[dict[int, dict[str, float]]] = []
    for env_info in camera.data.info:
        id_to_labels = env_info.get(segmentation_data_type, {}).get("idToLabels", {})
        env_materials: dict[int, dict[str, float]] = {}
        for segmentation_id, prim_path in id_to_labels.items():
            if not isinstance(prim_path, str):
                continue
            env_materials[int(segmentation_id)] = _lookup_material_properties(prim_path)
        materials_per_env.append(env_materials)
    return materials_per_env


def _object_local_coordinates(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return object-local normalized coordinates for visible pixels in a mask."""
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    if y_coords.numel() == 0:
        return None

    y_float = y_coords.to(dtype=torch.float32)
    x_float = x_coords.to(dtype=torch.float32)

    x_min, x_max = x_float.min(), x_float.max()
    y_min, y_max = y_float.min(), y_float.max()
    x_extent = torch.clamp(x_max - x_min, min=1.0)
    y_extent = torch.clamp(y_max - y_min, min=1.0)

    x_local = (x_float - x_min) / x_extent
    y_local = (y_float - y_min) / y_extent
    return y_local, x_local


def _temperature_structure_field(mask: torch.Tensor) -> torch.Tensor | None:
    """Create a smooth structured field from local mask coordinates."""
    coordinates = _object_local_coordinates(mask)
    if coordinates is None:
        return None
    y_local, x_local = coordinates

    x_centered = x_local - 0.5
    y_centered = y_local - 0.5
    radial_distance = torch.sqrt(x_centered.square() + y_centered.square()) / math.sqrt(0.5)
    radial_profile = (1.0 - radial_distance.clamp(0.0, 1.0)).pow(1.75)
    vertical_profile = -y_centered
    horizontal_profile = x_centered

    structure = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    structure[y_coords, x_coords] = radial_profile
    return structure, vertical_profile, horizontal_profile


def _fbm_values(
    seed: int,
    x_local: torch.Tensor,
    y_local: torch.Tensor,
    octaves: int = 5,
    lacunarity: float = 2.0,
    persistence: float = 0.55,
) -> torch.Tensor:
    """Sum of randomly-oriented sin-wave octaves with halving amplitudes (FBM-style)."""
    values = torch.zeros_like(x_local)
    amplitude = 1.0
    amplitude_sum = 0.0
    base_freq = _seeded_range(seed + 1, 0.8, 1.6)
    for octave in range(octaves):
        freq = base_freq * (lacunarity ** octave)
        angle = _seeded_range(seed + 100 + 4 * octave, 0.0, 2.0 * math.pi)
        phase = _seeded_range(seed + 101 + 4 * octave, 0.0, 2.0 * math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        term = torch.sin(2.0 * math.pi * freq * (cos_a * x_local + sin_a * y_local) + phase)
        values = values + amplitude * term
        amplitude_sum += amplitude
        amplitude *= persistence
    if amplitude_sum > 0.0:
        values = values / amplitude_sum
    return values.clamp(-1.0, 1.0)


def _streak_values(seed: int, x_local: torch.Tensor, y_local: torch.Tensor) -> torch.Tensor:
    """Wood-grain-like anisotropic stripes with low-frequency warp + cross modulation."""
    angle = _seeded_range(seed + 11, 0.0, math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u = cos_a * x_local + sin_a * y_local
    v = -sin_a * x_local + cos_a * y_local
    stripe_freq = _seeded_range(seed + 12, 7.0, 14.0)
    stripe_phase = _seeded_range(seed + 13, 0.0, 2.0 * math.pi)
    warp_freq = _seeded_range(seed + 14, 1.0, 2.5)
    warp_phase = _seeded_range(seed + 15, 0.0, 2.0 * math.pi)
    warp = 0.6 * torch.sin(2.0 * math.pi * warp_freq * v + warp_phase)
    stripes = torch.sin(2.0 * math.pi * stripe_freq * u + stripe_phase + warp)
    large_freq = _seeded_range(seed + 16, 0.4, 1.2)
    large_phase = _seeded_range(seed + 17, 0.0, 2.0 * math.pi)
    large_mod = 0.35 * torch.sin(2.0 * math.pi * large_freq * v + large_phase)
    return (0.7 * stripes + large_mod).clamp(-1.0, 1.0)


def _blotch_values(
    seed: int, x_local: torch.Tensor, y_local: torch.Tensor, num_blobs: int = 6
) -> torch.Tensor:
    """Sum of randomly-placed warm/cool Gaussian blobs in object-local coords."""
    values = torch.zeros_like(x_local)
    for blob in range(num_blobs):
        cx = _seeded_range(seed + 21 + 5 * blob, -0.1, 1.1)
        cy = _seeded_range(seed + 22 + 5 * blob, -0.1, 1.1)
        sigma = _seeded_range(seed + 23 + 5 * blob, 0.18, 0.40)
        sign = 1.0 if _hash_to_unit_float(seed + 24 + 5 * blob) > 0.5 else -1.0
        dx = x_local - cx
        dy = y_local - cy
        values = values + sign * torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma * sigma))
    max_abs = float(values.abs().max().item()) if values.numel() > 0 else 0.0
    if max_abs > 1e-6:
        values = values / max_abs
    return values.clamp(-1.0, 1.0)


def _pattern_noise(
    mask: torch.Tensor,
    env_index: int,
    segmentation_id: int,
    key_offset: int,
    pattern_type: str = "fbm",
) -> torch.Tensor | None:
    """Create deterministic patterned variation over a visible object mask."""
    coordinates = _object_local_coordinates(mask)
    if coordinates is None:
        return None
    y_local, x_local = coordinates
    seed = env_index * 1_000_003 + segmentation_id * 97_409 + key_offset * 65_537

    if pattern_type == "streaks":
        values = _streak_values(seed, x_local, y_local)
    elif pattern_type == "blotches":
        values = _blotch_values(seed, x_local, y_local)
    else:
        values = _fbm_values(seed, x_local, y_local)

    variation = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    variation[y_coords, x_coords] = values
    return variation


def _world_to_camera_frame(
    world_position: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:
    """Transform per-env world positions (N, 3) into per-env camera-frame positions."""
    relative = world_position - camera.data.pos_w
    return quat_apply_inverse(camera.data.quat_w_ros, relative)


def _back_project_pixels(
    y_coords: torch.Tensor,
    x_coords: torch.Tensor,
    depth_values: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Back-project pixels + depths to camera-frame 3-D points. Returns (P, 3)."""
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    z = depth_values
    x = (x_coords.to(torch.float32) - cx) * z / fx
    y = (y_coords.to(torch.float32) - cy) * z / fy
    return torch.stack([x, y, z], dim=-1)


def _heat_source_3d_at_mask(
    mask: torch.Tensor,
    depth_image: torch.Tensor,
    intrinsics: torch.Tensor,
    heat_source_cam: torch.Tensor,
    sigma_meters: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Compute a 3-D Gaussian heat-source falloff over the visible surface pixels of a mask."""
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    if y_coords.numel() == 0:
        return None
    depth_values = depth_image[y_coords, x_coords]
    valid = torch.isfinite(depth_values) & (depth_values > 0.01)
    if not torch.any(valid):
        return None
    y_coords = y_coords[valid]
    x_coords = x_coords[valid]
    depth_values = depth_values[valid]
    points_cam = _back_project_pixels(y_coords, x_coords, depth_values, intrinsics)
    deltas = points_cam - heat_source_cam.unsqueeze(0)
    distances_sq = deltas.square().sum(dim=-1)
    falloff = torch.exp(-distances_sq / (2.0 * sigma_meters * sigma_meters))
    return y_coords, x_coords, falloff


def _build_heat_source_camera_positions(
    env,
    camera: Camera,
) -> dict[str, torch.Tensor]:
    """Pre-compute camera-frame 3-D positions of each unique heat-source entity (per env)."""
    positions: dict[str, torch.Tensor] = {}
    scene_keys = set(env.scene.keys())
    for spec in _MATERIAL_SPECS:
        entity_name = spec.get("heat_source_entity_name")
        if entity_name is None or entity_name in positions or entity_name not in scene_keys:
            continue
        root_pos = env.scene[entity_name].data.root_pos_w  # (num_envs, 3)
        offset_meters = spec.get("heat_source_offset_meters", (0.0, 0.0, 0.0))
        offset = torch.tensor(offset_meters, device=root_pos.device, dtype=root_pos.dtype)
        world_position = root_pos + offset
        positions[entity_name] = _world_to_camera_frame(world_position, camera)
    return positions


def _radiance_from_graybody_terms(
    temperature_kelvin: torch.Tensor,
    emissivity: torch.Tensor,
    reflected_temperature_kelvin: torch.Tensor,
) -> torch.Tensor:
    """Compute apparent graybody radiance with object and reflected temperatures."""
    sigma = _STEFAN_BOLTZMANN_CONSTANT
    return (
        emissivity * sigma * temperature_kelvin.pow(4)
        + (1.0 - emissivity) * sigma * reflected_temperature_kelvin.pow(4)
    )


def _colorize_heatmap(normalized_radiance: torch.Tensor) -> torch.Tensor:
    """Map normalized radiance to a blue-yellow-red heatmap in [0, 1]."""
    cold = torch.tensor(_COLD_COLOR, device=normalized_radiance.device, dtype=torch.float32)
    mid = torch.tensor(_MID_COLOR, device=normalized_radiance.device, dtype=torch.float32)
    hot = torch.tensor(_HOT_COLOR, device=normalized_radiance.device, dtype=torch.float32)

    low_alpha = (normalized_radiance / 0.5).clamp(0.0, 1.0)
    high_alpha = ((normalized_radiance - 0.5) / 0.5).clamp(0.0, 1.0)
    low_rgb = cold + low_alpha.unsqueeze(-1) * (mid - cold)
    high_rgb = mid + high_alpha.unsqueeze(-1) * (hot - mid)
    rgb = torch.where((normalized_radiance <= 0.5).unsqueeze(-1), low_rgb, high_rgb)
    return rgb / 255.0


def graybody_pseudo_thermal_image(
    env,
    sensor_cfg: SceneEntityCfg,
    segmentation_data_type: str = _THERMAL_SEGMENTATION_DATA_TYPE,
    depth_data_type: str = _THERMAL_DEPTH_DATA_TYPE,
    global_reflected_temperature_kelvin: float = _GLOBAL_REFLECTED_TEMPERATURE_K,
    gaussian_kernel_size: int = _DEFAULT_GAUSSIAN_KERNEL_SIZE,
    gaussian_sigma: float = _DEFAULT_GAUSSIAN_SIGMA,
    enable_temperature_variation: bool = _TEMPERATURE_VARIATION_ENABLED,
    enable_emissivity_variation: bool = _EMISSIVITY_VARIATION_ENABLED,
) -> torch.Tensor:
    """Return a normalized pseudo-thermal image with graybody radiance, patterned noise, and 3-D heat sources."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    segmentation = camera.data.output[segmentation_data_type]
    if segmentation.dim() == 4 and segmentation.shape[-1] == 1:
        segmentation = segmentation.squeeze(-1)

    depth_image = camera.data.output.get(depth_data_type) if _HEAT_SOURCE_ENABLED else None
    if depth_image is not None and depth_image.dim() == 4 and depth_image.shape[-1] == 1:
        depth_image = depth_image.squeeze(-1)

    background_radiance = _STEFAN_BOLTZMANN_CONSTANT * _ENVIRONMENT_TEMPERATURE_K**4
    radiance = torch.full(
        segmentation.shape,
        fill_value=background_radiance,
        device=segmentation.device,
        dtype=torch.float32,
    )

    materials_per_env = _segmentation_id_materials(camera=camera, segmentation_data_type=segmentation_data_type)
    heat_source_cam_positions = (
        _build_heat_source_camera_positions(env, camera)
        if (_HEAT_SOURCE_ENABLED and depth_image is not None)
        else {}
    )
    intrinsics_per_env = camera.data.intrinsic_matrices if _HEAT_SOURCE_ENABLED else None
    for env_index, id_to_material in enumerate(materials_per_env):
        for segmentation_id, material in id_to_material.items():
            pixel_mask = segmentation[env_index] == segmentation_id
            if not torch.any(pixel_mask):
                continue

            pattern_type = str(material.get("pattern_type", "fbm"))
            temperature_map = torch.full(
                pixel_mask.shape,
                fill_value=material["temperature_kelvin"],
                device=segmentation.device,
                dtype=torch.float32,
            )
            emissivity_map = torch.full(
                pixel_mask.shape,
                fill_value=material["emissivity"],
                device=segmentation.device,
                dtype=torch.float32,
            )
            reflected_temperature_map = torch.full(
                pixel_mask.shape,
                fill_value=material.get("reflected_temperature_kelvin", global_reflected_temperature_kelvin),
                device=segmentation.device,
                dtype=torch.float32,
            )

            if enable_temperature_variation and material["temperature_noise_amplitude_kelvin"] > 0.0:
                temperature_variation = _pattern_noise(
                    pixel_mask, env_index, segmentation_id, key_offset=1, pattern_type=pattern_type
                )
                if temperature_variation is not None:
                    temperature_map = temperature_map + (
                        material["temperature_noise_amplitude_kelvin"] * temperature_variation
                    )

            temperature_structure = _temperature_structure_field(pixel_mask)
            if temperature_structure is not None:
                radial_profile, vertical_profile, horizontal_profile = temperature_structure
                y_coords, x_coords = torch.nonzero(pixel_mask, as_tuple=True)
                temperature_map[y_coords, x_coords] = (
                    temperature_map[y_coords, x_coords]
                    + material["temperature_radial_amplitude_kelvin"] * radial_profile[y_coords, x_coords]
                    + material["temperature_vertical_amplitude_kelvin"] * vertical_profile
                    + material["temperature_horizontal_amplitude_kelvin"] * horizontal_profile
                )

            heat_source_entity_name = material.get("heat_source_entity_name")
            heat_source_amplitude = float(material.get("heat_source_amplitude_kelvin", 0.0))
            if (
                _HEAT_SOURCE_ENABLED
                and depth_image is not None
                and intrinsics_per_env is not None
                and heat_source_entity_name is not None
                and heat_source_amplitude > 0.0
                and heat_source_entity_name in heat_source_cam_positions
            ):
                heat_source_cam = heat_source_cam_positions[heat_source_entity_name][env_index]
                if heat_source_cam[2].item() > 0.05:
                    sigma_meters = float(material.get("heat_source_sigma_meters", 0.05))
                    falloff_result = _heat_source_3d_at_mask(
                        pixel_mask,
                        depth_image[env_index],
                        intrinsics_per_env[env_index],
                        heat_source_cam,
                        sigma_meters,
                    )
                    if falloff_result is not None:
                        y_heat, x_heat, falloff_values = falloff_result
                        temperature_map[y_heat, x_heat] = (
                            temperature_map[y_heat, x_heat]
                            + heat_source_amplitude * falloff_values
                        )

            if enable_emissivity_variation and material["emissivity_noise_amplitude"] > 0.0:
                emissivity_variation = _pattern_noise(
                    pixel_mask, env_index, segmentation_id, key_offset=2, pattern_type=pattern_type
                )
                if emissivity_variation is not None:
                    emissivity_map = emissivity_map + (material["emissivity_noise_amplitude"] * emissivity_variation)

            emissivity_map = emissivity_map.clamp(0.02, 0.999)
            object_radiance = _radiance_from_graybody_terms(
                temperature_kelvin=temperature_map,
                emissivity=emissivity_map,
                reflected_temperature_kelvin=reflected_temperature_map,
            )
            radiance[env_index][pixel_mask] = object_radiance[pixel_mask]

    radiance = _smooth_image(radiance, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

    normalization_min_temperature = min(_NORMALIZATION_MIN_TEMPERATURE_K, _ENVIRONMENT_TEMPERATURE_K)
    normalization_max_temperature = max(_NORMALIZATION_MAX_TEMPERATURE_K, global_reflected_temperature_kelvin)
    radiance_min = _STEFAN_BOLTZMANN_CONSTANT * normalization_min_temperature**4
    radiance_max = _STEFAN_BOLTZMANN_CONSTANT * normalization_max_temperature**4
    normalized_radiance = ((radiance - radiance_min) / (radiance_max - radiance_min)).clamp(0.0, 1.0)
    return _colorize_heatmap(normalized_radiance)


def _configure_thermal_cameras(env_cfg):
    """Enable uncolorized instance-id segmentation and depth on thermal cameras."""
    for camera_name in ("table_cam", "wrist_cam"):
        camera_cfg = getattr(env_cfg.scene, camera_name)
        for data_type in (_THERMAL_SEGMENTATION_DATA_TYPE, _THERMAL_DEPTH_DATA_TYPE):
            if data_type not in camera_cfg.data_types:
                camera_cfg.data_types = [*camera_cfg.data_types, data_type]
        camera_cfg.colorize_instance_id_segmentation = False


@configclass
class GraybodyThermalObservationsCfg(hot_tea_droid_env_cfg.ObservationsCfg):
    """Observation specs with RGB and graybody pseudo-thermal table and wrist camera images."""

    @configclass
    class PolicyCfg(hot_tea_droid_env_cfg.ObservationsCfg.PolicyCfg):
        thermal_table_cam = ObsTerm(
            func=graybody_pseudo_thermal_image,
            params={"sensor_cfg": SceneEntityCfg("table_cam")},
        )
        thermal_wrist_cam = ObsTerm(
            func=graybody_pseudo_thermal_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidHotTeaJointPosGraybodyThermalEnvCfg(
    hot_tea_droid_env_cfg.DroidHotTeaJointPosVisuomotorEnvCfg
):
    """Hot tea task with joint-position control and graybody pseudo-thermal observations."""

    observations: GraybodyThermalObservationsCfg = GraybodyThermalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_thermal_cameras(self)


@configclass
class DroidHotTeaIkRelGraybodyThermalEnvCfg(
    hot_tea_droid_env_cfg.DroidHotTeaIkRelVisuomotorEnvCfg
):
    """Hot tea task with relative IK control and graybody pseudo-thermal observations."""

    observations: GraybodyThermalObservationsCfg = GraybodyThermalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_thermal_cameras(self)
