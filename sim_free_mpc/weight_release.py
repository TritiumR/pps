from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WeightReleaseDetector:
    """Latch a weight-task release phase from live EE motion and scale geometry."""

    ee_speed_threshold: float = 0.05
    scale_xy_radius: float = 0.12
    min_height: float = 0.0
    control_frequency: float = 15.0

    def __post_init__(self) -> None:
        if self.ee_speed_threshold <= 0.0:
            raise ValueError("ee_speed_threshold must be positive")
        if self.scale_xy_radius <= 0.0:
            raise ValueError("scale_xy_radius must be positive")
        if self.control_frequency <= 0.0:
            raise ValueError("control_frequency must be positive")
        self.reset()

    def reset(self) -> None:
        self.previous_eef_pos: np.ndarray | None = None
        self.release_object: str | None = None

    def update(
        self,
        *,
        eef_pos: Any,
        scale_top_pos: Any | None,
        subtasks: dict[str, bool],
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        current = np.asarray(eef_pos, dtype=np.float64).reshape(-1)[:3]
        if current.shape != (3,):
            raise ValueError(f"eef_pos must contain three coordinates, got {current.shape}")

        if self.previous_eef_pos is None:
            speed = float("inf")
        else:
            speed = float(
                np.linalg.norm(current - self.previous_eef_pos)
                * self.control_frequency
            )
        self.previous_eef_pos = current.copy()

        raw = {str(key): bool(value) for key, value in subtasks.items()}
        place_object = None
        if raw.get("grasp_apple", False):
            place_object = "apple"
        elif raw.get("grasp_pear", False):
            place_object = "pear"

        if self.release_object is not None and not raw.get(
            f"grasp_{self.release_object}", False
        ):
            self.release_object = None

        xy_distance = float("inf")
        vertical_clearance = float("-inf")
        above_scale = False
        if scale_top_pos is not None:
            scale_top = np.asarray(scale_top_pos, dtype=np.float64).reshape(-1)[:3]
            if scale_top.shape != (3,):
                raise ValueError(
                    f"scale_top_pos must contain three coordinates, got {scale_top.shape}"
                )
            xy_distance = float(np.linalg.norm(current[:2] - scale_top[:2]))
            vertical_clearance = float(current[2] - scale_top[2])
            above_scale = (
                xy_distance <= self.scale_xy_radius
                and vertical_clearance >= self.min_height
            )

        triggered = False
        if (
            self.release_object is None
            and place_object is not None
            and above_scale
            and speed <= self.ee_speed_threshold
        ):
            self.release_object = place_object
            triggered = True

        enriched = dict(raw)
        if self.release_object is not None:
            enriched[f"open_gripper_{self.release_object}"] = True

        return enriched, {
            "active": self.release_object is not None,
            "triggered": triggered,
            "release_object": self.release_object,
            "eef_speed_mps": speed,
            "speed_threshold_mps": float(self.ee_speed_threshold),
            "scale_xy_distance_m": xy_distance,
            "scale_xy_radius_m": float(self.scale_xy_radius),
            "scale_vertical_clearance_m": vertical_clearance,
            "scale_min_height_m": float(self.min_height),
            "above_scale": above_scale,
        }
