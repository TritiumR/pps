"""Grounding contract + source registry: a ``GroundingSource`` turns an env into a ``Grounding``
(scene objects + ordered stages). ``get_source`` resolves sources by name, imported lazily.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np

__all__ = ["Grounding", "GroundingSource", "SceneObject", "Stage", "get_source"]


@dataclasses.dataclass(frozen=True)
class SceneObject:
    """A scene object with a live ``pos`` accessor and half-extents ``(grip, keepout, half_height)``."""
    name: str
    pos: Callable[[], np.ndarray]
    extents: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Stage:
    """One task step: reach a live ``target`` with a ``gripper`` intent until the stage advances.

    ``grasp_obj``/``payload``/``place_target`` are collision-excluded; ``done_flag`` names the env
    flag gating the advance, ``done`` is the fallback predicate.
    """
    name: str
    target: Callable[[], np.ndarray]          # reference point (gripper proximity + place-release)
    gripper: str                              # "close" | "hold" | "open" | "place"
    grasp_obj: Optional[str] = None
    payload: Optional[str] = None
    place_target: Optional[str] = None        # object placed onto (collision-excluded during this stage)
    done_flag: Optional[str] = None           # env subtask_terms flag gating the advance (task progress)
    done: Callable[[], bool] = lambda: False
    orient: str = "down"
    # ReKep constraint-as-cost (optional): torch callables fn(ee_pos[K,H,3], kp[N,K,H,3]) -> [K,H].
    # When ``constraint`` is set it becomes J_task; ``held_idx`` are keypoints riding the gripper.
    constraint: Optional[Callable] = None
    path_fns: tuple = ()
    held_idx: tuple = ()


@dataclasses.dataclass(frozen=True)
class Grounding:
    """A grounded task: obstacles + ordered stages, plus the objects intentionally manipulated."""
    objects: list[SceneObject]
    stages: list[Stage]
    manipulated: frozenset[str] = frozenset()      # excluded from scene-disturbance reporting
    keypoints: Optional[Callable[[], np.ndarray]] = None   # live tracked keypoints [N,3] (ReKep grounding)


class GroundingSource(Protocol):
    """Turns an environment into a ``Grounding``; ``world`` is the only source of object state."""

    def ground(self, env, world) -> Grounding: ...


def get_source(name: str, **kwargs) -> GroundingSource:
    """Return a grounding source by name (``gt`` / ``rekep_fake`` / ``rekep_real``); lazy imports."""
    if name == "gt":
        from vlm_dp.grounding.gt import GTGrounding
        return GTGrounding(grasp_obj=kwargs.get("grasp_obj", "pear"), place_obj=kwargs.get("place_obj", "scale"),
                           grasp_objs=kwargs.get("grasp_objs"))
    if name in ("rekep_fake", "rekep_real"):
        from vlm_dp.grounding.rekep import RekepGrounding
        return RekepGrounding(vlm="real" if name == "rekep_real" else "fake",
                              task_key=kwargs.get("task_key", "weight"),
                              place_obj=kwargs.get("place_obj", "scale"),
                              perception=kwargs.get("perception"))
    raise ValueError(f"unknown grounding source: {name!r}")
