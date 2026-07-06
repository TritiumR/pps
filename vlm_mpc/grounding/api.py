"""Controller-agnostic grounding contract: what a front-end must produce for a base controller.

A ``GroundingSource`` (ReKep, VoxPoser, MOKA, or plain ground-truth) turns an environment into a
``Grounding`` -- scene obstacles plus an ordered list of stages. The base driver consumes only this
contract, so front-ends and controllers vary independently.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class SceneObject:
    """A scene object the controller may avoid or manipulate, with a live pose and geometry.

    ``pos`` is a live accessor (a keypoint tracker or a GT pose query) so the controller re-reads the
    current world position each chunk. ``extents`` are half-extents ``(grip, keepout, half_height)`` --
    the narrow horizontal one for grasping, the wide one for collision keepout.
    """
    name: str
    pos: Callable[[], np.ndarray]
    extents: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Stage:
    """One step of a task: reach a live ``target`` with a ``gripper`` intent until the stage advances.

    ``grasp_obj`` is the object the fingers straddle and that collision excludes while approaching;
    ``payload`` is a carried object, and ``place_target`` the object being placed onto -- both are
    collision-excluded so the pick/place can reach. ``done_flag`` names the env ``subtask_terms`` flag
    that gates the advance (actual task progress, e.g. ``"grasp_pear"`` / ``"pear_on_scale"``); when set
    the driver advances on that flag instead of the held-grasp heuristic. ``done`` remains the gripper
    release / fallback advance predicate (e.g. a place stage releases once its constraint is satisfied).
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
    """Turns an environment into a ``Grounding`` (implemented by ReKep/VoxPoser/MOKA/GT front-ends)."""

    def ground(self, env) -> Grounding: ...
