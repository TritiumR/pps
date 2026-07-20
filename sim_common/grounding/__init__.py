"""Grounding contract + source registry for base controllers.

A ``GroundingSource`` (ReKep, VoxPoser, MOKA, or plain ground-truth) turns an environment into a
``Grounding`` -- scene obstacles plus an ordered list of stages. The base driver consumes only this
contract, so front-ends and controllers vary independently. ``get_source`` resolves a source by name,
importing each source lazily so heavy front-end dependencies stay local to that source.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np

__all__ = ["Grounding", "GroundingSource", "SceneObject", "Stage", "get_source"]


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

    ``grasp_obj`` (fingers straddle it), ``payload`` (carried), and ``place_target`` (placed onto) are
    collision-excluded so the pick/place can reach. ``done_flag`` names the env ``subtask_terms`` flag that
    gates the advance (e.g. ``"grasp_pear"`` / ``"pear_on_scale"``); ``done`` is the fallback advance /
    gripper-release predicate.
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
    """Turns an environment into a ``Grounding`` (implemented by ReKep/VoxPoser/MOKA/GT front-ends).

    ``world`` is where object state comes from -- the simulator, or an estimate built from a camera and the
    joint encoders. A grounding never reads object state any other way, so the same front-end runs
    privileged or not.
    """

    def ground(self, env, world) -> Grounding: ...


def get_source(name: str, **kwargs) -> GroundingSource:
    """Return a grounding source by name (``gt`` / ``rekep_fake`` / ``rekep_real``; VoxPoser/MOKA to follow).

    Each source is imported lazily so a light source (``gt``) never pulls a heavy one's deps (``rekep`` ->
    DINOv2/clustering).
    """
    if name == "gt":
        from sim_common.grounding.gt import GTGrounding
        return GTGrounding(grasp_obj=kwargs.get("grasp_obj", "pear"), place_obj=kwargs.get("place_obj", "scale"),
                           grasp_objs=kwargs.get("grasp_objs"))
    if name in ("rekep_fake", "rekep_real"):
        from sim_common.grounding.rekep import RekepGrounding
        return RekepGrounding(vlm="real" if name == "rekep_real" else "fake",
                              task_key=kwargs.get("task_key", "weight"),
                              place_obj=kwargs.get("place_obj", "scale"),
                              perception=kwargs.get("perception"))
    raise ValueError(f"unknown grounding source: {name!r}")
