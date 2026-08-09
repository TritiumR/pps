"""Grounding contract and source registry.

A GroundingSource turns an env into a Grounding (scene objects and ordered stages). get_source resolves
sources by name with lazy imports.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np

__all__ = ["Grounding", "GroundingSource", "SceneObject", "Stage", "get_source"]


@dataclasses.dataclass(frozen=True)
class SceneObject:
    """A scene object with a live pos accessor and half-extents (grip, keepout, half_height)."""
    name: str
    pos: Callable[[], np.ndarray]
    extents: tuple[float, float, float]
    # Unit horizontal narrow-axis direction to close across. None if the object is round.
    axis: Optional[tuple] = None
    # Local half-width near the grasp keypoint, the grasp terms' feasibility radius when set. A thin lip
    # fits the gripper, where the whole-body extent never would.
    grasp_extent: Optional[float] = None
    # (axis[3], half_len): the graspable segment along the object's long horizontal axis. A point target
    # pins the grasp to one pose and leaves a steering proxy nothing to move. A region gives breadth over
    # where to grasp.
    grasp_region: Optional[tuple] = None


@dataclasses.dataclass(frozen=True)
class Stage:
    """One task step: reach a live target with a gripper intent until the stage advances.

    grasp_obj, payload and place_target are collision-excluded. done_flag names the env flag gating the
    advance, and done is the fallback predicate.
    """
    name: str
    target: Callable[[], np.ndarray]          # reference point (gripper proximity and place-release)
    gripper: str                              # close, hold, open or place
    grasp_obj: Optional[str] = None
    payload: Optional[str] = None
    place_target: Optional[str] = None        # object placed onto, collision-excluded during this stage
    done_flag: Optional[str] = None           # env subtask_terms flag gating the advance (task progress)
    done: Callable[[], bool] = lambda: False
    # Optional desired world-frame tool (+z) axis. Used when orient="axis"; a callable
    # keeps fixture-relative axes live without exposing simulator state to the cost.
    approach_axis: Optional[object] = None
    # Optional tool (+x) axis fixes gripper roll when finger direction matters.
    approach_x_axis: Optional[object] = None
    orient: str = "down"
    orientation_scale: float = 1.0
    grasp_slack: Optional[float] = None
    # Stay fully open in transit and close only once the measured TCP reaches
    # the grasp neighbourhood. Opt-in for plans whose sampled gripper channel
    # otherwise dithers around half-close.
    force_gripper_at_target: bool = False
    # Allow an opt-in grasp to advance after a latched close when raw visual
    # tracking confirms that the object rose with the hand.
    grasp_advance_on_visual_rise: bool = False
    # Optional object-relative waypoints used before approaching the ordinary target.
    grasp_transit_offsets: Optional[tuple] = None
    rise_confirm: Optional[float] = None
    place_point: Optional[Callable[[], np.ndarray]] = None    # calibrated top-surface seat point
    carry_z: Optional[Callable[[], float]] = None             # carry altitude for the place transit
    advance_on_payload_rise: bool = False     # hold stages: advance after the sensed payload rises
    advance_on_done: bool = False             # hold stages: advance on done() instead of the height gate
    # Called by the bridge on every entry (advance or backtrack) to this stage. Lets a stage latch a
    # world anchor at the moment it begins, rather than tracking a live estimate that its own motion moves.
    on_enter: Optional[Callable[[], None]] = None
    # ReKep constraint-as-cost (optional): torch callables fn(ee_pos[K,H,3], kp[N,K,H,3]) -> [K,H]. When
    # constraint is set it becomes J_task, and held_idx are keypoints riding the gripper.
    constraint: Optional[Callable] = None
    path_fns: tuple = ()
    held_idx: tuple = ()
    # pinch: straddle-grasp a free object. press: contact-and-hold an articulated part such as a lid, with
    # no pinch-certification, advancing on contact so the sub-goal drives it.
    contact: str = "pinch"
    # Articulated-fixture geometry from the joint the stage drives. FK is contact-blind, so these
    # describe the end-effector trajectory that WOULD produce it; keys are documented on
    # cost.terms.hook_pull / press_axis, the only readers. None on free-body stages.
    pull: Optional[Callable[[], Optional[dict]]] = None
    press: Optional[Callable[[], Optional[dict]]] = None
    # Insertion-corridor geometry, measured from the receptacle and the payload that passes through
    # it. The keys are documented on cost.terms._insert, the only reader. None on every stage whose
    # place is a set-down, and every configuration that does not list the insert terms is unaffected.
    insert: Optional[Callable[[], Optional[dict]]] = None
    # surface: set down on the destination's top. container: the VLM put it inside, and a container's top
    # is its rim, so the set-down terms must not seat it there.
    place_mode: str = "surface"
    # Steering authority policy, authored by the plan (never / on_failure / always). None means the
    # plan did not speak: the gate stays closed. The expert term may modify the sampler only under
    # this policy, evaluated by the bridge's failure gate.
    steer_policy: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Grounding:
    """A grounded task: obstacles and ordered stages, plus the objects intentionally manipulated."""
    objects: list[SceneObject]
    stages: list[Stage]
    manipulated: frozenset[str] = frozenset()      # excluded from scene-disturbance reporting
    keypoints: Optional[Callable[[], np.ndarray]] = None   # live tracked keypoints [N,3] (ReKep grounding)


class GroundingSource(Protocol):
    """Turns an environment into a Grounding. world is the only source of object state."""

    def ground(self, env, world) -> Grounding: ...


def get_source(name: str, **kwargs) -> GroundingSource:
    """Return a grounding source by name (gt, rekep_fake, rekep_real, and _vlm variants), lazy imports.

    There are no task-specific role defaults: a missing role fails loudly rather than silently binding
    another task's objects (a task without place_obj would otherwise default to the weight task's scale).
    """
    if name == "gt" and kwargs.get("task_key") == "capsule":
        from vlm_dp.grounding.capsule import CapsuleGrounding
        return CapsuleGrounding(grasp_obj=kwargs.get("grasp_obj"),
                                place_obj=kwargs.get("place_obj"))
    if name == "gt":
        from vlm_dp.grounding.gt import GTGrounding
        return GTGrounding(grasp_obj=kwargs.get("grasp_obj"), place_obj=kwargs.get("place_obj"),
                           grasp_objs=kwargs.get("grasp_objs"),
                           seat_shift=kwargs.get("seat_shift", True))
    if name in ("rekep_fake", "rekep_real", "rekep_fake_vlm", "rekep_real_vlm"):
        from vlm_dp.grounding.rekep import RekepGrounding
        # A _vlm suffix emits stages straight from the VLM constraints (opt-in). Otherwise the default
        # hand-coded grasp-lift-place template. Same front-end either way.
        return RekepGrounding(vlm="real" if name.startswith("rekep_real") else "fake",
                              stages="vlm" if name.endswith("_vlm") else "template",
                              task_key=kwargs.get("task_key"),
                              place_obj=kwargs.get("place_obj"),
                              grasp_objs=kwargs.get("grasp_objs"),
                              support=kwargs.get("support"),
                              perception=kwargs.get("perception"),
                              seat_shift=kwargs.get("seat_shift", True),
                              local_grasp=kwargs.get("local_grasp", False),
                              local_grasp_radius=kwargs.get("local_grasp_radius", 0.05),
                              kp_source=kwargs.get("kp_source", "perception"),
                              contact_criterion=kwargs.get("contact_criterion", "feasibility"),
                              subgoal_eps=kwargs.get("subgoal_eps", 0.06),
                              open_half=kwargs.get("open_half", 0.04),
                              rotate_grasp_offset=kwargs.get("rotate_grasp_offset", False),
                              lift_latch_xy=kwargs.get("lift_latch_xy", False),
                              seat_from_plane=kwargs.get("seat_from_plane", False))
    raise ValueError(f"unknown grounding source: {name!r}")
