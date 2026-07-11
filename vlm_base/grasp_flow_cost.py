"""Stage-driven adapter for Yixuan's GraspFlowStateCost.

Reuses her validated grasp/lift/place geometry (``sim_free_mpc.costs_grasp_flow``) verbatim, but
replaces her hardcoded weight-task dispatch (which reads ``grasp_pear`` / ``pear_on_scale`` env flags
and hardcodes the scale as the place target) with the base driver's *current stage*. The driver puts
the stage's roles in the cost context -- ``grasp_obj`` (grasp stage), ``payload`` (carried; lift + place
stages), ``place_target`` (place stage) -- and these overrides map them to her grasp/lift/place term
methods. Her cost geometry runs unchanged, so any task's ReKep/GT grounding drives the same validated
grasp.

The planner already accepts her ``tcp_pos`` call signature (``sim_free_mpc/planner.py`` tries it before
``ee_pos``), so no signature adapter is needed; the cost is dropped in as ``mpc.cost`` directly.
"""
from __future__ import annotations

from typing import Any

import torch

from sim_common.grounding import Grounding, Stage
from sim_free_mpc.costs_explore import _target_from_object
from sim_free_mpc.costs_grasp_flow import GraspFlowCostWeights, GraspFlowStateCost

_PLACE_CLEARANCE_Z = 0.04   # payload rests this far above the place object's top surface (her release clearance)
_DEFAULT_HALF_HEIGHT = 0.05


def coarsen_grounding(grounding: Grounding) -> Grounding:
    """Collapse a grounding's fine grasp/lift/place stages into one 'transport' stage per object.

    Yixuan's cost is a self-contained grasp->lift->place state machine (it decides the phase from the env
    grasp flag + the object's height). Feeding it the driver's *fine* stages fought that machine (the driver
    advanced lift->place before the object was lifted, dropping it). Instead we give it one coarse stage per
    object -- carrying the grasp object + its place target + the place done_flag -- and let her cost sequence
    the phases internally. A grasp stage (payload=None) opens a transport; the place stage (place_target set)
    closes it, contributing the place target + the advance flag.
    """
    transports, cur = [], None
    for s in grounding.stages:
        if s.payload is None and s.grasp_obj is not None:          # grasp stage -> open a new transport
            if cur is not None:
                transports.append(cur)
            cur = {"obj": s.grasp_obj, "place": None, "done_flag": None, "target": s.target, "done": s.done}
        else:                                                      # lift / place stage -> fold into the open one
            if cur is None:
                cur = {"obj": s.grasp_obj or s.payload, "place": None, "done_flag": None,
                       "target": s.target, "done": s.done}
            if s.place_target is not None:                         # the place stage: record the destination
                cur.update(place=s.place_target, done_flag=s.done_flag, target=s.target, done=s.done)
                transports.append(cur)
                cur = None
    if cur is not None:
        transports.append(cur)

    stages = [Stage(name=f"transport {t['obj']} -> {t['place']}", gripper="place", grasp_obj=t["obj"],
                    payload=t["obj"], place_target=t["place"], done_flag=t["done_flag"],
                    target=t["target"], done=t["done"]) for t in transports]
    return Grounding(objects=grounding.objects, stages=stages, manipulated=grounding.manipulated,
                     keypoints=grounding.keypoints)


class StageGraspFlowCost(GraspFlowStateCost):
    """``GraspFlowStateCost`` dispatched by the driver's stage instead of hardcoded weight flags.

    Pass ``task_name="weight"`` for the weight task to keep her weight-specific collision (``clear``)
    term; other tasks pass their own name (no ``clear`` term until collision is generalized).
    """

    def __init__(self, task_name: str = "stage", weights: GraspFlowCostWeights | None = None,
                 extents: dict | None = None, stage_dispatch: bool = True):
        super().__init__(task_name=task_name, weights=weights)
        self._extents = extents or {}
        self._stage_dispatch = stage_dispatch   # False -> use her native (weight-hardcoded) dispatch verbatim

    # --- dispatch: the driver's stage gives the object; the grasp flag gives grasp-vs-hold ------
    # This is her flag flip-flop (re-grasp the instant the object slips out) restricted to the driver's
    # current object, so a momentary grasp that then slips falls back to grasp instead of the lift
    # diverging on an unheld object. With stage_dispatch=False every hook defers to super() (her exact
    # weight-task dispatch), so the pipeline runs GraspFlowStateCost verbatim -- the faithfulness baseline.
    def _held(self, context: dict[str, Any], obj: str | None) -> bool:
        """Is ``obj`` currently grasped, per the env subtask flag (``grasp_<obj>``)?"""
        return bool(obj is not None and context.get("subtasks", {}).get(f"grasp_{obj}", False))

    def _grasp_object_name(self, context: dict[str, Any]) -> str | None:
        if not self._stage_dispatch:
            return super()._grasp_object_name(context)
        obj = context.get("grasp_obj")                # the transport's object (ReKep/driver provides it)
        return obj if (obj is not None and not self._held(context, obj)) else None

    def _place_object_name(self, context: dict[str, Any]) -> str | None:
        if not self._stage_dispatch:
            return super()._place_object_name(context)
        obj = context.get("grasp_obj")                # lift + place only once the object is actually held...
        return obj if self._held(context, obj) else None
    # NOTE: _needs_lift is NOT overridden -- her height-based check (lift to +0.20 m, then place) is what
    # sequences lift->place; tying it to a driver stage advance placed the object too early and dropped it.

    def _place_target(self, object_name: str, context: dict[str, Any], device, dtype):
        """Generic place target: the place object's top surface + the payload's half height + clearance.

        Replaces her scale-hardcoded ``_place_target``; for the weight task (place onto the scale) this
        resolves to the same top-of-scale point via the scene extents.
        """
        if not self._stage_dispatch:
            return super()._place_target(object_name, context, device, dtype)
        place_name = context.get("place_target")
        if place_name is None:
            return None
        place_pos = _target_from_object(context, place_name, device, dtype)
        if place_pos is None:
            return None
        objects = context.get("objects", {})
        stack = self._half_height(objects, place_name) + self._half_height(objects, object_name)
        clearance = float(context.get("place_clearance_z", _PLACE_CLEARANCE_Z))
        return place_pos + torch.tensor([0.0, 0.0, stack + clearance], device=device, dtype=dtype)

    def _half_height(self, objects: dict, name: str | None) -> float:
        """Object half-height from the scene extents (context first, then the passed-in fallback)."""
        obj = objects.get(name, {}) if name else {}
        ext = obj.get("extents") if isinstance(obj, dict) else None
        if ext is not None and len(ext) >= 3:
            return float(ext[2])
        return float(self._extents.get(name, (0.0, 0.0, _DEFAULT_HALF_HEIGHT))[2])

    # --- live target for the driver's distance readout (not used by the cost geometry itself) ----
    def target(self, context: dict[str, Any], device, dtype) -> torch.Tensor:
        payload = context.get("payload")
        if payload is not None:
            if context.get("place_target") is not None:
                place = self._place_target(payload, context, device, dtype)
                if place is not None:
                    return place
            obj = _target_from_object(context, payload, device, dtype)   # lift phase: the object itself
            if obj is not None:
                return obj
        grasp_obj = context.get("grasp_obj")
        if grasp_obj is not None:
            obj = _target_from_object(context, grasp_obj, device, dtype)
            if obj is not None:
                return obj
        return torch.as_tensor(context.get("target", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
