from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .costs_explore import (
    _WEIGHT_OBJECT_HALF_HEIGHT,
    _WEIGHT_OBJECT_HORIZONTAL_RADIUS,
    _WEIGHT_SCALE_CENTER_OFFSET,
    _WEIGHT_SCALE_TOP_OFFSET_Z,
    _apply_axis,
    _flag,
    _first_tensor,
    _get_vector,
    _obstacles_for_weight,
    _path_constraint_cost,
    _target_from_object,
    _transit_clearance_cost,
    _zero,
)

_GRASP_FLOW_TCP_TO_TIP_Z = 0.0
_GRASP_FLOW_CENTER_REGION_RADIUS_SCALE = 0.40
_GRASP_FLOW_PLACE_RELEASE_CLEARANCE_Z = 0.04
_GRASP_FLOW_PLACE_CARRY_CLEARANCE_Z = 0.20
_GRASP_FLOW_LIFT_HEIGHT = 0.15
_GRASP_FLOW_LIFT_DONE_TOLERANCE = 0.01


@dataclass(frozen=True)
class GraspFlowCostWeights:
    # Pre-grasp geometry intentionally follows ExploreStateCost.
    reach: float = 8.0
    terminal: float = 40.0
    orient: float = 8.0
    floor: float = 50.0
    smooth: float = 0.08
    local: float = 0.05
    yaw: float = 5.0
    straddle: float = 30.0
    tip_z: float = 80.0
    center_region: float = 120.0
    aperture_region: float = 80.0
    close_gripper: float = 20.0
    gripper_smooth: float = 0.5
    soft_grasp: float = 0.0
    lift_reach: float = 12.0
    lift_terminal: float = 80.0
    lift_xy: float = 80.0
    lift_z: float = 120.0
    lift_gripper: float = 40.0
    place_reach: float = 12.0
    place_terminal: float = 60.0
    place_xy: float = 120.0
    place_z: float = 80.0
    place_carry_height: float = 80.0
    place_gripper: float = 40.0
    clear: float = 0.1
    transit: float = 0.0
    path: float = 0.0


class GraspFlowStateCost:
    """Weight-task cost with hard subtask switches for grasp and place.

    Move terms are shared across stages.  Grasp and place terms are selected by
    environment subtask flags, without an internal latch.
    """

    def __init__(self, task_name: str, weights: GraspFlowCostWeights | None = None):
        self.task_name = task_name.lower()
        self.weights = weights or GraspFlowCostWeights()
        self.last_terms: dict[str, torch.Tensor] = {}
        self.last_stage = "init"
        self._lift_start_z: dict[str, float] = {}
        self._lift_start_frozen: set[str] = set()
        self._lift_done: dict[str, bool] = {}

    def reset(self) -> None:
        """Reset episode-local lift bookkeeping."""
        self._lift_start_z.clear()
        self._lift_start_frozen.clear()
        self._lift_done.clear()
        self.last_stage = "init"

    def _store(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_terms = {key: value.detach() for key, value in terms.items()}
        total = None
        for value in terms.values():
            total = value if total is None else total + value
        if total is None:
            raise ValueError("GraspFlowStateCost produced no terms.")
        return total

    def _move_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
        target_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        weights = self.weights
        batch = real_actions.shape[0]
        device = real_actions.device
        dtype = real_actions.dtype
        terms: dict[str, torch.Tensor] = {}

        if tcp_quat is not None:
            a_local = _get_vector(
                context,
                "a_local",
                default=(0.0, 0.0, 1.0),
                device=device,
                dtype=dtype,
            )
            target_axis = _get_vector(
                context,
                "target_axis",
                default=(0.0, 0.0, -1.0),
                device=device,
                dtype=dtype,
            )
            approach_axis = _apply_axis(tcp_quat, a_local)
            target_axis = target_axis / torch.clamp(torch.linalg.vector_norm(target_axis), min=1e-8)
            orient = 1.0 - torch.sum(approach_axis * target_axis.view(1, 1, 3), dim=-1)
            terms["orient"] = weights.orient * torch.mean(orient, dim=1)

        z_floor = float(context.get("z_floor", 0.0))
        floor = torch.clamp(z_floor - tcp_pos[..., 2], min=0.0).pow(2)
        terms["floor"] = weights.floor * torch.mean(floor, dim=1)

        q_traj = real_actions[..., :7]
        if q_traj.shape[1] > 1:
            smooth = torch.sum((q_traj[:, 1:] - q_traj[:, :-1]) ** 2, dim=(-1, -2))
        else:
            smooth = _zero(batch, device, dtype)
        terms["smooth"] = weights.smooth * smooth

        q_cur = _first_tensor(context, "joint_pos", device=device, dtype=dtype)
        if q_cur is not None:
            local = torch.sum((q_traj - q_cur[:7].view(1, 1, 7)) ** 2, dim=(-1, -2))
            terms["local"] = weights.local * local

        if "weight" in self.task_name:
            clear_terms = []
            ee_radius = float(context.get("tcp_radius", 0.035))
            clearance = float(context.get("clearance", 0.02))
            for obstacle_pos, obstacle_radius in _obstacles_for_weight(context, device, dtype):
                keepout = obstacle_radius + ee_radius + clearance
                dist = torch.linalg.vector_norm(tcp_pos - obstacle_pos.view(1, 1, 3), dim=-1)
                clear_terms.append(torch.clamp(keepout - dist, min=0.0).pow(2))
            if clear_terms:
                clear = torch.stack(clear_terms, dim=0).sum(dim=0)
                terms["clear"] = weights.clear * torch.mean(clear, dim=1)

        if weights.transit != 0.0:
            terms["transit"] = weights.transit * _transit_clearance_cost(tcp_pos, target_pos, context)

        if real_actions.shape[-1] > 7:
            gripper = real_actions[..., 7]
            if gripper.shape[1] > 1:
                gripper_smooth = torch.mean((gripper[:, 1:] - gripper[:, :-1]).pow(2), dim=1)
                terms["gripper_smooth"] = weights.gripper_smooth * gripper_smooth

        if weights.path != 0.0:
            terms["path"] = weights.path * _path_constraint_cost(context, batch, device, dtype)

        return terms

    def _grasp_terms(
        self,
        *,
        object_name: str,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        grasp_center = _target_from_object(context, object_name, device, dtype)
        if grasp_center is None:
            target_pos = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
            if target_pos is None:
                target_pos = torch.zeros(3, device=device, dtype=dtype)
            return {}, target_pos[:3]

        target_pos = grasp_center
        approach_axis_for_grasp = None
        tip_pos_for_grasp = None
        if tcp_quat is not None:
            approach_axis_for_grasp = _apply_axis(
                tcp_quat,
                _get_vector(
                    context,
                    "a_local",
                    default=(0.0, 0.0, 1.0),
                    device=device,
                    dtype=dtype,
                ),
            )
            tip_offset = float(context.get("tcp_to_tip_z", _GRASP_FLOW_TCP_TO_TIP_Z))
            tip_pos_for_grasp = tcp_pos + tip_offset * approach_axis_for_grasp

        reach_pos = tip_pos_for_grasp if tip_pos_for_grasp is not None else tcp_pos
        reach_target = grasp_center if tip_pos_for_grasp is not None else target_pos
        dist_sq = torch.sum((reach_pos - reach_target.view(1, 1, 3)) ** 2, dim=-1)
        terms: dict[str, torch.Tensor] = {
            "reach": weights.reach * torch.mean(dist_sq, dim=1),
            "terminal": weights.terminal * dist_sq[:, -1],
        }

        if tcp_quat is not None:
            approach_axis = approach_axis_for_grasp
            tip_pos = tip_pos_for_grasp
            center = grasp_center.view(1, 1, 3)
            tip_z = (tip_pos[..., 2] - grasp_center[2]).pow(2)
            terms["tip_z"] = weights.tip_z * torch.mean(tip_z, dim=1)

            closing_axis = _apply_axis(
                tcp_quat,
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            )
            lateral_axis = torch.cross(approach_axis, closing_axis, dim=-1)
            lateral_axis = lateral_axis / torch.clamp(
                torch.linalg.vector_norm(lateral_axis, dim=-1, keepdim=True),
                min=1e-8,
            )
            yaw = 1.0 - torch.maximum(closing_axis[..., 0].abs(), closing_axis[..., 1].abs())
            terms["yaw"] = weights.yaw * torch.mean(yaw, dim=1)

            object_radius = _WEIGHT_OBJECT_HORIZONTAL_RADIUS.get(object_name, 0.05)
            finger_r = float(context.get("finger_radius", 0.012))
            open_half = float(context.get("open_half", 0.04))
            keepout = object_radius + finger_r
            rel_to_tip = center - tip_pos
            closing_coord = torch.sum(rel_to_tip * closing_axis, dim=-1)
            lateral_coord = torch.sum(rel_to_tip * lateral_axis, dim=-1)
            approach_coord = torch.sum(rel_to_tip * approach_axis, dim=-1)
            center_err = torch.sqrt(
                closing_coord.pow(2)
                + lateral_coord.pow(2)
                + approach_coord.pow(2)
                + 1e-12
            )
            center_radius = float(
                context.get(
                    "gripper_center_region_radius",
                    _GRASP_FLOW_CENTER_REGION_RADIUS_SCALE * object_radius,
                )
            )
            center_region = torch.clamp(center_err - center_radius, min=0.0).pow(2)
            terms["center_region"] = weights.center_region * torch.mean(center_region, dim=1)

            aperture_margin = float(context.get("gripper_aperture_margin", 0.006))
            aperture_violation = torch.clamp(
                closing_coord.abs() + object_radius + aperture_margin - open_half,
                min=0.0,
            ).pow(2)
            terms["aperture_region"] = weights.aperture_region * torch.mean(aperture_violation, dim=1)

            finger_l = tcp_pos + open_half * closing_axis
            finger_r_pos = tcp_pos - open_half * closing_axis
            dist_l = torch.linalg.vector_norm(finger_l[..., :2] - center[..., :2], dim=-1)
            dist_r = torch.linalg.vector_norm(finger_r_pos[..., :2] - center[..., :2], dim=-1)
            penetration = (
                torch.clamp(keepout - dist_l, min=0.0).pow(2)
                + torch.clamp(keepout - dist_r, min=0.0).pow(2)
            )
            terms["straddle"] = weights.straddle * torch.mean(penetration, dim=1)

            if real_actions.shape[-1] > 7:
                z_scale = float(context.get("gripper_close_z_scale", 0.025))
                xy_scale = float(context.get("gripper_close_xy_scale", max(center_radius, 1e-3)))
                center_excess = torch.clamp(center_err - center_radius, min=0.0)
                z_excess = torch.clamp(torch.sqrt(tip_z + 1e-12) - z_scale, min=0.0)
                close_gate = torch.exp(
                    -center_excess.pow(2) / max(xy_scale * xy_scale, 1e-8)
                    -z_excess.pow(2) / max(z_scale * z_scale, 1e-8)
                )
                gripper = real_actions[..., 7]
                terms["close_gripper"] = weights.close_gripper * torch.mean(
                    (gripper - close_gate).pow(2),
                    dim=1,
                )

                if weights.soft_grasp != 0.0:
                    grasp_threshold = float(context.get("soft_grasp_distance", 0.08))
                    grasp_dist = torch.linalg.vector_norm(tcp_pos - center, dim=-1)
                    soft_grasp = torch.clamp(grasp_dist - grasp_threshold, min=0.0).pow(2)
                    soft_grasp = soft_grasp + (gripper - close_gate).pow(2)
                    terms["soft_grasp"] = weights.soft_grasp * torch.mean(soft_grasp, dim=1)

        return terms, target_pos

    def _place_target(self, object_name: str, context: dict[str, Any], device, dtype) -> torch.Tensor | None:
        scale_pos = _target_from_object(context, "scale", device, dtype)
        if scale_pos is None:
            return None
        half_height = _WEIGHT_OBJECT_HALF_HEIGHT.get(object_name, 0.05)
        x_offset = float(context.get("scale_place_x_offset", 0.0))
        y_offset = float(context.get("scale_place_y_offset", -0.05))
        clearance = float(context.get("place_clearance_z", _GRASP_FLOW_PLACE_RELEASE_CLEARANCE_Z))
        scale_top_center = scale_pos + _WEIGHT_SCALE_CENTER_OFFSET.to(device=device, dtype=dtype)
        scale_top_center = scale_top_center + torch.tensor(
            [0.0, 0.0, _WEIGHT_SCALE_TOP_OFFSET_Z],
            device=device,
            dtype=dtype,
        )
        return scale_top_center + torch.tensor(
            [x_offset, y_offset, half_height + clearance],
            device=device,
            dtype=dtype,
        )

    def _carried_object_pos(
        self,
        *,
        object_name: str,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
        device,
        dtype,
    ) -> torch.Tensor:
        object_pos = _target_from_object(context, object_name, device, dtype)
        current_tcp = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
        if object_pos is None or current_tcp is None:
            return tcp_pos
        return tcp_pos + (object_pos[:3] - current_tcp[:3]).view(1, 1, 3)

    def _lift_start_z_value(self, object_name: str, context: dict[str, Any], object_pos: torch.Tensor) -> float:
        key = f"{object_name}_lift_start_z"
        if key in context:
            value = context[key]
            if torch.is_tensor(value):
                value = value.detach().reshape(-1)[0].item()
            return float(value)
        if object_name not in self._lift_start_z:
            self._lift_start_z[object_name] = float(object_pos[2].detach().cpu())
        return self._lift_start_z[object_name]

    def _update_lift_start(self, object_name: str, context: dict[str, Any], device, dtype) -> None:
        """Track the object height before grasp, then freeze it for the episode."""
        object_pos = _target_from_object(context, object_name, device, dtype)
        if object_pos is None:
            return
        if _flag(context, f"grasp_{object_name}"):
            if object_name not in self._lift_start_z:
                self._lift_start_z[object_name] = float(object_pos[2].detach().cpu())
            self._lift_start_frozen.add(object_name)
        elif object_name not in self._lift_start_frozen:
            self._lift_start_z[object_name] = float(object_pos[2].detach().cpu())

    def _lift_target_z_value(self, object_name: str, context: dict[str, Any], object_pos: torch.Tensor) -> float:
        lift_height = float(context.get("grasp_flow_lift_height", _GRASP_FLOW_LIFT_HEIGHT))
        return self._lift_start_z_value(object_name, context, object_pos) + lift_height

    def _needs_lift(self, object_name: str, context: dict[str, Any], device, dtype) -> bool:
        object_pos = _target_from_object(context, object_name, device, dtype)
        if object_pos is None:
            return False
        if self._lift_done.get(object_name, False):
            return False
        target_z = self._lift_target_z_value(object_name, context, object_pos)
        done_tolerance = float(context.get("grasp_flow_lift_done_tolerance", _GRASP_FLOW_LIFT_DONE_TOLERANCE))
        if float(object_pos[2].detach().cpu()) >= target_z - done_tolerance:
            self._lift_done[object_name] = True
            return False
        return True

    def _reset_lift_progress(self, active_object: str | None) -> None:
        if active_object is None:
            self._lift_done.clear()
            return
        for object_name in list(self._lift_done):
            if object_name != active_object:
                self._lift_done.pop(object_name, None)

    def _lift_terms(
        self,
        *,
        object_name: str,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        object_pos = _target_from_object(context, object_name, device, dtype)
        if object_pos is None:
            target_pos = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
            if target_pos is None:
                target_pos = torch.zeros(3, device=device, dtype=dtype)
            return {}, target_pos[:3]

        target_z = self._lift_target_z_value(object_name, context, object_pos)
        target_pos = torch.stack(
            [
                object_pos[0],
                object_pos[1],
                torch.as_tensor(target_z, device=device, dtype=dtype),
            ]
        )
        carried_pos = self._carried_object_pos(
            object_name=object_name,
            tcp_pos=tcp_pos,
            context=context,
            device=device,
            dtype=dtype,
        )
        delta = carried_pos - target_pos.view(1, 1, 3)
        xy_dist = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        z_err = delta[..., 2]
        dist_sq = torch.sum(delta.pow(2), dim=-1)
        terms: dict[str, torch.Tensor] = {
            "lift_reach": weights.lift_reach * torch.mean(dist_sq, dim=1),
            "lift_terminal": weights.lift_terminal * dist_sq[:, -1],
            "lift_xy": weights.lift_xy * torch.mean(xy_dist.pow(2), dim=1),
            "lift_z": weights.lift_z * torch.mean(z_err.pow(2), dim=1),
        }
        if real_actions.shape[-1] > 7:
            gripper = real_actions[..., 7]
            terms["lift_gripper"] = weights.lift_gripper * torch.mean((gripper - 1.0).pow(2), dim=1)
        return terms, target_pos

    def _place_terms(
        self,
        *,
        object_name: str,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        target_pos = self._place_target(object_name, context, device, dtype)
        if target_pos is None:
            target_pos = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
            if target_pos is None:
                target_pos = torch.zeros(3, device=device, dtype=dtype)
            return {}, target_pos[:3]

        carried_pos = self._carried_object_pos(
            object_name=object_name,
            tcp_pos=tcp_pos,
            context=context,
            device=device,
            dtype=dtype,
        )
        xy_dist = torch.linalg.vector_norm(carried_pos[..., :2] - target_pos[:2].view(1, 1, 2), dim=-1)
        descend_radius = float(context.get("place_descend_radius", 0.08))
        outside_descend = (xy_dist > descend_radius).to(dtype=dtype)
        object_pos = _target_from_object(context, object_name, device, dtype)
        if object_pos is None:
            carry_z = target_pos[2] + float(context.get("place_carry_clearance_z", _GRASP_FLOW_PLACE_CARRY_CLEARANCE_Z))
        else:
            carry_z = torch.as_tensor(
                self._lift_target_z_value(object_name, context, object_pos),
                device=device,
                dtype=dtype,
            )
        z_target = target_pos[2] + outside_descend * (carry_z - target_pos[2])
        z_err = carried_pos[..., 2] - z_target
        dist_sq = torch.sum((carried_pos[..., :2] - target_pos[:2].view(1, 1, 2)).pow(2), dim=-1) + z_err.pow(2)
        terms: dict[str, torch.Tensor] = {
            "place_reach": weights.place_reach * torch.mean(dist_sq, dim=1),
            "place_terminal": weights.place_terminal * dist_sq[:, -1],
            "place_xy": weights.place_xy * torch.mean(xy_dist.pow(2), dim=1),
            "place_z": weights.place_z * torch.mean(z_err.pow(2), dim=1),
        }

        carry_height = torch.clamp(carry_z - carried_pos[..., 2], min=0.0).pow(2)
        terms["place_carry_height"] = weights.place_carry_height * torch.mean(
            outside_descend * carry_height,
            dim=1,
        )

        if real_actions.shape[-1] > 7:
            release_xy_radius = float(context.get("place_release_xy_radius", 0.06))
            release_z_radius = float(context.get("place_release_z_radius", 0.035))
            release_gate = torch.logical_and(
                xy_dist < release_xy_radius,
                z_err.abs() < release_z_radius,
            ).to(dtype=dtype)
            desired_gripper = 1.0 - release_gate
            gripper = real_actions[..., 7]
            terms["place_gripper"] = weights.place_gripper * torch.mean(
                (gripper - desired_gripper).pow(2),
                dim=1,
            )

        return terms, target_pos

    def _place_object_name(self, context: dict[str, Any]) -> str | None:
        if "weight" not in self.task_name:
            return None
        if _flag(context, "grasp_apple"):
            return "apple"
        if _flag(context, "grasp_pear"):
            return "pear"
        return None

    def _grasp_object_name(self, context: dict[str, Any]) -> str | None:
        if "weight" not in self.task_name:
            return None
        if _flag(context, "grasp_pear"):
            return None
        if _flag(context, "pear_on_scale"):
            return None if _flag(context, "grasp_apple") else "apple"
        return "pear"

    def compute_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        device = real_actions.device
        dtype = real_actions.dtype
        if "weight" in self.task_name:
            self._update_lift_start("pear", context, device, dtype)
            self._update_lift_start("apple", context, device, dtype)
        pick_object = self._grasp_object_name(context)
        place_object = self._place_object_name(context)
        if place_object is not None:
            self._reset_lift_progress(place_object)
            if self._needs_lift(place_object, context, device, dtype):
                self.last_stage = f"lift_{place_object}"
                terms, target_pos = self._lift_terms(
                    object_name=place_object,
                    real_actions=real_actions,
                    tcp_pos=tcp_pos,
                    context=context,
                )
            else:
                self.last_stage = f"place_{place_object}"
                terms, target_pos = self._place_terms(
                    object_name=place_object,
                    real_actions=real_actions,
                    tcp_pos=tcp_pos,
                    context=context,
                )
        elif pick_object is not None:
            self._reset_lift_progress(None)
            self.last_stage = f"grasp_{pick_object}"
            terms, target_pos = self._grasp_terms(
                object_name=pick_object,
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
        else:
            target_pos = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
            if target_pos is None:
                target_pos = torch.zeros(3, device=device, dtype=dtype)
            else:
                target_pos = target_pos[:3]
            terms = {}
            self._reset_lift_progress(None)
            self.last_stage = "idle"

        terms.update(
            self._move_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
                target_pos=target_pos,
            )
        )
        return terms

    def __call__(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> torch.Tensor:
        return self._store(
            self.compute_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
        )
