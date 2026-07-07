from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .fk import quat_apply_wxyz


_WEIGHT_SCALE_CENTER_OFFSET = torch.tensor([-0.0470425, 0.0, 0.0272255])
_WEIGHT_SCALE_TOP_OFFSET_Z = 0.0523800
_WEIGHT_PLACE_CLEARANCE_Z = 0.015
_WEIGHT_APPROACH_CLEARANCE_Z = 0.03
_WEIGHT_OBJECT_HALF_HEIGHT = {
    "apple": 0.037665,
    "pear": 0.0620635,
    "mango": 0.06,
    "cabbage": 0.055,
}
_WEIGHT_OBJECT_HORIZONTAL_RADIUS = {
    "apple": 0.0413,
    "pear": 0.0517,
    "mango": 0.06,
    "cabbage": 0.07,
}
_WEIGHT_COLLISION_OBJECTS = ("pear", "apple", "mango", "cabbage")
_WEIGHT_TCP_TO_TIP_Z = 0.03


@dataclass(frozen=True)
class ExploreCostWeights:
    # Common defaults follow costs_for_ref.py for the overlapping terms.
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
    close_gripper: float = 2.0
    clear: float = 0.1
    transit: float = 0.0
    gripper: float = 0.1
    path: float = 0.0


def _flag(context: dict[str, Any], key: str) -> bool:
    value = context.get("subtasks", {}).get(key, context.get(key, False))
    if torch.is_tensor(value):
        return bool(value.detach().flatten()[0].item())
    return bool(value)


def _first_tensor(context: dict[str, Any], *keys: str, device, dtype) -> torch.Tensor | None:
    for key in keys:
        value = context.get(key)
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=dtype)
        if value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        return value
    return None


def _target_from_object(context: dict[str, Any], name: str, device, dtype) -> torch.Tensor | None:
    obj = context.get("objects", {}).get(name)
    if obj is None:
        return None
    pos = obj.get("pos") if isinstance(obj, dict) else None
    if pos is None:
        return None
    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos)
    if pos.ndim > 1 and pos.shape[0] == 1:
        pos = pos[0]
    return pos.to(device=device, dtype=dtype)


def _approach_above(target: torch.Tensor, device, dtype) -> torch.Tensor:
    return target + torch.tensor([0.0, 0.0, _WEIGHT_APPROACH_CLEARANCE_Z], device=device, dtype=dtype)


def _scale_place_target(scale: torch.Tensor, object_name: str, device, dtype) -> torch.Tensor:
    half_height = _WEIGHT_OBJECT_HALF_HEIGHT.get(object_name, 0.05)
    scale_center = scale + _WEIGHT_SCALE_CENTER_OFFSET.to(device=device, dtype=dtype)
    return scale_center + torch.tensor(
        [0.0, 0.0, _WEIGHT_SCALE_TOP_OFFSET_Z + half_height + _WEIGHT_PLACE_CLEARANCE_Z],
        device=device,
        dtype=dtype,
    )


def _weight_pick_object_name(context: dict[str, Any]) -> str | None:
    if not _flag(context, "grasp_pear"):
        return "pear"
    if _flag(context, "pear_on_scale") and not _flag(context, "grasp_apple"):
        return "apple"
    return None


def _weight_manipulated_object_name(context: dict[str, Any]) -> str | None:
    if not _flag(context, "pear_on_scale"):
        return "pear"
    return "apple"


def _weight_is_place_phase(context: dict[str, Any]) -> bool:
    return (
        _flag(context, "grasp_pear")
        and not _flag(context, "pear_on_scale")
    ) or _flag(context, "grasp_apple")


def _default_target(context: dict[str, Any], device, dtype) -> torch.Tensor:
    eef = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
    if eef is not None:
        return eef[:3]
    return torch.zeros(3, device=device, dtype=dtype)


def _get_vector(
    context: dict[str, Any],
    key: str,
    *,
    default: tuple[float, float, float],
    device,
    dtype,
) -> torch.Tensor:
    value = context.get(key, default)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype).flatten()[:3]


def _apply_axis(tcp_quat: torch.Tensor, axis_local: torch.Tensor) -> torch.Tensor:
    local = axis_local.view(*((1,) * (tcp_quat.ndim - 1)), 3).expand(*tcp_quat.shape[:-1], 3)
    return quat_apply_wxyz(tcp_quat, local)


def _zero(batch: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch, device=device, dtype=dtype)


def _target_and_grasp_center(
    task_name: str,
    context: dict[str, Any],
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if "weight" not in task_name:
        return _default_target(context, device, dtype), None

    pick_object = _weight_pick_object_name(context)
    if pick_object is not None:
        obj = _target_from_object(context, pick_object, device, dtype)
        if obj is None:
            return _default_target(context, device, dtype), None
        return _approach_above(obj, device, dtype), obj

    scale = _target_from_object(context, "scale", device, dtype)
    if scale is None:
        return _default_target(context, device, dtype), None
    object_name = "pear" if not _flag(context, "pear_on_scale") else "apple"
    return _scale_place_target(scale, object_name, device, dtype), None


def _obstacles_for_weight(context: dict[str, Any], device, dtype) -> list[tuple[torch.Tensor, float]]:
    manipulated = _weight_manipulated_object_name(context)
    obstacles = []
    for object_name in _WEIGHT_COLLISION_OBJECTS:
        if object_name == manipulated:
            continue
        pos = _target_from_object(context, object_name, device, dtype)
        if pos is None:
            continue
        radius = _WEIGHT_OBJECT_HORIZONTAL_RADIUS.get(object_name, 0.06)
        obstacles.append((pos, radius))
    return obstacles


def _transit_clearance_cost(
    tcp_pos: torch.Tensor,
    target_pos: torch.Tensor,
    context: dict[str, Any],
) -> torch.Tensor:
    cfg = context.get("transit_clearance")
    if not cfg:
        return _zero(tcp_pos.shape[0], tcp_pos.device, tcp_pos.dtype)
    z_clear = float(cfg.get("z_clear", target_pos[2].detach().item() + 0.10))
    descend_r = float(cfg.get("descend_r", 0.08))
    xy_dist = torch.linalg.vector_norm(tcp_pos[..., :2] - target_pos[:2].view(1, 1, 2), dim=-1)
    outside_descend = (xy_dist > descend_r).to(dtype=tcp_pos.dtype)
    low = torch.clamp(z_clear - tcp_pos[..., 2], min=0.0).pow(2)
    return torch.mean(outside_descend * low, dim=1)


def _path_constraint_cost(context: dict[str, Any], batch: int, device, dtype) -> torch.Tensor:
    value = context.get("path_constraint_violation")
    if value is None:
        return _zero(batch, device, dtype)
    if callable(value):
        raise ValueError("Callable path_constraint_violation must be evaluated before cost construction.")
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        return torch.clamp(value, min=0.0).expand(batch)
    if value.ndim == 1:
        if value.shape[0] == batch:
            return torch.clamp(value, min=0.0)
        return torch.clamp(value, min=0.0).sum().expand(batch)
    return torch.clamp(value, min=0.0).reshape(batch, -1).sum(dim=1)


class ExploreStateCost:
    """Exploratory region-based grasp cost for PPS weight-task MPC."""

    def __init__(self, task_name: str, weights: ExploreCostWeights | None = None):
        self.task_name = task_name.lower()
        self.weights = weights or ExploreCostWeights()

    def __call__(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> torch.Tensor:
        weights = self.weights
        batch = real_actions.shape[0]
        device = real_actions.device
        dtype = real_actions.dtype
        target_pos, grasp_center = _target_and_grasp_center(self.task_name, context, device, dtype)

        approach_axis_for_grasp = None
        tip_pos_for_grasp = None
        if grasp_center is not None and tcp_quat is not None:
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
            tip_offset = float(context.get("tcp_to_tip_z", _WEIGHT_TCP_TO_TIP_Z))
            tip_pos_for_grasp = tcp_pos + tip_offset * approach_axis_for_grasp

        reach_pos = tip_pos_for_grasp if tip_pos_for_grasp is not None else tcp_pos
        reach_target = grasp_center if tip_pos_for_grasp is not None else target_pos
        dist_sq = torch.sum((reach_pos - reach_target.view(1, 1, 3)) ** 2, dim=-1)
        cost = weights.reach * torch.mean(dist_sq, dim=1)
        cost = cost + weights.terminal * dist_sq[:, -1]

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
            cost = cost + weights.orient * torch.mean(orient, dim=1)

        z_floor = float(context.get("z_floor", 0.0))
        floor = torch.clamp(z_floor - tcp_pos[..., 2], min=0.0).pow(2)
        cost = cost + weights.floor * torch.mean(floor, dim=1)

        q_traj = real_actions[..., :7]
        if q_traj.shape[1] > 1:
            smooth = torch.sum((q_traj[:, 1:] - q_traj[:, :-1]) ** 2, dim=(-1, -2))
        else:
            smooth = _zero(batch, device, dtype)
        cost = cost + weights.smooth * smooth

        q_cur = _first_tensor(context, "joint_pos", device=device, dtype=dtype)
        if q_cur is not None:
            local = torch.sum((q_traj - q_cur[:7].view(1, 1, 7)) ** 2, dim=(-1, -2))
            cost = cost + weights.local * local

        if grasp_center is not None and tcp_quat is not None:
            approach_axis = approach_axis_for_grasp
            tip_pos = tip_pos_for_grasp
            tip_z = (tip_pos[..., 2] - grasp_center[2]).pow(2)
            cost = cost + weights.tip_z * torch.mean(tip_z, dim=1)

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
            cost = cost + weights.yaw * torch.mean(yaw, dim=1)

            object_name = _weight_pick_object_name(context) or "object"
            object_radius = _WEIGHT_OBJECT_HORIZONTAL_RADIUS.get(object_name, 0.05)
            finger_r = float(context.get("finger_radius", 0.012))
            open_half = float(context.get("open_half", 0.04))
            keepout = object_radius + finger_r
            center = grasp_center.view(1, 1, 3)
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
            center_radius = float(context.get("gripper_center_region_radius", 0.35 * object_radius))
            center_region = torch.clamp(center_err - center_radius, min=0.0).pow(2)
            cost = cost + weights.center_region * torch.mean(center_region, dim=1)

            aperture_margin = float(context.get("gripper_aperture_margin", 0.006))
            aperture_violation = torch.clamp(
                closing_coord.abs() + object_radius + aperture_margin - open_half,
                min=0.0,
            ).pow(2)
            cost = cost + weights.aperture_region * torch.mean(aperture_violation, dim=1)

            finger_l = tcp_pos + open_half * closing_axis
            finger_r_pos = tcp_pos - open_half * closing_axis
            dist_l = torch.linalg.vector_norm(finger_l[..., :2] - center[..., :2], dim=-1)
            dist_r = torch.linalg.vector_norm(finger_r_pos[..., :2] - center[..., :2], dim=-1)
            penetration = (
                torch.clamp(keepout - dist_l, min=0.0).pow(2)
                + torch.clamp(keepout - dist_r, min=0.0).pow(2)
            )
            cost = cost + weights.straddle * torch.mean(penetration, dim=1)

            if (
                weights.close_gripper != 0.0
                and real_actions.shape[-1] > 7
                and bool(context.get("optimize_gripper", True))
            ):
                z_scale = float(context.get("gripper_close_z_scale", 0.025))
                xy_scale = float(context.get("gripper_close_xy_scale", max(center_radius, 1e-3)))
                close_gate = torch.exp(
                    -center_err.pow(2) / max(xy_scale * xy_scale, 1e-8)
                    - tip_z / max(z_scale * z_scale, 1e-8)
                )
                gripper = real_actions[..., 7]
                cost = cost + weights.close_gripper * torch.mean(close_gate * (gripper - 1.0).pow(2), dim=1)

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
                cost = cost + weights.clear * torch.mean(clear, dim=1)

        if weights.transit != 0.0:
            cost = cost + weights.transit * _transit_clearance_cost(tcp_pos, target_pos, context)

        if weights.gripper != 0.0 and real_actions.shape[-1] > 7:
            dist = torch.linalg.vector_norm(tcp_pos - target_pos.view(1, 1, 3), dim=-1)
            near_target = (dist < float(context.get("gripper_close_radius", 0.07))).to(dtype=dtype)
            desired = 1.0 - near_target if _weight_is_place_phase(context) else near_target
            gripper = real_actions[..., 7]
            cost = cost + weights.gripper * torch.mean((gripper - desired) ** 2, dim=1)

        if weights.path != 0.0:
            cost = cost + weights.path * _path_constraint_cost(context, batch, device, dtype)

        return cost
