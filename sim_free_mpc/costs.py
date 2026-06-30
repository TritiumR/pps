from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .fk import quat_apply_wxyz


_WEIGHT_SCALE_CENTER_OFFSET = torch.tensor([-0.0470425, 0.0, 0.0272255])
_WEIGHT_SCALE_TOP_OFFSET_Z = 0.0523800
_WEIGHT_OBJECT_HALF_HEIGHT = {
    "apple": 0.037665,
    "pear": 0.0620635,
}
_WEIGHT_PLACE_CLEARANCE_Z = 0.015
_WEIGHT_APPROACH_CLEARANCE_Z = 0.03


@dataclass(frozen=True)
class CostWeights:
    reach: float = 25.0
    terminal_reach: float = 40.0
    smooth: float = 0.03
    joint_delta: float = 0.005
    gripper: float = 0.1
    orientation: float = 0.25


def _first_tensor(context: dict[str, Any], *keys: str, device, dtype) -> torch.Tensor | None:
    for key in keys:
        if key in context and context[key] is not None:
            value = context[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(device=device, dtype=dtype)
            if value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            return value
    return None


def _flag(context: dict[str, Any], key: str) -> bool:
    value = context.get("subtasks", {}).get(key, context.get(key, False))
    if torch.is_tensor(value):
        return bool(value.detach().flatten()[0].item())
    return bool(value)


def _target_from_object(context: dict[str, Any], name: str, device, dtype) -> torch.Tensor | None:
    objects = context.get("objects", {})
    obj = objects.get(name)
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


def _reach_cost(ee_pos: torch.Tensor, target: torch.Tensor, weights: CostWeights) -> torch.Tensor:
    dist_sq = torch.sum((ee_pos - target.view(1, 1, 3)) ** 2, dim=-1)
    return weights.reach * dist_sq.mean(dim=1) + weights.terminal_reach * dist_sq[:, -1]


def _regularization(real_actions: torch.Tensor, context: dict[str, Any], weights: CostWeights) -> torch.Tensor:
    joints = real_actions[..., :7]
    smooth = torch.sum((joints[:, 1:] - joints[:, :-1]) ** 2, dim=(-1, -2))
    current = _first_tensor(context, "joint_pos", device=real_actions.device, dtype=real_actions.dtype)
    if current is not None:
        delta = torch.sum((joints - current[:7].view(1, 1, 7)) ** 2, dim=(-1, -2))
    else:
        delta = torch.zeros(real_actions.shape[0], device=real_actions.device, dtype=real_actions.dtype)
    return weights.smooth * smooth + weights.joint_delta * delta


def _weight_is_place_phase(context: dict[str, Any]) -> bool:
    return (
        _flag(context, "grasp_pear")
        and not _flag(context, "pear_on_scale")
    ) or _flag(context, "grasp_apple")


def _pot_is_place_phase(context: dict[str, Any]) -> bool:
    return (
        _flag(context, "grasp_cover")
        and not _flag(context, "lid_removed")
    ) or _flag(context, "grasp_egg")


def _gripper_cost(
    real_actions: torch.Tensor,
    ee_pos: torch.Tensor,
    target: torch.Tensor,
    context: dict[str, Any],
    task_name: str,
    weights: CostWeights,
) -> torch.Tensor:
    if real_actions.shape[-1] <= 7:
        return torch.zeros(real_actions.shape[0], device=real_actions.device, dtype=real_actions.dtype)

    task = task_name.lower()
    if "weight" in task:
        is_place_phase = _weight_is_place_phase(context)
    elif "pot" in task:
        is_place_phase = _pot_is_place_phase(context)
    else:
        is_place_phase = any(
            _flag(context, key)
            for key in ("grasp_teapot", "grasp_pod", "grasp_mug", "grasp_pen")
        )

    dist = torch.linalg.vector_norm(ee_pos - target.view(1, 1, 3), dim=-1)
    close_radius = 0.07
    near_target = (dist < close_radius).to(dtype=real_actions.dtype)
    if is_place_phase:
        # Carry while far from the placement target, then open near the target.
        desired = 1.0 - near_target
    else:
        # Approach with an open gripper, then close only when the EE is near the object.
        desired = near_target

    gripper = real_actions[..., 7]
    return weights.gripper * torch.mean((gripper - desired) ** 2, dim=1)


def _downward_orientation_cost(
    ee_quat: torch.Tensor | None,
    weights: CostWeights,
) -> torch.Tensor | None:
    if ee_quat is None:
        return None
    local_tool_axis = torch.zeros((*ee_quat.shape[:-1], 3), device=ee_quat.device, dtype=ee_quat.dtype)
    local_tool_axis[..., 2] = 1.0
    tool_axis_world = quat_apply_wxyz(ee_quat, local_tool_axis)
    desired_axis_world = torch.tensor([0.0, 0.0, -1.0], device=ee_quat.device, dtype=ee_quat.dtype)
    alignment = torch.sum(tool_axis_world * desired_axis_world.view(1, 1, 3), dim=-1)
    return weights.orientation * torch.mean(1.0 - alignment, dim=1)


def _default_target(context: dict[str, Any], device, dtype) -> torch.Tensor:
    ee = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
    if ee is not None:
        return ee[:3]
    return torch.zeros(3, device=device, dtype=dtype)


def _approach_above(target: torch.Tensor, z_offset: float, device, dtype) -> torch.Tensor:
    return target + torch.tensor([0.0, 0.0, z_offset], device=device, dtype=dtype)


def _scale_place_target(
    scale: torch.Tensor,
    object_name: str,
    device,
    dtype,
) -> torch.Tensor:
    half_height = _WEIGHT_OBJECT_HALF_HEIGHT.get(object_name, 0.05)
    scale_center = scale + _WEIGHT_SCALE_CENTER_OFFSET.to(device=device, dtype=dtype)
    return scale_center + torch.tensor(
        [0.0, 0.0, _WEIGHT_SCALE_TOP_OFFSET_Z + half_height + _WEIGHT_PLACE_CLEARANCE_Z],
        device=device,
        dtype=dtype,
    )


class PriorityStateCost:
    """Hand-designed task cost used until VLM-generated costs are connected."""

    def __init__(self, task_name: str, weights: CostWeights | None = None):
        self.task_name = task_name.lower()
        self.weights = weights or CostWeights()

    def target(self, context: dict[str, Any], device, dtype) -> torch.Tensor:
        task = self.task_name
        if "pot" in task:
            if not _flag(context, "grasp_cover"):
                target = _target_from_object(context, "cover", device, dtype)
                return target if target is not None else _default_target(context, device, dtype)
            if not _flag(context, "lid_removed"):
                cover = _target_from_object(context, "cover", device, dtype)
                if cover is not None:
                    return cover + torch.tensor([0.12, 0.0, 0.08], device=device, dtype=dtype)
            if not _flag(context, "grasp_egg"):
                target = _target_from_object(context, "egg", device, dtype)
                return target if target is not None else _default_target(context, device, dtype)
            pot = _target_from_object(context, "pot", device, dtype)
            if pot is not None:
                return pot + torch.tensor([0.0, 0.0, 0.08], device=device, dtype=dtype)

        if "weight" in task:
            if not _flag(context, "grasp_pear"):
                target = _target_from_object(context, "pear", device, dtype)
                return (
                    _approach_above(target, _WEIGHT_APPROACH_CLEARANCE_Z, device, dtype)
                    if target is not None
                    else _default_target(context, device, dtype)
                )
            if not _flag(context, "pear_on_scale"):
                scale = _target_from_object(context, "scale", device, dtype)
                if scale is not None:
                    return _scale_place_target(scale, "pear", device, dtype)
            if not _flag(context, "grasp_apple"):
                target = _target_from_object(context, "apple", device, dtype)
                return (
                    _approach_above(target, _WEIGHT_APPROACH_CLEARANCE_Z, device, dtype)
                    if target is not None
                    else _default_target(context, device, dtype)
                )
            scale = _target_from_object(context, "scale", device, dtype)
            if scale is not None:
                return _scale_place_target(scale, "apple", device, dtype)

        if "tea" in task:
            if not _flag(context, "grasp_teapot"):
                target = _target_from_object(context, "teapot", device, dtype)
                return target if target is not None else _default_target(context, device, dtype)
            cup = _target_from_object(context, "teacup", device, dtype)
            if cup is not None:
                return cup + torch.tensor([0.0, 0.0, 0.16], device=device, dtype=dtype)

        if "capsule" in task:
            if not _flag(context, "open_coffee_lid"):
                capsule = _target_from_object(context, "capsule", device, dtype)
                if capsule is not None:
                    return capsule + torch.tensor([0.0, 0.0, 0.35], device=device, dtype=dtype)
            if not _flag(context, "grasp_pod"):
                target = _target_from_object(context, "can", device, dtype)
                return target if target is not None else _default_target(context, device, dtype)
            capsule = _target_from_object(context, "capsule", device, dtype)
            if capsule is not None:
                return capsule + torch.tensor([0.0, 0.0, 0.27], device=device, dtype=dtype)

        return _default_target(context, device, dtype)

    def __call__(
        self,
        *,
        real_actions: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor | None = None,
        context: dict[str, Any],
    ) -> torch.Tensor:
        target = self.target(context, real_actions.device, real_actions.dtype)
        cost = _reach_cost(ee_pos, target, self.weights)
        cost = cost + _regularization(real_actions, context, self.weights)
        cost = cost + _gripper_cost(real_actions, ee_pos, target, context, self.task_name, self.weights)
        orientation_cost = _downward_orientation_cost(ee_quat, self.weights)
        if orientation_cost is not None:
            cost = cost + orientation_cost
        return cost
