from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .costs_explore import (
    _apply_axis,
    _flag,
    _first_tensor,
    _get_vector,
    _path_constraint_cost,
    _transit_clearance_cost,
    _zero,
)
from .fk import quat_apply_wxyz, quat_mul_wxyz


_CAPSULE_LID_GRIP_LOCAL = (0.0, -0.25, 0.0)
_CAPSULE_LID_HINGE_LOCAL = (-0.0073589, -0.0051588, 0.3963653)
_CAPSULE_LID_OPEN_THRESHOLD = -0.5
_CAPSULE_LID_OPEN_TARGET = -0.55
_CAPSULE_LID_ARC_STEP = 0.15
_CAPSULE_LID_GRASP_DISTANCE = 0.06
_CAPSULE_GRIPPER_CLOSED_THRESHOLD = 0.2
_GRIPPER_OPEN_ACTION = 0.0
_GRIPPER_CLOSED_ACTION = 1.0

# The can mesh is scaled by 0.55 in CapsuleSceneCfg.  These values describe
# the resulting root-to-center offset and horizontal radius.
_CAPSULE_POD_CENTER_LOCAL = (-0.0433, -0.0008, 0.0221)
_CAPSULE_POD_HORIZONTAL_RADIUS = 0.0202
_CAPSULE_POD_PLACE_LOCAL = (0.0, 0.0, 0.27)
_CAPSULE_PLACE_CARRY_CLEARANCE_Z = 0.10
_CAPSULE_PLACE_DESCEND_RADIUS = 0.06
_CAPSULE_PLACE_RELEASE_XY_RADIUS = 0.05
_CAPSULE_PLACE_RELEASE_Z_RADIUS = 0.03


@dataclass(frozen=True)
class CapsuleFlowCostWeights:
    reach: float = 8.0
    terminal: float = 40.0
    orient: float = 8.0
    floor: float = 50.0
    smooth: float = 0.08
    local: float = 0.05
    gripper_smooth: float = 0.5
    transit: float = 0.0
    path: float = 0.0

    open_reach: float = 12.0
    open_terminal: float = 80.0
    open_above_lid: float = 120.0
    open_hinge_align: float = 8.0
    open_gripper: float = 30.0

    grasp_tip_z: float = 80.0
    grasp_yaw: float = 5.0
    grasp_straddle: float = 30.0
    grasp_center_region: float = 120.0
    grasp_aperture_region: float = 80.0
    grasp_gripper: float = 20.0

    place_reach: float = 12.0
    place_terminal: float = 60.0
    place_xy: float = 120.0
    place_z: float = 80.0
    place_carry_height: float = 80.0
    place_gripper: float = 40.0


class CapsuleFlowStateCost:
    """Capsule-task cost with hard open, grasp, and place stage switches."""

    def __init__(self, task_name: str, weights: CapsuleFlowCostWeights | None = None):
        self.task_name = task_name.lower()
        self.weights = weights or CapsuleFlowCostWeights()
        self.last_terms: dict[str, torch.Tensor] = {}
        self.last_debug: dict[str, torch.Tensor] = {}
        self.last_stage = "init"

    def _store(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_terms = {key: value.detach() for key, value in terms.items()}
        total = None
        for value in terms.values():
            total = value if total is None else total + value
        if total is None:
            raise ValueError("CapsuleFlowStateCost produced no terms.")
        return total

    @staticmethod
    def _object_pose(
        context: dict[str, Any],
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        item = context.get("objects", {}).get(name)
        if not isinstance(item, dict) or item.get("pos") is None or item.get("quat") is None:
            return None
        pos = torch.as_tensor(item["pos"], device=device, dtype=dtype)
        quat = torch.as_tensor(item["quat"], device=device, dtype=dtype)
        if pos.ndim > 1 and pos.shape[0] == 1:
            pos = pos[0]
        if quat.ndim > 1 and quat.shape[0] == 1:
            quat = quat[0]
        return pos[:3], quat[:4]

    @staticmethod
    def _normalized(value: torch.Tensor) -> torch.Tensor:
        return value / torch.clamp(torch.linalg.vector_norm(value, dim=-1, keepdim=True), min=1e-8)

    @staticmethod
    def _rotate_about_axis(vector: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
        axis = axis / torch.clamp(torch.linalg.vector_norm(axis), min=1e-8)
        angle_tensor = torch.as_tensor(angle, device=vector.device, dtype=vector.dtype)
        cos_angle = torch.cos(angle_tensor)
        sin_angle = torch.sin(angle_tensor)
        return (
            vector * cos_angle
            + torch.cross(axis, vector, dim=-1) * sin_angle
            + axis * torch.sum(axis * vector) * (1.0 - cos_angle)
        )

    def _move_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
        target_pos: torch.Tensor,
        approach_target_axis: torch.Tensor | None = None,
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
            if approach_target_axis is None:
                approach_target_axis = _get_vector(
                    context,
                    "target_axis",
                    default=(0.0, 0.0, -1.0),
                    device=device,
                    dtype=dtype,
                )
            approach_target_axis = approach_target_axis / torch.clamp(
                torch.linalg.vector_norm(approach_target_axis), min=1e-8
            )
            approach_axis = _apply_axis(tcp_quat, a_local)
            orient = 1.0 - torch.sum(approach_axis * approach_target_axis.view(1, 1, 3), dim=-1)
            terms["orient"] = weights.orient * torch.mean(orient, dim=1)

        z_floor = float(context.get("z_floor", 0.0))
        floor = torch.clamp(z_floor - tcp_pos[..., 2], min=0.0).pow(2)
        terms["floor"] = weights.floor * torch.mean(floor, dim=1)

        q_traj = real_actions[..., :7]
        if q_traj.shape[1] > 1:
            smooth = torch.sum((q_traj[:, 1:] - q_traj[:, :-1]).pow(2), dim=(-1, -2))
        else:
            smooth = _zero(batch, device, dtype)
        terms["smooth"] = weights.smooth * smooth

        q_cur = _first_tensor(context, "joint_pos", device=device, dtype=dtype)
        if q_cur is not None:
            local = torch.sum((q_traj - q_cur[:7].view(1, 1, 7)).pow(2), dim=(-1, -2))
            terms["local"] = weights.local * local

        if real_actions.shape[-1] > 7 and real_actions.shape[1] > 1:
            gripper = real_actions[..., 7]
            gripper_smooth = torch.mean((gripper[:, 1:] - gripper[:, :-1]).pow(2), dim=1)
            terms["gripper_smooth"] = weights.gripper_smooth * gripper_smooth

        if weights.transit != 0.0:
            terms["transit"] = weights.transit * _transit_clearance_cost(tcp_pos, target_pos, context)
        if weights.path != 0.0:
            terms["path"] = weights.path * _path_constraint_cost(context, batch, device, dtype)
        return terms

    def _lid_state(
        self,
        context: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        lid_joint = _first_tensor(context, "capsule_lid_joint_pos", device=device, dtype=dtype)
        if lid_joint is None:
            raise ValueError("CapsuleFlowStateCost open_lid stage requires capsule_lid_joint_pos.")

        lid_angle_tensor = lid_joint.reshape(-1)[0]
        lid_angle = float(lid_angle_tensor.detach().cpu())
        lid_pose = self._object_pose(context, "capsule_lid", device, dtype)
        if lid_pose is not None:
            return lid_pose[0], lid_pose[1], lid_angle

        capsule_pose = self._object_pose(context, "capsule", device, dtype)
        if capsule_pose is None:
            raise ValueError(
                "CapsuleFlowStateCost open_lid stage requires either "
                "objects['capsule_lid'].{pos,quat} or objects['capsule'].{pos,quat}."
            )
        capsule_pos, capsule_quat = capsule_pose
        hinge_local = torch.as_tensor(
            context.get("capsule_lid_hinge_local", _CAPSULE_LID_HINGE_LOCAL),
            device=device,
            dtype=dtype,
        )
        hinge_pos = capsule_pos + quat_apply_wxyz(capsule_quat, hinge_local)
        half_angle = 0.5 * lid_angle_tensor.to(device=device, dtype=dtype)
        lid_joint_quat = torch.stack(
            (
                torch.cos(half_angle),
                torch.sin(half_angle),
                torch.zeros_like(half_angle),
                torch.zeros_like(half_angle),
            )
        )
        lid_quat = quat_mul_wxyz(capsule_quat, lid_joint_quat)
        return hinge_pos, lid_quat, lid_angle

    def _open_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        batch = real_actions.shape[0]
        hinge_pos, lid_quat, lid_angle = self._lid_state(context, device, dtype)

        grip_local = torch.as_tensor(
            context.get("capsule_lid_grip_local", _CAPSULE_LID_GRIP_LOCAL),
            device=device,
            dtype=dtype,
        )
        radial = quat_apply_wxyz(lid_quat, grip_local)
        grip_pos = hinge_pos + radial
        hinge_axis = self._normalized(
            quat_apply_wxyz(lid_quat, torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype))
        )
        lid_normal = self._normalized(
            quat_apply_wxyz(lid_quat, torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype))
        )

        current_tcp = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
        gripper_pos = _first_tensor(context, "gripper_pos", device=device, dtype=dtype)
        grasp_distance = float(context.get("capsule_lid_grasp_distance", _CAPSULE_LID_GRASP_DISTANCE))
        closed_threshold = float(
            context.get("capsule_gripper_closed_threshold", _CAPSULE_GRIPPER_CLOSED_THRESHOLD)
        )
        current_near = current_tcp is not None and float(
            torch.linalg.vector_norm(current_tcp[:3] - grip_pos).detach().cpu()
        ) <= grasp_distance
        current_closed = gripper_pos is not None and float(gripper_pos.abs().mean().detach().cpu()) >= closed_threshold

        open_threshold = float(context.get("capsule_lid_open_threshold", _CAPSULE_LID_OPEN_THRESHOLD))
        if lid_angle <= open_threshold:
            open_mode = "release"
            target_pos = grip_pos
        elif current_near and current_closed:
            open_mode = "pull"
            arc_step = float(context.get("capsule_lid_arc_step", _CAPSULE_LID_ARC_STEP))
            open_target = float(context.get("capsule_lid_open_target", _CAPSULE_LID_OPEN_TARGET))
            desired_angle = max(open_target, lid_angle - arc_step)
            target_pos = hinge_pos + self._rotate_about_axis(radial, hinge_axis, desired_angle - lid_angle)
        else:
            open_mode = "approach"
            target_pos = grip_pos

        dist_sq = torch.sum((tcp_pos - target_pos.view(1, 1, 3)).pow(2), dim=-1)
        terms: dict[str, torch.Tensor] = {
            "open_reach": weights.open_reach * torch.mean(dist_sq, dim=1),
            "open_terminal": weights.open_terminal * dist_sq[:, -1],
        }

        # Being equally far above or below the lid is not equivalent: a TCP
        # above the contact trajectory cannot get behind the lid to push it.
        # Use the active arc target during pull so this constraint does not
        # oppose the intended upward hinge motion.
        above_lid = torch.clamp(tcp_pos[..., 2] - target_pos[2], min=0.0).pow(2)
        terms["open_above_lid"] = weights.open_above_lid * torch.mean(above_lid, dim=1)

        if tcp_quat is not None:
            closing_axis = _apply_axis(
                tcp_quat,
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            )
            hinge_align = 1.0 - torch.abs(torch.sum(closing_axis * hinge_axis.view(1, 1, 3), dim=-1))
            terms["open_hinge_align"] = weights.open_hinge_align * torch.mean(hinge_align, dim=1)

        if real_actions.shape[-1] > 7:
            gripper = real_actions[..., 7]
            if open_mode == "release":
                desired_gripper = torch.full_like(gripper, _GRIPPER_OPEN_ACTION)
            else:
                # Closed fingertips form the rigid pusher used to contact and
                # sweep the lid, so they must already be closed on approach.
                desired_gripper = torch.full_like(gripper, _GRIPPER_CLOSED_ACTION)
            terms["open_gripper"] = weights.open_gripper * torch.mean(
                (gripper - desired_gripper).pow(2), dim=1
            )

        self.last_debug = {
            "lid_approach_mode": torch.full(
                (batch,), float(open_mode == "approach"), device=device, dtype=dtype
            ),
            "lid_pull_mode": torch.full((batch,), float(open_mode == "pull"), device=device, dtype=dtype),
            "lid_release_mode": torch.full(
                (batch,), float(open_mode == "release"), device=device, dtype=dtype
            ),
        }
        return terms, target_pos, lid_normal

    def _pod_grasp_center(
        self,
        context: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        can_pose = self._object_pose(context, "can", device, dtype)
        if can_pose is None:
            return None
        center_local = torch.as_tensor(
            context.get("capsule_pod_center_local", _CAPSULE_POD_CENTER_LOCAL),
            device=device,
            dtype=dtype,
        )
        return can_pose[0] + quat_apply_wxyz(can_pose[1], center_local)

    def _grasp_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        grasp_center = self._pod_grasp_center(context, device, dtype)
        if grasp_center is None:
            raise ValueError("CapsuleFlowStateCost grasp_pod stage requires objects['can'].{pos,quat}.")

        dist_sq = torch.sum((tcp_pos - grasp_center.view(1, 1, 3)).pow(2), dim=-1)
        terms: dict[str, torch.Tensor] = {
            "reach": weights.reach * torch.mean(dist_sq, dim=1),
            "terminal": weights.terminal * dist_sq[:, -1],
        }

        if tcp_quat is not None:
            approach_axis = _apply_axis(
                tcp_quat,
                _get_vector(
                    context,
                    "a_local",
                    default=(0.0, 0.0, 1.0),
                    device=device,
                    dtype=dtype,
                ),
            )
            closing_axis = _apply_axis(
                tcp_quat,
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            )
            lateral_axis = self._normalized(torch.cross(approach_axis, closing_axis, dim=-1))
            center = grasp_center.view(1, 1, 3)
            rel = center - tcp_pos
            closing_coord = torch.sum(rel * closing_axis, dim=-1)
            lateral_coord = torch.sum(rel * lateral_axis, dim=-1)
            approach_coord = torch.sum(rel * approach_axis, dim=-1)
            center_err = torch.sqrt(
                closing_coord.pow(2) + lateral_coord.pow(2) + approach_coord.pow(2) + 1e-12
            )

            tip_z = (tcp_pos[..., 2] - grasp_center[2]).pow(2)
            terms["grasp_tip_z"] = weights.grasp_tip_z * torch.mean(tip_z, dim=1)
            yaw = 1.0 - torch.maximum(closing_axis[..., 0].abs(), closing_axis[..., 1].abs())
            terms["grasp_yaw"] = weights.grasp_yaw * torch.mean(yaw, dim=1)

            pod_radius = float(context.get("capsule_pod_horizontal_radius", _CAPSULE_POD_HORIZONTAL_RADIUS))
            center_radius = float(context.get("capsule_pod_center_radius", 0.4 * pod_radius))
            center_region = torch.clamp(center_err - center_radius, min=0.0).pow(2)
            terms["grasp_center_region"] = weights.grasp_center_region * torch.mean(center_region, dim=1)

            open_half = float(context.get("open_half", 0.04))
            aperture_margin = float(context.get("gripper_aperture_margin", 0.006))
            aperture = torch.clamp(
                closing_coord.abs() + pod_radius + aperture_margin - open_half,
                min=0.0,
            ).pow(2)
            terms["grasp_aperture_region"] = weights.grasp_aperture_region * torch.mean(aperture, dim=1)

            finger_radius = float(context.get("finger_radius", 0.012))
            finger_l = tcp_pos + open_half * closing_axis
            finger_r = tcp_pos - open_half * closing_axis
            keepout = pod_radius + finger_radius
            dist_l = torch.linalg.vector_norm(finger_l[..., :2] - center[..., :2], dim=-1)
            dist_r = torch.linalg.vector_norm(finger_r[..., :2] - center[..., :2], dim=-1)
            straddle = (
                torch.clamp(keepout - dist_l, min=0.0).pow(2)
                + torch.clamp(keepout - dist_r, min=0.0).pow(2)
            )
            terms["grasp_straddle"] = weights.grasp_straddle * torch.mean(straddle, dim=1)

            if real_actions.shape[-1] > 7:
                close_scale = float(context.get("capsule_pod_close_scale", max(center_radius, 1e-3)))
                center_excess = torch.clamp(center_err - center_radius, min=0.0)
                close_gate = torch.exp(-center_excess.pow(2) / max(close_scale * close_scale, 1e-8))
                gripper = real_actions[..., 7]
                terms["grasp_gripper"] = weights.grasp_gripper * torch.mean(
                    (gripper - close_gate).pow(2), dim=1
                )

        return terms, grasp_center

    def _place_target(
        self,
        context: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        capsule_pose = self._object_pose(context, "capsule", device, dtype)
        if capsule_pose is None:
            raise ValueError("CapsuleFlowStateCost place_pod stage requires objects['capsule'].{pos,quat}.")
        target_local = torch.as_tensor(
            context.get("capsule_pod_place_local", _CAPSULE_POD_PLACE_LOCAL),
            device=device,
            dtype=dtype,
        )
        return capsule_pose[0] + quat_apply_wxyz(capsule_pose[1], target_local)

    def _carried_pod_pos(
        self,
        *,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        can_pose = self._object_pose(context, "can", device, dtype)
        current_tcp = _first_tensor(context, "eef_pos", device=device, dtype=dtype)
        if can_pose is None or current_tcp is None:
            return tcp_pos
        return tcp_pos + (can_pose[0] - current_tcp[:3]).view(1, 1, 3)

    def _place_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.weights
        device = real_actions.device
        dtype = real_actions.dtype
        target_pos = self._place_target(context, device, dtype)
        carried_pos = self._carried_pod_pos(
            tcp_pos=tcp_pos,
            context=context,
            device=device,
            dtype=dtype,
        )

        final_delta = carried_pos - target_pos.view(1, 1, 3)
        xy_dist = torch.linalg.vector_norm(final_delta[..., :2], dim=-1)
        descend_radius = float(context.get("capsule_place_descend_radius", _CAPSULE_PLACE_DESCEND_RADIUS))
        outside_descend = (xy_dist > descend_radius).to(dtype=dtype)
        carry_clearance = float(
            context.get("capsule_place_carry_clearance_z", _CAPSULE_PLACE_CARRY_CLEARANCE_Z)
        )
        carry_z = target_pos[2] + carry_clearance
        z_target = target_pos[2] + outside_descend * carry_clearance
        z_err = carried_pos[..., 2] - z_target
        dist_sq = torch.sum(final_delta[..., :2].pow(2), dim=-1) + z_err.pow(2)
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
            release_xy = float(
                context.get("capsule_place_release_xy_radius", _CAPSULE_PLACE_RELEASE_XY_RADIUS)
            )
            release_z = float(
                context.get("capsule_place_release_z_radius", _CAPSULE_PLACE_RELEASE_Z_RADIUS)
            )
            release_gate = torch.logical_and(
                xy_dist < release_xy,
                final_delta[..., 2].abs() < release_z,
            ).to(dtype=dtype)
            desired_gripper = _GRIPPER_CLOSED_ACTION - release_gate
            gripper = real_actions[..., 7]
            terms["place_gripper"] = weights.place_gripper * torch.mean(
                (gripper - desired_gripper).pow(2), dim=1
            )
        return terms, target_pos

    def compute_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        self.last_debug = {}
        if not _flag(context, "open_coffee_lid"):
            self.last_stage = "open_lid"
            terms, target_pos, approach_target_axis = self._open_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
        elif not _flag(context, "grasp_pod"):
            self.last_stage = "grasp_pod"
            terms, target_pos = self._grasp_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
            approach_target_axis = None
        else:
            self.last_stage = "place_pod"
            terms, target_pos = self._place_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                context=context,
            )
            approach_target_axis = None

        terms.update(
            self._move_terms(
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
                target_pos=target_pos,
                approach_target_axis=approach_target_axis,
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
