from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .costs_explore import (
    _WEIGHT_OBJECT_HORIZONTAL_RADIUS,
    _apply_axis,
    _first_tensor,
    _get_vector,
    _obstacles_for_weight,
    _target_from_object,
    _zero,
    _flag,
)
from .costs_grasp_flow import GraspFlowCostWeights, GraspFlowStateCost


@dataclass(frozen=True)
class WeightKeyposeCostWeights:
    """Endpoint-only weights derived from the weight grasp-flow objective."""

    reach: float = 8.0
    terminal: float = 40.0
    orient: float = 8.0
    floor: float = 50.0
    yaw: float = 5.0
    straddle: float = 30.0
    tip_z: float = 80.0
    center_region: float = 120.0
    aperture_region: float = 80.0
    close_gripper: float = 20.0
    place_terminal: float = 60.0
    place_xy: float = 120.0
    place_z: float = 80.0
    place_gripper: float = 40.0
    clear: float = 0.1


def _as_grasp_flow_weights(weights: WeightKeyposeCostWeights) -> GraspFlowCostWeights:
    """Reuse the tested grasp geometry while disabling every trajectory term."""

    return GraspFlowCostWeights(
        reach=weights.reach,
        terminal=weights.terminal,
        orient=weights.orient,
        floor=weights.floor,
        smooth=0.0,
        local=0.0,
        yaw=weights.yaw,
        straddle=weights.straddle,
        tip_z=weights.tip_z,
        center_region=weights.center_region,
        aperture_region=weights.aperture_region,
        close_gripper=weights.close_gripper,
        gripper_smooth=0.0,
        soft_grasp=0.0,
        lift_reach=0.0,
        lift_terminal=0.0,
        lift_xy=0.0,
        lift_z=0.0,
        lift_gripper=0.0,
        place_reach=0.0,
        place_terminal=weights.place_terminal,
        place_xy=weights.place_xy,
        place_z=weights.place_z,
        place_carry_height=0.0,
        place_gripper=weights.place_gripper,
        clear=weights.clear,
        transit=0.0,
        path=0.0,
    )


class WeightKeyposeStateCost(GraspFlowStateCost):
    """Weight-task objective evaluated only on the final keypose token.

    The action prefix is deliberately invisible to this task cost. There is no
    lift stage, carry-height target, locality, velocity, smoothness, transit, or
    path term. During placement, the object pose is estimated from the live
    object-to-TCP offset on every replan and rigidly attached only for this one
    endpoint evaluation.
    """

    def __init__(
        self,
        task_name: str,
        weights: WeightKeyposeCostWeights | None = None,
    ):
        self.keypose_weights = weights or WeightKeyposeCostWeights()
        super().__init__(
            task_name,
            weights=_as_grasp_flow_weights(self.keypose_weights),
        )

    @staticmethod
    def _endpoint(values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        if values.ndim < 2 or values.shape[1] < 1:
            raise ValueError(
                "Weight keypose cost expects a non-empty temporal dimension; "
                f"got {tuple(values.shape)}"
            )
        return values[:, -1:, ...]

    def _endpoint_move_terms(
        self,
        *,
        active_object: str | None,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        weights = self.keypose_weights
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
            target_axis = target_axis / torch.clamp(
                torch.linalg.vector_norm(target_axis),
                min=1e-8,
            )
            orient = 1.0 - torch.sum(
                approach_axis * target_axis.view(1, 1, 3),
                dim=-1,
            )
            terms["orient"] = weights.orient * orient[:, -1]

        z_floor = float(context.get("z_floor", 0.0))
        floor = torch.clamp(z_floor - tcp_pos[:, -1, 2], min=0.0).pow(2)
        terms["floor"] = weights.floor * floor

        if "weight" in self.task_name and weights.clear != 0.0:
            clear = _zero(batch, device, dtype)
            ee_radius = float(context.get("tcp_radius", 0.035))
            clearance = float(context.get("clearance", 0.02))
            active_pos = (
                _target_from_object(context, active_object, device, dtype)
                if active_object is not None
                else None
            )
            for obstacle_pos, obstacle_radius in _obstacles_for_weight(
                context,
                device,
                dtype,
            ):
                # The grasp target or rigidly carried object is not an obstacle
                # to its own endpoint. Match by live position because the shared
                # obstacle helper intentionally returns only positions/radii.
                if active_pos is not None and torch.allclose(
                    obstacle_pos[:3],
                    active_pos[:3],
                    atol=1e-6,
                    rtol=0.0,
                ):
                    continue
                keepout = obstacle_radius + ee_radius + clearance
                distance = torch.linalg.vector_norm(
                    tcp_pos[:, -1, :] - obstacle_pos.view(1, 3),
                    dim=-1,
                )
                clear = clear + torch.clamp(
                    keepout - distance,
                    min=0.0,
                ).pow(2)
            terms["clear"] = weights.clear * clear

        return terms

    def _place_endpoint_terms(
        self,
        *,
        object_name: str,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        weights = self.keypose_weights
        device = real_actions.device
        dtype = real_actions.dtype
        target_pos = self._place_target(object_name, context, device, dtype)
        if target_pos is None:
            fallback = _first_tensor(
                context,
                "eef_pos",
                device=device,
                dtype=dtype,
            )
            target_pos = (
                torch.zeros(3, device=device, dtype=dtype)
                if fallback is None
                else fallback[:3]
            )
            return {}, target_pos

        carried_pos = self._carried_object_pos(
            object_name=object_name,
            tcp_pos=tcp_pos,
            context=context,
            device=device,
            dtype=dtype,
        )[:, -1, :]
        delta = carried_pos - target_pos.view(1, 3)
        xy_sq = torch.sum(delta[:, :2].pow(2), dim=-1)
        z_sq = delta[:, 2].pow(2)
        terms: dict[str, torch.Tensor] = {
            "place_terminal": weights.place_terminal * (xy_sq + z_sq),
            "place_xy": weights.place_xy * xy_sq,
            "place_z": weights.place_z * z_sq,
        }

        if real_actions.shape[-1] > 7:
            xy_distance = torch.sqrt(xy_sq + 1e-12)
            release_xy_radius = max(
                float(context.get("place_release_xy_radius", 0.06)),
                _WEIGHT_OBJECT_HORIZONTAL_RADIUS.get(object_name, 0.05),
            )
            release_z_radius = float(
                context.get("place_release_z_radius", 0.035)
            )
            release_gate = torch.logical_and(
                xy_distance < release_xy_radius,
                delta[:, 2].abs() < release_z_radius,
            ).to(dtype=dtype)
            desired_gripper = 1.0 - release_gate
            gripper = real_actions[:, -1, 7]
            terms["place_gripper"] = weights.place_gripper * (
                gripper - desired_gripper
            ).pow(2)

        return terms, target_pos

    @staticmethod
    def _release_object_name(context: dict[str, Any]) -> str | None:
        if _flag(context, "open_gripper_apple"):
            return "apple"
        if _flag(context, "open_gripper_pear"):
            return "pear"
        return None

    def compute_terms(
        self,
        *,
        real_actions: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor | None,
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        real_actions = self._endpoint(real_actions)
        tcp_pos = self._endpoint(tcp_pos)
        tcp_quat = self._endpoint(tcp_quat)
        assert real_actions is not None
        assert tcp_pos is not None

        release_object = self._release_object_name(context)
        pick_object = self._grasp_object_name(context)
        place_object = self._place_object_name(context)
        active_object = release_object or place_object or pick_object
        if release_object is not None:
            self.last_stage = f"open_gripper_{release_object}_keypose"
            gripper = real_actions[:, -1, 7]
            terms = {
                "open_gripper": self.keypose_weights.place_gripper
                * gripper.pow(2)
            }
        elif place_object is not None:
            self.last_stage = f"place_{place_object}_keypose"
            terms, _ = self._place_endpoint_terms(
                object_name=place_object,
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                context=context,
            )
        elif pick_object is not None:
            self.last_stage = f"grasp_{pick_object}_keypose"
            terms, _ = self._grasp_terms(
                object_name=pick_object,
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
        else:
            self.last_stage = "idle_keypose"
            terms = {}

        terms.update(
            self._endpoint_move_terms(
                active_object=active_object,
                real_actions=real_actions,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                context=context,
            )
        )
        return terms
