"""Ground-truth pick-and-place grounding from simulator object poses."""

from __future__ import annotations

import numpy as np

from sim_free_mpc.costs_explore import (
    _WEIGHT_SCALE_CENTER_OFFSET,
    _WEIGHT_SCALE_TOP_OFFSET_Z,
)
from vlm_dp.grounding import Grounding, SceneObject, Stage
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_PLACE_CLEARANCE = 0.10
_LIFT_HEIGHT = 0.15
_LIFT_CONFIRM = 0.05
_PLACE_XY = 0.12

# Separates objects on the platform from objects knocked onto the table.
_PLACE_Z_SANITY = 0.10

_SEAT_CLEARANCE = 0.01


def place_success_xy(place_obj, root_xy):
    """Return the reference point used by the task success predicate."""
    if place_obj == "scale":
        return np.asarray(root_xy, dtype=np.float64) + np.array([0.0, -0.05])
    return np.asarray(root_xy, dtype=np.float64)


def place_platform_offset(place_obj, place_half_height):
    """Return the top-surface offset from the place-object root."""
    if place_obj == "scale":
        offset = _WEIGHT_SCALE_CENTER_OFFSET.cpu().numpy().astype(np.float64)
        return offset + np.array([0.0, -0.05, _WEIGHT_SCALE_TOP_OFFSET_Z])
    return np.array([0.0, 0.0, place_half_height])


def _shifted_seat(seat, own_r, others, footprint=None):
    """Move a seat away from occupied positions while staying on the platform."""
    for pos, radius in others:
        delta = seat[:2] - pos[:2]
        required = radius + own_r + _SEAT_CLEARANCE
        distance = float(np.linalg.norm(delta))
        if distance >= required:
            continue

        direction = (
            delta / distance
            if distance > 1e-6
            else np.array([0.0, 1.0])
        )
        seat = seat.copy()

        if footprint is None:
            seat[:2] = pos[:2] + direction * required
            continue

        center = np.asarray(footprint[0], dtype=np.float64)[:2]
        room = np.maximum(
            np.asarray(footprint[1], dtype=np.float64)[:2] - own_r,
            1e-3,
        )

        def _slack(candidate):
            target = pos[:2] + candidate * required
            return float(np.min(room - np.abs(target - center)))

        if distance <= 0.5 * required or _slack(direction) < 0.0:
            axes = [
                np.array([1.0, 0.0]),
                np.array([-1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, -1.0]),
            ]
            direction = max(axes, key=_slack)

        seat[:2] = np.clip(
            pos[:2] + direction * required,
            center - room,
            center + room,
        )

    return seat


class GTGrounding:
    """Ground each grasp object onto the target using ground-truth poses."""

    def __init__(
        self,
        grasp_obj: str = "pear",
        place_obj: str = "scale",
        grasp_objs=None,
        seat_shift: bool = True,
    ):
        self.grasp_objs = list(grasp_objs) if grasp_objs else [grasp_obj]
        self.place_obj = place_obj
        self.seat_shift = seat_shift

    def ground(self, env, world) -> Grounding:
        scene_objects = list(
            getattr(env.env.scene, "rigid_objects", {}) or {}
        )
        extents = usd_extents(env, scene_objects)

        # Object positions come from the selected world model.
        obj_pos = lambda name: world.object_pose(name)[0]

        objects = [
            SceneObject(
                name=name,
                pos=lambda name=name: obj_pos(name),
                extents=extents.get(name, _DEFAULT_EXTENT),
            )
            for name in scene_objects
        ]

        platform_offset = place_platform_offset(
            self.place_obj,
            extents.get(self.place_obj, _DEFAULT_EXTENT)[2],
        )

        def base_seat():
            return obj_pos(self.place_obj) + platform_offset

        def seat_for(name, prior):
            """Return a live seat that avoids previously placed objects."""

            def _seat():
                others = [
                    (
                        np.asarray(obj_pos(other), dtype=np.float64),
                        extents.get(other, _DEFAULT_EXTENT)[1],
                    )
                    for other in prior
                ]
                seat = np.asarray(base_seat(), dtype=np.float64)
                place_extents = extents.get(
                    self.place_obj,
                    _DEFAULT_EXTENT,
                )
                half_xy = (
                    float(place_extents[0]) + float(place_extents[1])
                ) / 2.0
                return _shifted_seat(
                    seat,
                    extents.get(name, _DEFAULT_EXTENT)[1],
                    others,
                    footprint=(seat[:2], (half_xy, half_xy)),
                )

            return _seat

        def hover_for(seat_fn):
            def _target():
                position = seat_fn().copy()
                position[2] += _PLACE_CLEARANCE
                return position

            return _target

        def placed(name, seat_fn):
            """Return whether the object satisfies the estimated place predicate."""

            def _placed():
                reference = place_success_xy(
                    self.place_obj,
                    obj_pos(self.place_obj)[:2],
                )
                delta = obj_pos(name)[:2] - reference
                seat_z = (
                    float(seat_fn()[2])
                    + extents.get(name, _DEFAULT_EXTENT)[2]
                )
                near_z = (
                    abs(float(obj_pos(name)[2]) - seat_z)
                    < _PLACE_Z_SANITY
                )
                return bool(
                    float(np.linalg.norm(delta)) < _PLACE_XY
                    and near_z
                )

            return _placed

        stages = []

        for index, name in enumerate(self.grasp_objs):
            prior = self.grasp_objs[:index] if self.seat_shift else []
            seat = seat_for(name, prior)

            grasp_position = obj_pos(name).copy()
            grasp_z = float(grasp_position[2])

            # Keep the lift target aligned with the object's live XY position.
            lift_target = (
                lambda name=name, z=grasp_z + _LIFT_HEIGHT: np.array(
                    [*np.asarray(obj_pos(name))[:2], z],
                    dtype=np.float64,
                )
            )

            stages += [
                Stage(
                    name=f"grasp {name}",
                    gripper="close",
                    grasp_obj=name,
                    payload=None,
                    target=lambda name=name: obj_pos(name),
                    done_flag=f"grasp_{name}",
                ),
                Stage(
                    name=f"lift {name}",
                    gripper="hold",
                    grasp_obj=name,
                    payload=name,
                    target=lift_target,
                    done=(
                        lambda name=name, z=grasp_z: float(
                            obj_pos(name)[2]
                        )
                        > z + _LIFT_CONFIRM
                    ),
                ),
                Stage(
                    name=f"place {name} on {self.place_obj}",
                    gripper="place",
                    grasp_obj=None,
                    payload=name,
                    place_target=self.place_obj,
                    target=hover_for(seat),
                    done=placed(name, seat),
                    place_point=seat,
                    carry_z=lambda z=grasp_z + _LIFT_HEIGHT: z,
                    done_flag=f"{name}_on_{self.place_obj}",
                ),
            ]

        return Grounding(
            objects=objects,
            stages=stages,
            manipulated=frozenset(self.grasp_objs) | {self.place_obj},
        )