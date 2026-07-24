"""Ground-truth grounding: pick-and-place from simulator object poses (the privileged baseline)."""
from __future__ import annotations

import numpy as np

from vlm_dp.grounding import Grounding, SceneObject, Stage
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents
from sim_free_mpc.costs_explore import (_WEIGHT_SCALE_CENTER_OFFSET, _WEIGHT_SCALE_TOP_OFFSET_Z)

_PLACE_CLEARANCE = 0.10   # hover height above the place surface (m)
_LIFT_HEIGHT = 0.15       # raise the grasped object this high before carrying to the place (m)
_LIFT_CONFIRM = 0.05      # the object must rise at least this much for the lift to be done (grasp confirmed) (m)
_PLACE_XY = 0.12          # the task's own success threshold (SCALE_XY_THRESHOLD)
_PLACE_Z_SANITY = 0.10    # m, separates on-the-platform from knocked-to-the-table. No tighter: the env's
                          # predicate has no z term, and a seat-tight gate mispredicts lying objects.
_SEAT_CLEARANCE = 0.01    # xy gap kept between seated objects sharing the place surface (m)


def place_success_xy(place_obj, root_xy):
    """The xy point the task's own success predicate measures from (scale: root + its y_offset).

    Aiming uses the calibrated seat. Confirming must mirror the env or a placed object never reads as
    placed, since the seat sits about 5 cm from the env's reference.
    """
    if place_obj == "scale":
        return np.asarray(root_xy, dtype=np.float64) + np.array([0.0, -0.05])
    return np.asarray(root_xy, dtype=np.float64)


def place_platform_offset(place_obj, place_half_height):
    """Top-surface seat offset from the place object's root (calibrated for the scale fixture)."""
    if place_obj == "scale":
        off = _WEIGHT_SCALE_CENTER_OFFSET.cpu().numpy().astype(np.float64)
        return off + np.array([0.0, -0.05, _WEIGHT_SCALE_TOP_OFFSET_Z])
    return np.array([0.0, 0.0, place_half_height])


def _shifted_seat(seat, own_r, others):
    """Nudge a seat xy away from occupying objects [(pos, radius)] until it clears them."""
    for pos, radius in others:
        d = seat[:2] - pos[:2]
        need = radius + own_r + _SEAT_CLEARANCE
        dist = float(np.linalg.norm(d))
        if dist < need:
            direction = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
            seat = seat.copy()
            seat[:2] = pos[:2] + direction * need
    return seat


class GTGrounding:
    """Pick-and-place grounded in GT poses: each of grasp_objs onto place_obj."""

    def __init__(self, grasp_obj: str = "pear", place_obj: str = "scale", grasp_objs=None,
                 seat_shift: bool = True):
        # grasp_objs wins when given. grasp_obj stays for the single-object callers.
        self.grasp_objs = list(grasp_objs) if grasp_objs else [grasp_obj]
        self.place_obj = place_obj
        self.seat_shift = seat_shift

    def ground(self, env, world) -> Grounding:
        scene_objects = list(getattr(env.env.scene, "rigid_objects", {}) or {})
        extents = usd_extents(env, scene_objects)
        obj_pos = lambda n: world.object_pose(n)[0]   # object positions come from the world model, not the sim
        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)),
                               extents=extents.get(n, _DEFAULT_EXTENT)) for n in scene_objects]

        platform_off = place_platform_offset(self.place_obj,
                                             extents.get(self.place_obj, _DEFAULT_EXTENT)[2])

        def base_seat():
            return obj_pos(self.place_obj) + platform_off      # top-surface seat point

        def seat_for(name, prior):
            """Per-object seat: the calibrated point, nudged live off objects placed before this one."""
            def _p():
                others = [(np.asarray(obj_pos(o), dtype=np.float64),
                           extents.get(o, _DEFAULT_EXTENT)[1]) for o in prior]
                return _shifted_seat(np.asarray(base_seat(), dtype=np.float64),
                                     extents.get(name, _DEFAULT_EXTENT)[1], others)
            return _p

        def hover_for(seat_fn):
            def _t():
                pos = seat_fn().copy()
                pos[2] += _PLACE_CLEARANCE                      # hover above the seat
                return pos
            return _t

        def placed(name, seat_fn):
            """The task's own on-place predicate from our estimates: xy about the env's reference
            point, plus a z band separating the platform from the table (the env has no z term)."""
            def _p():
                ref = place_success_xy(self.place_obj, obj_pos(self.place_obj)[:2])
                d = obj_pos(name)[:2] - ref
                seat_z = float(seat_fn()[2]) + extents.get(name, _DEFAULT_EXTENT)[2]
                near_z = abs(float(obj_pos(name)[2]) - seat_z) < _PLACE_Z_SANITY
                return bool(float(np.linalg.norm(d)) < _PLACE_XY and near_z)
            return _p

        stages = []
        for i, name in enumerate(self.grasp_objs):
            seat = seat_for(name, self.grasp_objs[:i] if self.seat_shift else [])
            grasp_pos0 = obj_pos(name).copy()            # resting pose at grounding time (static until grasped)
            grasp_z0 = float(grasp_pos0[2])
            # Lift target: the object's LIVE xy at a fixed height. A frozen xy drags the object
            # toward its old resting spot while gripped (it slips), and re-grasps after a drop
            # would lift toward the wrong place entirely.
            lift_target = (lambda n=name, z=grasp_z0 + _LIFT_HEIGHT:
                           np.array([*np.asarray(obj_pos(n))[:2], z], dtype=np.float64))
            stages += [
                Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None,
                      target=(lambda n=name: obj_pos(n)),
                      done_flag=f"grasp_{name}"),   # advance on the real grasp, not a proximity/hold timeout
                Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                      target=lift_target,
                      done=(lambda n=name, z=grasp_z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)),
                Stage(name=f"place {name} on {self.place_obj}", gripper="place", grasp_obj=None, payload=name,
                      place_target=self.place_obj, target=hover_for(seat), done=placed(name, seat),
                      place_point=seat,
                      carry_z=(lambda z=grasp_z0 + _LIFT_HEIGHT: z),   # transit at the lift altitude
                      done_flag=f"{name}_on_{self.place_obj}"),   # names the place object for the place cost
            ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset(self.grasp_objs) | {self.place_obj})
