"""Ground-truth grounding: targets and obstacles from the simulator's object poses.

The pre-ReKep baseline -- a two-stage pick-and-place (grasp an object, place it on a location) with
geometry read from the USD bounding boxes. It reproduces the original ``minimal_base`` behaviour and
serves as the reference a perception/VLM front-end is swapped in against.
"""
from __future__ import annotations

from sim_common.grounding.api import Grounding, SceneObject, Stage
from sim_common.scene_extents import usd_extents

_DEFAULT_EXTENT = (0.05, 0.05, 0.05)
_PLACE_CLEARANCE = 0.10   # hover height above the place surface (m)


class GTGrounding:
    """Two-stage pick-and-place grounded in ground-truth object poses (``grasp_obj`` -> ``place_obj``)."""

    def __init__(self, grasp_obj: str = "pear", place_obj: str = "scale"):
        self.grasp_obj = grasp_obj
        self.place_obj = place_obj

    def ground(self, env) -> Grounding:
        scene_objects = list(getattr(env.env.scene, "rigid_objects", {}) or {})
        extents = usd_extents(env, scene_objects)
        objects = [SceneObject(name=n, pos=(lambda n=n: env.object_pose(n)[0]),
                               extents=extents.get(n, _DEFAULT_EXTENT)) for n in scene_objects]

        place_half_height = extents.get(self.place_obj, _DEFAULT_EXTENT)[2]

        def place_target():
            pos = env.object_pose(self.place_obj)[0].copy()
            pos[2] += place_half_height + _PLACE_CLEARANCE   # hover above the place surface
            return pos

        stages = [
            Stage(name=f"grasp {self.grasp_obj}", gripper="close", grasp_obj=self.grasp_obj, payload=None,
                  target=(lambda: env.object_pose(self.grasp_obj)[0])),
            Stage(name=f"place on {self.place_obj}", gripper="hold", grasp_obj=self.grasp_obj,
                  payload=self.grasp_obj, target=place_target),
        ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({self.grasp_obj, self.place_obj}))
