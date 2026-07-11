"""Ground-truth grounding: two-stage pick-and-place from the simulator's object poses.

Grasp an object then place it on a location, with obstacles + geometry read straight from the USD
bounding boxes. The privileged baseline a perception/VLM front-end is compared against.
"""
from __future__ import annotations

from sim_common.grounding import Grounding, SceneObject, Stage
from sim_common.geometry import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_PLACE_CLEARANCE = 0.10   # hover height above the place surface (m)
_LIFT_HEIGHT = 0.15       # raise the grasped object this high before carrying to the place (m)
_LIFT_CONFIRM = 0.05      # the object must rise at least this much for the lift to be done (grasp confirmed) (m)


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

        grasp_pos0 = env.object_pose(self.grasp_obj)[0].copy()   # resting pose at grounding time
        lift_pos = grasp_pos0.copy()
        lift_pos[2] += _LIFT_HEIGHT                              # a fixed point directly above the grasp
        grasp_z0 = float(grasp_pos0[2])

        def lifted():
            return float(env.object_pose(self.grasp_obj)[0][2]) > grasp_z0 + _LIFT_CONFIRM   # object rose -> grasped

        stages = [
            Stage(name=f"grasp {self.grasp_obj}", gripper="close", grasp_obj=self.grasp_obj, payload=None,
                  target=(lambda: env.object_pose(self.grasp_obj)[0]),
                  done_flag=f"grasp_{self.grasp_obj}"),   # advance on the real grasp, not a proximity/hold timeout
            Stage(name=f"lift {self.grasp_obj}", gripper="hold", grasp_obj=self.grasp_obj,
                  payload=self.grasp_obj, target=(lambda: lift_pos), done=lifted),
            Stage(name=f"place on {self.place_obj}", gripper="place", grasp_obj=None,
                  payload=self.grasp_obj, place_target=self.place_obj, target=place_target,
                  done_flag=f"{self.grasp_obj}_on_{self.place_obj}"),   # names the place object (grasp_flow cost needs it)
        ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({self.grasp_obj, self.place_obj}))
