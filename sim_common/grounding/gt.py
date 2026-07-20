"""Ground-truth grounding: pick-and-place from the simulator's object poses.

Grasp one or more objects and place each on a location, with obstacles + geometry read straight from the
USD bounding boxes. The privileged baseline a perception/VLM front-end is compared against. With several
grasp objects it is the privileged upper bound for a multi-object task (e.g. both fruits on the scale),
which is what makes it directly comparable to a hand-tuned reference on that task.
"""
from __future__ import annotations

import numpy as np

from sim_common.grounding import Grounding, SceneObject, Stage
from sim_common.geometry import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_PLACE_CLEARANCE = 0.10   # hover height above the place surface (m)
_LIFT_HEIGHT = 0.15       # raise the grasped object this high before carrying to the place (m)
_LIFT_CONFIRM = 0.05      # the object must rise at least this much for the lift to be done (grasp confirmed) (m)
_PLACE_XY = 0.09          # object within this xy distance of the place-surface centre before release/advance:
                          # tighter than the 0.12 success tolerance so the object is set down WELL ONTO the
                          # surface, not at its edge (releasing at the 12cm edge let a round object roll off).
                          # The place stage ALSO requires z-seating below, so it completes only once set down.
_PLACE_SEAT_Z = 0.04      # object seated: its centre within this of the resting height (bottom on the surface
                          # top) before release/advance; stops the release-from-hover drop


class GTGrounding:
    """Pick-and-place grounded in ground-truth object poses: each of ``grasp_objs`` -> ``place_obj``.

    A single grasp object gives the original pear -> scale baseline; several give the multi-object one. The
    per-object stages are built identically, and every object is placed at the same place-surface point, so
    the multi-object case is the honest extension of the single one rather than a re-tuned variant.
    """

    def __init__(self, grasp_obj: str = "pear", place_obj: str = "scale", grasp_objs=None):
        # grasp_objs wins when given; grasp_obj stays for the single-object callers.
        self.grasp_objs = list(grasp_objs) if grasp_objs else [grasp_obj]
        self.place_obj = place_obj

    def ground(self, env, world) -> Grounding:
        scene_objects = list(getattr(env.env.scene, "rigid_objects", {}) or {})
        extents = usd_extents(env, scene_objects)
        obj_pos = lambda n: world.object_pose(n)[0]   # object positions come from the world model, not the sim
        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)),
                               extents=extents.get(n, _DEFAULT_EXTENT)) for n in scene_objects]

        place_half_height = extents.get(self.place_obj, _DEFAULT_EXTENT)[2]

        def place_target():
            pos = obj_pos(self.place_obj).copy()
            pos[2] += place_half_height + _PLACE_CLEARANCE   # hover above the place surface
            return pos

        def placed(name):
            """The object is resting on the place surface: near it in xy AND seated on top in z.

            A geometric fact read from the world model, so it holds for GT and sensed state alike and does
            not depend on the env exposing a per-object placement flag (the weight env, for one, has no
            apple_on_scale term, so a flag-only check can never see the apple land). The z-seating test makes
            release/advance wait until the object is set down: without it the stage completed while the object
            was still xy-aligned at carry height, releasing it into a drop and advancing blind to the landing.
            """
            def _p():
                d = obj_pos(name)[:2] - obj_pos(self.place_obj)[:2]
                seat_z = float(obj_pos(self.place_obj)[2]) + place_half_height + extents.get(name, _DEFAULT_EXTENT)[2]
                seated = abs(float(obj_pos(name)[2]) - seat_z) < _PLACE_SEAT_Z
                return bool(float(np.linalg.norm(d)) < _PLACE_XY and seated)
            return _p

        stages = []
        for name in self.grasp_objs:
            grasp_pos0 = obj_pos(name).copy()            # resting pose at grounding time (static until grasped)
            lift_pos = grasp_pos0.copy()
            lift_pos[2] += _LIFT_HEIGHT                   # a fixed point directly above the grasp
            grasp_z0 = float(grasp_pos0[2])
            stages += [
                Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None,
                      target=(lambda n=name: obj_pos(n)),
                      done_flag=f"grasp_{name}"),   # advance on the real grasp, not a proximity/hold timeout
                Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                      target=(lambda lp=lift_pos: lp),
                      done=(lambda n=name, z=grasp_z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)),
                Stage(name=f"place {name} on {self.place_obj}", gripper="place", grasp_obj=None, payload=name,
                      place_target=self.place_obj, target=place_target, done=placed(name),
                      done_flag=f"{name}_on_{self.place_obj}"),   # names the place object (grasp_flow cost needs it)
            ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset(self.grasp_objs) | {self.place_obj})
