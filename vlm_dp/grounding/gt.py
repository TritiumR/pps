"""Ground-truth grounding: pick-and-place from simulator object poses (the privileged baseline)."""
from __future__ import annotations

import numpy as np

from vlm_dp.grounding import Grounding, SceneObject, Stage
from sim_common.geometry import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_PLACE_CLEARANCE = 0.10   # hover height above the place surface (m)
_LIFT_HEIGHT = 0.15       # raise the grasped object this high before carrying to the place (m)
_LIFT_CONFIRM = 0.05      # the object must rise at least this much for the lift to be done (grasp confirmed) (m)
_PLACE_XY = 0.09          # object within this xy distance of the place-surface centre before release/advance:
                          # tighter than the success tolerance so the object seats well onto the surface, not at its edge
_PLACE_SEAT_Z = 0.04      # object seated: its centre within this of the resting height (bottom on the surface
                          # top) before release/advance; stops the release-from-hover drop


class GTGrounding:
    """Pick-and-place grounded in GT poses: each of ``grasp_objs`` -> ``place_obj``."""

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
            """Resting on the place surface: near in xy AND seated in z (no env placement flag needed)."""
            def _p():
                d = obj_pos(name)[:2] - obj_pos(self.place_obj)[:2]
                seat_z = float(obj_pos(self.place_obj)[2]) + place_half_height + extents.get(name, _DEFAULT_EXTENT)[2]
                seated = abs(float(obj_pos(name)[2]) - seat_z) < _PLACE_SEAT_Z
                return bool(float(np.linalg.norm(d)) < _PLACE_XY and seated)
            return _p

        stages = []
        for name in self.grasp_objs:
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
                      place_target=self.place_obj, target=place_target, done=placed(name),
                      done_flag=f"{name}_on_{self.place_obj}"),   # names the place object (grasp_flow cost needs it)
            ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset(self.grasp_objs) | {self.place_obj})
