"""ReKep grounding: per-stage relational keypoint constraints as the controller's objective.

Runs the ReKep front-end (``propose_keypoints`` perception, fake/real VLM metadata + constraint files,
live ``KeypointTracker``) and packages each stage's subgoal + path constraints (loaded to torch via
``constraints``) into the ``Grounding`` contract as ``constraint`` / ``path_fns`` (``held_idx`` = keypoints
riding the gripper). The cost optimizes the constraint directly -- no subgoal solve, so nothing is reduced
to a single resolved point.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from rekep import grounding as rk_grounding
from rekep.constraint_generation import ConstraintGenerator
from rekep.keypoint_tracking import KeypointTracker
from rekep.utils import get_callable_grasping_cost_fn, load_default_config
from sim_common.grounding import Grounding, SceneObject, Stage, fake_vlm, masks
from sim_common.geometry import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents
from sim_common.constraints import TorchNumpyShim, load_torch_constraints, make_torch_constraint

_PLACE_HOVER = (0.0, 0.0, 0.10)   # reference height above the placement (gripper proximity + place-release)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RekepGrounding:
    """Grounding from the ReKep front-end: tracked keypoints + per-stage relational constraints."""

    def __init__(self, vlm: str = "fake", task_key: str = "weight", place_obj: str = "scale",
                 clearance: float = 0.015, release_tol: float = 0.05):
        self.vlm = vlm
        self.task_key = task_key
        self.place_obj = place_obj
        self.clearance = clearance
        self.release_tol = release_tol

    def ground(self, env) -> Grounding:
        dev = env.device
        config = load_default_config()
        grounded = rk_grounding.propose_keypoints(env.cam, env.env, config)
        keypoints = grounded["keypoints"]
        if len(keypoints) == 0:
            raise SystemExit("[rekep-grounding] no keypoints proposed")
        scene_objects = list(getattr(env.env.scene, "rigid_objects", {}) or {})
        extents = usd_extents(env, scene_objects)
        tracker = KeypointTracker(env.env, keypoints)
        shim = TorchNumpyShim(dev)

        vlm_dir = os.path.join(_REPO, "results", "vlm_mpc", "vlm_base", f"vlm_query_{self.task_key}")
        if self.vlm == "fake":
            metadata, _ = fake_vlm.generate(self.task_key, vlm_dir, keypoints, grounded, env.env, self.clearance)
        else:
            metadata = self._real_constraints(vlm_dir, grounded, config)
        bad = [i for i in metadata["grasp_keypoints"] + metadata["release_keypoints"] if not -1 <= i < len(keypoints)]
        if bad:                                   # untrusted VLM output -- fail fast, not with an IndexError mid-rollout
            raise SystemExit(f"[rekep] VLM referenced out-of-range keypoint(s) {bad} (have {len(keypoints)})")

        # Live per-object center for collision/straddle: its nearest tracked keypoint moved to the
        # perception centroid by a fixed offset (keypoint tracks motion; the offset removes the
        # surface-keypoint bias off the object center). GT pose as fallback when no masked points.
        kp_of, centroid_off = {}, {}
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is not None:
                kp = masks._nearest_kp(keypoints, pts.mean(axis=0))
                kp_of[name], centroid_off[name] = kp, pts.mean(axis=0) - keypoints[kp]

        def obj_pos(name):
            kp = kp_of.get(name)
            return tracker.get_positions()[kp] + centroid_off[name] if kp is not None else env.object_pose(name)[0]

        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)), extents=extents.get(n, _DEFAULT_EXTENT))
                   for n in scene_objects]
        scale_kp = kp_of.get(self.place_obj)   # placement keypoint = the place-onto object's tracked keypoint

        def load_stage(idx, held):
            """Load stage idx's subgoal + path constraints as torch callables (held kps ride the gripper)."""
            grasp_fn = get_callable_grasping_cost_fn(list(held))
            subgoal = make_torch_constraint(load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_subgoal_constraints.txt"), grasp_fn, shim))
            path_fns = tuple(make_torch_constraint([c]) for c in load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_path_constraints.txt"), grasp_fn, shim))
            return subgoal, path_fns

        def satisfied(subgoal):
            """The subgoal met at the current single TCP + live keypoints (place-stage release/advance)."""
            def _done():
                tcp = torch.tensor(env.tcp(), device=dev, dtype=torch.float32).reshape(1, 1, 3)
                kps = torch.tensor(tracker.get_positions(), device=dev, dtype=torch.float32)
                return float(subgoal(tcp, kps).reshape(-1)[0]) < self.release_tol
            return _done

        stages, manipulated, grasped_body = [], {self.place_obj}, None
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = masks.object_for_keypoint(grounded, env.env, keypoints[grasp_kp],
                                                           names=tuple(scene_objects))
                manipulated.add(name)
                stages.append(Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None,
                                    held_idx=held, target=(lambda gk=grasp_kp: tracker.get_positions()[gk]),
                                    constraint=subgoal, path_fns=path_fns, done_flag=f"grasp_{name}"))
                grasped_body = tracker.owners[grasp_kp]
            elif release_kp >= 0:
                name = masks.object_for_keypoint(grounded, env.env, keypoints[release_kp],
                                                           names=tuple(scene_objects))
                manipulated.add(name)
                place_ref = scale_kp if scale_kp is not None else release_kp   # place-onto kp, else the held kp
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name,
                                    place_target=self.place_obj, done_flag=f"{name}_on_{self.place_obj}",
                                    held_idx=held, done=satisfied(subgoal), constraint=subgoal, path_fns=path_fns,
                                    target=(lambda rk=place_ref: tracker.get_positions()[rk] + np.asarray(_PLACE_HOVER))))
                grasped_body = None

        kps_probe = torch.as_tensor(keypoints, device=dev, dtype=torch.float32)
        tcp_probe = kps_probe[:1].reshape(1, 1, 3)
        for st in stages:                         # probe-eval each constraint so a bad index in a body fails here
            for fn in filter(None, (st.constraint, *st.path_fns)):
                try:
                    fn(tcp_probe, kps_probe)
                except Exception as exc:
                    raise SystemExit(f"[rekep] stage '{st.name}' constraint failed to evaluate "
                                     f"(bad keypoint index?): {exc}")

        return Grounding(objects=objects, stages=stages, manipulated=frozenset(manipulated),
                         keypoints=(lambda: tracker.get_positions()))

    def _real_constraints(self, vlm_dir, grounded, config):
        """Query the real GPT-4o ``ConstraintGenerator`` for this task; return the parsed metadata.

        Writes the same artifacts the fake stub does (``metadata.json`` + per-stage constraint files) into
        ``vlm_dir`` from the keypoint-annotated image + the task instruction, so the stage building above is
        identical to the fake path.
        """
        with open(os.path.join(_REPO, "task_prompts.json"), encoding="utf-8") as f:
            instruction = json.load(f)[self.task_key]["prompt"]
        ConstraintGenerator(config["constraint_generator"]).generate(
            grounded["projected"], instruction, {}, vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), encoding="utf-8") as f:
            return json.load(f)
