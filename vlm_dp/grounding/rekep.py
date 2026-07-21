"""ReKep grounding: tracked keypoints + per-stage relational constraints as the objective.

Runs the ReKep front-end (keypoint proposal, fake/real VLM constraints, live tracker) and
packages subgoal/path constraints into the ``Grounding`` contract.
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
from vlm_dp.grounding import Grounding, SceneObject, Stage, fake_vlm, masks
from sim_common.geometry import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents
from sim_common.constraints import TorchNumpyShim, load_torch_constraints, make_torch_constraint

_PLACE_HOVER = (0.0, 0.0, 0.10)   # reference height above the placement (gripper proximity + place-release)
_LIFT_HEIGHT = 0.15               # raise the grasped object this high before the place (m)
_LIFT_CONFIRM = 0.05              # the object must rise at least this much for the lift to be done (m)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RekepGrounding:
    """Grounding from the ReKep front-end: tracked keypoints + per-stage relational constraints."""

    def __init__(self, vlm: str = "fake", task_key: str = "weight", place_obj: str = "scale",
                 clearance: float = 0.015, release_tol: float = 0.05, perception=None):
        self.vlm = vlm
        self.task_key = task_key
        self.place_obj = place_obj
        self.clearance = clearance
        self.release_tol = release_tol
        self.perception = perception   # when set, masks and object points come from a segmenter, not the sim

    def ground(self, env, world) -> Grounding:
        dev = env.device
        config = load_default_config()
        grounded = rk_grounding.propose_keypoints(env.cam, env.env, config, perception=self.perception)
        keypoints = grounded["keypoints"]
        if len(keypoints) == 0:
            raise SystemExit("[rekep-grounding] no keypoints proposed")
        scene_objects = list(world.names)
        if self.perception is not None:
            # Sizes from the segmented cloud, not the USD model; unsized objects get a generic default.
            extents = {n: e for n in scene_objects if (e := self.perception.object_extents(n)) is not None}
            usd = usd_extents(env, scene_objects)   # logged only, to see how far the cloud size is from the model
            for _n in scene_objects:
                if _n in extents and _n in usd:
                    print(f"[ext] {_n}: cloud(grip,keep,h)={tuple(round(x,3) for x in extents[_n])} "
                          f"usd={tuple(round(x,3) for x in usd[_n])}", flush=True)
        else:
            extents = usd_extents(env, scene_objects)
        tracker = KeypointTracker(world, keypoints)
        shim = TorchNumpyShim(dev)

        vlm_dir = os.path.join(_REPO, "results", "vlm_mpc", "vlm_base", f"vlm_query_{self.task_key}")
        if self.vlm == "fake":
            metadata, _ = fake_vlm.generate(self.task_key, vlm_dir, keypoints, grounded, env.env, self.clearance)
        else:
            metadata = self._real_constraints(vlm_dir, grounded, config)
        bad = [i for i in metadata["grasp_keypoints"] + metadata["release_keypoints"] if not -1 <= i < len(keypoints)]
        if bad:                                   # untrusted VLM output -- fail fast, not with an IndexError mid-rollout
            raise SystemExit(f"[rekep] VLM referenced out-of-range keypoint(s) {bad} (have {len(keypoints)})")

        # Nearest tracked keypoint + fixed offset to a geometry-based center; GT pose only as last fallback.
        kp_of, centroid_off = {}, {}
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is not None:
                kp = masks._nearest_kp(keypoints, pts.mean(axis=0))
                bbox_xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0   # extent center (robust to near-side bias)
                # Depth sees the top shell; a low z-percentile reads the silhouette edge (the grasp height).
                grasp_z = float(np.percentile(pts[:, 2], 15))
                grasp_center = np.array([bbox_xy[0], bbox_xy[1], grasp_z], dtype=np.float64)
                kp_of[name], centroid_off[name] = kp, grasp_center - keypoints[kp]

        def obj_pos(name):
            """Live grasp centre: tracked keypoint minus its proposal offset. No simulator fallback."""
            kp = kp_of.get(name)
            if kp is None:
                raise KeyError(f"[rekep] no grasp centre for {name!r}: it has no keypoint of its own.")
            return tracker.get_positions()[kp] + centroid_off[name]

        for _n in scene_objects:   # estimate vs ground truth, logged for scoring only (never read back)
            if _n in kp_of:
                err = np.linalg.norm(obj_pos(_n) - env.object_pose(_n)[0]) * 1000
                print(f"[rekep-dbg] {_n}: grasp_center={np.round(obj_pos(_n), 3)} "
                      f"gt={np.round(env.object_pose(_n)[0], 3)} err={err:.1f}mm", flush=True)

        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)), extents=extents.get(n, _DEFAULT_EXTENT))
                   for n in scene_objects]

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

        def name_for(kp_idx):
            """Owner of keypoint kp_idx: tracker's nearest-centre assignment, else nearest masked point."""
            return tracker.owners[kp_idx] or masks.object_for_keypoint(
                grounded, env.env, keypoints[kp_idx], names=tuple(scene_objects))

        stages, manipulated, grasped_body = [], {self.place_obj}, None
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                _alt = masks.object_for_keypoint(grounded, env.env, keypoints[grasp_kp], names=tuple(scene_objects))
                if _alt != name:   # the boundary/support-surface mis-map this fix corrects (e.g. pear -> "cabbage"/"board")
                    print(f"[rekep] grasp_kp={grasp_kp} -> {name} (nearest-surface would mis-say '{_alt}')", flush=True)
                manipulated.add(name)
                stages.append(Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None,
                                    held_idx=held, target=(lambda nm=name: obj_pos(nm)),
                                    done_flag=f"grasp_{name}"))   # reach the object centroid (graspable), like GT
                grasped_body = tracker.owners[grasp_kp]
                # Lift the just-grasped object before the carry (grasp_flow's grasp -> lift -> place). Reaches a
                # fixed point above the grasp with the gripper held closed; done once the object physically rises.
                held_after = tuple(j for j, o in enumerate(tracker.owners) if o == grasped_body)
                # Lift target: the object's LIVE corrected center xy at a fixed height (a stale xy
                # drags the gripped object toward its old resting spot and it slips).
                z0 = float(obj_pos(name)[2])
                lift_target = (lambda n=name, z=z0 + _LIFT_HEIGHT:
                               np.array([*np.asarray(obj_pos(n))[:2], z], dtype=np.float64))
                stages.append(Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                                    held_idx=held_after, target=lift_target,
                                    done=(lambda n=name, z=z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)))
            elif release_kp >= 0:
                name = name_for(release_kp)
                manipulated.add(name)
                # Hover over the corrected CENTER: nearest-keypoint ties on a wide object are float noise and
                # move the target between runs; the centroid is tie-free.
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name,
                                    place_target=self.place_obj, done_flag=f"{name}_on_{self.place_obj}",
                                    held_idx=held, done=satisfied(subgoal), constraint=subgoal, path_fns=path_fns,
                                    target=(lambda: obj_pos(self.place_obj) + np.asarray(_PLACE_HOVER))))
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
        """Query the real GPT-4o constraint generator; writes the same artifacts as the fake stub."""
        with open(os.path.join(_REPO, "task_prompts.json"), encoding="utf-8") as f:
            instruction = json.load(f)[self.task_key]["prompt"]
        ConstraintGenerator(config["constraint_generator"]).generate(
            grounded["projected"], instruction, {}, vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), encoding="utf-8") as f:
            return json.load(f)
