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
            # Object sizes from the segmented point cloud, not the USD model: the base then runs on image
            # and instruction alone. An object perception could not size falls to a generic default rather
            # than to a model lookup, so no per-object geometry is assumed.
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

        # Live per-object graspable center: its nearest tracked keypoint moved by a fixed offset to a
        # geometry-based center (keypoint tracks motion; the offset removes the surface bias). The center
        # is the xy bounding-box center (robust to the visible-near-side bias in the point mean) at a
        # base + half-height grasp depth. GT pose as fallback when no masked points.
        kp_of, centroid_off = {}, {}
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is not None:
                kp = masks._nearest_kp(keypoints, pts.mean(axis=0))
                bbox_xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0   # extent center (robust to near-side bias)
                # single-cam depth sees the top surface down to the silhouette edge (the widest visible cross-section,
                # ~the object center + best grasp height); the point-MEAN sits ~1cm high on the top shell. Use a low
                # percentile of z (~the silhouette edge) instead.
                grasp_z = float(np.percentile(pts[:, 2], 15))
                grasp_center = np.array([bbox_xy[0], bbox_xy[1], grasp_z], dtype=np.float64)
                kp_of[name], centroid_off[name] = kp, grasp_center - keypoints[kp]

        def obj_pos(name):
            """Live grasp centre: the object's tracked keypoint, offset to the centre the gripper wants.

            The keypoint carries the object's motion; the offset removes the surface bias it was proposed
            with. No fallback to simulator state: an object perception never found has no position, and
            saying so is better than quietly substituting the truth.
            """
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
            """Object owning keypoint kp_idx: the tracker's nearest-center assignment (object root within
            0.35m), robust to boundary depth-bleed and to the large support surface that fool a
            nearest-masked-point lookup; falls back to nearest masked-point only when the tracker left it
            unowned."""
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
                # Lift straight up from the GRASP CENTER, not from the raw keypoint. The raw keypoint is a
                # surface point (the near-side/top-shell bias that centroid_off exists to remove), so a lift
                # target built on it is laterally offset from where the gripper actually holds the object --
                # the arm then drags the object sideways as it lifts and the grasp slips. obj_pos is the
                # corrected center, which is what the GT grounding lifts from too.
                lift_pos = obj_pos(name) + np.asarray([0.0, 0.0, _LIFT_HEIGHT], dtype=np.float64)
                z0 = float(obj_pos(name)[2])
                stages.append(Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                                    held_idx=held_after, target=(lambda lp=lift_pos: lp),
                                    done=(lambda n=name, z=z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)))
            elif release_kp >= 0:
                name = name_for(release_kp)
                manipulated.add(name)
                # Hover over the place object's corrected CENTER, not over one of its raw keypoints. A wide
                # object like the scale owns several keypoints that are near-equidistant from its centroid,
                # so `_nearest_kp` picks between them on a float-noise tie-break: the same scene resolved
                # the scale to kp4 on one run and kp6 on the next, moving the place target and splitting
                # otherwise identical rollouts. The centroid is tie-free, so the target is reproducible.
                # (The VLM subgoal is already invariant to this: its offset is measured from whichever
                # keypoint was chosen, so kp[scale] + off lands on the same scale-top point either way.)
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
