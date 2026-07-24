"""ReKep grounding: tracked keypoints and per-stage relational constraints as the objective.

Runs the ReKep front-end (keypoint proposal, fake or real VLM constraints, live tracker) and packages
subgoal and path constraints into the Grounding contract.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from rekep import grounding as rk_grounding
from rekep.constraint_generation import ConstraintGenerator
from rekep.keypoint_tracking import KeypointTracker
from rekep.utils import get_callable_grasping_cost_fn, load_default_config
from vlm_dp.grounding import Grounding, SceneObject, Stage, fake_vlm, masks
# The env-mirroring predicates (_PLACE_XY, _PLACE_Z_SANITY, place_success_xy) are deliberately not
# imported: they belong to the privileged GT rung. This path judges placement from its own geometry.
from vlm_dp.grounding.gt import _LIFT_HEIGHT as _GT_LIFT, _shifted_seat
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, center_from_points, usd_extents
from vlm_dp.sim_helpers import TorchNumpyShim, load_torch_constraints, make_torch_constraint

_PLACE_HOVER = (0.0, 0.0, 0.10)   # reference height above the placement (gripper proximity and release)
# The seat is measured directly off the destination's own cloud (see seat_point), so there is no
# per-fixture calibration table and no per-object constant.
_LIFT_HEIGHT = 0.15               # raise the grasped object this high before the place (m)
_LIFT_CONFIRM = 0.05              # the object must rise at least this much for the lift to be done (m)
# Placement tolerances are our own sensing slack, deliberately not the env's success thresholds: a
# policy whose done-predicate mirrors the evaluator is tuned to the answer key and will not transfer.
_PLACE_MARGIN = 0.02              # m, slack added to the destination's measured footprint (grounding error)
_REST_TOL = 0.10                  # m, how far an object's centre may sit from its seat height and still
                                  # read as resting. A tighter gate mispredicts objects that settle lying over.
_MIN_LOCAL_PTS = 20               # local grasp: min cloud points near the keypoint before trusting the ball
_MIN_GRASP_EXT = 0.012            # local grasp: extent floor (a finger radius, cannot grasp more precisely)
_RESTS_ON_SUPPORT = 0.03          # local grasp: cloud bottom within this of the support gives support-relative z
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _annotate_object_names(img, name_masks):
    """Write each segmented object's name at its mask centroid on the keypoint overlay."""
    import cv2
    out = np.ascontiguousarray(img.copy())
    for name, mask in name_masks.items():
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        at = (int(xs.mean()) - 4 * len(name), int(ys.mean()))
        cv2.putText(out, name, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, name, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _referenced_kps(txt_path):
    """Keypoint indices a constraint file references (for the collision-excluded place target)."""
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, encoding="utf-8") as f:
        src = f.read()
    return sorted({int(m) for m in re.findall(r"keypoints\[(\d+)\]", src)})


# Tokens that mark a sub-goal as constraining an orientation (an angle between keypoint vectors, an
# upright or tilt condition) rather than only a position.
_ROTATION_TOKENS = ("arccos", "arctan", "cross(", "angle", "upright", "tilt", "parallel", "perpendicular")


def _places_into(txt_path, place_target):
    """True when the VLM's sub-goal puts the payload inside the destination rather than on it.

    A container's top surface is its rim, so the set-down terms would seat the object on the rim and
    never in the cavity. When the VLM says inside it has already specified the drop point itself.
    """
    if not os.path.exists(txt_path):
        return False
    with open(txt_path, encoding="utf-8") as f:
        src = f.read().lower()
    if any(tok in src for tok in ("inside", "into", "within")):
        return True
    tgt = (place_target or "").lower()
    return bool(tgt) and (f"in the {tgt}" in src or f"in {tgt}" in src)


_KP_MENTION = re.compile(r"keypoints?\s*[\[\s]\s*(\d+)\s*\]?")


def keypoint_claims(text, names):
    """Map keypoint index to the object the constraint text attributes it to, from the VLM's own prose.

    The VLM names each keypoint as it uses it (the teapot spout, keypoint 23, and the cup opening,
    keypoint 25), which is an independent statement of identity from the tracker's. The tracker assigns
    owners by nearest object centre within a radius, a rule that quietly mis-assigns a keypoint on a
    large fixture or a part. Two independent sources let a disagreement be detected instead of silently
    steering the arm at the wrong thing. Each index is attributed to the last object name mentioned
    before it.
    """
    claims = {}
    for chunk in re.split(r"(?<=[.;])\s+|\n", text or ""):
        cursor = 0
        for match in _KP_MENTION.finditer(chunk):
            before = chunk[cursor:match.start()].lower()
            hit = max(((before.rfind(n.lower()), n) for n in names if n.lower() in before), default=None)
            if hit is not None and hit[0] >= 0:
                claims[int(match.group(1))] = hit[1]
            cursor = match.end()
    return claims


def _constrains_orientation(txt_path):
    """True when the VLM's sub-goal commands a rotation (pour, reorient, swing to an angle).

    The controller's downward tool-axis prior must then stand down: it is a grasp-approach convenience,
    and left on it fights the very rotation the VLM asked for, so a pour could never tilt.
    """
    if not os.path.exists(txt_path):
        return False
    with open(txt_path, encoding="utf-8") as f:
        src = f.read().lower()
    return any(tok in src for tok in _ROTATION_TOKENS)


class RekepGrounding:
    """Grounding from the ReKep front-end: tracked keypoints + per-stage relational constraints."""

    def __init__(self, vlm: str = "fake", task_key: str | None = None, place_obj: str | None = None,
                 clearance: float = 0.015, perception=None, seat_shift: bool = True,
                 grasp_objs=None, support=None, stages: str = "template", subgoal_eps: float = 0.06,
                 local_grasp: bool = False, local_grasp_radius: float = 0.05,
                 kp_source: str = "perception", contact_criterion: str = "feasibility",
                 open_half: float = 0.04):
        self.vlm = vlm
        self.task_key = task_key
        self.place_obj = place_obj
        self.grasp_objs = list(grasp_objs) if grasp_objs else []
        self.support = support   # surface the graspables rest on (their centre is the top-support midpoint)
        self.clearance = clearance
        self.perception = perception   # when set, masks and object points come from a segmenter, not the sim
        self.seat_shift = seat_shift
        # Local grasp affordance (opt-in): centre and extent from the cloud near the VLM grasp keypoint,
        # not the whole-object centroid, so it grasps a part (lid lip) and makes its pinch feasible. Radius
        # is aperture-scaled. Default off, leaving the whole-object centroid path unchanged.
        self.local_grasp = bool(local_grasp)
        self.local_grasp_radius = float(local_grasp_radius)
        # perception (default) proposes keypoints from the camera. gt injects task keypoints at GT part
        # poses, a diagnostic that de-confounds occlusion and perception from the downstream cost.
        self.kp_source = kp_source
        # template (default) is the hand-coded grasp-lift-place ladder. vlm emits stages straight from the
        # VLM constraints, where the sub-goal is both the objective and the advance predicate.
        self.stages = stages
        self.subgoal_eps = float(subgoal_eps)   # m, sub-goal-satisfied tolerance for stage advance
        # Contact mode: feasibility (default) measures whether the gripper fits, plan keeps the old
        # plan-structural inference. open_half is the gripper's aperture half-width.
        self.contact_criterion = contact_criterion
        self.open_half = float(open_half)

    def ground(self, env, world) -> Grounding:
        # Roles have no task defaults: missing ones fail here rather than silently binding another task's
        # objects. The VLM-driven path needs no place_obj, deriving the destination from the sub-goal.
        if not self.task_key:
            raise SystemExit("[rekep] task_key is required, pass it from the task entry (no task default).")
        if self.stages != "vlm" and not self.place_obj:
            raise SystemExit("[rekep] the template path needs place_obj, pass it from the task entry, or use "
                             "a _vlm grounding, which derives the destination from the VLM sub-goal.")
        dev = env.device
        config = load_default_config()
        grounded = rk_grounding.propose_keypoints(env.cam, env.env, config, perception=self.perception)
        keypoints = grounded["keypoints"]
        virtual_kps, gt_meta = set(), None
        if self.kp_source == "gt":
            # Diagnostic: swap perception keypoints for task keypoints at GT part poses (occlusion-free),
            # so the driven cost is tested on correct grounding. Task-specific, a privileged rung.
            if self.task_key == "capsule":
                from vlm_dp.grounding.capsule import gt_keypoints as _gt_kps
                keypoints, gt_meta = _gt_kps(env)
                grounded["keypoints"] = keypoints
                virtual_kps = gt_meta["virtual"]
                print(f"[rekep-gt] injected {len(keypoints)} GT keypoints "
                      f"(owners={gt_meta['owners']}, virtual={sorted(virtual_kps)})", flush=True)
            else:
                raise SystemExit(f"[rekep-grounding] kp_source=gt has no keypoints for task {self.task_key!r}")
        if len(keypoints) == 0:
            raise SystemExit("[rekep-grounding] no keypoints proposed")
        scene_objects = list(world.names)
        if self.perception is not None:
            # Sizes from the segmented cloud, not the USD model. Unsized objects get a generic default.
            extents = {n: e for n in scene_objects if (e := self.perception.object_extents(n)) is not None}
            usd = usd_extents(env, scene_objects)   # logged only, to compare cloud size against the model
            diag = os.environ.get("VLMDP_EXT_DIAG")   # opt-in extent diagnostics (see extent_diagnostics)
            for _n in scene_objects:
                if _n in extents and _n in usd:
                    print(f"[ext] {_n}: cloud(grip,keep,h)={tuple(round(x,3) for x in extents[_n])} "
                          f"usd={tuple(round(x,3) for x in usd[_n])}", flush=True)
                    if diag and (d := self.perception.extent_diagnostics(_n)) is not None:
                        print(f"[ext-diag] {_n}: {d}", flush=True)
        else:
            extents = usd_extents(env, scene_objects)
        tracker = KeypointTracker(world, keypoints)
        if gt_meta is not None:
            # GT identities: force owner and re-registration per keypoint. The tracker associates by
            # nearest object centre within a radius, which cannot place a keypoint on a large fixture's
            # lid (its centre is far away). With injected GT keypoints the identity is known, so set it.
            for _i, _o in enumerate(gt_meta["owners"]):
                if not (0 <= _i < len(tracker.owners)):
                    continue
                if _o is None or _i in virtual_kps:
                    tracker.owners[_i] = None
                    tracker.registrations[_i] = (None, np.asarray(keypoints[_i], dtype=np.float64).copy())
                else:
                    _p, _r = world.object_pose(_o)
                    tracker.owners[_i] = _o
                    tracker.registrations[_i] = (_o, _r.T @ (np.asarray(keypoints[_i], dtype=np.float64) - _p))
        shim = TorchNumpyShim(dev)

        # Per-process artifact dir: parallel runs otherwise clobber each other's constraint files mid-read.
        vlm_dir = os.path.join(_REPO, "results", "vlm_mpc", "vlm_base",
                               f"vlm_query_{self.task_key}_p{os.getpid()}")
        if self.vlm == "fake":
            metadata, _ = fake_vlm.generate(self.task_key, vlm_dir, keypoints, grounded, env.env, self.clearance)
        else:
            metadata = self._real_constraints(vlm_dir, grounded, config)
        bad = [i for i in metadata["grasp_keypoints"] + metadata["release_keypoints"] if not -1 <= i < len(keypoints)]
        if bad:                                   # untrusted VLM output, fail fast not with a mid-rollout IndexError
            raise SystemExit(f"[rekep] VLM referenced out-of-range keypoint(s) {bad} (have {len(keypoints)})")

        # Nearest tracked keypoint plus a fixed offset to a geometry-based centre.
        support_top = None
        if self.support:
            spts = masks._masked_points(grounded, env.env, self.support)
            if spts is not None:
                support_top = float(np.percentile(spts[:, 2], 95))
        # The VLM's grasp keypoints are the grasp decision, and their identity comes from perception (the
        # tracker's object assignment), not a hardcoded grasp_objs allow-list. Objects the VLM chose to
        # grasp get the support-relative centre and narrow-axis, the rest do not.
        grasped_names = {tracker.owners[k] for k in metadata["grasp_keypoints"]
                         if 0 <= k < len(tracker.owners) and tracker.owners[k]}
        # The VLM's grasp keypoint per grasped object, its chosen affordance point and the local focus.
        grasp_kp_of = {}
        for k in metadata["grasp_keypoints"]:
            if 0 <= k < len(tracker.owners) and tracker.owners[k]:
                grasp_kp_of.setdefault(tracker.owners[k], k)
        kp_of, centroid_off, grasp_axis, grasp_ext_of, seat_off = {}, {}, {}, {}, {}
        grasp_region_of = {}
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is None:
                continue
            gk = grasp_kp_of.get(name)
            # Support height for this object's grasp centre: the declared entity if named, else measured
            # geometrically from the ring around its footprint (A.2). Without either, center_from_points
            # falls back to the visible-extent midpoint, biased ~1-2cm high.
            obj_support = support_top
            if obj_support is None and self.perception is not None and name in grasped_names:
                obj_support = self.perception.support_height(name)
            if self.local_grasp and gk is not None:
                # Local grasp: centre, extent and axis from the cloud within an aperture ball of the VLM
                # keypoint, not the whole-object centroid, so it grasps a part (lid lip) or offset
                # affordance and its narrow local extent makes the pinch terms satisfiable. Degrades to
                # the body centre for a compact object, where the ball then holds the whole object.
                near = pts[np.linalg.norm(pts - keypoints[gk], axis=1) <= self.local_grasp_radius]
                src = near if near.shape[0] >= _MIN_LOCAL_PTS else pts
                lo, hi = np.percentile(src[:, :2], [5, 95], axis=0)
                grasp_ext_of[name] = max(float(np.min(hi - lo)) / 2.0, _MIN_GRASP_EXT)   # narrow half-width
                rests = (obj_support is not None
                         and float(np.percentile(src[:, 2], 5)) - obj_support < _RESTS_ON_SUPPORT)
                grasp_center = center_from_points(src, obj_support if rests else None)
                kp = gk
            else:
                src = pts
                kp = masks._nearest_kp(keypoints, pts.mean(axis=0))
                grasp_center = center_from_points(pts, obj_support if name in grasped_names else None)
            kp_of[name], centroid_off[name] = kp, grasp_center - keypoints[kp]
            if name == self.place_obj:
                # Destination seat measured straight off its own cloud: the top face's xy centre at the
                # top z, tracked as an offset from the same keypoint like the grasp centre. This needs no
                # privileged root and no per-fixture constant.
                cap = pts[pts[:, 2] >= np.percentile(pts[:, 2], 90)]      # the top face
                cap_xy = (cap[:, :2].min(axis=0) + cap[:, :2].max(axis=0)) / 2.0
                top_z = float(np.percentile(pts[:, 2], 95))
                seat_off[name] = np.array([cap_xy[0], cap_xy[1], top_z], dtype=np.float64) - keypoints[kp]
            if name in grasped_names:                  # close across the narrow axis (None if round)
                grasp_axis[name] = masks.narrow_axis(src)
                print(f"[grasp-axis] {name}: {grasp_axis[name]}", flush=True)
                # Graspable segment: the span along the object's long horizontal axis (perpendicular to
                # the narrow closing axis) over which the gripper can still close. A point target pins the
                # grasp to one pose. A region leaves the base free to choose where, which is what a
                # steering proxy needs in order to shift the grasp at all.
                n = grasp_axis[name]
                if n is not None:
                    long_axis = np.array([-float(n[1]), float(n[0]), 0.0], dtype=np.float64)
                    t = (src[:, :2] - grasp_center[:2]) @ long_axis[:2]
                    half_len = float(np.percentile(np.abs(t), 80))   # robust span, ignores cloud outliers
                    grasp_region_of[name] = (long_axis.tolist(), half_len)
                    print(f"[grasp-region] {name}: half_len={half_len:.3f}m along {np.round(long_axis, 2)}",
                          flush=True)

        if gt_meta is not None:
            # GT keypoints are the grasp targets: grasp at the grasp keypoint (a large fixture's lid has
            # no maskable cloud to refine from), with the GT grasp extent. Overrides the perception centre.
            for k in metadata["grasp_keypoints"]:
                o = gt_meta["owners"][k] if 0 <= k < len(gt_meta["owners"]) else None
                if k < 0 or o is None:
                    continue
                kp_of[o], centroid_off[o] = k, np.zeros(3)
                if k in gt_meta["grasp_extent"]:
                    grasp_ext_of[o] = gt_meta["grasp_extent"][k]
                if o not in scene_objects:
                    scene_objects.append(o)

        def obj_pos(name):
            """Live grasp centre: tracked keypoint minus its proposal offset. No simulator fallback."""
            kp = kp_of.get(name)
            if kp is None:
                raise KeyError(f"[rekep] no grasp centre for {name!r}: it has no keypoint of its own.")
            return tracker.get_positions()[kp] + centroid_off[name]

        def seat_point():
            """Live destination seat: the MEASURED top surface, tracked off the place object's keypoint."""
            kp, off = kp_of.get(self.place_obj), seat_off.get(self.place_obj)
            if kp is None or off is None:
                raise KeyError(f"[rekep] no measured seat for place object {self.place_obj!r}: "
                               "perception found no cloud for it.")
            return tracker.get_positions()[kp] + off

        for _n in scene_objects:   # estimate vs ground truth, logged for scoring only (never read back)
            if _n in kp_of:
                err = np.linalg.norm(obj_pos(_n) - env.object_pose(_n)[0]) * 1000
                print(f"[rekep-dbg] {_n}: grasp_center={np.round(obj_pos(_n), 3)} "
                      f"gt={np.round(env.object_pose(_n)[0], 3)} err={err:.1f}mm", flush=True)

        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)), extents=extents.get(n, _DEFAULT_EXTENT),
                               axis=grasp_axis.get(n), grasp_extent=grasp_ext_of.get(n),
                               grasp_region=grasp_region_of.get(n))
                   for n in scene_objects]
        # Unnamed scene geometry as obstacles. Once the vocabulary names only the instruction's
        # referents, a distractor is no longer a scene object, and one the arm cannot see is worse than
        # one it might grasp. An obstacle needs extent, not identity, so leftover cloud enters as
        # anonymous blobs. The keepout terms exclude by name, so a blob matches no role and can never be
        # selected as a target. Static like the fixtures, measured once at grounding.
        if self.perception is not None:
            for i, (centre, ext) in enumerate(self.perception.unexplained_obstacles(support_top)):
                objects.append(SceneObject(name=f"_obstacle{i}", pos=(lambda p=centre: p), extents=ext,
                                           axis=None, grasp_extent=None, grasp_region=None))
            if len(objects) > len(scene_objects):
                print(f"[rekep] {len(objects) - len(scene_objects)} unnamed obstacles from scene geometry",
                      flush=True)

        def load_stage(idx, held):
            """Load stage idx's subgoal and path constraints as torch callables (held kps ride the gripper)."""
            grasp_fn = get_callable_grasping_cost_fn(list(held))
            subgoal = make_torch_constraint(load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_subgoal_constraints.txt"), grasp_fn, shim))
            path_fns = tuple(make_torch_constraint([c]) for c in load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_path_constraints.txt"), grasp_fn, shim))
            return subgoal, path_fns

        def placed(name, seat_fn):
            """Resting on the destination, judged from our own geometry: the object sits over the
            destination's measured footprint and at its seat height.

            Deliberately not the env's success predicate. Mirroring the evaluator's thresholds and its
            per-fixture offset tunes the policy to the answer key, inflates measured success, and does
            not transfer off this bench.
            """
            def _done():
                seat = np.asarray(seat_fn(), dtype=np.float64)
                obj = obj_pos(name)
                foot = extents.get(self.place_obj, _DEFAULT_EXTENT)[1]      # destination's own half-footprint
                near_xy = float(np.linalg.norm(obj[:2] - seat[:2])) < foot + _PLACE_MARGIN
                seat_z = float(seat[2]) + extents.get(name, _DEFAULT_EXTENT)[2]
                return bool(near_xy and abs(float(obj[2]) - seat_z) < _REST_TOL)
            return _done

        def name_for(kp_idx):
            """Owner of keypoint kp_idx: tracker's nearest-centre assignment, else nearest masked point."""
            return tracker.owners[kp_idx] or masks.object_for_keypoint(
                grounded, env.env, keypoints[kp_idx], names=tuple(scene_objects))

        def seat_for(nm, prior):
            """Per-object seat on the place surface, nudged live off objects placed before this one."""
            def _p():
                seat = np.asarray(seat_point(), dtype=np.float64)
                others = [(np.asarray(obj_pos(o), dtype=np.float64),
                           extents.get(o, _DEFAULT_EXTENT)[1]) for o in prior]
                return _shifted_seat(seat, extents.get(nm, _DEFAULT_EXTENT)[1], others)
            return _p

        # No identity guard: the VLM's grasp keypoints are trusted as-is and their identity is read from
        # perception (name_for), not remapped onto a hardcoded grasp_objs list. The old guard mislabeled
        # a part of a larger segmented object (a coffee-maker lid relabeled as the pod). Robustness to
        # GPT-4o index-misjoins belongs in perception and constraint-text verification, not a task-noun
        # allow-list.

        # VLM-driven path: emit exactly num_stages stages from the constraints, with no hand-coded lift,
        # hover or seat (the lift emerges from the sub-goal, collision and floor). Opt-in. The template
        # path below is the default and is unchanged.
        if self.stages == "vlm":
            stages, manipulated = self._vlm_stages(metadata, tracker, keypoints, name_for,
                                                   load_stage, objects, env, dev, vlm_dir)
            return Grounding(objects=objects, stages=stages, manipulated=frozenset(manipulated),
                             keypoints=(lambda: tracker.get_positions()))

        stages, manipulated, grasped_body = [], {self.place_obj}, None
        placed_names, last_z0 = [], None
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                _alt = masks.object_for_keypoint(grounded, env.env, keypoints[grasp_kp], names=tuple(scene_objects))
                if _alt != name:   # boundary or support-surface mis-map this fix corrects
                    print(f"[rekep] grasp_kp={grasp_kp} -> {name} (nearest-surface would mis-say {_alt})", flush=True)
                manipulated.add(name)
                stages.append(Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None,
                                    held_idx=held, target=(lambda nm=name: obj_pos(nm)),
                                    done_flag=f"grasp_{name}"))   # reach the graspable object centroid, like GT
                grasped_body = tracker.owners[grasp_kp]
                # Lift the just-grasped object before the carry. Reaches a fixed point above the grasp
                # with the gripper held closed, done once the object physically rises.
                held_after = tuple(j for j, o in enumerate(tracker.owners) if o == grasped_body)
                # Lift target: the object's live corrected centre xy at a fixed height. A stale xy would
                # drag the gripped object toward its old resting spot and it slips.
                z0 = float(obj_pos(name)[2])
                last_z0 = z0
                lift_target = (lambda n=name, z=z0 + _LIFT_HEIGHT:
                               np.array([*np.asarray(obj_pos(n))[:2], z], dtype=np.float64))
                stages.append(Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name,
                                    held_idx=held_after, target=lift_target,
                                    done=(lambda n=name, z=z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)))
            elif release_kp >= 0:
                name = name_for(release_kp)
                manipulated.add(name)
                seat = seat_for(name, tuple(placed_names) if self.seat_shift else ())
                # Hover over the corrected centre. Nearest-keypoint ties on a wide object are float noise
                # and move the target between runs, whereas the centroid is tie-free.
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name,
                                    place_target=self.place_obj, done_flag=f"{name}_on_{self.place_obj}",
                                    held_idx=held, done=placed(name, seat), constraint=subgoal, path_fns=path_fns,
                                    target=(lambda s=seat: np.asarray(s()) + np.asarray(_PLACE_HOVER)),
                                    place_point=seat,
                                    carry_z=(None if last_z0 is None
                                             else (lambda z=last_z0 + _GT_LIFT: z))))
                placed_names.append(name)
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

    def _contact_for(self, name, extents):
        """Contact mode from measured geometry: can the gripper close across this object as a unit?

        extents[..][0] is the object's narrow horizontal half-width, the axis the gripper closes across
        (grasp_axis yaws it there), measured off its own cloud. open_half is the aperture half-width. So
        this is a direct feasibility test, not an inference:

        - fits gives pinch. The gripper can take the object as a unit, so the aperture sensor's stall test
          is a valid hold certificate and the pinch-certification terms are satisfiable.
        - too big gives press. It cannot be taken as a unit, so whatever the gripper does at the VLM's
          keypoint is contact against a larger structure: certification stands down and advance runs on
          contact. This is right whether the structure is hinged (a capsule lid) or free (a pot cover),
          since the contact mode only says do not certify a free-body pinch here, and the sub-goal still
          owns what happens next.

        The criterion this replaces read the plan (grasp X, then displace X with no external destination),
        so it could not tell a hinged part from a free one and called the pot's cover an articulation.
        Measured separation is wide: every pinchable object is <= 30 mm, every press case >= 76 mm,
        against a 40 mm aperture.
        """
        grip = float(extents.get(name, _DEFAULT_EXTENT)[0])
        return ("pinch", grip) if grip <= self.open_half else ("press", grip)

    def _vlm_stages(self, metadata, tracker, keypoints, name_for, load_stage, objects, env, dev, vlm_dir):
        """Emit exactly the VLM's stages, whatever num_stages and grasp-release pattern it returns, with
        no assumed structure and no inserted stages.

        Each stage's sub-goal is both its objective and its advance predicate, and grasp or release
        keypoints set the gripper intent. The grasp centres on the object body. The place is driven by
        the VLM sub-goal, and the carry routes around obstacles via payload collision (carry_clear), an
        emergent lift-over rather than a hard-coded lift.
        """
        obj_names = [o.name for o in objects]
        obj_center = {o.name: o.pos for o in objects}   # geometric body centre (the grasp pose, not a kp)
        obj_ext = {o.name: o.extents for o in objects}

        def kp_point(k):
            return lambda k=k: tracker.get_positions()[k]

        def subgoal_done(subgoal):
            """The VLM sub-goal is satisfied at the current sensed state (its own advance predicate).

            Held keypoints are already carried by the world model, so the live tracker positions give the
            true current constraint value.
            """
            eps = self.subgoal_eps

            def _d():
                ee = torch.as_tensor(env.tcp(), device=dev, dtype=torch.float32).reshape(1, 1, 3)
                kp = torch.as_tensor(tracker.get_positions(), device=dev, dtype=torch.float32)[:, None, None, :]
                return float(torch.as_tensor(subgoal(ee, kp)).reshape(-1)[0]) < eps
            return _d

        def place_target_for(stage_idx, payload_owner):
            """The object the place sub-goal references, other than the payload (collision-excluded
            and the carry-hold destination)."""
            refs = _referenced_kps(os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt"))
            for k in refs:
                owner = tracker.owners[k] if k < len(tracker.owners) else None
                if owner is not None and owner != payload_owner and owner in obj_names:
                    return owner
            return None

        def place_mode_for(stage_idx, place_target):
            """container when the VLM puts the payload inside the destination, so the set-down terms
            (which aim at the destination's top, its rim) stand down and the sub-goal owns the drop."""
            path = os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt")
            if _places_into(path, place_target):
                print(f"[rekep-vlm] stage {stage_idx + 1} places into {place_target}, container mode",
                      flush=True)
                return "container"
            return "surface"

        def orient_for(stage_idx):
            """free when this stage's VLM sub-goal commands a rotation, so the downward tool-axis prior
            stands down. down (the default grasp-approach convenience) otherwise."""
            path = os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt")
            if _constrains_orientation(path):
                print(f"[rekep-vlm] stage {stage_idx + 1} constrains orientation, free tool axis", flush=True)
                return "free"
            return "down"

        def self_displace_next(grasp_i, owner):
            """True when the stage after the grasp displaces the just-grasped object's own keypoint with
            no external destination (an articulation such as opening a lid), rather than transporting it
            onto another object. Signals a press contact rather than a straddle-pinch (the GT capsule
            rung)."""
            j = grasp_i + 1
            if j >= metadata["num_stages"] or metadata["grasp_keypoints"][j] >= 0:
                return False                          # next is another grasp, not a displacement
            return metadata["release_keypoints"][j] >= 0 and place_target_for(j, owner) is None

        stages, manipulated, grasped_body, pressed = [], set(), None, False
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                owner = tracker.owners[grasp_kp]
                if self.contact_criterion == "plan":   # opt-out: the old plan-structural inference
                    press = self_displace_next(i, owner)
                    print(f"[rekep-vlm] {name}: contact={'press' if press else 'pinch'} (plan structure)",
                          flush=True)
                else:
                    mode, grip = self._contact_for(name, obj_ext)
                    press = mode == "press"
                    print(f"[rekep-vlm] {name}: contact={mode} (grip half-width {grip * 1e3:.0f}mm vs "
                          f"{self.open_half * 1e3:.0f}mm aperture)", flush=True)
                manipulated.add(name)
                # Grasp is execution. Fits the aperture: centre on the body and straddle-pinch (the grasp
                # sensor certifies the hold). Too big to take as a unit: press-contact the VLM keypoint,
                # the only place the VLM said to touch, with no pinch-certification, advancing on contact
                # so the next stage's sub-goal owns what the contact achieves.
                stages.append(Stage(name=f"{'press' if press else 'grasp'} {name}", gripper="close",
                                    grasp_obj=name, payload=None, held_idx=held,
                                    target=(kp_point(grasp_kp) if press else obj_center[name]),
                                    contact=("press" if press else "pinch")))
                grasped_body = owner
                pressed = press           # the displacement that follows is still a press-contact hold
            elif release_kp >= 0:
                name = name_for(release_kp)
                place_target = place_target_for(i, grasped_body)
                manipulated.update({name} | ({place_target} if place_target else set()))
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name,
                                    place_target=place_target, target=kp_point(release_kp), held_idx=held,
                                    constraint=subgoal, path_fns=path_fns, done=subgoal_done(subgoal),
                                    orient=orient_for(i),
                                    place_mode=place_mode_for(i, place_target),
                                    contact=("press" if pressed else "pinch")))
                grasped_body = None
                pressed = False
            else:                                 # a move or hold sub-goal with no grasp or release event
                refs = _referenced_kps(os.path.join(vlm_dir, f"stage{i + 1}_subgoal_constraints.txt"))
                # pressed survives this stage: the contact mode belongs to what the hand is holding, so it
                # runs from the grasp to the release. Left off, a move after a press demands a certified
                # pinch the press can never produce and backtracks every step.
                stages.append(Stage(name=f"move {i}", gripper=("hold" if grasped_body else "open"),
                                    grasp_obj=None, payload=grasped_body,
                                    target=kp_point(refs[0] if refs else 0), held_idx=held,
                                    constraint=subgoal, path_fns=path_fns, done=subgoal_done(subgoal),
                                    orient=orient_for(i),
                                    contact=("press" if pressed else "pinch")))

        tcp_probe = torch.as_tensor(keypoints, device=dev, dtype=torch.float32)[:1].reshape(1, 1, 3)
        kps_probe = torch.as_tensor(keypoints, device=dev, dtype=torch.float32)
        for st in stages:                         # a bad keypoint index fails here, not mid-rollout
            for fn in filter(None, (st.constraint, *st.path_fns)):
                try:
                    fn(tcp_probe, kps_probe)
                except Exception as exc:
                    raise SystemExit(f"[rekep-vlm] stage '{st.name}' constraint failed to evaluate: {exc}")
        self._check_keypoint_identity(metadata["num_stages"], vlm_dir, tracker, obj_names)
        print(f"[rekep-vlm] emitted {len(stages)} VLM-driven stages: "
              f"{[s.name for s in stages]}", flush=True)
        return stages, manipulated

    def _check_keypoint_identity(self, num_stages, vlm_dir, tracker, obj_names):
        """Cross-check the VLM's stated keypoint identities against the tracker's owner assignment.

        Two independent sources: the constraint prose says which object each keypoint is on, the tracker
        assigns owners by nearest object centre. Agreement is evidence the grounding is real, and
        disagreement means the arm is about to be steered at something other than what the VLM named.
        Reported, not fatal: the tracker's radius rule is the weaker of the two and legitimately declines
        on parts, so a hard failure here would reject good plans.
        """
        disagreements = []
        for i in range(num_stages):
            path = os.path.join(vlm_dir, f"stage{i + 1}_subgoal_constraints.txt")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                claims = keypoint_claims(f.read(), obj_names)
            for kp, claimed in claims.items():
                owner = tracker.owners[kp] if 0 <= kp < len(tracker.owners) else None
                if owner is not None and owner != claimed:
                    disagreements.append(f"stage{i + 1} kp{kp}: VLM says {claimed!r}, tracker says {owner!r}")
        if disagreements:
            print("[rekep-vlm] keypoint identity disagreement (the plan may target the wrong object): "
                  + ", ".join(disagreements), flush=True)

    def _real_constraints(self, vlm_dir, grounded, config):
        """Query the real GPT-4o constraint generator. Writes the same artifacts as the fake stub."""
        with open(os.path.join(_REPO, "task_prompts.json"), encoding="utf-8") as f:
            instruction = json.load(f)[self.task_key]["prompt"]
        img = grounded["projected"]
        if self.perception is not None:
            # The segmenter already knows each region's name. Withholding it makes the VLM re-derive
            # identity from anonymous dots, its dominant failure mode of grabbing the wrong object.
            img = _annotate_object_names(img, self.perception.masks)
            instruction += " (object names are written on the image)"
        ConstraintGenerator(config["constraint_generator"]).generate(img, instruction, {}, vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), encoding="utf-8") as f:
            return json.load(f)
