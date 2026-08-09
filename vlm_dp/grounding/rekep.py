"""Build ReKep grounding from tracked keypoints and stage constraints."""
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


from vlm_dp.grounding.gt import _LIFT_HEIGHT as _GT_LIFT, _shifted_seat
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, center_from_points, usd_extents
from vlm_dp.sim_helpers import TorchNumpyShim, load_torch_constraints, make_torch_constraint

_PLACE_HOVER = (0.0, 0.0, 0.10)


_LIFT_HEIGHT = 0.15
_LIFT_CONFIRM = 0.05


_PLACE_MARGIN = 0.02
_REST_TOL = 0.10

_SEAT_BAND = 0.005
_MIN_LOCAL_PTS = 20
_MIN_GRASP_EXT = 0.012

# Radius of the cloud patch used to measure an object's width at its grasp keypoint.
_GRASP_PROBE_R = 0.03
# A segmentation sanity bound: how much wider than its own USD bounding box an object's observed
# cloud may be before we call the mask wrong rather than the object big.
_GROSS_EXT_FACTOR, _GROSS_EXT_PAD = 3.0, 0.05
# How far a grasp keypoint may sit from the nearest point of the object it is supposed to be on.
_KP_ON_OBJECT = 0.10
_RESTS_ON_SUPPORT = 0.03
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _annotate_object_names(img, name_masks):
    """Draw object names at mask centroids."""
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
    """Return keypoint indices referenced by a constraint file."""
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, encoding="utf-8") as f:
        src = f.read()
    return sorted({int(m) for m in re.findall(r"keypoints\[(\d+)\]", src)})


_ROTATION_TOKENS = ("arccos", "arctan", "cross(", "angle", "upright", "tilt", "parallel", "perpendicular")


def _places_into(txt_path, place_target):
    """Return whether a subgoal places the payload inside the destination."""
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
    """Map keypoint indices to object names stated in constraint text."""
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
    """Return whether a subgoal constrains orientation."""
    if not os.path.exists(txt_path):
        return False
    with open(txt_path, encoding="utf-8") as f:
        src = f.read().lower()
    return any(tok in src for tok in _ROTATION_TOKENS)


class RekepGrounding:
    """Build ReKep grounding from tracked keypoints and relational constraints."""

    def __init__(self, vlm: str = "fake", task_key: str | None = None, place_obj: str | None = None,
                 clearance: float = 0.015, perception=None, seat_shift: bool = True,
                 grasp_objs=None, support=None, stages: str = "template", subgoal_eps: float = 0.06,
                 local_grasp: bool = False, local_grasp_radius: float = 0.05,
                 kp_source: str = "perception", contact_criterion: str = "feasibility",
                 open_half: float = 0.04, rotate_grasp_offset: bool = False,
                 lift_latch_xy: bool = False, seat_from_plane: bool = False):
        self.vlm = vlm
        self.task_key = task_key
        self.place_obj = place_obj
        self.grasp_objs = list(grasp_objs) if grasp_objs else []
        self.support = support
        self.clearance = clearance
        self.perception = perception
        self.seat_shift = seat_shift


        self.local_grasp = bool(local_grasp)
        self.local_grasp_radius = float(local_grasp_radius)


        self.kp_source = kp_source


        self.stages = stages
        self.subgoal_eps = float(subgoal_eps)


        self.contact_criterion = contact_criterion
        self.open_half = float(open_half)


        self.rotate_grasp_offset = bool(rotate_grasp_offset)
        self.lift_latch_xy = bool(lift_latch_xy)
        self.seat_from_plane = bool(seat_from_plane)

    def ground(self, env, world) -> Grounding:


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


            if self.task_key == "capsule":
                from vlm_dp.grounding.capsule import gt_keypoints as _gt_kps
                keypoints, gt_meta = _gt_kps(env)
            else:
                from vlm_dp.grounding.gt_points import GT_KEYPOINTS
                if self.task_key not in GT_KEYPOINTS:
                    raise SystemExit(
                        f"[rekep-grounding] kp_source=gt has no keypoints for task {self.task_key!r} "
                        f"(have: capsule, {sorted(GT_KEYPOINTS)})")
                keypoints, gt_meta = GT_KEYPOINTS[self.task_key](env.env, grounded)
            grounded["keypoints"] = keypoints
            grounded["gt_meta"] = gt_meta
            virtual_kps = gt_meta["virtual"]
            print(f"[rekep-gt] injected {len(keypoints)} GT keypoints "
                  f"(owners={gt_meta['owners']}, virtual={sorted(virtual_kps)})", flush=True)
        if len(keypoints) == 0:
            raise SystemExit("[rekep-grounding] no keypoints proposed")
        scene_objects = list(world.names)
        if self.perception is not None:

            extents = {n: e for n in scene_objects if (e := self.perception.object_extents(n)) is not None}
            usd = usd_extents(env, scene_objects)
            diag = os.environ.get("VLMDP_EXT_DIAG")
            for _n in scene_objects:
                if _n in extents and _n in usd:
                    print(f"[ext] {_n}: cloud(grip,keep,h)={tuple(round(x,3) for x in extents[_n])} "
                          f"usd={tuple(round(x,3) for x in usd[_n])}", flush=True)
                    if diag and (d := self.perception.extent_diagnostics(_n)) is not None:
                        print(f"[ext-diag] {_n}: {d}", flush=True)
        else:
            extents = usd_extents(env, scene_objects)
            usd = extents
        tracker = KeypointTracker(world, keypoints)
        if gt_meta is not None:


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


        vlm_dir = os.path.join(_REPO, "results", "vlm_mpc", "vlm_base",
                               f"vlm_query_{self.task_key}_p{os.getpid()}")


        if os.path.isdir(vlm_dir):
            for _f in os.listdir(vlm_dir):
                if _f.endswith("_constraints.txt") or _f == "metadata.json":
                    os.remove(os.path.join(vlm_dir, _f))
        roles, declared = {}, {}
        if self.vlm == "fake":
            metadata, roles, extra_kps = fake_vlm.generate(self.task_key, vlm_dir, keypoints, grounded,
                                                           env.env, self.clearance)
            if extra_kps:
                # Points the plan needs that no proposal stands for (a thin lid rim, a place in
                # mid-air, a recess inside a fixture). Appended after the tracker exists, so they
                # are registered here by hand rather than re-running the association.
                first = len(keypoints)
                keypoints = np.concatenate(
                    [np.asarray(keypoints, dtype=np.float64),
                     np.stack([np.asarray(p, dtype=np.float64) for p, _ in extra_kps])], axis=0)
                grounded["keypoints"] = keypoints
                static_declared = set(metadata.get("static_keypoints", ()))

                for point, owner in extra_kps:
                    point = np.asarray(point, dtype=np.float64)
                    kp_idx = len(tracker.owners)
                    if owner is None or kp_idx in static_declared:
                        tracker.registrations.append((None, point))
                    else:
                        _p, _r = world.object_pose(owner)
                        tracker.registrations.append((owner, _r.T @ (point - _p)))
                    declared[kp_idx] = owner
                    tracker.owners.append(owner)
                print(f"[rekep] fake VLM declared keypoints {list(range(first, len(keypoints)))} "
                      f"owners={[o for _, o in extra_kps]}", flush=True)
        else:
            metadata = self._real_constraints(vlm_dir, grounded, config)
        bad = [i for i in metadata["grasp_keypoints"] + metadata["release_keypoints"] if not -1 <= i < len(keypoints)]
        if bad:
            raise SystemExit(f"[rekep] VLM referenced out-of-range keypoint(s) {bad} (have {len(keypoints)})")


        support_top = None
        if self.support:
            spts = masks._masked_points(grounded, env.env, self.support)
            if spts is not None:
                support_top = float(np.percentile(spts[:, 2], 95))


        grasped_names = {tracker.owners[k] for k in metadata["grasp_keypoints"]
                         if 0 <= k < len(tracker.owners) and tracker.owners[k]}

        grasp_kp_of = {}
        for k in metadata["grasp_keypoints"]:
            if 0 <= k < len(tracker.owners) and tracker.owners[k]:
                grasp_kp_of.setdefault(tracker.owners[k], k)
        def _obj_rot(kp):
            """Return the owner's rotation, or identity for an unowned keypoint."""
            owner = tracker.owners[kp] if 0 <= kp < len(tracker.owners) else None
            return world.object_pose(owner)[1] if owner is not None else np.eye(3)

        def _register_off(kp, vec):
            """Store an offset in the owner frame when rotation tracking is enabled."""
            return (_obj_rot(kp).T @ vec) if self.rotate_grasp_offset else np.asarray(vec)

        def _read_off(kp, off):
            return (_obj_rot(kp) @ off) if self.rotate_grasp_offset else off

        kp_of, centroid_off, grasp_axis, grasp_ext_of, seat_off = {}, {}, {}, {}, {}
        grasp_region_of = {}
        clouds, probe_ext = {}, {}
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is None:
                continue
            clouds[name] = pts
            gk = grasp_kp_of.get(name)

            # What the gripper actually closes on: the object's width *at the grasp point*, not
            # the width of the whole object. A teapot is 96mm across and its handle is not; a
            # coffee machine is 166mm across and its lid rim is not. Measured unconditionally
            # (it is a few hundred distance computations) but only consulted when the
            # whole-object width would rule a graspable feature un-graspable.
            if gk is not None:
                near = pts[np.linalg.norm(pts - keypoints[gk], axis=1) <= _GRASP_PROBE_R]
                if near.shape[0] >= _MIN_LOCAL_PTS:
                    lo, hi = np.percentile(near[:, :2], [5, 95], axis=0)
                    probe_ext[name] = max(float(np.min(hi - lo)) / 2.0, _MIN_GRASP_EXT)


            obj_support = support_top
            if obj_support is None and self.perception is not None and name in grasped_names:
                obj_support = self.perception.support_height(name)
            if self.local_grasp and gk is not None:


                near = pts[np.linalg.norm(pts - keypoints[gk], axis=1) <= self.local_grasp_radius]
                src = near if near.shape[0] >= _MIN_LOCAL_PTS else pts
                lo, hi = np.percentile(src[:, :2], [5, 95], axis=0)
                grasp_ext_of[name] = max(float(np.min(hi - lo)) / 2.0, _MIN_GRASP_EXT)
                rests = (obj_support is not None
                         and float(np.percentile(src[:, 2], 5)) - obj_support < _RESTS_ON_SUPPORT)
                grasp_center = center_from_points(src, obj_support if rests else None)
                kp = gk
                print(f"[rekep-ground] {name}: local_pts={near.shape[0]}/{pts.shape[0]} "
                      f"src={'local' if near.shape[0] >= _MIN_LOCAL_PTS else 'FALLBACK-whole-object'} "
                      f"ext={grasp_ext_of[name]:.3f} center={np.round(grasp_center, 3)} "
                      f"kp={np.round(keypoints[gk], 3)}", flush=True)
            else:
                src = pts
                kp = masks._nearest_kp(keypoints, pts.mean(axis=0))
                grasp_center = center_from_points(pts, obj_support if name in grasped_names else None)
            kp_of[name], centroid_off[name] = kp, _register_off(kp, grasp_center - keypoints[kp])
            if name == self.place_obj:


                cap = pts[pts[:, 2] >= np.percentile(pts[:, 2], 90)]
                cap_xy = (cap[:, :2].min(axis=0) + cap[:, :2].max(axis=0)) / 2.0
                top_z = float(np.percentile(pts[:, 2], 95))
                if self.seat_from_plane:


                    zs = pts[:, 2]
                    span = float(zs.max() - zs.min())
                    if span > _SEAT_BAND:
                        hist, edges = np.histogram(zs, bins=max(8, int(np.ceil(span / _SEAT_BAND))))
                        k = int(np.argmax(hist))
                        top_z = float((edges[k] + edges[k + 1]) / 2.0)
                        band = pts[np.abs(zs - top_z) <= _SEAT_BAND]
                        if band.shape[0] >= _MIN_LOCAL_PTS:
                            cap_xy = (band[:, :2].min(axis=0) + band[:, :2].max(axis=0)) / 2.0
                seat_off[name] = _register_off(
                    kp, np.array([cap_xy[0], cap_xy[1], top_z], dtype=np.float64) - keypoints[kp])
            if name in grasped_names:
                grasp_axis[name] = masks.narrow_axis(src)
                print(f"[grasp-axis] {name}: {grasp_axis[name]}", flush=True)


                n = grasp_axis[name]
                if n is not None:
                    long_axis = np.array([-float(n[1]), float(n[0]), 0.0], dtype=np.float64)
                    t = (src[:, :2] - grasp_center[:2]) @ long_axis[:2]
                    half_len = float(np.percentile(np.abs(t), 80))
                    grasp_region_of[name] = (long_axis.tolist(), half_len)
                    print(f"[grasp-region] {name}: half_len={half_len:.3f}m along {np.round(long_axis, 2)}",
                          flush=True)

        # A declared grasp keypoint IS the grasp point (a lid rim, not the machine's centroid), so
        # it overrides the cloud-centroid estimate its owner would otherwise be grasped at. Same
        # rule the gt_meta block below applies to privileged keypoints.
        for k, owner in declared.items():
            if owner is not None and k in metadata["grasp_keypoints"]:
                kp_of[owner], centroid_off[owner] = k, np.zeros(3)

        for owner, geometry in (metadata.get("grasp_geometry") or {}).items():
            axis = np.asarray(geometry.get("axis"), dtype=np.float64)
            if axis.shape != (3,) or float(np.linalg.norm(axis)) < 1e-6:
                raise ValueError(f"invalid declared grasp axis for {owner!r}: {axis}")
            axis = axis / np.linalg.norm(axis)
            extent = float(geometry.get("extent", 0.0))
            if not 0.0 < extent <= self.open_half:
                raise ValueError(f"invalid declared grasp extent for {owner!r}: {extent}")
            grasp_axis[owner] = tuple(float(v) for v in axis)
            grasp_ext_of[owner] = extent
            print(
                f"[rekep-ground] {owner}: declared local grasp axis={np.round(axis, 3)} "
                f"half_width={extent * 1e3:.0f}mm",
                flush=True,
            )

        if gt_meta is not None:


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
            """Return the live tracked grasp center for an object."""
            kp = kp_of.get(name)
            if kp is None:
                raise KeyError(f"[rekep] no grasp centre for {name!r}: it has no keypoint of its own.")
            return tracker.get_positions()[kp] + _read_off(kp, centroid_off[name])

        def seat_point():
            """Return the live measured destination seat."""
            kp, off = kp_of.get(self.place_obj), seat_off.get(self.place_obj)
            if kp is None or off is None:
                raise KeyError(f"[rekep] no measured seat for place object {self.place_obj!r}: "
                               "perception found no cloud for it.")
            return tracker.get_positions()[kp] + _read_off(kp, off)

        for _n in scene_objects:
            if _n in kp_of:
                try:
                    _gt = env.object_pose(_n)[0]
                except Exception:
                    # A named distractor need not be a simulator entity (it exists only to give
                    # the segmenter somewhere to put a competing mask). No truth to compare to.
                    print(f"[rekep-dbg] {_n}: grasp_center={np.round(obj_pos(_n), 3)} "
                          f"(no simulator pose, distractor only)", flush=True)
                    continue
                err = np.linalg.norm(obj_pos(_n) - _gt) * 1000
                print(f"[rekep-dbg] {_n}: grasp_center={np.round(obj_pos(_n), 3)} "
                      f"gt={np.round(_gt, 3)} err={err:.1f}mm", flush=True)

        tracked_scene_objects = [n for n in scene_objects if n in kp_of]
        missing_geometry = [n for n in scene_objects if n not in kp_of]
        if missing_geometry:
            print(f"[rekep] ignoring perceived distractors without usable point geometry: "
                  f"{missing_geometry}", flush=True)
        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)), extents=extents.get(n, _DEFAULT_EXTENT),
                               axis=grasp_axis.get(n), grasp_extent=grasp_ext_of.get(n),
                               grasp_region=grasp_region_of.get(n))
                   for n in tracked_scene_objects]


        if self.perception is not None:
            for i, (centre, ext) in enumerate(self.perception.unexplained_obstacles(support_top)):
                objects.append(SceneObject(name=f"_obstacle{i}", pos=(lambda p=centre: p), extents=ext,
                                           axis=None, grasp_extent=None, grasp_region=None))
            if len(objects) > len(tracked_scene_objects):
                print(f"[rekep] {len(objects) - len(tracked_scene_objects)} unnamed obstacles from scene geometry",
                      flush=True)

        def load_stage(idx, held):
            """Load a stage's subgoal and path constraints."""
            grasp_fn = get_callable_grasping_cost_fn(list(held))
            subgoal = make_torch_constraint(load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_subgoal_constraints.txt"), grasp_fn, shim))
            path_fns = tuple(make_torch_constraint([c]) for c in load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_path_constraints.txt"), grasp_fn, shim))
            return subgoal, path_fns

        def placed(name, seat_fn):
            """Return a predicate for resting on the measured destination."""
            def _done():
                seat = np.asarray(seat_fn(), dtype=np.float64)
                obj = obj_pos(name)
                foot = extents.get(self.place_obj, _DEFAULT_EXTENT)[1]
                near_xy = float(np.linalg.norm(obj[:2] - seat[:2])) < foot + _PLACE_MARGIN
                seat_z = float(seat[2]) + extents.get(name, _DEFAULT_EXTENT)[2]
                return bool(near_xy and abs(float(obj[2]) - seat_z) < _REST_TOL)
            return _done

        def name_for(kp_idx):
            """Resolve the object that owns a keypoint."""
            return tracker.owners[kp_idx] or masks.object_for_keypoint(
                grounded, env.env, keypoints[kp_idx], names=tuple(scene_objects))

        def seat_for(nm, prior):
            """Return a live seat that avoids previously placed objects."""
            def _p():
                seat = np.asarray(seat_point(), dtype=np.float64)
                others = [(np.asarray(obj_pos(o), dtype=np.float64),
                           extents.get(o, _DEFAULT_EXTENT)[1]) for o in prior]
                pext = extents.get(self.place_obj, _DEFAULT_EXTENT)
                h = (float(pext[0]) + float(pext[1])) / 2.0
                return _shifted_seat(seat, extents.get(nm, _DEFAULT_EXTENT)[1], others,
                                     footprint=(seat[:2], (h, h)))
            return _p


        if self.stages == "vlm":
            stages, manipulated = self._vlm_stages(metadata, tracker, keypoints, name_for,
                                                   load_stage, objects, env, dev, vlm_dir,
                                                   probe_ext, clouds, usd, roles)
            return Grounding(objects=objects, stages=stages, manipulated=frozenset(manipulated),
                             keypoints=(lambda: tracker.get_positions()))

        stages, manipulated, grasped_body = [], {self.place_obj}, None
        placed_names, last_z0 = [], None
        steer_policies = metadata.get("steer_policies") or []
        def _pol(idx):
            return steer_policies[idx] if idx < len(steer_policies) else None
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                _alt = masks.object_for_keypoint(grounded, env.env, keypoints[grasp_kp], names=tuple(scene_objects))
                if _alt != name:
                    print(f"[rekep] grasp_kp={grasp_kp} -> {name} (nearest-surface would mis-say {_alt})", flush=True)
                manipulated.add(name)
                stages.append(Stage(name=f"grasp {name}", gripper="close", grasp_obj=name, payload=None, steer_policy=_pol(i),
                                    held_idx=held, target=(lambda nm=name: obj_pos(nm)),
                                    done_flag=f"grasp_{name}"))
                grasped_body = tracker.owners[grasp_kp]


                held_after = tuple(j for j, o in enumerate(tracker.owners) if o == grasped_body)


                z0 = float(obj_pos(name)[2])
                last_z0 = z0


                lift_cell = {}

                def lift_target(n=name, z=z0 + _LIFT_HEIGHT, cell=lift_cell):
                    if self.lift_latch_xy:
                        if "xy" not in cell:
                            cell["xy"] = np.asarray(obj_pos(n), dtype=np.float64)[:2].copy()
                        xy = cell["xy"]
                    else:
                        xy = np.asarray(obj_pos(n), dtype=np.float64)[:2]
                    return np.array([xy[0], xy[1], z], dtype=np.float64)

                stages.append(Stage(name=f"lift {name}", gripper="hold", grasp_obj=name, payload=name, steer_policy=_pol(i),
                                    held_idx=held_after, target=lift_target,
                                    on_enter=(lambda cell=lift_cell: cell.pop("xy", None)),
                                    done=(lambda n=name, z=z0: float(obj_pos(n)[2]) > z + _LIFT_CONFIRM)))
            elif release_kp >= 0:
                name = name_for(release_kp)
                manipulated.add(name)
                seat = seat_for(name, tuple(placed_names) if self.seat_shift else ())


                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name, steer_policy=_pol(i),
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
        for st in stages:
            for fn in filter(None, (st.constraint, *st.path_fns)):
                try:
                    fn(tcp_probe, kps_probe)
                except Exception as exc:
                    raise SystemExit(f"[rekep] stage '{st.name}' constraint failed to evaluate "
                                     f"(bad keypoint index?): {exc}")

        return Grounding(objects=objects, stages=stages, manipulated=frozenset(manipulated),
                         keypoints=(lambda: tracker.get_positions()))

    def _contact_for(self, name, extents, local_grip=None):
        """Choose pinch or press from the measured grasp width."""
        grip = (float(local_grip) if local_grip is not None
                else float(extents.get(name, _DEFAULT_EXTENT)[0]))
        return ("pinch", grip) if grip <= self.open_half else ("press", grip)

    def _effective_contact(self, name, obj_ext, local_ext, probe_ext):
        """Decide pinch vs press, falling back to the width at the grasp point."""
        mode, grip = self._contact_for(name, obj_ext, local_grip=local_ext.get(name))
        if mode == "press" and probe_ext.get(name) is not None:
            alt_mode, alt_grip = self._contact_for(name, obj_ext, local_grip=probe_ext[name])
            if alt_mode == "pinch":
                return alt_mode, alt_grip, f"width at the grasp point, r={_GRASP_PROBE_R:g}m"
        return mode, grip, "whole-object width"

    def _preflight(self, metadata, keypoints, name_for, obj_ext, local_ext, probe_ext,
                   usd_ext, clouds, roles):
        """Refuse to roll out on grounding the plan cannot act on.

        Every failure here used to be discovered ~50 minutes later as an inexplicably bad rollout.
        The checks are all on quantities the compiler has already computed, so healthy grounding
        pays nothing for them.
        """
        problems, claimed = [], {}
        for role, kp in sorted((roles or {}).items()):
            if not isinstance(kp, (int, np.integer)) or int(kp) < 0:
                continue
            kp = int(kp)
            if kp in claimed:
                problems.append(f"roles {claimed[kp]!r} and {role!r} both resolved to keypoint "
                                f"{kp} -- the plan would drive two different things to one point")
            claimed[kp] = role
            # A role named after an object has to land on that object. Without this the plan can
            # compile cleanly around a keypoint that belongs to something else entirely.
            pts = clouds.get(role)
            if pts is not None and 0 <= kp < len(keypoints):
                off = float(np.linalg.norm(pts - keypoints[kp], axis=1).min())
                if off > _KP_ON_OBJECT:
                    problems.append(f"role {role!r} resolved to keypoint {kp}, which is "
                                    f"{off * 1e3:.0f}mm off the nearest point of {role!r} itself")

        # Segmentation sanity, for every object at once: a mask that has run off its object onto
        # the scene shows up as a cloud far wider than the object's own bounding box.
        for name, pts in sorted(clouds.items()):
            usd = usd_ext.get(name)
            if usd is None or pts.shape[0] < _MIN_LOCAL_PTS:
                continue
            grip = float(obj_ext.get(name, _DEFAULT_EXTENT)[0])
            bound = max(_GROSS_EXT_FACTOR * float(usd[0]), float(usd[0]) + _GROSS_EXT_PAD)
            if grip > bound:
                problems.append(f"{name!r} segments {grip * 1e3:.0f}mm wide against a "
                                f"{float(usd[0]) * 1e3:.0f}mm USD half-width: the mask has run off "
                                f"the object onto the scene")

        for i, gk in enumerate(metadata["grasp_keypoints"]):
            if gk < 0:
                continue
            name = name_for(gk)
            pts = clouds.get(name)
            if name is None or pts is None or pts.shape[0] < _MIN_LOCAL_PTS:
                problems.append(f"stage {i + 1} grasps keypoint {gk}, whose object "
                                f"({name!r}) has no usable point cloud "
                                f"({0 if pts is None else int(pts.shape[0])} points): the segmenter "
                                f"did not find it, or the keypoint landed on nothing")
                continue
            span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            if span < 2 * _MIN_GRASP_EXT:
                problems.append(f"{name!r} (grasped by stage {i + 1}) has a degenerate cloud: "
                                f"largest span {span * 1e3:.0f}mm")
                continue
            off = float(np.linalg.norm(pts - keypoints[gk], axis=1).min())
            if off > _KP_ON_OBJECT:
                problems.append(f"stage {i + 1} grasp keypoint {gk} sits {off * 1e3:.0f}mm off the "
                                f"nearest point of {name!r} -- the role did not land on its object")
            forced_contact = (metadata.get("contact_modes") or {}).get(str(i))
            if forced_contact not in (None, "pinch", "press"):
                problems.append(f"stage {i + 1} declares unknown contact mode {forced_contact!r}")
            if self.contact_criterion != "plan" and forced_contact is None:
                mode, width, src = self._effective_contact(name, obj_ext, local_ext, probe_ext)
                if mode == "press":
                    problems.append(
                        f"{name!r} is grasped by stage {i + 1} but its grip half-width is "
                        f"{width * 1e3:.0f}mm ({src}) against a {self.open_half * 1e3:.0f}mm "
                        f"gripper aperture, so the stage would compile as a press: either the mask "
                        f"is wrong or the role resolved to the wrong object")
        if self.task_key == "pot":
            pot_pts, cover_pts = clouds.get("pot"), clouds.get("cover")
            if pot_pts is None or cover_pts is None:
                problems.append("pot preflight requires both 'pot' and 'cover' point clouds")
            else:
                pot_xy = np.median(pot_pts[:, :2], axis=0)
                cover_xy = np.median(cover_pts[:, :2], axis=0)
                lid_dxy = float(np.linalg.norm(cover_xy - pot_xy))
                pot_rim_z = float(np.percentile(pot_pts[:, 2], 95))
                cover_z = float(np.median(cover_pts[:, 2]))
                lid_dz = cover_z - pot_rim_z
                if lid_dxy > 0.10:
                    problems.append(
                        f"pot cover cloud is {lid_dxy * 1e3:.0f}mm from the pot centre; "
                        "the cover detector likely selected the robot or background")
                if not -0.05 <= lid_dz <= 0.08:
                    problems.append(
                        f"pot cover median is {lid_dz * 1e3:.0f}mm relative to the observed rim; "
                        "the pot/cover masks are geometrically inconsistent")
        if problems:
            raise ValueError("[rekep-preflight] grounding is not usable for this plan; refusing to "
                             "roll out:\n  - " + "\n  - ".join(problems))
        print(f"[rekep-preflight] OK: {len(claimed)} distinct roles, grasp objects "
              f"{sorted({name_for(k) for k in metadata['grasp_keypoints'] if k >= 0})} all "
              f"pinchable within the {self.open_half * 1e3:.0f}mm aperture", flush=True)

    def _vlm_stages(self, metadata, tracker, keypoints, name_for, load_stage, objects, env, dev, vlm_dir,
                    probe_ext=None, clouds=None, usd_ext=None, roles=None):
        """Build stages directly from VLM constraints."""
        obj_names = [o.name for o in objects]
        obj_center = {o.name: o.pos for o in objects}
        obj_ext = {o.name: o.extents for o in objects}
        local_ext = {o.name: o.grasp_extent for o in objects if o.grasp_extent is not None}
        probe_ext, clouds, usd_ext = probe_ext or {}, clouds or {}, usd_ext or {}
        self._preflight(metadata, keypoints, name_for, obj_ext, local_ext, probe_ext,
                        usd_ext, clouds, roles)

        def kp_point(k):
            return lambda k=k: tracker.get_positions()[k]

        def subgoal_done(subgoal):
            """Return a predicate that checks the current subgoal value."""
            eps = self.subgoal_eps

            def _d():
                ee = torch.as_tensor(env.tcp(), device=dev, dtype=torch.float32).reshape(1, 1, 3)
                kp = torch.as_tensor(tracker.get_positions(), device=dev, dtype=torch.float32)[:, None, None, :]
                return float(torch.as_tensor(subgoal(ee, kp)).reshape(-1)[0]) < eps
            return _d

        def target_done(target):
            """Return a sensed completion predicate for an explicit end-effector target."""
            eps = self.subgoal_eps

            def _d():
                return float(np.linalg.norm(
                    np.asarray(env.tcp(), dtype=np.float64)
                    - np.asarray(target(), dtype=np.float64)
                )) < eps
            return _d

        def place_target_for(stage_idx, payload_owner):
            """Return the non-payload object referenced by a place subgoal."""
            refs = _referenced_kps(os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt"))
            for k in refs:
                owner = tracker.owners[k] if k < len(tracker.owners) else None
                if owner is not None and owner != payload_owner and owner in obj_names:
                    return owner
            return None

        def place_mode_for(stage_idx, place_target):
            """Choose surface or container placement from the subgoal text."""
            path = os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt")
            if _places_into(path, place_target):
                print(f"[rekep-vlm] stage {stage_idx + 1} places into {place_target}, container mode",
                      flush=True)
                return "container"
            return "surface"

        def orient_for(stage_idx):
            """Choose whether the stage may control tool orientation."""
            axis = (metadata.get("approach_axes") or {}).get(str(stage_idx))
            if axis is not None:
                print(
                    f"[rekep-vlm] stage {stage_idx + 1} uses declared tool axis "
                    f"{np.round(axis, 3)}",
                    flush=True,
                )
                return "axis"
            path = os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt")
            if _constrains_orientation(path):
                print(f"[rekep-vlm] stage {stage_idx + 1} constrains orientation, free tool axis", flush=True)
                return "free"
            return "down"

        def approach_for(stage_idx):
            return (metadata.get("approach_axes") or {}).get(str(stage_idx))

        def orientation_scale_for(stage_idx):
            return float((metadata.get("approach_axis_scales") or {}).get(str(stage_idx), 1.0))

        def contact_slack_for(stage_idx):
            value = (metadata.get("contact_slack") or {}).get(str(stage_idx))
            return None if value is None else float(value)

        def rise_confirm_for(name):
            value = (metadata.get("rise_confirm") or {}).get(name)
            return None if value is None else float(value)

        def self_displace_next(grasp_i, owner):
            """Return whether the next stage displaces the grasped object's keypoint."""
            j = grasp_i + 1
            if j >= metadata["num_stages"] or metadata["grasp_keypoints"][j] >= 0:
                return False
            return metadata["release_keypoints"][j] >= 0 and place_target_for(j, owner) is None

        stages, manipulated, grasped_body, pressed = [], set(), None, False
        steer_policies = metadata.get("steer_policies") or []
        def _pol(idx):
            return steer_policies[idx] if idx < len(steer_policies) else None
        release_done_targets = {int(idx) for idx in metadata.get("release_done_targets", [])}
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                owner = tracker.owners[grasp_kp]
                forced_contact = (metadata.get("contact_modes") or {}).get(str(i))
                if forced_contact is not None:
                    press = forced_contact == "press"
                    print(f"[rekep-vlm] {name}: contact={forced_contact} (declared by plan)", flush=True)
                elif self.contact_criterion == "plan":
                    press = self_displace_next(i, owner)
                    print(f"[rekep-vlm] {name}: contact={'press' if press else 'pinch'} (plan structure)",
                          flush=True)
                else:
                    mode, grip, src = self._effective_contact(name, obj_ext, local_ext, probe_ext)
                    press = mode == "press"
                    print(f"[rekep-vlm] {name}: contact={mode} (grip half-width {grip * 1e3:.0f}mm vs "
                          f"{self.open_half * 1e3:.0f}mm aperture, {src})", flush=True)
                manipulated.add(name)


                stages.append(Stage(name=f"{'press' if press else 'grasp'} {name}", gripper="close", steer_policy=_pol(i),
                                    grasp_obj=name, payload=None, held_idx=held,
                                    target=(kp_point(grasp_kp) if press or
                                            (metadata.get("grasp_targets") or {}).get(str(i)) == "keypoint"
                                            else obj_center[name]),
                                    orient=orient_for(i), approach_axis=approach_for(i),
                                    orientation_scale=orientation_scale_for(i),
                                    grasp_slack=contact_slack_for(i),
                                    contact=("press" if press else "pinch")))
                grasped_body = owner
                pressed = press
            elif release_kp >= 0:
                name = name_for(release_kp)
                target_kp = int((metadata.get("release_targets") or {}).get(str(i), release_kp))
                if not 0 <= target_kp < len(tracker.owners):
                    raise ValueError(f"stage {i + 1} release target keypoint {target_kp} is invalid")
                target = kp_point(target_kp)
                place_target = place_target_for(i, grasped_body)
                manipulated.update({name} | ({place_target} if place_target else set()))
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name, steer_policy=_pol(i),
                                    place_target=place_target, target=target, held_idx=held,
                                    constraint=subgoal, path_fns=path_fns,
                                    done=(target_done(target) if i in release_done_targets else subgoal_done(subgoal)),
                                    orient=orient_for(i), approach_axis=approach_for(i),
                                    orientation_scale=orientation_scale_for(i),
                                    rise_confirm=rise_confirm_for(name),
                                    place_mode=place_mode_for(i, place_target),
                                    contact=("press" if pressed else "pinch")))
                grasped_body = None
                pressed = False
            else:
                refs = _referenced_kps(os.path.join(vlm_dir, f"stage{i + 1}_subgoal_constraints.txt"))


                stages.append(Stage(name=f"move {i}", gripper=("hold" if grasped_body else "open"), steer_policy=_pol(i),
                                    grasp_obj=None, payload=grasped_body,
                                    target=kp_point(refs[0] if refs else 0), held_idx=held,
                                    constraint=subgoal, path_fns=path_fns, done=subgoal_done(subgoal),
                                    orient=orient_for(i), approach_axis=approach_for(i),
                                    orientation_scale=orientation_scale_for(i),
                                    rise_confirm=rise_confirm_for(grasped_body),
                                    contact=("press" if pressed else "pinch")))

        tcp_probe = torch.as_tensor(keypoints, device=dev, dtype=torch.float32)[:1].reshape(1, 1, 3)
        kps_probe = torch.as_tensor(keypoints, device=dev, dtype=torch.float32)
        for st in stages:
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
        """Compare VLM keypoint claims with tracker ownership."""
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
        """Generate and load constraints with the real VLM."""
        with open(os.path.join(_REPO, "task_prompts.json"), encoding="utf-8") as f:
            instruction = json.load(f)[self.task_key]["prompt"]
        img = grounded["projected"]
        if self.perception is not None:


            img = _annotate_object_names(img, self.perception.masks)
            instruction += " (object names are written on the image)"
        ConstraintGenerator(config["constraint_generator"]).generate(img, instruction, {}, vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), encoding="utf-8") as f:
            return json.load(f)
