"""Build ReKep grounding from tracked keypoints and stage constraints."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from rekep import grounding as rk_grounding
# ConstraintGenerator (real-VLM codegen, needs `parse` + `openai`) and KeypointTracker are imported
# lazily where used: the artifact path -- manually authored or previously generated constraints --
# must load without the live-VLM and perception dependencies installed.
from rekep.utils import get_callable_grasping_cost_fn, load_default_config
from vlm_dp.grounding import Grounding, SceneObject, Stage, fake_vlm, masks

from .. import paths
from .gt import CP_EXTENTS, EXTENTS, HC_EXTENTS, MC_EXTENTS, TASKS


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
_RESTS_ON_SUPPORT = 0.03
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Per-task extent tables: mug_cleanup's mug lives in MC_EXTENTS, coffee_prep's in CP_EXTENTS,
# hammer_cleanup's in HC_EXTENTS. Reading only EXTENTS left those objects with no synthetic cloud,
# so the template builder raised "no grasp centre for 'mug'".
# gt.py names some parts by role while MuJoCo names them by hierarchy; without the alias the
# object has no pose, hence no cloud, hence "perception found no cloud for it".
_BODY_ALIASES = {
    "coffee_pod_holder": ("coffee_machine_pod_holder_holder", "coffee_machine_pod_holder"),
    "coffee_machine": ("coffee_machine_body",),
}

_ALL_EXTENTS = {**EXTENTS, **MC_EXTENTS, **HC_EXTENTS, **CP_EXTENTS,
                # The drawer's sliding front. Tracked as a movable so rollout telemetry records
                # the articulated state, so it also needs a box for the synthetic cloud.
                "drawer_link": (0.12, 0.12, 0.04)}


def _box_points_fn(world, per_axis=7):
    """Return points_of(name) -> a box point cloud from the object's live pose and extents."""
    if world is None:
        return None
    grid = np.linspace(-1.0, 1.0, per_axis)
    unit = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)

    def points_of(name):
        half = _ALL_EXTENTS.get(name)
        if half is None:
            return None
        pos = rot = None
        for candidate in (name,) + _BODY_ALIASES.get(name, ()):
            try:
                pos, rot = world.object_pose(candidate)[:2]
                break
            except Exception:
                continue
        if pos is None:
            return None
        local = unit * np.asarray(half, dtype=np.float64)
        return np.asarray(pos, dtype=np.float64) + local @ np.asarray(rot, dtype=np.float64).T

    return points_of


def load_rekep_context(path, world=None):
    """Load a written rekep_context.json into the dict `propose_keypoints` would have returned.

    The artifact stores each keypoint as {owner, offset_local, world_at_capture} -- i.e. RIGIDLY
    ATTACHED to a named body, not as a frozen world point. `world_at_capture` came from whatever
    episode wrote the artifact, so replaying it verbatim would target where the objects USED to be;
    with `world` supplied, each owned keypoint is recomputed as owner_pos + owner_rot @ offset_local
    against the current scene. Unowned keypoints keep their captured position.

    This is the standing keypoint assumption: rigid attachment to known objects, no visual
    re-detection. Point tracking is deliberately out of scope.

    Camera calibration (fovy, focal length, pose, near/far) rides along for later image-space work;
    only `keypoints` and `projected` are consumed downstream.
    """
    if not path:
        raise SystemExit("[rekep] kp_source=artifact needs --rekep_context <rekep_context.json>")
    if not os.path.exists(path):
        raise SystemExit(f"[rekep] no rekep context at {path}")
    with open(path) as fh:
        raw = json.load(fh)

    entries = raw["keypoints"]
    points, owners, rebased = [], [], 0
    for entry in entries:
        if not isinstance(entry, dict):                      # plain [x, y, z] form
            points.append(np.asarray(entry, dtype=np.float64))
            owners.append(None)
            continue
        owner = entry.get("owner")
        captured = np.asarray(entry["world_at_capture"], dtype=np.float64)
        offset = np.asarray(entry.get("offset_local", (0.0, 0.0, 0.0)), dtype=np.float64)
        if owner and world is not None:
            try:
                pos, rot = world.object_pose(owner)[:2]
                points.append(np.asarray(pos, dtype=np.float64) + np.asarray(rot) @ offset)
                rebased += 1
            except Exception:                                # owner absent from this scene
                points.append(captured)
        else:
            points.append(captured)
        owners.append(owner)

    keypoints = np.stack(points) if points else np.zeros((0, 3))
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise SystemExit(f"[rekep] context keypoints must resolve to [N, 3], got "
                         f"{keypoints.shape} in {path}")
    print(f"[rekep] loaded {len(keypoints)} keypoints from {path} "
          f"({rebased} rebased onto live object poses, owners={owners})", flush=True)
    return {
        "keypoints": keypoints,
        "owners": owners,
        # masks._masked_points prefers this hook over IsaacLab's scene/prim lookup, which does not
        # exist under MuJoCo. Synthesising the cloud from the live pose + known extents gives the
        # seat/centroid/local-grasp geometry downstream something real to measure.
        "points_of": _box_points_fn(world),
        "projected": raw.get("projected"),
        "names": raw.get("names"),
        # The ladder the generator walks: stage -> target keypoint, plus intents and payloads.
        "stage_keypoints": raw.get("stage_keypoints"),
        "stage_names": raw.get("stage_names"),
        "stage_gripper": raw.get("stage_gripper"),
        "stage_payload": raw.get("stage_payload"),
        "camera": raw.get("camera"),
        "masks": None,
        "points": None,
        "id_to_prim": raw.get("id_to_prim"),
    }


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
                 lift_latch_xy: bool = False, seat_from_plane: bool = False,
                 context_path: str | None = None, constraints_path: str | None = None):
        # kp_source="artifact" replays these instead of proposing from a live camera.
        self.context_path = context_path
        self.constraints_path = constraints_path
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
        # MuJoCoEnv has no .device (an IsaacLab/torch attribute); everything here is CPU there.
        self._env_for_vlm = env
        dev = getattr(env, "device", "cpu")
        config = load_default_config()
        if self.kp_source == "artifact":
            # Replay a previously written rekep_context.json instead of proposing keypoints from a
            # live camera. propose_keypoints needs IsaacLab perception, which does not exist in the
            # MuJoCo eval env, so this is the only path by which manually authored or
            # previously generated ReKep artifacts can reach the planner here.
            grounded = load_rekep_context(self.context_path, world)
        else:
            grounded = rk_grounding.propose_keypoints(
                env.cam, env.env, config, perception=self.perception)
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
        # world.names is the task's `movable` set, which often omits the destination (coffee tracks
        # only the pod, not the holder). The seat/extent loop below then measures nothing for it and
        # the template builder raises "perception found no cloud". Add the declared roles.
        # Only roles we can LOCALIZE: a SceneObject must answer pos(), and that comes from the
        # object's own keypoint, which only exists if it has a cloud. can's bin2_q3 has no body at
        # all, so adding it unconditionally crashed context-build on the first replan.
        _spec = TASKS.get(self.task_key, {})
        for _role in list(_spec.get("grasp_objs", ())) + (
                [_spec["place_obj"]] if _spec.get("place_obj") else []):
            if _role and _role not in scene_objects:
                if masks._masked_points(grounded, env.env, _role) is None:
                    print(f"[rekep] role {_role!r} has no cloud; not a scene object", flush=True)
                    continue
                scene_objects.append(_role)
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
        # usd_extents needs `omni` (Isaac only). Under MuJoCo it returns nothing and every object
        # falls back to DEFAULT_EXTENT = 0.05 half-width, which exceeds open_half (0.04) -- so
        # _contact_for typed EVERY grasp as a "press" and the arm pressed cubeA (true half-width
        # 0.02) instead of pinching it. gt.EXTENTS carries the real numbers for these tasks.
        for _n in scene_objects:
            if _n in EXTENTS and _n not in extents:
                extents[_n] = EXTENTS[_n]
        from rekep.keypoint_tracking import KeypointTracker
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


        # Generated constraints are scratch. _REPO/results is root-owned in the container image,
        # so honour MUJOCO_EVAL_RESULTS when set (the mujoco eval always sets it) and fall back to
        # the repo only when it is not.
        _vlm_root = os.environ.get("MUJOCO_EVAL_RESULTS") or os.path.join(_REPO, "results")
        vlm_dir = os.path.join(_vlm_root, "vlm_mpc", "vlm_base",
                               f"vlm_query_{self.task_key}_p{os.getpid()}")


        if os.path.isdir(vlm_dir):
            for _f in os.listdir(vlm_dir):
                if _f.endswith("_constraints.txt") or _f == "metadata.json":
                    os.remove(os.path.join(vlm_dir, _f))
        if self.vlm == "fake":
            # Route to whichever fake VLM knows this task: vlm_dp.fake_vlm covers the Isaac
            # tasks, fake_rekep the mujoco ones. Both emit the same artifact layout, so the
            # loader below does not care which produced it.
            from . import fake_rekep
            gen = fake_rekep if self.task_key in fake_rekep.supported_tasks() else fake_vlm
            metadata, _ = gen.generate(self.task_key, vlm_dir, keypoints, grounded, env.env,
                                       self.clearance)
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
        for name in scene_objects:
            pts = masks._masked_points(grounded, env.env, name)
            if pts is None:
                continue
            gk = grasp_kp_of.get(name)


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
                err = np.linalg.norm(obj_pos(_n) - env.object_pose(_n)[0]) * 1000
                print(f"[rekep-dbg] {_n}: grasp_center={np.round(obj_pos(_n), 3)} "
                      f"gt={np.round(env.object_pose(_n)[0], 3)} err={err:.1f}mm", flush=True)

        objects = [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)), extents=extents.get(n, _DEFAULT_EXTENT),
                               axis=grasp_axis.get(n), grasp_extent=grasp_ext_of.get(n),
                               grasp_region=grasp_region_of.get(n))
                   for n in scene_objects]


        if self.perception is not None:
            for i, (centre, ext) in enumerate(self.perception.unexplained_obstacles(support_top)):
                objects.append(SceneObject(name=f"_obstacle{i}", pos=(lambda p=centre: p), extents=ext,
                                           axis=None, grasp_extent=None, grasp_region=None))
            if len(objects) > len(scene_objects):
                print(f"[rekep] {len(objects) - len(scene_objects)} unnamed obstacles from scene geometry",
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
                                                   load_stage, objects, env, dev, vlm_dir)
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

    def _vlm_stages(self, metadata, tracker, keypoints, name_for, load_stage, objects, env, dev, vlm_dir):
        """Build stages directly from VLM constraints."""
        obj_names = [o.name for o in objects]
        obj_center = {o.name: o.pos for o in objects}
        obj_ext = {o.name: o.extents for o in objects}
        local_ext = {o.name: o.grasp_extent for o in objects if o.grasp_extent is not None}

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
            path = os.path.join(vlm_dir, f"stage{stage_idx + 1}_subgoal_constraints.txt")
            if _constrains_orientation(path):
                print(f"[rekep-vlm] stage {stage_idx + 1} constrains orientation, free tool axis", flush=True)
                return "free"
            return "down"

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
        for i in range(metadata["num_stages"]):
            grasp_kp, release_kp = metadata["grasp_keypoints"][i], metadata["release_keypoints"][i]
            held = tuple(j for j, o in enumerate(tracker.owners) if grasped_body is not None and o == grasped_body)
            subgoal, path_fns = load_stage(i, held)
            if grasp_kp >= 0:
                name = name_for(grasp_kp)
                owner = tracker.owners[grasp_kp]
                if self.contact_criterion == "plan":
                    press = self_displace_next(i, owner)
                    print(f"[rekep-vlm] {name}: contact={'press' if press else 'pinch'} (plan structure)",
                          flush=True)
                else:
                    mode, grip = self._contact_for(name, obj_ext, local_grip=local_ext.get(name))
                    press = mode == "press"
                    print(f"[rekep-vlm] {name}: contact={mode} (grip half-width {grip * 1e3:.0f}mm vs "
                          f"{self.open_half * 1e3:.0f}mm aperture)", flush=True)
                manipulated.add(name)


                # The authored stage-N subgoal ("reach the grasp keypoint") was loaded and then
                # dropped here, leaving the grasp stage with no constraint while the place stage
                # below carried one -- so rekep_subgoal/rekep_path scored nothing until the final
                # stage. `done` is deliberately NOT the subgoal: a grasp advances on the hold
                # sensor, not on proximity, or it would advance before the gripper closes.
                stages.append(Stage(name=f"{'press' if press else 'grasp'} {name}", gripper="close", steer_policy=_pol(i),
                                    grasp_obj=name, payload=None, held_idx=held,
                                    target=(kp_point(grasp_kp) if press else obj_center[name]),
                                    constraint=subgoal, path_fns=path_fns,
                                    contact=("press" if press else "pinch")))
                grasped_body = owner
                pressed = press
            elif release_kp >= 0:
                name = name_for(release_kp)
                place_target = place_target_for(i, grasped_body)
                manipulated.update({name} | ({place_target} if place_target else set()))
                stages.append(Stage(name=f"place {name}", gripper="place", grasp_obj=None, payload=name, steer_policy=_pol(i),
                                    place_target=place_target, target=kp_point(release_kp), held_idx=held,
                                    constraint=subgoal, path_fns=path_fns, done=subgoal_done(subgoal),
                                    orient=orient_for(i),
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
                                    orient=orient_for(i),
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
        img = grounded.get("projected")
        if img is None:
            # Artifact path: no stored overlay. ReKep prompts with a NUMBERED-keypoint image, so
            # render one from the live scene -- the same frame mujoco_eval.viz draws for debugging.
            from .. import viz
            img = viz.annotate_keypoints(self._env_for_vlm, grounded["keypoints"], hw=512)
        if self.perception is not None:


            img = _annotate_object_names(img, self.perception.masks)
            instruction += " (object names are written on the image)"
        from rekep.constraint_generation import ConstraintGenerator
        ConstraintGenerator(config["constraint_generator"]).generate(img, instruction, {}, vlm_dir)
        with open(os.path.join(vlm_dir, "metadata.json"), encoding="utf-8") as f:
            return json.load(f)


class MGRekepGroundingSource:
    """ReKep grounding for mujoco_eval, replayed from written artifacts.

    The `--ground rekep` source. Mirrors MGGroundingSource's interface (`movable`, `roles`,
    `ground(env, world)`) so the runner, bridge, costs and steering modes need no branch: whichever
    source is built, the planner receives the same Grounding object.

    Keypoints and camera calibration come from `rekep_context.json`, constraints from a
    `rekep_constraints/` directory. Nothing here touches a live camera, a segmenter or a VLM, so it
    runs under the MuJoCo eval env where none of those are installed.

    KEYPOINT ASSUMPTION: keypoints are loaded once per episode and then ride their owning bodies
    through KeypointTracker. There is no visual re-detection -- point tracking is deliberately out
    of scope for now.
    """

    _MODE = "template"

    def __init__(self, task, context_path=None, constraints_path=None, vlm="fake"):
        if task not in TASKS:
            raise ValueError(f"unknown mg task {task!r} (have {sorted(TASKS)})")
        spec = TASKS[task]
        self.task = task
        self.movable = list(spec["movable"])
        self.roles = {"grasp_obj": spec["grasp_objs"][0], "grasp_objs": spec["grasp_objs"],
                      "place_obj": spec["place_obj"]}
        self.context_path = context_path or self._default(task, "rekep_context.json")
        self.constraints_path = constraints_path or self._default(task, "rekep_constraints")
        self._grounding = RekepGrounding(
            task_key=task, place_obj=spec["place_obj"],
            grasp_objs=spec["grasp_objs"], stages=self._MODE, kp_source="artifact", vlm=vlm,
            context_path=self.context_path, constraints_path=self.constraints_path)

    @staticmethod
    def _default(task, leaf):
        """Artifacts live beside the task's demos, as grounding/propose.py writes them."""
        return str(paths.task_data(task, leaf))

    def ground(self, env, world):
        return self._grounding.ground(env, world)


class MGRekepVlmGroundingSource(MGRekepGroundingSource):
    """`--ground rekep_vlm`: stages emitted from the VLM metadata rather than the task ladder.

    Same artifacts and the same interface; only the stage-construction mode differs, so the two
    sources stay a one-word change rather than parallel implementations.
    """

    _MODE = "vlm"
