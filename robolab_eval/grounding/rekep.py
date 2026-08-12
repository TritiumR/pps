"""ReKep grounding for RoboLab: measured keypoints, VLM-emitted stages.

The RoboLab counterpart of mujoco_eval.grounding.rekep.MGRekepVlmGroundingSource, and it reuses
that module wherever the code is scene-agnostic -- the context loader, the constraint-text readers
and, most importantly, `RekepGrounding._vlm_stages`, which is the piece that decides what a stage
IS: its constraint, its path functions, its held keypoints and its `subgoal_done` advance
predicate. Inheriting it is the point: RoboLab and MimicGen then advance stages by the same rule,
and a difference in behaviour cannot be a difference in stage semantics.

WHY `ground` IS OVERRIDDEN AT ALL. The parent resolves object geometry through
`mujoco_eval.grounding.rekep.extent_table(task_key)`, a MimicGen-only table. RoboLab bodies are
absent from it, so every synthetic cloud would come back None and every object would end up
without a grasp centre. `load_rekep_context` already takes the table as a parameter, so the fix is
to pass ours -- measured by robolab_eval.grounding.make_context and carried in the context
artifact itself -- rather than to edit the MimicGen table. The rest of this method is the parent's
artifact path with the branches that only exist for MimicGen (gt keypoints, the drawer, live
perception, the hand-written place template) left out.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import torch

from mujoco_eval.grounding.rekep import load_rekep_context
from rekep.utils import get_callable_grasping_cost_fn, load_default_config
from vlm_dp.grounding import Grounding, SceneObject, fake_vlm, masks
from vlm_dp.sim_helpers import (DEFAULT_EXTENT, TorchNumpyShim, center_from_points,
                                load_torch_constraints, make_torch_constraint)

from .. import paths
from ..tasks import scene_objects, spec


_INSERT_FIELDS = ("mouth", "insert_depth", "insert_hover", "mouth_radius",
                  "seat_radius", "insert_capture")


def attach_standard_insertion(grounding):
    """Attach a measured insertion corridor to container-place stages.

    The shared VLM/ReKep compiler deliberately has no RoboLab task table.  A canned/live plan can
    nevertheless declare a standard insertion receipt through ``render_fields.json``.  This
    adapter turns those semantic fields into the existing :class:`Stage.insert` contract, using
    the live tracked mouth keypoint on every call.  There is no simulator-pose fallback.
    """
    fields = grounding.plan_fields or {}
    missing = [name for name in _INSERT_FIELDS if name not in fields]
    if missing:
        return grounding

    mouth_idx = int(fields["mouth"])
    depth = float(fields["insert_depth"])
    hover = float(fields["insert_hover"])
    mouth_radius = float(fields["mouth_radius"])
    seat_radius = float(fields["seat_radius"])
    capture = float(fields["insert_capture"])

    def geometry():
        keypoints = np.asarray(grounding.keypoints(), dtype=np.float64)
        if not 0 <= mouth_idx < len(keypoints):
            raise RuntimeError(f"insertion mouth kp{mouth_idx} absent from {len(keypoints)} keypoints")
        mouth = keypoints[mouth_idx]
        seat = mouth - np.array([0.0, 0.0, depth], dtype=np.float64)
        return {
            "axis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "seat": seat,
            "goal": seat,
            "height": hover + depth,
            "r_mouth": mouth_radius,
            "r_seat": seat_radius,
            "capture": capture,
        }

    stages, decorated = [], []
    for i, stage in enumerate(grounding.stages):
        if stage.place_mode == "container" and stage.payload is not None:
            stage = dataclasses.replace(stage, insert=geometry)
            decorated.append(i + 1)
        stages.append(stage)
    if not decorated:
        raise SystemExit("[robolab-rekep] insertion fields were declared, but no container-place "
                         "stage consumed them")
    print(f"[robolab-rekep] measured insertion corridor on stages {decorated}: "
          f"mouth=kp{mouth_idx} depth={depth:.3f}m radii={seat_radius:.3f}->"
          f"{mouth_radius:.3f}m capture={capture:.3f}m", flush=True)
    return dataclasses.replace(grounding, stages=stages)


class RoboLabRekepVlmGrounding:
    """Build a Grounding from a measured context and an authored ReKep plan."""

    def __init__(self, task, context_path=None, subgoal_eps=0.06, clearance=0.015,
                 open_half=0.04, vlm="fake"):
        from mujoco_eval.grounding.rekep import RekepGrounding

        self.task = task
        self.spec = spec(task)
        self.context_path = str(context_path or paths.task_data(task, "rekep_context.json"))
        self.clearance = float(clearance)
        # Bodies whose pose the episode reads. Fixtures are included on purpose: the plan's
        # transport targets are offsets from a FIXTURE keypoint, so the bowl and the bins have to
        # be tracked or those targets would be frozen at their capture pose.
        self.movable = scene_objects(task)
        self.roles = {"grasp_obj": self.spec["grasp_objs"][0],
                      "grasp_objs": list(self.spec["grasp_objs"]),
                      "place_obj": self.spec["place_obj"]}
        # The shared stage builder. Only `_vlm_stages` and the geometry helpers on it are used;
        # its own `ground` is replaced below.
        self._stager = RekepGrounding(
            vlm=vlm, task_key=task, place_obj=self.spec["place_obj"],
            grasp_objs=self.spec["grasp_objs"], stages="vlm", kp_source="artifact",
            subgoal_eps=subgoal_eps, open_half=open_half, context_path=self.context_path)

    def _tables(self):
        """Read the measured geometry the context carries, with tasks.py as fallback."""
        with open(self.context_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        fallback = dict(self.spec["extents"])
        extents = {n: tuple(v) for n, v in (raw.get("extents") or {}).items()} or fallback
        box_half = {n: tuple(v) for n, v in (raw.get("box_half") or {}).items()}
        centres = {n: np.asarray(v, dtype=np.float64)
                   for n, v in (raw.get("box_center_local") or {}).items()}
        if not box_half:
            # No measured box: fall back to the declared triple. That triple is sorted by size,
            # so the cloud is axis-swapped for any body whose narrow side is not local x -- say so
            # rather than hide it.
            print("[robolab-rekep] context carries no box_half; the synthetic cloud falls back to "
                  "the (grip, keepout, half_height) triple and may be mis-oriented", flush=True)
            box_half = {n: tuple(v) for n, v in {**fallback, **extents}.items()}
        return extents, box_half, centres

    @staticmethod
    def _box_cloud(world, box_half, centres, per_axis=7):
        """Return points_of(name) -> a box cloud from the body's live pose and measured box.

        mujoco_eval's `_box_points_fn` builds the box around the body ORIGIN. That is right for
        MimicGen, whose bodies are centred, and wrong here: a grey bin's origin lies on its base,
        so its cloud would sit half a bin underground -- the clearance terms would plan around
        empty space and the seat measurement would land inside the table. This is the same
        builder plus the measured centre offset.
        """
        grid = np.linspace(-1.0, 1.0, per_axis)
        unit = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)

        def points_of(name):
            half = box_half.get(name)
            if half is None:
                return None
            try:
                pos, rot = world.object_pose(name)[:2]
            except Exception:                       # body absent from this scene
                return None
            local = unit * np.asarray(half, dtype=np.float64) + centres.get(name, np.zeros(3))
            return np.asarray(pos, dtype=np.float64) + local @ np.asarray(rot, dtype=np.float64).T

        return points_of

    def _vlm_dir(self):
        """Scratch directory the generated per-stage constraint files are written to."""
        root = os.environ.get("ROBOLAB_EVAL_RESULTS") or str(paths.RESULTS)
        out = os.path.join(root, "_vlm_query", f"{self.task}_p{os.getpid()}")
        os.makedirs(out, exist_ok=True)
        for leaf in os.listdir(out):
            if leaf.endswith("_constraints.txt") or leaf == "metadata.json":
                os.remove(os.path.join(out, leaf))
        return out

    def ground(self, env, world) -> Grounding:
        """Return the grounded task: scene objects plus the plan's ordered stages."""
        from rekep.keypoint_tracking import KeypointTracker

        load_default_config()                        # validates the ReKep config is installed
        extents, box_half, centres = self._tables()
        grounded = load_rekep_context(self.context_path, world, box_half)
        grounded["points_of"] = self._box_cloud(world, box_half, centres)
        keypoints = np.asarray(grounded["keypoints"], dtype=np.float64)
        if len(keypoints) == 0:
            raise SystemExit("[robolab-rekep] the context proposed no keypoints")
        dev = getattr(env, "device", "cpu")
        names = [n for n in (grounded.get("names") or self.movable)]

        # The CONTEXT is the authority on attachment, exactly as in the MimicGen path: the tracker
        # re-derives ownership by nearest centre, which would re-own a static container keypoint
        # onto whatever payload passes near it -- and since `held_idx` is "every keypoint owned by
        # the grasped body", a re-owned place target would then ride the gripper and satisfy the
        # place rule for free.
        tracker = KeypointTracker(world, keypoints)
        for i, owner in enumerate(grounded.get("owners") or []):
            if not 0 <= i < len(tracker.owners):
                continue
            pos, rot = world.object_pose(owner)
            tracker.owners[i] = owner
            tracker.registrations[i] = (owner, np.asarray(rot).T @ (keypoints[i] - np.asarray(pos)))
        print(f"[robolab-rekep] tracker ownership from context: {tracker.owners}", flush=True)

        vlm_dir = self._vlm_dir()
        metadata, roles, extra_kps = fake_vlm.generate(
            self.task, vlm_dir, keypoints, grounded, env, self.clearance)
        declared_ext = {}
        if extra_kps:
            # The Spoon plan declares thin features that the generic keypoint proposer is not
            # expected to sample: the neck grasp, both utensil ends, and the holder mouth. Append
            # and rigidly register them exactly as the shared ReKep grounding does.
            first = len(keypoints)
            extra_kps = [tuple(e) + (None,) * (3 - len(e)) for e in extra_kps]
            keypoints = np.concatenate(
                [keypoints, np.stack([np.asarray(p, dtype=np.float64)
                                      for p, _, _ in extra_kps])], axis=0)
            grounded["keypoints"] = keypoints
            local_extents = list(grounded.get("local_extents") or [None] * first)
            local_extents.extend(
                [None if ext is None else [float(ext)] * 3 for _, _, ext in extra_kps])
            grounded["local_extents"] = local_extents
            for point, owner, ext in extra_kps:
                point = np.asarray(point, dtype=np.float64)
                if owner is None:
                    tracker.registrations.append((None, point))
                else:
                    pos, rot = world.object_pose(owner)
                    tracker.registrations.append(
                        (owner, np.asarray(rot).T @ (point - np.asarray(pos))))
                idx = len(tracker.owners)
                tracker.owners.append(owner)
                if ext is not None:
                    declared_ext[idx] = float(ext)
            print(f"[robolab-rekep] VLM declared keypoints "
                  f"{list(range(first, len(keypoints)))} "
                  f"owners={[owner for _, owner, _ in extra_kps]}", flush=True)
        bad = [i for i in metadata["grasp_keypoints"] + metadata["release_keypoints"]
               if not -1 <= i < len(keypoints)]
        if bad:
            raise SystemExit(f"[robolab-rekep] the plan referenced out-of-range keypoint(s) {bad} "
                             f"(have {len(keypoints)})")

        objects = self._scene_objects(
            grounded, env, names, keypoints, tracker, metadata, extents, declared_ext)
        shim = TorchNumpyShim(dev)

        def load_stage(idx, held):
            """Load a stage's summed sub-goal, its per-rule sub-goals, and its path rules."""
            grasp_fn = get_callable_grasping_cost_fn(list(held))
            rules = load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_subgoal_constraints.txt"), grasp_fn, shim)
            path_fns = tuple(make_torch_constraint([c]) for c in load_torch_constraints(
                os.path.join(vlm_dir, f"stage{idx + 1}_path_constraints.txt"), grasp_fn, shim))
            return make_torch_constraint(rules), path_fns, tuple(
                make_torch_constraint([c]) for c in rules)

        obj_names = tuple(o.name for o in objects)

        def name_for(kp_idx):
            """Resolve the object that owns a keypoint."""
            return tracker.owners[kp_idx] or masks.object_for_keypoint(
                grounded, env, keypoints[kp_idx], names=obj_names)

        stages, manipulated = self._stager._vlm_stages(
            metadata, tracker, keypoints, name_for, load_stage, objects, env, dev, vlm_dir)
        print(f"[robolab-rekep] {self.task}: roles={roles} objects={list(obj_names)}", flush=True)
        return Grounding(
            objects=objects, stages=stages, manipulated=frozenset(manipulated),
            keypoints=(lambda: tracker.get_positions()),
            keypoint_metadata=(lambda: [
                {"owner": owner, "registered": tracker.registrations[i] is not None}
                for i, owner in enumerate(tracker.owners)
            ]),
        )

    def _scene_objects(self, grounded, env, names, keypoints, tracker, metadata, extents,
                       declared_ext=None):
        """Measure each body's grasp centre, closing axis and graspable span."""
        declared_ext = declared_ext or {}
        grasp_kp_of, grasped = {}, set()
        for k in metadata["grasp_keypoints"]:
            if 0 <= k < len(tracker.owners) and tracker.owners[k]:
                grasp_kp_of.setdefault(tracker.owners[k], k)
                grasped.add(tracker.owners[k])

        ctx_local = grounded.get("local_extents") or []
        kp_of, centroid_off, grasp_axis, grasp_ext, grasp_region = {}, {}, {}, {}, {}
        for name in names:
            pts = masks._masked_points(grounded, env, name)
            if pts is None:
                print(f"[robolab-rekep] {name!r} has no cloud; not a scene object", flush=True)
                continue
            gk = grasp_kp_of.get(name)
            kp = gk if gk is not None else masks._nearest_kp(keypoints, pts.mean(axis=0))
            # A declared grasp feature is itself the grasp point. Do not silently translate it
            # back to the owner's whole-body centroid.
            centre = keypoints[kp] if kp in declared_ext else center_from_points(pts, None)
            kp_of[name], centroid_off[name] = kp, centre - keypoints[kp]
            if name in grasped:
                grasp_axis[name] = masks.narrow_axis(pts)
                axis = grasp_axis[name]
                if axis is not None:
                    # The graspable SPAN along the body's long horizontal axis. A point target
                    # pins the grasp to one pose; a region leaves the sampler somewhere to move.
                    long_axis = np.array([-float(axis[1]), float(axis[0]), 0.0])
                    t = (pts[:, :2] - centre[:2]) @ long_axis[:2]
                    grasp_region[name] = (long_axis.tolist(),
                                          float(np.percentile(np.abs(t), 80)))
                if gk in declared_ext:
                    grasp_ext[name] = float(declared_ext[gk])
                elif gk is not None and 0 <= gk < len(ctx_local) and ctx_local[gk] is not None:
                    grasp_ext[name] = float(ctx_local[gk][0])
                print(f"[robolab-rekep] {name}: axis={grasp_axis[name]} "
                      f"grip_half={grasp_ext.get(name)} region={grasp_region.get(name)}",
                      flush=True)

        def obj_pos(name):
            kp = kp_of.get(name)
            if kp is None:
                raise KeyError(f"[robolab-rekep] no grasp centre for {name!r}: no cloud.")
            return tracker.get_positions()[kp] + centroid_off[name]

        for name in kp_of:
            # Against the body ORIGIN, which is not the body centre for every asset -- a grey
            # bin's origin sits 5.25 cm below its own centre, so a ~52 mm reading there is the
            # measured offset, not grounding error.
            err = float(np.linalg.norm(obj_pos(name) - env.object_pose(name)[0])) * 1e3
            print(f"[robolab-rekep] {name}: centre={np.round(obj_pos(name), 3)} "
                  f"origin={np.round(env.object_pose(name)[0], 3)} d={err:.1f}mm", flush=True)

        return [SceneObject(name=n, pos=(lambda n=n: obj_pos(n)),
                            extents=tuple(extents.get(n, DEFAULT_EXTENT)),
                            axis=grasp_axis.get(n), grasp_extent=grasp_ext.get(n),
                            grasp_region=grasp_region.get(n))
                for n in kp_of]


def _R_to_quat(rot):
    """Rotation matrix -> wxyz quaternion."""
    m = np.asarray(rot, dtype=np.float64)
    q = np.array([
        np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])),
        np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])), m[2, 1] - m[1, 2]),
        np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])), m[0, 2] - m[2, 0]),
        np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])), m[1, 0] - m[0, 1]),
    ]) / 2.0
    return q / max(float(np.linalg.norm(q)), 1e-9)


def probe_terms(grounding, env, geom, n=64, sigma=0.05, tilt=0.15, seed=0):
    """Report whether the ReKep terms are LIVE on every stage of a grounding.

    The RoboLab port of agent_tests/_vlmwire_terms_probe.py. A term that reads zero spread across
    a random candidate batch is inert on that stage whatever the stage table says, and a plan
    whose rules are all constant over the candidate set steers nothing.
    """
    from vlm_dp.cost.terms import TERMS

    class _Inputs:
        def __init__(self, ee_pos, ctx, ee_quat=None, geom=None, extents=None):
            self.ee_pos = ee_pos
            self.ee_quat = ee_quat
            self.real_actions = torch.zeros(*ee_pos.shape[:2], 8)
            self.extents = extents or {}
            self.geom = geom
            self.context = ctx

    from sim_free_mpc.fk import quat_mul_wxyz

    objects = {o.name: {"extents": o.extents, "axis": o.axis, "grasp_extent": o.grasp_extent,
                        "grasp_region": o.grasp_region} for o in grounding.objects}
    extents = {n: o["extents"] for n, o in objects.items()}
    tcp = np.asarray(env.tcp(), dtype=np.float64)
    tcp_rot = np.asarray(env.tcp_rot(), dtype=np.float64)
    torch.manual_seed(seed)
    ee = torch.as_tensor(tcp, dtype=torch.float32).view(1, 1, 3) + sigma * torch.randn(n, 8, 3)
    # The LIVE tool orientation turned by a random yaw about its own approach axis -- the one
    # rotational freedom a top-down grasp has. It must be the live orientation and not an
    # arbitrary tool-down frame: `_capture_held` stores held offsets in the GRIPPER frame and the
    # cost re-applies them through `ee_quat`, so a probe that pairs a world-frame offset with an
    # unrelated quaternion flings the carried object somewhere the rollout never puts it. That is
    # what made the altitude guards read a flat zero: the "held" banana was being reflected to
    # twice the hand's height, where a floor at the pick height cannot bite.
    # Free yaw about the approach axis plus a small tilt about the other two: that is the shape
    # of the wrist variation a joint-space candidate cloud actually produces, and a rule stated on
    # two held keypoints (a bottle's body and its top) is invisible to yaw alone.
    live = torch.as_tensor(_R_to_quat(tcp_rot), dtype=torch.float32).view(1, 1, 4)
    w = torch.cat([tilt * torch.randn(n, 2), torch.rand(n, 1) * 2 * np.pi], dim=-1)
    angle = torch.linalg.vector_norm(w, dim=-1, keepdim=True).clamp(min=1e-8)
    spin = torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * w / angle], dim=-1)
    quat = quat_mul_wxyz(live.expand(n, 8, 4), spin[:, None, :].expand(n, 8, 4))

    rows = []
    for i, st in enumerate(grounding.stages):
        kps = grounding.keypoints()
        # Exactly vlm_dp.stage._capture_held: the offset lives in the gripper frame.
        held_off = (np.stack([tcp_rot.T @ (kps[j] - tcp) for j in st.held_idx])
                    if st.held_idx else None)
        target = np.asarray(st.target(), dtype=np.float32)
        ctx = {"constraint": st.constraint, "path_fns": st.path_fns, "keypoints": kps,
               "held_idx": st.held_idx, "held_offset": held_off, "objects": objects,
               "grasp_obj": st.grasp_obj, "payload": st.payload,
               "place_target": st.place_target, "contact": st.contact, "orient": st.orient,
               "target": target}
        # Two clouds, because a CLAMPED inequality reads zero wherever it is already satisfied and
        # that is not the same thing as being inert. `start` is the arm's current pose (where the
        # stage is entered from); `target` is the pose the stage is trying to reach, which is
        # where a keep-out or an altitude floor actually bites. A rule flat on both is inert.
        clouds = {"start": ee,
                  "target": torch.as_tensor(target).view(1, 1, 3) + sigma * torch.randn(n, 8, 3)}

        def spread(term, override=None, cloud=None):
            v = TERMS[term](_Inputs(cloud, {**ctx, **(override or {})}, ee_quat=quat, geom=geom,
                                    extents=extents))
            return float(v.max() - v.min())

        best = {k: max(spread(k, cloud=c) for c in clouds.values())
                for k in ("rekep_subgoal", "rekep_keypose", "rekep_path")}
        print(f"[terms] stage {i} {st.name!r}: held={list(st.held_idx)} "
              f"subgoal={'yes' if st.constraint is not None else 'NONE'} "
              f"path={len(st.path_fns)} contact={st.contact} " + " ".join(
                  f"{k}=spread {v:.4f}" for k, v in best.items()), flush=True)
        for j, fn in enumerate(st.subgoal_fns or ()):
            at = {k: spread("rekep_subgoal", {"constraint": fn}, c) for k, c in clouds.items()}
            print(f"[terms]     subgoal rule {j + 1}: " + " ".join(
                f"{k}={v:.4f}" for k, v in at.items()), flush=True)
        for j, fn in enumerate(st.path_fns or ()):
            at = {k: spread("rekep_path", {"path_fns": (fn,)}, c) for k, c in clouds.items()}
            print(f"[terms]     path rule {j + 1}: " + " ".join(
                f"{k}={v:.4f}" for k, v in at.items()), flush=True)
        rows.append({"stage": st.name, **{k: round(v, 5) for k, v in best.items()}})
    return rows
