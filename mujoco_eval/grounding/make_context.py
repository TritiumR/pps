"""Write rekep_context.json for any mujoco task, from ground-truth object poses.

Only `stack` shipped a ReKep context, and the tool that produced it (grounding/propose.py) needs
IsaacLab perception -- which does not exist in the MuJoCo eval env. So `--ground rekep` could not
run on eleven of the twelve tasks for want of a keypoint file, not for want of planner support.

This closes that without perception. `gt.py` already computes each task's semantic points (the nut
handle, the peg top, the drawer handle, the seat), so every keypoint is attributed to the nearest
scene object and stored as {owner, offset_local, world_at_capture} -- the same rigid-attachment
schema the perception path writes, and the one `load_rekep_context` rebases onto live poses.

    python -m mujoco_eval.grounding.make_context --task square
    python -m mujoco_eval.grounding.make_context --all

The result is an ORACLE grounding in ReKep's clothing: real keypoints, real owners, no VLM and no
segmenter. That is the honest description -- it makes the ReKep code path runnable and testable
everywhere, and leaves keypoint *proposal* (which needs image->world) as separate work.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .. import paths
from ..eval import _DATA_NAME, _FK_FIT_NAME
from .gt import TASKS, MGGroundingSource


def _task_paths(task):
    """Resolve demo + fk_fit exactly as eval.resolve_defaults does, so a context is built for the
    same scene the rollout will run."""
    data_dir = paths.DATA / _DATA_NAME.get(task, f"{task}_d0")
    fk = paths.fk_fit(_FK_FIT_NAME.get(task, "fk_fit_stack_d0.json"))
    return str(data_dir / "demo.hdf5"), str(fk), data_dir


def _owner_of(point, poses, extents):
    """Attribute a keypoint to the object whose BOX SURFACE it lies nearest, in metres.

    Surface distance rather than normalised centre distance: several tasks place keypoints at
    functional offsets from a body -- can's drop point sits 18 cm above the bin, square's peg top
    above the peg -- and a centre metre normalised by a thin extent scores those worse than a
    distant compact object, silently reassigning the drop point to the payload. Distance to the
    box the object actually occupies keeps offset points with the thing they are an offset OF.
    """
    best, best_score = None, (np.inf, np.inf)
    for name, (pos, _rot) in poses.items():
        half = np.asarray(extents.get(name, (0.05, 0.05, 0.05)), dtype=np.float64)
        delta = np.abs(np.asarray(point) - np.asarray(pos))
        # Ties are the normal case for NESTED bodies -- the drawer's tray lies inside both the
        # sliding front's box and the cabinet's -- and dict order used to settle them, which handed
        # moving parts to the static shell. The tighter box wins instead: it is the more specific
        # claim, and it is the body the point actually rides.
        score = (float(np.linalg.norm(np.maximum(delta - half, 0.0))),   # 0 inside the box
                 float(np.linalg.norm(half)))
        if score < best_score:
            best, best_score = name, score
    return best, best_score[0]


# Beyond this surface distance a keypoint is not rigidly attached to anything: can's drop
# point sits 80 cm from the can and its destination (bin2_q3) has no MuJoCo body at all, so
# forcing attribution would make a fixed world target ride the payload.
_UNOWNED_M = 0.15

# A box describes the geometry AROUND a keypoint only if the keypoint lies on it. Past this
# surface distance the box is something the point is merely near, and its half-widths say nothing
# about what the fingers would close on there.
_LOCAL_PART_M = 0.02


def _local_extents_of(point, parts):
    """Return (half_extents, part_name) describing the geometry local to one keypoint.

    THE INTERFACE IS "local extents around a keypoint". Here the backend is the declared part
    table -- every SceneObject the task grounding publishes, which includes sub-body parts such as
    a drawer handle with its own box. On a real robot the same triple would be measured from the
    perceived point cloud inside a small ball around the keypoint; nothing downstream knows or
    cares which produced it, so no part- or task-name string is needed to consume it.

    Selection is the same rule `_owner_of` uses -- nearest box surface, smallest box breaks ties --
    restricted to boxes the keypoint actually lies on (within `_LOCAL_PART_M`). A keypoint that
    lies on no declared part falls back to its owner body's box, which is the coarsest honest
    answer and what every consumer read before.
    """
    best, best_score = None, (np.inf, np.inf)
    for name, (pos, half) in parts.items():
        delta = np.abs(np.asarray(point) - np.asarray(pos))
        score = (float(np.linalg.norm(np.maximum(delta - np.asarray(half), 0.0))),
                 float(np.linalg.norm(half)))
        if score < best_score:
            best, best_score = name, score
    if best is None or best_score[0] > _LOCAL_PART_M:
        return None, None
    return tuple(float(x) for x in parts[best][1]), best


def build_context(task, seed=101, hw=512, camera="agentview"):
    """Return the rekep_context.json payload for one task."""
    from ..env.mujoco_env import MuJoCoEnv, MGWorld

    demo, fk, _ = _task_paths(task)
    env = MuJoCoEnv(demo, fk)
    env.reset(seed)

    # A context describes the SCENE, so it must not encode anything the task specification is
    # entitled to decide. Only sort_can has such a choice (which coloured bin), and neutral mode
    # is a no-op for every other task -- see MGGroundingSource._ground_sort_can.
    source = MGGroundingSource(task, semantic_neutral=True)
    world = MGWorld(env, sensor=None, names=source.movable)
    grounding = source.ground(env, world)
    if grounding.keypoints is None:
        raise SystemExit(f"[make-context] task {task!r} has no gt keypoints to convert")

    points = list(np.asarray(grounding.keypoints(), dtype=np.float64))
    # Also register each stage's own target point. gt.py already computes every semantic goal the
    # ladder needs (the drawer handle at full stroke, the retreat point, the seat), and a
    # constraint generator can only reference what the context carries -- without these, only
    # grasp/place stages were expressible and articulated ladders had to be refused.
    stage_kp = {}
    for idx, stage in enumerate(grounding.stages):
        try:
            target = np.asarray(stage.target(), dtype=np.float64).reshape(3)
        except Exception:
            continue
        match = next((j for j, p in enumerate(points)
                      if float(np.linalg.norm(p - target)) < 1e-6), None)
        if match is None:
            points.append(target)
            match = len(points) - 1
        stage_kp[idx] = match
    points = np.asarray(points, dtype=np.float64)
    names = sorted({o.name for o in grounding.objects} | set(
        list(TASKS[task]["grasp_objs"]) + ([TASKS[task]["place_obj"]]
                                            if TASKS[task].get("place_obj") else [])))
    poses, extents = {}, {}
    from .gt import EXTENTS
    spec = TASKS[task]
    # grounding.objects does not always carry the destination -- can's ladder builds a SceneObject
    # for the can but not for bin2_q3 -- so its drop point had nothing to attribute to and fell to
    # the payload. Query the task's declared roles too.
    declared = list(spec["grasp_objs"]) + ([spec["place_obj"]] if spec.get("place_obj") else [])
    # The task's TRACKED bodies are candidates too, or a keypoint attached to a moving part is
    # attributed to whatever static body happens to sit nearby: mug_cleanup's drawer handle was
    # owned by the cabinet rather than by the sliding front, so nothing downstream -- constraint,
    # cost or advance predicate -- could observe the drawer moving at all.
    from .rekep import extent_table
    known = extent_table(task)
    candidates = {o.name: o.extents for o in grounding.objects}
    for name in list(spec["movable"]) + declared:
        candidates.setdefault(name, known.get(name, EXTENTS.get(name)))
    for name, ext in candidates.items():
        try:
            pos, rot = world.object_pose(name)[:2]
        except Exception:
            continue
        poses[name] = (np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64))
        extents[name] = ext if ext is not None else EXTENTS.get(name, (0.05, 0.05, 0.05))

    # Declared PARTS, for the keypoint-local extents below. grounding.objects publishes sub-body
    # parts the tracked-body table has no entry for -- mug_cleanup's drawer handle is a
    # SceneObject with its own 9 mm box while the body it rides is a 24 cm drawer front -- and it
    # answers pos() for parts that have no MuJoCo body of their own, which `poses` cannot.
    parts = {}
    for obj in grounding.objects:
        if obj.extents is None:
            continue
        try:
            parts[obj.name] = (np.asarray(obj.pos(), dtype=np.float64).reshape(3),
                               np.asarray(obj.extents, dtype=np.float64).reshape(3))
        except Exception:                       # no live position: not a localizable part
            continue

    entries = []
    for point in points:
        owner, score = _owner_of(point, poses, extents)
        if owner is not None and score > _UNOWNED_M:
            owner = None                       # static world target
        local, local_from = _local_extents_of(point, parts)
        if local is None and owner is not None:
            local, local_from = tuple(float(x) for x in extents[owner]), owner
        if owner is None:
            entries.append({"owner": None, "offset_local": [0.0, 0.0, 0.0],
                            "world_at_capture": point.tolist()})
            continue
        pos, rot = poses[owner]
        entries.append({
            "owner": owner,
            # Local offset so the keypoint rides its body: world = pos + rot @ offset_local.
            "offset_local": (rot.T @ (point - pos)).tolist(),
            "world_at_capture": point.tolist(),
            "attribution_score": round(float(score), 3),
            # Half-extents of the geometry AT this keypoint, in the (grip, keepout, half_height)
            # convention every extents triple uses. `local_extents_from` is provenance for the log
            # only; no consumer branches on the name.
            "local_extents": [round(float(x), 6) for x in local],
            "local_extents_from": local_from,
        })

    model = env.env.env.sim.model if hasattr(env.env, "env") else env.env.sim.model
    return {
        "mode": "gt_oracle",
        "mask_source": "gt_pose",
        "names": names,
        "keypoints": entries,
        "static_unowned": sum(1 for e in entries if e["owner"] is None),
        # stage index -> keypoint index, so a generator can walk the ladder.
        "stage_keypoints": {str(k): v for k, v in stage_kp.items()},
        "stage_names": [st.name for st in grounding.stages],
        "stage_gripper": [st.gripper for st in grounding.stages],
        "stage_payload": [st.payload for st in grounding.stages],
        "camera": {"camera": camera, "hw": hw,
                   "fovy": float(model.cam_fovy[_cam_id(model, camera)])},
        "source": "mujoco_eval.grounding.make_context",
        "seed": int(seed),
    }


def _cam_id(model, camera):
    import mujoco

    return mujoco.mj_name2id(model._model if hasattr(model, "_model") else model,
                             mujoco.mjtObj.mjOBJ_CAMERA, camera)


def write(task, seed=101):
    payload = build_context(task, seed=seed)
    out = _task_paths(task)[2] / "rekep_context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    owners = [e["owner"] for e in payload["keypoints"]]
    local = [e.get("local_extents_from") for e in payload["keypoints"]]
    print(f"[make-context] {task}: {len(owners)} keypoints owners={owners} "
          f"local_extents_from={local} -> {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, choices=sorted(TASKS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if not args.task and not args.all:
        raise SystemExit("pass --task <name> or --all")
    for task in (sorted(TASKS) if args.all else [args.task]):
        try:
            write(task, seed=args.seed)
        except Exception as exc:                    # one bad task must not stop the sweep
            print(f"[make-context] {task}: FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
