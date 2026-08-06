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
    best, best_score = None, np.inf
    for name, (pos, _rot) in poses.items():
        half = np.asarray(extents.get(name, (0.05, 0.05, 0.05)), dtype=np.float64)
        delta = np.abs(np.asarray(point) - np.asarray(pos))
        score = float(np.linalg.norm(np.maximum(delta - half, 0.0)))   # 0 inside the box
        if score < best_score:
            best, best_score = name, score
    return best, best_score


# Beyond this surface distance a keypoint is not rigidly attached to anything: can's drop
# point sits 80 cm from the can and its destination (bin2_q3) has no MuJoCo body at all, so
# forcing attribution would make a fixed world target ride the payload.
_UNOWNED_M = 0.15


def build_context(task, seed=101, hw=512, camera="agentview"):
    """Return the rekep_context.json payload for one task."""
    from ..env.mujoco_env import MuJoCoEnv, MGWorld

    demo, fk, _ = _task_paths(task)
    env = MuJoCoEnv(demo, fk)
    env.reset(seed)

    source = MGGroundingSource(task)
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
    candidates = {o.name: o.extents for o in grounding.objects}
    for name in declared:
        candidates.setdefault(name, EXTENTS.get(name))
    for name, ext in candidates.items():
        try:
            pos, rot = world.object_pose(name)[:2]
        except Exception:
            continue
        poses[name] = (np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64))
        extents[name] = ext if ext is not None else EXTENTS.get(name, (0.05, 0.05, 0.05))

    entries = []
    for point in points:
        owner, score = _owner_of(point, poses, extents)
        if owner is not None and score > _UNOWNED_M:
            owner = None                       # static world target
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
    print(f"[make-context] {task}: {len(owners)} keypoints owners={owners} -> {out}", flush=True)
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
