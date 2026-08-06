"""Fake VLM for mujoco_eval tasks: author ReKep constraints from the task ladder.

`vlm_dp.grounding.fake_vlm` covers only the Isaac tasks (weight, capsule, tea, pot) -- its
`_FAKE_VLMS` registry has zero overlap with the twelve mujoco tasks, so `--ground rekep` died at
dispatch even once the import chain was fixed.

Rather than hand-write one generator per task, this derives the constraint program from the task's
own stage ladder, which `make_context` records in rekep_context.json (stage names, gripper intents,
payloads, and each stage's target keypoint):

    no payload   ||ee - kp(stage target)||                  -> put the gripper here
    payload      ||kp(payload) - kp(stage target) + off||   -> put the carried object here

The emitted text is real ReKep source, parsed by the same loader a real VLM's output goes through,
so the fake and real paths converge on one Grounding -- which is the property that makes switching
between them a flag rather than a code branch.

It WALKS the ladder rather than applying a fixed grasp/place template, so articulated and
multi-stage tasks (mug_cleanup's open/close drawer, kitchen's buttons) are covered by the same
code: every stage has a target point, and what moves toward it is the end effector or the carried
object depending on whether the stage holds a payload.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .gt import TASKS

def supported_tasks():
    """Every mujoco task: the ladder walk is task-agnostic."""
    return tuple(sorted(TASKS))


def _payload_kp(owners, payload):
    """Index of a keypoint owned by the carried object, or None."""
    return next((i for i, o in enumerate(owners or ()) if o == payload), None)


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write ReKep metadata + constraint files by WALKING the task's stage ladder.

    One subgoal per stage, not a fixed grasp/place template. What moves toward the stage target is
    the end effector when nothing is held, and the carried object's keypoint once there is a
    payload -- which is the difference between "put the gripper here" and "put the mug here", and
    the reason the same walk covers articulated ladders (open drawer, release handle, close drawer)
    as well as plain pick-and-place.

    Every target point is already in the context: make_context registers each stage's own
    `target()` as a keypoint. Mirrors vlm_dp.grounding.fake_vlm.generate's signature and return.
    """
    del env
    if task_key not in TASKS:
        raise SystemExit(f"[fake-rekep] unknown task {task_key!r} (have {sorted(TASKS)})")

    owners = grounded.get("owners")
    stage_kp = {int(k): int(v) for k, v in (grounded.get("stage_keypoints") or {}).items()}
    names = grounded.get("stage_names") or []
    payloads = grounded.get("stage_payload") or []
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if not stage_kp:
        raise SystemExit(
            "[fake-rekep] the rekep context carries no stage_keypoints; regenerate it with "
            "`python -m mujoco_eval.grounding.make_context --task " + task_key + "`")

    os.makedirs(out_dir, exist_ok=True)

    def write(stage, kind, body):
        with open(os.path.join(out_dir, f"stage{stage}_{kind}_constraints.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(body)

    n = len(names) or (max(stage_kp) + 1)
    grasp_keypoints, release_keypoints = [], []
    for idx in range(n):
        target = stage_kp.get(idx)
        label = names[idx] if idx < len(names) else f"stage {idx}"
        payload = payloads[idx] if idx < len(payloads) else None
        moving = _payload_kp(owners, payload) if payload else None
        if target is None:
            write(idx + 1, "subgoal", "")
        elif moving is None:
            write(idx + 1, "subgoal", f'''def stage{idx + 1}_subgoal_constraint1(end_effector, keypoints):
    """{label}: bring the end-effector to the stage target."""
    return np.linalg.norm(end_effector - keypoints[{target}])
''')
        else:
            off = np.array([0.0, 0.0, float(clearance)]).tolist()
            write(idx + 1, "subgoal", f'''def stage{idx + 1}_subgoal_constraint1(end_effector, keypoints):
    """{label}: bring the carried {payload} to the stage target."""
    return np.linalg.norm(keypoints[{moving}] - (keypoints[{target}] + np.array({off})))
''')
        write(idx + 1, "path", "")
        # ReKep's convention: grasp_keypoints marks where a stage ACQUIRES an object, release
        # where it lets one go. Read them off the ladder's payload transitions.
        prev = payloads[idx - 1] if 0 < idx < len(payloads) else None
        nxt = payloads[idx + 1] if idx + 1 < len(payloads) else None
        acquires = payload is None and nxt is not None
        releases = payload is not None and nxt is None and prev is not None
        grasp_keypoints.append(_payload_kp(owners, nxt) if acquires else -1)
        release_keypoints.append(_payload_kp(owners, payload) if releases else -1)

    metadata = {"num_stages": n, "grasp_keypoints": grasp_keypoints,
                "release_keypoints": release_keypoints}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[fake-rekep] {task_key}: {n} stages from the ladder "
          f"({', '.join(names) if names else 'unnamed'})", flush=True)
    return metadata, {"stage_keypoints": stage_kp}
