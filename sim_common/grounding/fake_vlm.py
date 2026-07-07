"""Fake ReKep VLM stubs, per task -- validate the pipeline without calling GPT-4o.

Each registered task writes the *exact* artifacts ``rekep.constraint_generation.ConstraintGenerator``
produces -- a ``metadata.json`` (num_stages, grasp/release keypoints) + per-stage
``stage{i}_{subgoal,path}_constraints.txt`` (numpy functions of ``end_effector, keypoints``) -- so a
driver loads them identically to the real VLM. Add a task by writing a
``_<task>(out_dir, keypoints, grounded, env, clearance)`` and registering it in ``_FAKE_VLMS``.

What the VLM would supply, stubbed: which keypoint is which object (resolved from GT instance masks),
and any placement offsets (derived from the object/target point clouds).
"""
import json
import os

import numpy as np

from sim_common.grounding.masks import _masked_points, _nearest_kp


def _weight_roles(keypoints, grounded, env, clearance):
    """Weight task roles: map pear/apple/scale to nearest keypoints + the placement offset onto the scale.

    Derived from GT-masked perception -- the keypoint selection (object -> nearest keypoint) and the
    placement offset (scale top + object half-height + clearance). Returns ``(roles, off)`` with
    ``off[name]`` a 3-vector added to the scale keypoint.
    """
    roles, half_h, scale_top = {}, {}, None
    for name in ("pear", "apple", "scale"):
        pts = _masked_points(grounded, env, name)
        if pts is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        roles[name] = _nearest_kp(keypoints, pts.mean(axis=0))
        if name == "scale":
            scale_top = np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].max()])
        else:
            half_h[name] = float((pts[:, 2].max() - pts[:, 2].min()) / 2.0)  # half-height from the cloud
    scale_kp = keypoints[roles["scale"]]
    off = {n: (np.array([scale_top[0], scale_top[1], scale_top[2] + half_h[n] + clearance]) - scale_kp).tolist()
           for n in ("pear", "apple")}
    return roles, off


def _weight(out_dir, keypoints, grounded, env, clearance):
    """Fake VLM for the weight task: grasp pear -> place on scale, grasp apple -> place on scale."""
    roles, off = _weight_roles(keypoints, grounded, env, clearance)
    p, a, s = roles["pear"], roles["apple"], roles["scale"]
    metadata = {"num_stages": 4, "grasp_keypoints": [p, -1, a, -1], "release_keypoints": [-1, p, -1, a]}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    def write(stage, kind, body):
        with open(os.path.join(out_dir, f"stage{stage}_{kind}_constraints.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    write(1, "subgoal", f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    """Grasp the pear: align the end-effector with the pear keypoint."""
    return np.linalg.norm(end_effector - keypoints[{p}])
''')
    write(2, "subgoal", f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    """Place the pear on the scale (pear keypoint at the scale-top placement point)."""
    return np.linalg.norm(keypoints[{p}] - (keypoints[{s}] + np.array({off["pear"]})))
''')
    write(3, "subgoal", f'''def stage3_subgoal_constraint1(end_effector, keypoints):
    """Grasp the apple: align the end-effector with the apple keypoint."""
    return np.linalg.norm(end_effector - keypoints[{a}])
''')
    write(4, "subgoal", f'''def stage4_subgoal_constraint1(end_effector, keypoints):
    """Place the apple on the scale (apple keypoint at the scale-top placement point)."""
    return np.linalg.norm(keypoints[{a}] - (keypoints[{s}] + np.array({off["apple"]})))
''')
    for st in range(1, 5):
        write(st, "path", "")  # no path constraints in this v1 stub

    print(f"[fake-vlm] weight roles pear=kp{p} apple=kp{a} scale=kp{s}", flush=True)
    return metadata, roles


_FAKE_VLMS = {"weight": _weight}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write ``task_key``'s fake ReKep VLM output (metadata + constraint files). Returns (metadata, roles).

    Dispatches to the stub registered in ``_FAKE_VLMS``; register a new task's ``_<task>`` to extend.
    """
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}; register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    return _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
