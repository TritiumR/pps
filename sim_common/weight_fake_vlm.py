"""Fake ReKep VLM output for the weight task.

Writes the *exact* artifacts `rekep.constraint_generation.ConstraintGenerator` produces -- a
`metadata.json` (num_stages, grasp_keypoints, release_keypoints) and per-stage
`stage{i}_{subgoal,path}_constraints.txt` (numpy functions of `end_effector, keypoints`). The driver
loads these identically to the real thing, so going live = call the VLM to fill the same dir instead
of calling `generate` here.

The two things the VLM would supply, stubbed:
  - which keypoint is the pear / apple / scale -> resolved from GT instance masks (the "selection"),
  - the placement offset -> derived from the object/scale point clouds (object half-height + clearance).
"""
import json
import os
import re

import numpy as np


def _masked_points(grounded, env, name):
    """World points belonging to `name`'s GT instance mask (real table_cam depth, GT-segmented)."""
    rel = re.sub(r"^/World/envs/env_[^/]*/", "", env.scene[name].cfg.prim_path)
    ids = [i for i, prim in grounded["id_to_prim"].items() if rel and rel in prim]
    sel = np.isin(grounded["masks"], ids) & np.isfinite(grounded["points"]).all(axis=-1)
    return grounded["points"][sel] if int(sel.sum()) else None


def _nearest_kp(keypoints, point):
    return int(np.argmin(np.linalg.norm(keypoints - point, axis=1)))


def object_for_keypoint(grounded, env, point, names=("pear", "apple", "scale", "mango", "cabbage")):
    """Which scene object a keypoint sits on, by GT-mask grounding.

    The real VLM picks keypoint *indices*; this maps a chosen keypoint's world position back to the
    object whose masked depth points it is nearest to. Lets the grasp target that object's local
    centroid (the perception grasp center) in live mode, where the fake VLM's role table is gone.
    Returns the object name, or None if no candidate has masked points.
    """
    point = np.asarray(point)
    best, best_d = None, np.inf
    for name in names:
        pts = _masked_points(grounded, env, name)
        if pts is None:
            continue
        d = float(np.linalg.norm(pts - point, axis=1).min())
        if d < best_d:
            best, best_d = name, d
    return best


def local_centroid(grounded, env, name, near=None, radius=0.06):
    """Perception-derived grasp center: the centroid of `name`'s masked depth points, restricted to
    those within `radius` of `near` (the local graspable region) when given. Generalizes to parts
    (handle) as well as whole objects. Single-camera, so it's the visible-surface centroid (a known
    bias) -- a multi-view/shape-fit center would remove it; this is the cheap principled version.
    """
    pts = _masked_points(grounded, env, name)
    if pts is None:
        return None
    if near is not None:
        local = pts[np.linalg.norm(pts - np.asarray(near), axis=1) < radius]
        if len(local):
            pts = local
    return pts.mean(axis=0)


def resolve_roles(keypoints, grounded, env, clearance=0.015):
    """Map pear/apple/scale to their nearest keypoints and the placement offset onto the scale.

    The two things the VLM would supply for the weight task, derived from GT-masked perception: the
    keypoint selection (object -> nearest keypoint) and the placement offset (scale top + object
    half-height + clearance) that carries the scale keypoint to each object's placement point.
    Returns ``(roles, off)`` with ``off[name]`` a 3-vector added to the scale keypoint. Reused by both
    the constraint-file writer below and the ``RekepGrounding`` source.
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


def generate(out_dir, keypoints, grounded, env, clearance=0.015):
    """Write the weight task's metadata + constraint files. Returns (metadata, resolved_roles)."""
    os.makedirs(out_dir, exist_ok=True)
    roles, off = resolve_roles(keypoints, grounded, env, clearance)
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

    print(f"[fake-vlm] roles pear=kp{p} apple=kp{a} scale=kp{s}", flush=True)
    return metadata, roles
