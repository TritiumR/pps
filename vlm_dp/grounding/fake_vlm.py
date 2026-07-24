"""Fake ReKep VLM stubs, one per task.

Each writes the exact artifacts ConstraintGenerator produces (metadata.json and per-stage constraint
files) without calling GPT-4o. Register a task in _FAKE_VLMS.
"""
import json
import os

import numpy as np

from vlm_dp.grounding.masks import _masked_points, _nearest_kp


def _weight_roles(keypoints, grounded, env, clearance):
    """Weight task: pear/apple/scale -> nearest keypoints + the placement offset onto the scale."""
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


def _capsule(out_dir, keypoints, grounded, env, clearance):
    """Fake VLM for capsule, paired with GT keypoints in the fixed order [lip, open_goal, pod, bay, body].

    A known-correct plan so the driven cost is tested in isolation from perception and the VLM: grasp
    the lid lip, lift it to the open goal, grasp the pod, place it in the bay. Requires kp_source=gt for
    the GT keypoint order. With perception keypoints the indices would be meaningless.
    """
    lip, open_goal, pod, bay = 0, 1, 2, 3
    metadata = {"num_stages": 4, "grasp_keypoints": [lip, -1, pod, -1],
                "release_keypoints": [-1, lip, -1, pod]}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    def write(stage, kind, body):
        with open(os.path.join(out_dir, f"stage{stage}_{kind}_constraints.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    write(1, "subgoal", f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    """Grasp the coffee-maker lid: align the end-effector with the lid lip."""
    return np.linalg.norm(end_effector - keypoints[{lip}])
''')
    write(2, "subgoal", f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    """Open the lid: bring the lip to the open goal above it (rotates the lid up on its hinge)."""
    return np.linalg.norm(keypoints[{lip}] - keypoints[{open_goal}])
''')
    write(2, "path", f'''def stage2_path_constraint1(end_effector, keypoints):
    """Keep grasping the lid lip while opening."""
    return get_grasping_cost_by_keypoint_idx({lip})
''')
    write(3, "subgoal", f'''def stage3_subgoal_constraint1(end_effector, keypoints):
    """Grasp the pod: align the end-effector with the pod keypoint."""
    return np.linalg.norm(end_effector - keypoints[{pod}])
''')
    write(4, "subgoal", f'''def stage4_subgoal_constraint1(end_effector, keypoints):
    """Place the pod in the bay (pod keypoint resting at the bay opening, ~2cm above)."""
    return np.linalg.norm(keypoints[{pod}] - (keypoints[{bay}] + np.array([0.0, 0.0, 0.02])))
''')
    write(4, "path", f'''def stage4_path_constraint1(end_effector, keypoints):
    """Keep grasping the pod while placing."""
    return get_grasping_cost_by_keypoint_idx({pod})
''')
    for st, kind in ((1, "path"), (3, "path")):
        write(st, kind, "")   # grasp stages: no path constraints

    print(f"[fake-vlm] capsule GT plan: grasp lid kp{lip} -> open to kp{open_goal} -> "
          f"grasp pod kp{pod} -> place at bay kp{bay}", flush=True)
    return metadata, {"lid": lip, "pod": pod, "bay": bay}


_FAKE_VLMS = {"weight": _weight, "capsule": _capsule}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write task_key's fake VLM output (metadata and constraint files). Returns (metadata, roles)."""
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}, register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    return _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
