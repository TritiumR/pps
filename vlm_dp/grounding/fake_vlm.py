"""Generate deterministic ReKep constraint artifacts for supported tasks."""
import json
import os

import numpy as np

from vlm_dp.grounding.masks import _masked_points, _nearest_kp


def _weight_roles(keypoints, grounded, env, clearance):
    """Resolve weight-task keypoints and placement offsets."""
    roles, half_h, scale_top = {}, {}, None
    for name in ("pear", "apple", "scale"):
        pts = _masked_points(grounded, env, name)
        if pts is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        roles[name] = _nearest_kp(keypoints, pts.mean(axis=0))
        if name == "scale":
            scale_top = np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].max()])
        else:
            half_h[name] = float((pts[:, 2].max() - pts[:, 2].min()) / 2.0)
    scale_kp = keypoints[roles["scale"]]
    off = {n: (np.array([scale_top[0], scale_top[1], scale_top[2] + half_h[n] + clearance]) - scale_kp).tolist()
           for n in ("pear", "apple")}
    return roles, off


def _weight(out_dir, keypoints, grounded, env, clearance):
    """Generate constraints for placing the pear and apple on the scale."""
    roles, off = _weight_roles(keypoints, grounded, env, clearance)
    p, a, s = roles["pear"], roles["apple"], roles["scale"]
    metadata = {"num_stages": 4, "grasp_keypoints": [p, -1, a, -1], "release_keypoints": [-1, p, -1, a],


                "steer_policies": ["on_failure", "on_failure", "on_failure", "on_failure"]}
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
        write(st, "path", "")

    print(f"[fake-vlm] weight roles pear=kp{p} apple=kp{a} scale=kp{s}", flush=True)
    return metadata, roles


def _capsule(out_dir, keypoints, grounded, env, clearance):
    """Generate the fixed capsule-task constraint plan."""
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
        write(st, kind, "")

    print(f"[fake-vlm] capsule GT plan: grasp lid kp{lip} -> open to kp{open_goal} -> "
          f"grasp pod kp{pod} -> place at bay kp{bay}", flush=True)
    return metadata, {"lid": lip, "pod": pod, "bay": bay}


def _quat_rotate_wxyz(quat, vec):
    w, x, y, z = [float(v) for v in quat]
    q = np.array([x, y, z])
    return vec + 2.0 * np.cross(q, np.cross(q, vec) + w * vec)


def _nearest_kp_distinct(keypoints, point, taken):
    """Return the nearest unclaimed keypoint."""
    order = np.argsort(np.linalg.norm(keypoints - np.asarray(point)[None], axis=-1))
    for idx in order:
        if int(idx) not in taken:
            return int(idx)
    return int(order[0])


def _rim_point(pts):
    """Return a graspable rim point from a point cloud."""
    c = pts[:, :2].mean(axis=0)
    return pts[int(np.argmax(np.linalg.norm(pts[:, :2] - c[None], axis=-1)))]


def _tea(out_dir, keypoints, grounded, env, clearance):
    """Generate constraints for grasping, carrying, and pouring the teapot."""
    roles = {}
    pts = {}
    for name in ("teapot", "teacup"):
        p = _masked_points(grounded, env, name)
        if p is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        pts[name] = p
    pot = env.scene["teapot"].data
    mouth_world = pot.root_pos_w[0].cpu().numpy() + _quat_rotate_wxyz(
        pot.root_quat_w[0].cpu().numpy(), np.array([0.0, 0.05847, -0.06146]))
    if grounded.get("gt_meta") is not None:

        roles["teapot"], roles["mouth"], roles["teacup"] = 0, 1, 2
    else:


        tp = pts["teapot"]
        d_mouth = np.linalg.norm(tp[:, :2] - mouth_world[None, :2], axis=-1)
        handle_world = tp[int(np.argmax(d_mouth))]
        taken = set()
        roles["teapot"] = _nearest_kp_distinct(keypoints, handle_world, taken)
        taken.add(roles["teapot"])
        roles["teacup"] = _nearest_kp_distinct(keypoints, pts["teacup"].mean(axis=0), taken)
        taken.add(roles["teacup"])
        roles["mouth"] = _nearest_kp_distinct(keypoints, mouth_world, taken)
    cup_top = np.array([pts["teacup"][:, 0].mean(), pts["teacup"][:, 1].mean(),
                        pts["teacup"][:, 2].max()])
    cup_off = (cup_top - keypoints[roles["teacup"]]).tolist()
    h, m, c = roles["teapot"], roles["mouth"], roles["teacup"]


    lever = float(np.linalg.norm(keypoints[m] - keypoints[h]))
    rest_dz = float(keypoints[m][2] - keypoints[h][2])
    pour_margin = rest_dz - max(0.03, 0.5 * lever)

    metadata = {"num_stages": 3, "grasp_keypoints": [h, -1, -1],
                "release_keypoints": [-1, -1, -1]}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    def write(stage, kind, body):
        with open(os.path.join(out_dir, f"stage{stage}_{kind}_constraints.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    write(1, "subgoal", f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    """Grasp the teapot: align the end-effector with the teapot keypoint."""
    return np.linalg.norm(end_effector - keypoints[{h}])
''')
    write(1, "path", "")
    write(2, "subgoal", f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    """Carry the teapot mouth above the cup (10 cm over the rim)."""
    return np.linalg.norm(keypoints[{m}] - (keypoints[{c}] + np.array({cup_off}) + np.array([0.0, 0.0, 0.10])))
''')
    write(2, "path", f'''def stage2_path_constraint1(end_effector, keypoints):
    """Keep grasping the teapot while carrying."""
    return get_grasping_cost_by_keypoint_idx({h})
''')
    write(3, "subgoal", f'''def stage3_subgoal_constraint1(end_effector, keypoints):
    """Pour: bring the mouth just above the cup rim."""
    return np.linalg.norm(keypoints[{m}] - (keypoints[{c}] + np.array({cup_off}) + np.array([0.0, 0.0, 0.04])))

def stage3_subgoal_constraint2(end_effector, keypoints):
    """Pour: drop the mouth below its rest offset by the pour-angle lever arm."""
    return np.maximum(0.0, keypoints[{m}][..., 2] - keypoints[{h}][..., 2] - ({pour_margin}))
''')
    write(3, "path", f'''def stage3_path_constraint1(end_effector, keypoints):
    """Keep grasping the teapot while pouring."""
    return get_grasping_cost_by_keypoint_idx({h})
''')
    print(f"[fake-vlm] tea roles teapot=kp{h} mouth=kp{m} teacup=kp{c}", flush=True)
    return metadata, roles


def _pot(out_dir, keypoints, grounded, env, clearance):
    """Generate constraints for moving the lid and placing the egg."""
    roles = {}
    pts = {}
    for name in ("pot", "cover", "egg"):
        p = _masked_points(grounded, env, name)
        if p is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        pts[name] = p
    if grounded.get("gt_meta") is not None:

        roles["cover"], roles["egg"], roles["pot"] = 0, 1, 2
    else:


        taken = set()
        roles["pot"] = _nearest_kp_distinct(keypoints, pts["pot"].mean(axis=0), taken)
        taken.add(roles["pot"])
        roles["cover"] = _nearest_kp_distinct(keypoints, _rim_point(pts["cover"]), taken)
        taken.add(roles["cover"])
        roles["egg"] = _nearest_kp_distinct(keypoints, pts["egg"].mean(axis=0), taken)
    lid, egg, pot = roles["cover"], roles["egg"], roles["pot"]


    lid_dz = float(pts["cover"][:, 2].mean() - pts["pot"][:, 2].max())
    lid_dxy = float(np.linalg.norm(pts["cover"][:, :2].mean(axis=0)
                                   - pts["pot"][:, :2].mean(axis=0)))
    print(f"[fake-vlm] pot lid offset: dz_above_rim={lid_dz:.3f} dxy={lid_dxy:.3f} "
          f"(removal thresholds ~0.06 dz / 0.08 dxy)", flush=True)
    table_z = float(pts["pot"][:, 2].min())
    lid_half = float((pts["cover"][:, 2].max() - pts["cover"][:, 2].min()) / 2.0)
    lid_xy = pts["cover"][:, :2].mean(axis=0)
    egg_xy = pts["egg"][:, :2].mean(axis=0)
    u = lid_xy - egg_xy
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0])
    aside = np.array([lid_xy[0] + 0.18 * u[0], lid_xy[1] + 0.18 * u[1],
                      table_z + lid_half + clearance])

    lid_off = (aside - keypoints[pot]).tolist()
    rim_top = np.array([pts["pot"][:, 0].mean(), pts["pot"][:, 1].mean(),
                        pts["pot"][:, 2].max()])
    egg_off = (rim_top + np.array([0.0, 0.0, 0.02]) - keypoints[pot]).tolist()

    metadata = {"num_stages": 4, "grasp_keypoints": [lid, -1, egg, -1],
                "release_keypoints": [-1, lid, -1, egg]}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    def write(stage, kind, body):
        with open(os.path.join(out_dir, f"stage{stage}_{kind}_constraints.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    write(1, "subgoal", f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    """Grasp the pot lid: align the end-effector with the lid keypoint."""
    return np.linalg.norm(end_effector - keypoints[{lid}])
''')
    write(1, "path", "")
    write(2, "subgoal", f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    """Set the lid aside on the table, clear of the pot."""
    return np.linalg.norm(keypoints[{lid}] - (keypoints[{pot}] + np.array({lid_off})))
''')
    write(2, "path", f'''def stage2_path_constraint1(end_effector, keypoints):
    """Keep grasping the lid while carrying it aside."""
    return get_grasping_cost_by_keypoint_idx({lid})
''')
    write(3, "subgoal", f'''def stage3_subgoal_constraint1(end_effector, keypoints):
    """Grasp the egg: align the end-effector with the egg keypoint."""
    return np.linalg.norm(end_effector - keypoints[{egg}])
''')
    write(3, "path", "")
    write(4, "subgoal", f'''def stage4_subgoal_constraint1(end_effector, keypoints):
    """Place the egg into the pot (egg keypoint just above the rim centre)."""
    return np.linalg.norm(keypoints[{egg}] - (keypoints[{pot}] + np.array({egg_off})))
''')
    write(4, "path", f'''def stage4_path_constraint1(end_effector, keypoints):
    """Keep grasping the egg while placing."""
    return get_grasping_cost_by_keypoint_idx({egg})
''')
    print(f"[fake-vlm] pot roles lid=kp{lid} egg=kp{egg} pot=kp{pot}", flush=True)
    return metadata, roles


_FAKE_VLMS = {"weight": _weight, "capsule": _capsule, "tea": _tea, "pot": _pot}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write fake VLM metadata and constraint files for a task."""
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}, register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    return _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
