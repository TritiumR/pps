"""Generate deterministic ReKep constraint artifacts for supported tasks."""
import json
import os

import numpy as np

from vlm_dp.grounding.masks import _masked_points, _nearest_kp

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_vlm_output")

# Height an object is raised before transport. Matches the lift the `template` compiler injects
# (rekep._LIFT_HEIGHT), so a plan carrying its own explicit lift stage is structurally equal to
# the templated one -- with the difference that every stage keeps its constraints under `_vlm`.
_LIFT_HEIGHT = 0.15


def _render(task_key, out_dir, **fields):
    """Render a task's raw.txt plan into the per-stage files the loader reads.

    gt_vlm_output/<task>/raw.txt is the single source of truth: the whole plan in one file,
    in the same shape a real GPT-4o response takes. This splits it exactly as
    rekep/constraint_generation.py splits a live response -- functions run from a column-0
    `def ` to their `    return `, and group by the stage prefix (the trailing part of the
    name is the constraint index).

    {placeholders} carry the values only resolvable at runtime: keypoint indices resolved
    against the live scene, and offsets measured from the observed point cloud.

    Returns the parsed metadata (num_stages / grasp_keypoints / release_keypoints).
    """
    raw_path = os.path.join(_TEMPLATE_DIR, task_key, "raw.txt")
    if not os.path.isfile(raw_path):
        raise SystemExit(f"[fake-vlm] no raw.txt for task {task_key!r} at {raw_path}")
    with open(raw_path, encoding="utf-8") as f:
        output = f.read().format(**fields)

    # --- split into function blocks (same convention as the real parser) ---
    lines, functions, start, name = output.split("\n"), {}, None, None
    for i, line in enumerate(lines):
        if line.startswith("def "):
            start, name = i, line.split("(")[0].split("def ")[1]
        elif line.startswith("    return ") and name is not None:
            functions[name] = lines[start:i + 1]
            start, name = None, None

    grouped = {}
    for fn_name in functions:                      # stage3_subgoal_constraint2 -> stage3_subgoal
        grouped.setdefault("_".join(fn_name.split("_")[:-1]), []).append(fn_name)

    # --- metadata ---
    def _line(key):
        for line in lines:
            if line.startswith(f"{key} = "):
                return line.split(" = ", 1)[1].strip()
        raise SystemExit(f"[fake-vlm] {key} not found in {raw_path}")

    def _int_list(text):
        return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",")]

    metadata = {"num_stages": int(_line("num_stages")),
                "grasp_keypoints": _int_list(_line("grasp_keypoints")),
                "release_keypoints": _int_list(_line("release_keypoints"))}

    # --- write one file per (stage, kind), including the empty ones ---
    # load_stage() reads every stage{N}_path_constraints.txt unconditionally, so a stage with
    # no path constraint still needs the file to exist.
    for idx in range(1, metadata["num_stages"] + 1):
        for kind in ("subgoal", "path"):
            key = f"stage{idx}_{kind}"
            body = "\n\n".join("\n".join(functions[n]) for n in sorted(grouped.get(key, [])))
            with open(os.path.join(out_dir, f"{key}_constraints.txt"), "w", encoding="utf-8") as f:
                f.write(body + "\n" if body else "")
    return metadata


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
    # Lift targets: the object's pick-up position raised _LIFT_HEIGHT, expressed as an offset
    # from the scale keypoint because that one is on a fixture and does not move. Anchoring to
    # the carried object's own keypoint would be degenerate (the target would track the object).
    lift = {n: (keypoints[k] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[s]).tolist()
            for n, k in (("pear", p), ("apple", a))}
    metadata = _render("weight", out_dir, p=p, a=a, s=s,
                       off_pear=off["pear"], off_apple=off["apple"],
                       lift_pear=lift["pear"], lift_apple=lift["apple"])
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] weight roles pear=kp{p} apple=kp{a} scale=kp{s}", flush=True)
    return metadata, roles


def _capsule(out_dir, keypoints, grounded, env, clearance):
    """Generate the fixed capsule-task constraint plan."""
    lip, open_goal, pod, bay = 0, 1, 2, 3
    lift_pod = (keypoints[pod] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[bay]).tolist()
    metadata = _render("capsule", out_dir, lip=lip, open_goal=open_goal, pod=pod, bay=bay,
                       lift_pod=lift_pod)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

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

    lift_teapot = (keypoints[h] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[c]).tolist()
    metadata = _render("tea", out_dir, h=h, m=m, c=c, cup_off=cup_off, pour_margin=pour_margin,
                       lift_teapot=lift_teapot)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
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

    metadata = _render("pot", out_dir, lid=lid, egg=egg, pot=pot, lid_off=lid_off, egg_off=egg_off)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
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
