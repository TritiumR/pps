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

# Transport geometry the plan references: how high the hover point sits above the place
# point, and how much slack the carry-height path rule allows below the lift height.
_CARRY_HOVER, _CARRY_SLACK = 0.10, 0.03

# Capsule geometry, in the machine-root frame. Same numbers vlm_dp/grounding/capsule.py and
# vlm_dp/offline_context.py use: the pod bay sits on the machine's vertical axis, and the lid
# travels roughly this far up when it swings open.
_CAPSULE_BAY_LOCAL = np.array([0.0, 0.0, 0.27])
_CAPSULE_LID_OPEN_LIFT = np.array([0.0, 0.0, 0.12])
# Thickness of the "top slab" of the machine cloud the lid lip is searched in.
_CAPSULE_LID_BAND = 0.04


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
    # Hover point for the transport stage: the place point raised by the carry clearance, so
    # the descent in the following stage is straight down.
    hover = {n: (np.array(off[n]) + np.array([0.0, 0.0, _CARRY_HOVER])).tolist()
             for n in ("pear", "apple")}
    # Absolute world height the plan's path rule forbids dropping below while carrying.
    carry_z = {n: float(keypoints[k][2] + _LIFT_HEIGHT - _CARRY_SLACK)
               for n, k in (("pear", p), ("apple", a))}
    # Absolute world height the LIFT stage's sub-goal measures against. The stage states its
    # sub-goal as a one-sided vertical shortfall rather than a 3-D distance to `lift`, so it
    # needs the target height as a scalar. `lift` is still supplied: the offset form is what
    # any plan wanting the full 3-D lift point reads, and dropping it would break them.
    lift_z = {n: float(keypoints[k][2] + _LIFT_HEIGHT)
              for n, k in (("pear", p), ("apple", a))}
    # Absolute world height of the hover point, for the descent stages' gated path rule: it
    # forbids dropping below this until the object is horizontally over the place point.
    hover_z = {n: float(keypoints[s][2] + hover[n][2]) for n in ("pear", "apple")}
    metadata = _render("weight", out_dir, p=p, a=a, s=s,
                       off_pear=off["pear"], off_apple=off["apple"],
                       lift_pear=lift["pear"], lift_apple=lift["apple"],
                       hover_pear=hover["pear"], hover_apple=hover["apple"],
                       carry_z_pear=carry_z["pear"], carry_z_apple=carry_z["apple"],
                       lift_z_pear=lift_z["pear"], lift_z_apple=lift_z["apple"],
                       hover_z_pear=hover_z["pear"], hover_z_apple=hover_z["apple"])
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] weight roles pear=kp{p} apple=kp{a} scale=kp{s}", flush=True)
    return metadata, roles


def _capsule(out_dir, keypoints, grounded, env, clearance):
    """Resolve the capsule roles against the live scene, then render the fixed plan.

    Three of the four roles have no proposed keypoint to snap to: the lid lip is a thin rim the
    proposer rarely samples, the open goal is a place in mid-air, and the bay is a recess inside
    the machine. They are *declared* instead -- returned as extra keypoints the caller appends and
    registers (the same trick `kp_source=gt` uses for its virtual points), so the plan can name
    them without any of them being a keypoint index the proposer happened to hand us.

    Only the pod is resolved by snapping, because the can is a real, segmented, movable object
    and its keypoint has to travel with it.
    """
    pts = {}
    for name in ("can", "capsule"):
        p = _masked_points(grounded, env, name)
        if p is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        pts[name] = p

    # The machine is a declared fixture: perception.calibrate() already reads its mask from the
    # simulator rather than the segmenter, so reading its (static) root pose here grants no
    # privilege the perception path did not already have. Everything below is measured from it or
    # from the machine's own point cloud -- no keypoint is copied from vlm_dp.grounding.capsule.
    data = env.scene["capsule"].data
    root = data.root_pos_w[0].cpu().numpy().astype(np.float64)
    quat = data.root_quat_w[0].cpu().numpy()
    front = _quat_rotate_wxyz(quat, np.array([0.0, -1.0, 0.0]))

    # Lid lip: the front-most point of the machine's top slab, from the observed cloud.
    mach = pts["capsule"]
    band = mach[mach[:, 2] >= float(np.percentile(mach[:, 2], 99)) - _CAPSULE_LID_BAND]
    lip_world = band[int(np.argmax((band - root) @ front))].astype(np.float64)
    open_world = lip_world + _CAPSULE_LID_OPEN_LIFT
    bay_world = root + _quat_rotate_wxyz(quat, _CAPSULE_BAY_LOCAL)

    pod = _nearest_kp_distinct(keypoints, pts["can"].mean(axis=0), set())
    n = len(keypoints)
    lip, open_goal, bay = n, n + 1, n + 2
    extra = [(lip_world, "capsule"), (open_world, None), (bay_world, "capsule")]

    kps = np.concatenate([np.asarray(keypoints, dtype=np.float64),
                          np.stack([lip_world, open_world, bay_world])], axis=0)
    lift_pod = (kps[pod] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - kps[bay]).tolist()
    metadata = _render("capsule", out_dir, lip=lip, open_goal=open_goal, pod=pod, bay=bay,
                       lift_pod=lift_pod)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] capsule roles pod=kp{pod} (snapped, {np.round(kps[pod], 3)}) "
          f"lip=kp{lip} (declared, {np.round(lip_world, 3)}) "
          f"open_goal=kp{open_goal} (declared, {np.round(open_world, 3)}) "
          f"bay=kp{bay} (declared, {np.round(bay_world, 3)})", flush=True)
    _capsule_lip_diagnostic(env, lip_world)
    return metadata, {"lid": lip, "open_goal": open_goal, "pod": pod, "bay": bay}, extra


def _capsule_lip_diagnostic(env, lip_world):
    """Report how far the measured lid lip sits from the privileged one (diagnostic only)."""
    try:
        from vlm_dp.grounding.capsule import gt_keypoints
        from types import SimpleNamespace

        gt_kps, _ = gt_keypoints(SimpleNamespace(env=env))
        err = float(np.linalg.norm(np.asarray(gt_kps[0]) - lip_world)) * 1e3
        print(f"[fake-vlm] capsule lip check: measured {np.round(lip_world, 3)} vs privileged "
              f"{np.round(np.asarray(gt_kps[0]), 3)} ({err:.0f}mm) -- diagnostic only, "
              f"the plan uses the measured point", flush=True)
    except Exception as exc:                       # never let a diagnostic break grounding
        print(f"[fake-vlm] capsule lip check unavailable ({exc})", flush=True)


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

    # Lift target: the handle's pick-up position raised _LIFT_HEIGHT, expressed as an offset from
    # the TEACUP keypoint because that one is on a fixture and does not move. Anchoring to the
    # carried teapot's own keypoint would be degenerate (the target would track the teapot).
    lift_teapot = (keypoints[h] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[c]).tolist()
    # Absolute world heights the plan's scalar rules measure against. The lift stage states its
    # sub-goal as a one-sided vertical shortfall rather than a 3-D distance to `lift_teapot`, so
    # it needs the target height as a scalar; `lift_teapot` is still supplied for any plan
    # wanting the full 3-D lift point. Both are on the HANDLE keypoint -- the grasped feature,
    # whose height the hand controls directly -- not the mouth at the end of the lever arm.
    lift_z_teapot = float(keypoints[h][2] + _LIFT_HEIGHT)
    carry_z_teapot = float(keypoints[h][2] + _LIFT_HEIGHT - _CARRY_SLACK)
    # Absolute world height of the pour hover point (the cup rim raised by the carry clearance
    # the stage-3 sub-goal already adds), for stage 3's completion predicate.
    hover_z_mouth = float(keypoints[c][2] + cup_off[2] + _CARRY_HOVER)
    metadata = _render("tea", out_dir, h=h, m=m, c=c, cup_off=cup_off, pour_margin=pour_margin,
                       lift_teapot=lift_teapot, lift_z_teapot=lift_z_teapot,
                       carry_z_teapot=carry_z_teapot, hover_z_mouth=hover_z_mouth)
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
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

    # Lift targets: pick-up pose raised _LIFT_HEIGHT, anchored to the pot keypoint (the pot
    # stays put; anchoring to the carried object's own keypoint would be degenerate).
    lift_lid = (keypoints[lid] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[pot]).tolist()
    lift_egg = (keypoints[egg] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[pot]).tolist()
    # Hover points for the two transport stages: each place point raised by the carry clearance,
    # so the descent in the following stage is straight down. Same pot anchor as the place points.
    hover = {"lid": (np.array(lid_off) + np.array([0.0, 0.0, _CARRY_HOVER])).tolist(),
             "egg": (np.array(egg_off) + np.array([0.0, 0.0, _CARRY_HOVER])).tolist()}
    # Absolute world heights the plan's scalar rules measure against: the LIFT stages' one-sided
    # vertical sub-goals, the transport stages' carry-height floors, and the descent gates' hover
    # heights. `lift_lid` / `lift_egg` are still supplied for any plan wanting the 3-D lift point.
    lift_z = {n: float(keypoints[k][2] + _LIFT_HEIGHT) for n, k in (("lid", lid), ("egg", egg))}
    carry_z = {n: float(keypoints[k][2] + _LIFT_HEIGHT - _CARRY_SLACK)
               for n, k in (("lid", lid), ("egg", egg))}
    hover_z = {n: float(keypoints[pot][2] + hover[n][2]) for n in ("lid", "egg")}
    metadata = _render("pot", out_dir, lid=lid, egg=egg, pot=pot, lid_off=lid_off, egg_off=egg_off,
                       lift_lid=lift_lid, lift_egg=lift_egg,
                       hover_lid=hover["lid"], hover_egg=hover["egg"],
                       lift_z_lid=lift_z["lid"], lift_z_egg=lift_z["egg"],
                       carry_z_lid=carry_z["lid"], carry_z_egg=carry_z["egg"],
                       hover_z_lid=hover_z["lid"], hover_z_egg=hover_z["egg"])
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] pot roles lid=kp{lid} egg=kp{egg} pot=kp{pot}", flush=True)
    return metadata, roles


_FAKE_VLMS = {"weight": _weight, "capsule": _capsule, "tea": _tea, "pot": _pot}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write fake VLM metadata and constraint files for a task.

    Returns ``(metadata, roles, extra_keypoints)``. ``extra_keypoints`` is a list of
    ``(world_point, owner_name_or_None)`` the task declared: points the plan needs that no
    proposed keypoint stands for. The caller appends them and registers them for tracking.
    """
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}, register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    out = _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
    return (out[0], out[1], out[2] if len(out) > 2 else ())
