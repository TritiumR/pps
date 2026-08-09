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
# Closed-lid contact lip in the calibrated machine-root frame, measured from the asset geometry.
# Unlike the front-most visible slab point, this remains on the articulated lip under camera occlusion.
_CAPSULE_LIP_LOCAL = np.array([-0.0784, -0.2392, 0.3924])


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
    metadata = _render("weight", out_dir, p=p, a=a, s=s,
                       off_pear=off["pear"], off_apple=off["apple"],
                       lift_pear=lift["pear"], lift_apple=lift["apple"],
                       hover_pear=hover["pear"], hover_apple=hover["apple"],
                       carry_z_pear=carry_z["pear"], carry_z_apple=carry_z["apple"])
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

    # The machine is a calibrated static fixture; this never reads its articulated joint.
    data = env.scene["capsule"].data
    root = data.root_pos_w[0].cpu().numpy().astype(np.float64)
    quat = data.root_quat_w[0].cpu().numpy()
    lid_lip_world = root + _quat_rotate_wxyz(quat, _CAPSULE_LIP_LOCAL)
    # Match the successful task-policy posture: descend vertically around the front
    # lip, close the fingers, then lift the lip while retaining the vertical tool axis.
    lip_world = lid_lip_world + np.array([0.0, 0.0, -0.012])
    open_world = lip_world + np.array([0.0, 0.0, 0.12])
    approach_axis = np.array([0.0, 0.0, -1.0])
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
    metadata["contact_modes"] = {"0": "press"}
    metadata["contact_slack"] = {"0": 0.012}
    # Stage 2 is the release stage syntactically, but physically it must keep contact
    # while lifting the lid lip to the open goal and release only after arriving.
    metadata["release_targets"] = {"1": open_goal}
    metadata["release_done_targets"] = [1]
    metadata["approach_axes"] = {
        "0": approach_axis.tolist(), "1": approach_axis.tolist()
    }
    metadata["approach_axis_scales"] = {"0": 4.0, "1": 6.0}
    metadata["static_keypoints"] = [lip, bay]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] capsule roles pod=kp{pod} (snapped, {np.round(kps[pod], 3)}) "
          f"lip=kp{lip} (declared, {np.round(lip_world, 3)}) "
          f"open_goal=kp{open_goal} (declared, {np.round(open_world, 3)}) "
          f"bay=kp{bay} (declared, {np.round(bay_world, 3)})", flush=True)
    print(f"[fake-vlm] capsule approach axis={np.round(approach_axis, 3)}", flush=True)
    _capsule_lip_diagnostic(env, lid_lip_world)
    return metadata, {"lid": lip, "open_goal": open_goal, "pod": pod, "bay": bay}, extra


def _capsule_lip_diagnostic(env, lip_world):
    """Report how far the measured lid lip sits from the privileged one (diagnostic only)."""
    try:
        data = env.scene["capsule"].data
        from vlm_dp.grounding.capsule import gt_keypoints
        from types import SimpleNamespace

        gt_kps, _ = gt_keypoints(SimpleNamespace(env=env))
        err = float(np.linalg.norm(np.asarray(gt_kps[0]) - lip_world)) * 1e3
        root = data.root_pos_w[0].cpu().numpy().astype(np.float64)
        quat = data.root_quat_w[0].cpu().numpy()
        inv_quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        local = _quat_rotate_wxyz(inv_quat, np.asarray(gt_kps[0]) - root)
        print(f"[fake-vlm] capsule lip check: measured {np.round(lip_world, 3)} vs privileged "
              f"{np.round(np.asarray(gt_kps[0]), 3)} ({err:.0f}mm), root-local "
              f"{np.round(local, 4)} -- diagnostic only, "
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

    lift_teapot = (keypoints[h] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - keypoints[c]).tolist()
    metadata = _render("tea", out_dir, h=h, m=m, c=c, cup_off=cup_off, pour_margin=pour_margin,
                       lift_teapot=lift_teapot)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] tea roles teapot=kp{h} mouth=kp{m} teacup=kp{c}", flush=True)
    return metadata, roles


def _pot_handle_point(cover_pts):
    """Locate the compact lid handle protruding above the broad lid disk."""
    pts = np.asarray(cover_pts, dtype=np.float64)
    rim_z = float(np.percentile(pts[:, 2], 95))
    raised = pts[pts[:, 2] > rim_z + 0.005]
    prominence = float(np.max(pts[:, 2]) - rim_z)
    if raised.shape[0] < 3 or prominence < 0.012:
        raise ValueError(
            f"[fake-vlm] pot-lid handle is not visibly resolved: {raised.shape[0]} raised "
            f"points, {prominence * 1e3:.0f}mm prominence"
        )
    # The handle is a thin arch. Its visible depth samples are heavily biased toward one
    # post, so their median can put the TCP on that post (or beside the crossbar). The
    # midpoint of the raised patch bounds is stable under the oblique camera view.
    handle = 0.5 * (raised.min(axis=0) + raised.max(axis=0))
    handle[2] = 0.5 * (
        float(np.percentile(raised[:, 2], 10)) + float(np.percentile(raised[:, 2], 90))
    )
    xy = raised[:, :2] - np.mean(raised[:, :2], axis=0)
    _, vectors = np.linalg.eigh(xy.T @ xy)
    axis = vectors[:, 0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    projection = xy @ axis
    extent = max(float(np.percentile(projection, 95) - np.percentile(projection, 5)) / 2.0, 0.005)
    print(
        f"[fake-vlm] pot handle: {raised.shape[0]} raised points, "
        f"{prominence * 1e3:.0f}mm prominence, centre={np.round(handle, 3)}, "
        f"narrow_axis={np.round(axis, 3)}, half_width={extent * 1e3:.0f}mm",
        flush=True,
    )
    return handle, (float(axis[0]), float(axis[1]), 0.0), extent


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
        extra = ()
        handle_axis, handle_extent = None, None
    else:


        taken = set()
        roles["pot"] = _nearest_kp_distinct(keypoints, pts["pot"].mean(axis=0), taken)
        taken.add(roles["pot"])
        roles["egg"] = _nearest_kp_distinct(keypoints, pts["egg"].mean(axis=0), taken)
        points_of_raw = grounded.get("points_of_raw")
        raw_cover = points_of_raw("cover") if points_of_raw is not None else pts["cover"]
        handle, handle_axis, handle_extent = _pot_handle_point(raw_cover)
        roles["cover"] = len(keypoints)
        extra = [(handle, "cover")]
    lid, egg, pot = roles["cover"], roles["egg"], roles["pot"]

    kps = np.concatenate([np.asarray(keypoints), np.asarray([extra[0][0]])]) if extra else keypoints
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
    aside = np.array([lid_xy[0] + 0.10 * u[0], lid_xy[1] + 0.10 * u[1],
                      table_z + lid_half + clearance])

    lid_off = (aside - kps[pot]).tolist()
    rim_top = np.array([pts["pot"][:, 0].mean(), pts["pot"][:, 1].mean(),
                        pts["pot"][:, 2].max()])
    egg_off = (rim_top + np.array([0.0, 0.0, 0.02]) - kps[pot]).tolist()

    # Lift targets: pick-up pose raised _LIFT_HEIGHT, anchored to the pot keypoint (the pot
    # stays put; anchoring to the carried object's own keypoint would be degenerate).
    lift_lid = (kps[lid] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - kps[pot]).tolist()
    lift_egg = (kps[egg] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - kps[pot]).tolist()
    metadata = _render("pot", out_dir, lid=lid, egg=egg, pot=pot, lid_off=lid_off, egg_off=egg_off,
                       lift_lid=lift_lid, lift_egg=lift_egg)
    metadata["grasp_targets"] = {"0": "keypoint"}
    metadata["contact_modes"] = {"0": "pinch"}
    metadata["rise_confirm"] = {"cover": 0.003}
    metadata["approach_axes"] = {"0": [0.0, 0.0, -1.0]}
    metadata["approach_axis_scales"] = {"0": 4.0}
    if handle_axis is not None:
        metadata["grasp_geometry"] = {
            "cover": {"axis": list(handle_axis), "extent": float(handle_extent)}
        }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] pot roles lid=kp{lid} egg=kp{egg} pot=kp{pot}", flush=True)
    return metadata, roles, extra


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
