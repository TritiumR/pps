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

# Clearance a carried object holds above its seat before it descends onto it, and the slack under
# the transport altitude before the carry-height path rule starts charging.
_HOVER_HEIGHT = 0.10
_CARRY_TOL = 0.02


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

    def _opt_line(key):
        """Same as _line, but None when the plan does not state the field."""
        for line in lines:
            if line.startswith(f"{key} = "):
                return line.split(" = ", 1)[1].strip()
        return None

    def _str_list(text):
        return [x.strip().strip('"').strip("'")
                for x in text.replace("[", "").replace("]", "").split(",")]

    metadata = {"num_stages": int(_line("num_stages")),
                "grasp_keypoints": _int_list(_line("grasp_keypoints")),
                "release_keypoints": _int_list(_line("release_keypoints"))}

    # Optional DECLARED per-stage semantics. Without them the consumer falls back to grepping
    # this file's prose for "inside"/"into" and for rotation tokens, which makes an English
    # docstring load-bearing; a plan that states the fields is read for what it says instead.
    # Absent (every plan written before this) the metadata simply lacks the keys and nothing
    # downstream changes.
    for key in ("stage_place_mode", "stage_orient"):
        raw_value = _opt_line(key)
        if raw_value is not None:
            metadata[key] = _str_list(raw_value)

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


# --- MimicGen tasks (square, can, coffee, mug_cleanup, stack) -------------------------------
#
# These scenes reach the generator through mujoco_eval's rekep context, which carries a per-
# keypoint `owners` list (the body each keypoint is rigidly attached to) alongside a synthetic
# `points_of` cloud built from live poses and known extents. Keypoint ORDER is not contracted, so
# every index below is resolved from that context -- by owner first, by cloud proximity otherwise --
# and never hardcoded.


def _cloud(grounded, env, name):
    """Observed points for a named object, or None when this scene does not expose it."""
    try:
        return _masked_points(grounded, env, name)
    except Exception:                                # Isaac lookup on a name this scene lacks
        return None


def _half(pts):
    """Axis-aligned half-extents of a point cloud."""
    return (pts.max(axis=0) - pts.min(axis=0)) / 2.0


def _principal_half(pts):
    """Half-extents along the cloud's own principal axes, ascending.

    Rotation invariant, so a plate lying at an angle (the coffee machine lid, open) still reports
    its true thickness as the smallest value.
    """
    centred = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    projected = centred @ vectors
    return np.sort((projected.max(axis=0) - projected.min(axis=0)) / 2.0)


def _top(pts):
    """Centre of the top face of a point cloud."""
    return np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].max()])


def _object_kps(keypoints, grounded, env, name):
    """Indices of the keypoints belonging to an object.

    Prefers the context's `owners` attribution; falls back to every keypoint lying within reach of
    the object's observed cloud, and finally to the single nearest keypoint.
    """
    owners = grounded.get("owners") or []
    owned = [i for i, o in enumerate(owners) if o == name and i < len(keypoints)]
    if owned:
        return owned
    pts = _cloud(grounded, env, name)
    if pts is None:
        raise SystemExit(f"[fake-vlm] cannot locate {name!r}: no owner attribution and no points")
    reach = float(np.linalg.norm(_half(pts))) + 0.02
    near = [i for i, k in enumerate(keypoints)
            if float(np.linalg.norm(pts - k, axis=-1).min()) <= reach]
    return near or [_nearest_kp(keypoints, pts.mean(axis=0))]


def _pick(keypoints, candidates, probe, farthest=False):
    """Nearest (or farthest) candidate keypoint to a probe point."""
    candidates = list(candidates)
    d = np.linalg.norm(keypoints[candidates] - np.asarray(probe, dtype=np.float64)[None], axis=-1)
    return int(candidates[int(np.argmax(d) if farthest else np.argmin(d))])


def _role_kp(keypoints, grounded, env, name, probe=None):
    """Keypoint standing for an object: the one of its own nearest a probe (its centroid)."""
    candidates = _object_kps(keypoints, grounded, env, name)
    if probe is None:
        pts = _cloud(grounded, env, name)
        probe = pts.mean(axis=0) if pts is not None else keypoints[candidates].mean(axis=0)
    return _pick(keypoints, candidates, probe)


def _transport(obj_kp, anchor_kp, seat_point):
    """Lift / carry / seat offsets from a STATIC anchor, plus the two altitude scalars.

    Every offset is measured from `anchor_kp` (a fixture keypoint) rather than from the carried
    object's own keypoint: a target anchored on the payload rides the payload and is degenerate.
    """
    lift_point = obj_kp + np.array([0.0, 0.0, _LIFT_HEIGHT])
    carry_point = seat_point + np.array([0.0, 0.0, _HOVER_HEIGHT])
    floor = min(float(lift_point[2]), float(carry_point[2])) - _CARRY_TOL
    return {"lift": (lift_point - anchor_kp).tolist(),
            "carry": (carry_point - anchor_kp).tolist(),
            "seat": (seat_point - anchor_kp).tolist(),
            "pick_z": float(obj_kp[2] - anchor_kp[2]),
            "carry_z": float(floor - anchor_kp[2])}


def _finish(out_dir, metadata, roles, note):
    """Write metadata.json and report the resolved roles."""
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] {note}", flush=True)
    return metadata, roles


def _stack(out_dir, keypoints, grounded, env, clearance):
    """Constraints for stacking cubeA on cubeB. Fixture anchor: cubeB."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    a = _role_kp(keypoints, grounded, env, "cubeA")
    b = _role_kp(keypoints, grounded, env, "cubeB")
    if a == b:
        raise SystemExit("[fake-vlm] stack: cubeA and cubeB resolved to the same keypoint")
    a_pts, b_pts = _cloud(grounded, env, "cubeA"), _cloud(grounded, env, "cubeB")
    a_half = float(_half(a_pts)[2]) if a_pts is not None else 0.02
    b_top = _top(b_pts) if b_pts is not None else keypoints[b] + np.array([0.0, 0.0, 0.025])
    seat = b_top + np.array([0.0, 0.0, a_half + clearance])
    t = _transport(keypoints[a], keypoints[b], seat)
    metadata = _render("stack", out_dir, a=a, b=b, lift_a=t["lift"], carry_a=t["carry"],
                       seat_a=t["seat"], pick_z=t["pick_z"], carry_z=t["carry_z"])
    return _finish(out_dir, metadata, {"cubeA": a, "cubeB": b},
                   f"stack roles cubeA=kp{a} cubeB=kp{b} seat={np.round(seat, 3).tolist()}")


def _square(out_dir, keypoints, grounded, env, clearance):
    """Constraints for threading the square nut onto the peg. Fixture anchor: the peg."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    nut_pts = _cloud(grounded, env, "nut")
    nut_kps = _object_kps(keypoints, grounded, env, "nut")
    centre = nut_pts.mean(axis=0) if nut_pts is not None else keypoints[nut_kps].mean(axis=0)
    n = _pick(keypoints, nut_kps, centre)
    # The handle is the nut keypoint offset sideways from the ring. Candidates are restricted to
    # the nut's own height band first: a lift TARGET registered on the nut sits far above it and
    # would otherwise win a plain "farthest from the centre" vote.
    band = 3.0 * float(_half(nut_pts)[2]) if nut_pts is not None else 0.05
    level = [i for i in nut_kps if abs(float(keypoints[i][2] - centre[2])) <= band] or nut_kps
    h = _pick(keypoints, level, centre, farthest=True)
    peg_pts = _cloud(grounded, env, "peg1")
    peg_probe = _top(peg_pts) if peg_pts is not None else None
    p = _role_kp(keypoints, grounded, env, "peg1", probe=peg_probe)
    if p in nut_kps:
        raise SystemExit("[fake-vlm] square: the peg keypoint resolved onto the nut")
    # The nut ends up resting at the height it started from -- both the nut and the peg's base sit
    # on the same table -- so the seating rule is one-sided about that height: lower is free.
    seat_dz = float(centre[2] - keypoints[p][2])
    t = _transport(keypoints[h], keypoints[p], keypoints[p] + np.array([0.0, 0.0, clearance]))
    metadata = _render("square", out_dir, h=h, n=n, p=p, lift_h=t["lift"],
                       carry_n=[0.0, 0.0, _HOVER_HEIGHT], pick_z=t["pick_z"],
                       carry_z=t["carry_z"], seat_dz=seat_dz)
    return _finish(out_dir, metadata, {"nut": n, "handle": h, "peg1": p},
                   f"square roles handle=kp{h} ring=kp{n} peg_top=kp{p} seat_dz={seat_dz:.3f}")


def _can(out_dir, keypoints, grounded, env, clearance):
    """Constraints for dropping the can into the bin. Fixture anchor: the bin release point."""
    del clearance                                    # the can is released above the bin, not seated
    keypoints = np.asarray(keypoints, dtype=np.float64)
    can_kps = _object_kps(keypoints, grounded, env, "can")
    c = _role_kp(keypoints, grounded, env, "can")
    # The bin has no body of its own, so its release point is the keypoint that belongs to nothing
    # the can owns and lies farthest away in the ground plane; of the points stacked over the bin
    # (release, hover) the release one is the lower.
    others = [i for i in range(len(keypoints)) if i not in set(can_kps)]
    if not others:
        raise SystemExit("[fake-vlm] can: no keypoint over the bin (every one is owned by the can)")
    span = {i: float(np.linalg.norm(keypoints[i][:2] - keypoints[c][:2])) for i in others}
    far = max(span.values())
    d = min([i for i in others if span[i] > far - 0.01], key=lambda i: float(keypoints[i][2]))
    t = _transport(keypoints[c], keypoints[d], keypoints[d])
    metadata = _render("can", out_dir, c=c, d=d, lift_c=t["lift"],
                       carry_c=[0.0, 0.0, _HOVER_HEIGHT], pick_z=t["pick_z"],
                       carry_z=t["carry_z"])
    return _finish(out_dir, metadata, {"can": c, "bin": d},
                   f"can roles can=kp{c} bin_release=kp{d}")


def _coffee(out_dir, keypoints, grounded, env, clearance):
    """Constraints for inserting the pod and closing the lid. Fixture anchor: the pod holder."""
    del clearance                                    # the pod drops into the holder mouth
    keypoints = np.asarray(keypoints, dtype=np.float64)
    # The machine's own bounding box swallows both the holder and the lid, so the context
    # attributes their keypoints to the machine and owner lookup cannot separate them. Claim the
    # three roles in turn against the geometry instead: the pod first, then the point sitting on
    # the holder's rim, then whatever is left lying on the lid.
    c = _role_kp(keypoints, grounded, env, "coffee_pod")
    pod_kps = set(_object_kps(keypoints, grounded, env, "coffee_pod")) | {c}
    holder_pts = _cloud(grounded, env, "coffee_pod_holder")
    free = [i for i in range(len(keypoints)) if i not in pod_kps]
    if holder_pts is not None and free:
        d = _pick(keypoints, free, _top(holder_pts))
    else:
        d = _role_kp(keypoints, grounded, env, "coffee_pod_holder")
    lid_pts = _cloud(grounded, env, "coffee_machine_lid")
    rest = [i for i in free if i != d]
    if lid_pts is not None and rest:
        lid = min(rest, key=lambda i: float(np.linalg.norm(lid_pts - keypoints[i], axis=-1).min()))
    else:
        lid = _role_kp(keypoints, grounded, env, "coffee_machine_lid")
    if len({c, d, lid}) != 3:
        raise SystemExit(f"[fake-vlm] coffee: pod/holder/lid collapsed onto {sorted({c, d, lid})}")
    # Where the lid ends up when it is shut: flat over the holder, one plate thickness above its
    # rim, with the press point another thickness above that. Measured, so it survives a scene
    # whose machine sits somewhere else.
    thickness = float(_principal_half(lid_pts)[0]) if lid_pts is not None else 0.01
    rim = _top(holder_pts) if holder_pts is not None else keypoints[d]
    closed = np.array([rim[0], rim[1], float(rim[2]) + 2.0 * thickness])
    t = _transport(keypoints[c], keypoints[d], keypoints[d])
    metadata = _render("coffee", out_dir, c=c, d=d, l=lid, lift_c=t["lift"],
                       carry_c=[0.0, 0.0, _HOVER_HEIGHT], pick_z=t["pick_z"],
                       carry_z=t["carry_z"], lid_closed=(closed - keypoints[d]).tolist())
    return _finish(out_dir, metadata, {"coffee_pod": c, "coffee_pod_holder": d,
                                       "coffee_machine_lid": lid},
                   f"coffee roles pod=kp{c} holder=kp{d} lid=kp{lid} "
                   f"lid_thickness={thickness:.4f}")


def _mug_cleanup(out_dir, keypoints, grounded, env, clearance):
    """Constraints for the open-drawer / stow-mug / close-drawer ladder.

    Two drawer keypoints, and they must NOT be the same one:

      handle  rides the sliding front (owner "drawer_link"). It is a GRASP keypoint -- pulling a
              drawer is a grasp of its handle -- so it is held through the pull, which is what
              lets the pull sub-goal be written on the handle and still steer candidates.
      anchor  sits on the cabinet BODY (owner "drawer"). It never moves, so every offset target
              here is measured from it. Anchored on the handle instead, the pull target would
              recede exactly as fast as the drawer travels.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    m = _role_kp(keypoints, grounded, env, "mug")
    body_pts = _cloud(grounded, env, "drawer")
    if body_pts is None:
        raise SystemExit("[fake-vlm] mug_cleanup: no points for the drawer body")
    centre = body_pts.mean(axis=0)
    owners = grounded.get("owners") or []
    link_kps = [i for i, o in enumerate(owners) if o == "drawer_link" and i < len(keypoints)]
    body_kps = [i for i, o in enumerate(owners) if o == "drawer" and i < len(keypoints)]
    if not link_kps or not body_kps:
        # Without both, the plan cannot separate what moves from what it is measured against, and
        # every rule it could write on the drawer would be either frozen or self-referential.
        raise SystemExit(
            "[fake-vlm] mug_cleanup: the context attributes no keypoint to 'drawer_link' "
            f"(sliding front: {link_kps}) or none to 'drawer' (cabinet body: {body_kps}). "
            "Rebuild it with `python -m mujoco_eval.grounding.make_context --task mug_cleanup`.")
    # The handle is the point on the sliding front that sticks farthest out in the ground plane;
    # the other front-owned points (the tray seat) sit over the cabinet footprint.
    a = max(link_kps, key=lambda i: float(np.linalg.norm(keypoints[i][:2] - centre[:2])))
    s = _pick(keypoints, body_kps, centre)
    # The drawer opens towards its own handle, and travels by its own depth along that direction.
    direction = keypoints[a][:2] - centre[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise SystemExit("[fake-vlm] mug_cleanup: the handle keypoint sits on the drawer axis")
    axis = np.array([direction[0] / norm, direction[1] / norm, 0.0])
    projected = body_pts @ axis
    pull = float((projected.max() - projected.min()) / 2.0)
    lateral = np.array([-axis[1], axis[0], 0.0])
    # Where the handle sits with the drawer shut, as an offset from the static anchor: travel is
    # measured from that line, so both the pull and the push read zero at the closed pose.
    closed = keypoints[a] - keypoints[s]
    # Seat: over the drawer tray once it has slid out. The tray (drawer_link) gives its height
    # directly; without it, fall back to the cabinet centre, which is lower and still inside.
    tray_pts = _cloud(grounded, env, "drawer_link")
    tray = tray_pts.mean(axis=0) if tray_pts is not None else centre
    mug_pts = _cloud(grounded, env, "mug")
    mug_half = float(_half(mug_pts)[2]) if mug_pts is not None else 0.031
    seat = np.array([tray[0], tray[1], float(tray[2]) + mug_half + clearance]) + pull * axis
    t = _transport(keypoints[m], keypoints[s], seat)
    retreat = closed + (pull + 0.05) * axis + np.array([0.0, 0.0, _HOVER_HEIGHT])
    metadata = _render("mug_cleanup", out_dir, a=a, s=s, m=m, open_axis=axis.tolist(),
                       lateral_axis=lateral.tolist(), pull=pull, closed=closed.tolist(),
                       retreat=retreat.tolist(),
                       lift_m=t["lift"], carry_m=t["carry"], seat_m=t["seat"],
                       pick_z=t["pick_z"], carry_z=t["carry_z"])
    return _finish(out_dir, metadata, {"mug": m, "handle": a, "drawer": s},
                   f"mug_cleanup roles handle=kp{a} (sliding front) anchor=kp{s} (cabinet) "
                   f"mug=kp{m} pull={pull:.3f} axis={np.round(axis, 3).tolist()} "
                   f"seat={np.round(seat, 3).tolist()}")


# --- RoboLab tasks (banana_in_bowl, mustard_left_bin) ---------------------------------------
#
# Same contract as the MimicGen block above: the scene reaches the generator through a rekep
# context carrying per-keypoint `owners` and a synthetic `points_of` cloud, so every index is
# resolved from that context and never hardcoded. What differs is only that these scenes come
# from RoboLab (robolab_eval), which supplies the context and the extents table.


def _extreme_pair(keypoints, candidates):
    """The two candidates lying farthest apart -- an elongated body's ends."""
    best, far = (candidates[0], candidates[-1]), -1.0
    for i in candidates:
        for j in candidates:
            if j <= i:
                continue
            d = float(np.linalg.norm(keypoints[i] - keypoints[j]))
            if d > far:
                best, far = (i, j), d
    return best


def _mouth_radius(pts, fallback=0.06):
    """Half-width of the narrowest horizontal side of a container's mouth."""
    if pts is None:
        return fallback
    half = _half(pts)
    return float(min(half[0], half[1]))


def _release_seat(mouth_kp, payload_pts, clearance):
    """Release point over a container mouth: the mouth keypoint, one payload half-height up.

    The gripper cannot follow a payload down to a container's floor -- the walls are in the way --
    so the last few centimetres are a drop, exactly as the `can` plan does it. The seat is
    therefore written at the RIM, not at the bottom of the container.

    Anchored on the MOUTH KEYPOINT rather than the top of the container's cloud: the keypoint is
    measured on the rim by the context builder, whereas a cloud built around a body whose origin
    is not its centre puts its "top" somewhere inside the container.
    """
    half_h = float(_half(payload_pts)[2]) if payload_pts is not None else 0.02
    return np.asarray(mouth_kp, dtype=np.float64) + np.array([0.0, 0.0, half_h + clearance])


def _banana_in_bowl(out_dir, keypoints, grounded, env, clearance):
    """Constraints for putting the banana in the bowl. Fixture anchor: the bowl mouth."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    banana_kps = _object_kps(keypoints, grounded, env, "banana")
    banana_pts, bowl_pts = _cloud(grounded, env, "banana"), _cloud(grounded, env, "bowl")
    centre = (banana_pts.mean(axis=0) if banana_pts is not None
              else keypoints[banana_kps].mean(axis=0))
    b = _pick(keypoints, banana_kps, centre)
    rest = [i for i in banana_kps if i != b]
    if len(rest) < 2:
        # The stage-4 mouth rule is stated on the two ENDS; without them the plan would have to
        # fall back to the centroid, which cannot see a yaw that hangs one end outside the bowl.
        raise SystemExit(f"[fake-vlm] banana_in_bowl: the context gives the banana "
                         f"{len(banana_kps)} keypoints ({banana_kps}); it needs a middle and two "
                         f"ends. Rebuild it with robolab_eval.grounding.make_context.")
    e1, e2 = _extreme_pair(keypoints, rest)
    w = _role_kp(keypoints, grounded, env, "bowl",
                 probe=_top(bowl_pts) if bowl_pts is not None else None)
    if w in banana_kps:
        raise SystemExit("[fake-vlm] banana_in_bowl: the bowl keypoint resolved onto the banana")
    seat = _release_seat(keypoints[w], banana_pts, clearance)
    t = _transport(keypoints[b], keypoints[w], seat)
    mouth_r = _mouth_radius(bowl_pts)
    metadata = _render("banana_in_bowl", out_dir, b=b, e1=e1, e2=e2, w=w,
                       lift_b=t["lift"], carry_b=t["carry"], seat_b=t["seat"],
                       pick_z=t["pick_z"], carry_z=t["carry_z"], mouth_r=round(mouth_r, 4))
    return _finish(out_dir, metadata, {"banana": b, "banana_end1": e1, "banana_end2": e2,
                                       "bowl": w},
                   f"banana_in_bowl roles banana=kp{b} ends=kp{e1}/kp{e2} bowl=kp{w} "
                   f"seat={np.round(seat, 3).tolist()} mouth_r={mouth_r:.3f}")


def _mustard_left_bin(out_dir, keypoints, grounded, env, clearance):
    """Constraints for dropping the mustard bottle into the LEFT bin. Anchor: the left bin."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    mustard_kps = _object_kps(keypoints, grounded, env, "mustard")
    mustard_pts = _cloud(grounded, env, "mustard")
    left_pts, right_pts = (_cloud(grounded, env, "grey_bin_left"),
                           _cloud(grounded, env, "grey_bin_right"))
    centre = (mustard_pts.mean(axis=0) if mustard_pts is not None
              else keypoints[mustard_kps].mean(axis=0))
    m = _pick(keypoints, mustard_kps, centre)
    rest = [i for i in mustard_kps if i != m]
    if not rest:
        raise SystemExit(f"[fake-vlm] mustard_left_bin: the context gives the bottle "
                         f"{len(mustard_kps)} keypoints ({mustard_kps}); it needs a body and a "
                         f"top. Rebuild it with robolab_eval.grounding.make_context.")
    tk = max(rest, key=lambda i: float(keypoints[i][2]))
    left = _role_kp(keypoints, grounded, env, "grey_bin_left",
                    probe=_top(left_pts) if left_pts is not None else None)
    right = _role_kp(keypoints, grounded, env, "grey_bin_right",
                     probe=_top(right_pts) if right_pts is not None else None)
    if len({m, tk, left, right}) != 4:
        raise SystemExit(f"[fake-vlm] mustard_left_bin: roles collapsed onto "
                         f"{sorted({m, tk, left, right})} (bottle/top/left bin/right bin)")
    seat = _release_seat(keypoints[left], mustard_pts, clearance)
    t = _transport(keypoints[m], keypoints[left], seat)
    mouth_r = _mouth_radius(left_pts)
    # Keep-out around the wrong bin: its own mouth plus the bottle's widest horizontal half, so a
    # transit that clips the right bin's airspace is charged before anything can land in it.
    payload_half = float(max(_half(mustard_pts)[:2])) if mustard_pts is not None else 0.04
    keepout_r = _mouth_radius(right_pts) + payload_half
    metadata = _render("mustard_left_bin", out_dir, m=m, t=tk, l=left, r=right,
                       lift_m=t["lift"], carry_m=t["carry"], seat_m=t["seat"],
                       pick_z=t["pick_z"], carry_z=t["carry_z"],
                       mouth_r=round(mouth_r, 4), keepout_r=round(keepout_r, 4))
    return _finish(out_dir, metadata, {"mustard": m, "mustard_top": tk,
                                       "grey_bin_left": left, "grey_bin_right": right},
                   f"mustard_left_bin roles mustard=kp{m} top=kp{tk} left_bin=kp{left} "
                   f"right_bin=kp{right} seat={np.round(seat, 3).tolist()} "
                   f"mouth_r={mouth_r:.3f} keepout_r={keepout_r:.3f}")


# --- Two-bin sorting (sort_can) --------------------------------------------------------------
#
# The only mujoco task whose destination is not fixed by the scene: two walled quadrants are
# painted red and blue, the assignment is redrawn every episode, and the INSTRUCTION names the
# colour. So the plan is written once and rendered per instruction; what the instruction picks is
# the pair of keypoint indices ({d} requested bin, {o} the other one).


def _mj_model(env):
    """The MuJoCo model behind whatever env wrapper was handed in, or None."""
    for attr_chain in (("sim",), ("raw", "sim"), ("env", "sim"), ("env", "env", "sim")):
        obj = env
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and getattr(obj, "model", None) is not None:
            return obj
    return None


def _pads(env, colours=("red", "blue")):
    """Return {quadrant id: (world xyz, colour, half extents)} for the coloured overlay pads.

    THIS IS A MODEL READ, NOT VISION. The pad's pose, size and rgba are taken from the MuJoCo
    model rather than segmented out of the rendered image. It stands in for the detector a
    deployed front end would run, and it is exactly the seam a real perception stack replaces --
    everything downstream of here (which keypoint the colour names, the rendered plan, the
    compiled constraint, the selected goal) is the real path and is unaffected by how the pad
    was found.
    """
    sim = _mj_model(env)
    if sim is None:
        raise SystemExit("[fake-vlm] sort_can: no MuJoCo sim behind the env; cannot read the pads")
    palette = {"red": np.array([0.85, 0.10, 0.10, 1.0]),
               "blue": np.array([0.10, 0.20, 0.85, 1.0])}
    out = {}
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid)
        if not name or not name.startswith("sortcan_pad_q"):
            continue
        qid = int(name[len("sortcan_pad_q"):])
        rgba = np.asarray(sim.model.geom_rgba[gid], dtype=np.float64)
        colour = min(colours, key=lambda c: float(np.linalg.norm(rgba - palette[c])))
        out[qid] = (np.asarray(sim.data.geom_xpos[gid], dtype=np.float64), colour,
                    np.asarray(sim.model.geom_size[gid], dtype=np.float64))
    if len(out) < 2:
        raise SystemExit(f"[fake-vlm] sort_can: found {len(out)} coloured pads, need 2")
    return out


def _requested_colour(instruction, colours=("red", "blue")):
    """The colour the instruction asks for. Ambiguity and silence are both fatal."""
    if not instruction:
        raise SystemExit("[fake-vlm] sort_can: no instruction given; the destination colour is "
                         "the task specification and cannot be guessed")
    text = str(instruction).lower()
    hits = [c for c in colours if c in text]
    if len(hits) != 1:
        raise SystemExit(f"[fake-vlm] sort_can: instruction {instruction!r} names {hits} of "
                         f"{list(colours)}; exactly one colour is required")
    return hits[0]


def _sort_can(out_dir, keypoints, grounded, env, clearance):
    """Constraints for dropping the can into the bin the INSTRUCTION names.

    Anchor: the requested bin's release point. The other bin becomes a keep-out, so the plan
    states what it must avoid as well as what it must reach.
    """
    del clearance                                    # the can is released above the bin, not seated
    keypoints = np.asarray(keypoints, dtype=np.float64)
    can_kps = set(_object_kps(keypoints, grounded, env, "can"))
    c = _role_kp(keypoints, grounded, env, "can")
    others = [i for i in range(len(keypoints)) if i not in can_kps]
    if len(others) < 2:
        raise SystemExit(f"[fake-vlm] sort_can: the context gives {len(others)} keypoint(s) not "
                         f"owned by the can; it needs one release point per bin. Rebuild it with "
                         f"`python -m mujoco_eval.grounding.make_context --task sort_can`.")

    # Each pad claims the release keypoint nearest it in the ground plane. Claimed in a fixed
    # order and without replacement, so two pads can never collapse onto one keypoint.
    pads, kp_of_pad, taken = _pads(env), {}, set()
    for qid in sorted(pads):
        free = [i for i in others if i not in taken]
        k = min(free, key=lambda i: float(np.linalg.norm(keypoints[i][:2] - pads[qid][0][:2])))
        kp_of_pad[qid], _ = k, taken.add(k)
    if len(set(kp_of_pad.values())) != len(kp_of_pad):
        raise SystemExit(f"[fake-vlm] sort_can: pads collapsed onto {sorted(kp_of_pad.values())}")

    want = _requested_colour(grounded.get("instruction"))
    match = [q for q in kp_of_pad if pads[q][1] == want]
    if len(match) != 1:
        raise SystemExit(f"[fake-vlm] sort_can: {len(match)} pads are {want!r}; need exactly one")
    q_want = match[0]
    q_other = next(q for q in kp_of_pad if q != q_want)
    d, o = kp_of_pad[q_want], kp_of_pad[q_other]

    # Keep-out around the wrong bin: its own painted half-width plus the can's horizontal half,
    # so a transit that clips the other bin's airspace is charged before anything can land in it.
    local = (grounded.get("local_extents") or [])
    can_half = float(local[c][0]) if c < len(local) and local[c] else 0.025
    keepout_r = float(min(pads[q_other][2][:2])) + can_half

    t = _transport(keypoints[c], keypoints[d], keypoints[d])
    metadata = _render("sort_can", out_dir, c=c, d=d, o=o, colour=want, lift_c=t["lift"],
                       carry_c=[0.0, 0.0, _HOVER_HEIGHT], pick_z=t["pick_z"],
                       carry_z=t["carry_z"], keepout_r=round(keepout_r, 4))
    # Provenance for the log and for the receipts the caller writes: what the instruction was
    # taken to mean, and every association it rests on.
    metadata["task_spec"] = {"instruction": grounded.get("instruction"), "colour": want}
    metadata["resolved"] = {
        "can_kp": int(c), "requested_kp": int(d), "other_kp": int(o),
        "requested_quadrant": int(q_want), "other_quadrant": int(q_other),
        "pad_colour": {str(q): pads[q][1] for q in sorted(pads)},
        "kp_of_quadrant": {str(q): int(k) for q, k in sorted(kp_of_pad.items())},
        "keepout_r": round(keepout_r, 4),
    }
    return _finish(out_dir, metadata, {"can": c, "requested_bin": d, "other_bin": o},
                   f"sort_can roles can=kp{c} requested={want} q{q_want}=kp{d} "
                   f"other=q{q_other}=kp{o} keepout_r={keepout_r:.3f}")


# --- Continuous-goal tray placement (sort_can_tray) --------------------------------------------
#
# The sibling of sort_can with the discrete choice removed: one undivided tray, and the
# destination is a continuous position marked by an inert disc lying on the tray floor. Nothing
# has to be chosen from a menu, so no instruction parsing happens here -- the task specification
# is entirely "which keypoint is the marker", which the context's owner attribution answers.

_MARKER_OWNER = "traymarker"
_CAN_HALF_HEIGHT_FALLBACK = 0.0407


def _sort_can_tray(out_dir, keypoints, grounded, env, clearance):
    """Constraints for seating the can on the tray's marker. Fixture anchor: the marker.

    The seat offset is the can's own half-height taken from the context's measured extents, so
    "resting on the marker" is a statement about the observed can rather than a constant.
    """
    del clearance                                    # the seat height IS the can's half-height
    keypoints = np.asarray(keypoints, dtype=np.float64)
    can_kps = _object_kps(keypoints, grounded, env, "can")
    c = _role_kp(keypoints, grounded, env, "can")
    m = _role_kp(keypoints, grounded, env, _MARKER_OWNER)
    if m in set(can_kps):
        raise SystemExit("[fake-vlm] sort_can_tray: the marker resolved onto a can keypoint; "
                         "rebuild the context so the marker owns its own keypoint")

    local = (grounded.get("local_extents") or [])
    can_half_h = float(local[c][2]) if c < len(local) and local[c] else _CAN_HALF_HEIGHT_FALLBACK
    seat = keypoints[m] + np.array([0.0, 0.0, can_half_h])
    t = _transport(keypoints[c], keypoints[m], seat)
    metadata = _render("sort_can_tray", out_dir, c=c, m=m, lift_c=t["lift"], carry_c=t["carry"],
                       seat_c=t["seat"], pick_z=t["pick_z"], carry_z=t["carry_z"])
    metadata["task_spec"] = {"instruction": grounded.get("instruction"),
                             "destination": "the marker disc in the tray"}
    metadata["resolved"] = {"can_kp": int(c), "marker_kp": int(m),
                            "can_half_height": round(can_half_h, 5),
                            "seat_world": np.round(seat, 6).tolist()}
    return _finish(out_dir, metadata, {"can": c, "marker": m},
                   f"sort_can_tray roles can=kp{c} marker=kp{m} "
                   f"seat={np.round(seat, 3).tolist()}")


# --- Semantic-programmability tray families (sort_can_tray_{region,relational,constrained}) -----
#
# Three NEW objectives over the SAME scene, the same keypoints and the same grasp/lift/carry
# stages as `_sort_can_tray`. Only the terminal stage differs, so what changes between them is the
# PROGRAM and nothing else: the question they exist to answer is whether a frozen goal-conditioned
# expert can be re-tasked by editing the ReKep program alone.
#
# Every family's terminal stage is written as `semantic hinges + _SEMPROG_TIE_W * transport`, a
# shared lexicographic objective: the hinges are exactly zero on the feasible set and grow at one
# metre per metre outside it, and the tie-break -- the horizontal distance from where the can was
# picked up -- orders the satisfying positions without ever being able to buy a violation
# (it shrinks at most _SEMPROG_TIE_W metres per metre). The tie-break's anchor is the can's
# OBSERVED pick-up position, carried into the program as an offset from the static marker, because
# the payload keypoint itself rides the candidate and cannot report where the can started.
#
# `traymarker` and `bin2` (the tray floor, whose centre is "the tray centre") are static fixtures
# and each owns its own keypoint, so both survive the payload substitution unmoved.

_TRAY_OWNER = "bin2"
_SEMPROG_TIE_W = 0.001
_SEMPROG_REGION_RADIUS = 0.06
_SEMPROG_SIDE_MARGIN = 0.03
_SEMPROG_NEAR_RADIUS = 0.08
_SEMPROG_CENTRE_CLEARANCE = 0.05
# Instruction token -> the sign the terminal hinge is written with. +1 charges positions on the
# +x side of the marker, so it demands the -x side, which this task calls "left".
_SEMPROG_SIDES = {"left": 1.0, "right": -1.0}


def _semprog_roles(keypoints, grounded, env, need_tray=False):
    """Resolve the tray families' shared roles and transport offsets.

    Returns (fields, roles, extra) where `fields` are the placeholders every family's raw.txt
    shares. The carry waypoint is stated over the marker: the terminal stage, not the carry, is
    what fixes the final position, and a transit target cannot be written against a set.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    can_kps = _object_kps(keypoints, grounded, env, "can")
    c = _role_kp(keypoints, grounded, env, "can")
    m = _role_kp(keypoints, grounded, env, _MARKER_OWNER)
    if m in set(can_kps):
        raise SystemExit("[fake-vlm] semprog: the marker resolved onto a can keypoint; rebuild "
                         "the context so the marker owns its own keypoint")
    roles = {"can": c, "marker": m}
    if need_tray:
        t = _role_kp(keypoints, grounded, env, _TRAY_OWNER)
        if t in set(can_kps) or t == m:
            raise SystemExit("[fake-vlm] semprog: the tray centre resolved onto the can or the "
                             "marker; rebuild the context so bin2 owns its own keypoint")
        roles["tray"] = t

    local = (grounded.get("local_extents") or [])
    can_half_h = float(local[c][2]) if c < len(local) and local[c] else _CAN_HALF_HEIGHT_FALLBACK
    seat = keypoints[m] + np.array([0.0, 0.0, can_half_h])
    tr = _transport(keypoints[c], keypoints[m], seat)
    fields = {"c": c, "m": m, "lift_c": tr["lift"], "carry_c": tr["carry"],
              "pick_z": tr["pick_z"], "carry_z": tr["carry_z"],
              "start_c": (keypoints[c] - keypoints[m]).tolist(), "w": _SEMPROG_TIE_W}
    if need_tray:
        fields["t"] = roles["tray"]
    extra = {"can_half_height": round(can_half_h, 5),
             "start_world": np.round(keypoints[c], 6).tolist(),
             "tie_break_weight": _SEMPROG_TIE_W}
    return fields, roles, extra


def _sort_can_tray_region(out_dir, keypoints, grounded, env, clearance):
    """REGION family: the can ends up anywhere within a fixed radius of the marker."""
    del clearance
    fields, roles, extra = _semprog_roles(keypoints, grounded, env)
    metadata = _render("sort_can_tray_region", out_dir,
                       radius=_SEMPROG_REGION_RADIUS, **fields)
    metadata["task_spec"] = {"instruction": grounded.get("instruction"),
                             "family": "region",
                             "destination": f"within {_SEMPROG_REGION_RADIUS} m of the marker"}
    metadata["resolved"] = {"can_kp": int(fields["c"]), "marker_kp": int(fields["m"]),
                            "region_radius": _SEMPROG_REGION_RADIUS, **extra}
    return _finish(out_dir, metadata, roles,
                   f"sort_can_tray_region roles can=kp{fields['c']} marker=kp{fields['m']} "
                   f"radius={_SEMPROG_REGION_RADIUS}")


def _sort_can_tray_relational(out_dir, keypoints, grounded, env, clearance):
    """RELATIONAL family: the can ends up on the side of the marker the INSTRUCTION named."""
    del clearance
    instruction = (grounded.get("instruction") or "").lower()
    named = [s for s in _SEMPROG_SIDES if s in instruction]
    if len(named) != 1:
        raise SystemExit(f"[fake-vlm] semprog relational: the instruction must name exactly one "
                         f"of {sorted(_SEMPROG_SIDES)}, got {named} in {instruction!r}")
    side = named[0]
    fields, roles, extra = _semprog_roles(keypoints, grounded, env)
    metadata = _render("sort_can_tray_relational", out_dir, side=side,
                       s=_SEMPROG_SIDES[side], margin=_SEMPROG_SIDE_MARGIN, **fields)
    metadata["task_spec"] = {"instruction": grounded.get("instruction"),
                             "family": "relational",
                             "destination": f"{side} of the marker by >= "
                                            f"{_SEMPROG_SIDE_MARGIN} m along x"}
    metadata["resolved"] = {"can_kp": int(fields["c"]), "marker_kp": int(fields["m"]),
                            "side": side, "sign": _SEMPROG_SIDES[side],
                            "side_margin": _SEMPROG_SIDE_MARGIN, **extra}
    return _finish(out_dir, metadata, roles,
                   f"sort_can_tray_relational roles can=kp{fields['c']} marker=kp{fields['m']} "
                   f"side={side} sign={_SEMPROG_SIDES[side]:+.0f}")


def _sort_can_tray_constrained(out_dir, keypoints, grounded, env, clearance):
    """CONSTRAINED family: near the marker AND clear of the tray centre -- a lens."""
    del clearance
    fields, roles, extra = _semprog_roles(keypoints, grounded, env, need_tray=True)
    metadata = _render("sort_can_tray_constrained", out_dir, r_near=_SEMPROG_NEAR_RADIUS,
                       r_far=_SEMPROG_CENTRE_CLEARANCE, **fields)
    metadata["task_spec"] = {"instruction": grounded.get("instruction"),
                             "family": "constrained",
                             "destination": f"within {_SEMPROG_NEAR_RADIUS} m of the marker and "
                                            f"at least {_SEMPROG_CENTRE_CLEARANCE} m from the "
                                            f"tray centre"}
    metadata["resolved"] = {"can_kp": int(fields["c"]), "marker_kp": int(fields["m"]),
                            "tray_kp": int(fields["t"]), "near_radius": _SEMPROG_NEAR_RADIUS,
                            "centre_clearance": _SEMPROG_CENTRE_CLEARANCE, **extra}
    return _finish(out_dir, metadata, roles,
                   f"sort_can_tray_constrained roles can=kp{fields['c']} "
                   f"marker=kp{fields['m']} tray=kp{fields['t']} "
                   f"near={_SEMPROG_NEAR_RADIUS} clear={_SEMPROG_CENTRE_CLEARANCE}")


_FAKE_VLMS = {"weight": _weight, "capsule": _capsule, "tea": _tea, "pot": _pot,
              "stack": _stack, "square": _square, "can": _can, "coffee": _coffee,
              "mug_cleanup": _mug_cleanup, "sort_can": _sort_can,
              "sort_can_tray": _sort_can_tray,
              "sort_can_tray_region": _sort_can_tray_region,
              "sort_can_tray_relational": _sort_can_tray_relational,
              "sort_can_tray_constrained": _sort_can_tray_constrained,
              "banana_in_bowl": _banana_in_bowl, "mustard_left_bin": _mustard_left_bin}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write fake VLM metadata and constraint files for a task."""
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}, register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    return _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
