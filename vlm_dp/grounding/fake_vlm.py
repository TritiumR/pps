"""Generate deterministic ReKep constraint artifacts for supported tasks."""
import json
import os

import numpy as np

from vlm_dp.grounding import predicates
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
# Transport geometry the plan references: how high the hover point sits above the place
# point, and how much slack the carry-height path rule allows below the lift height.
_CARRY_HOVER, _CARRY_SLACK = 0.10, 0.03
# Tea: the band separating "mouth still clear above the cup" from "mouth descended onto it".
# Same 5cm the other plans' place predicates use; authored here because the mouth is not a
# body the plan carries as far as the runtime's ownership rule is concerned (see _tea).
_TEA_MOUTH_CLEAR = 0.05
# Authored feature anchors in the teapot root frame. Keep these two points explicit: the asset's
# handle is on negative local z and its mouth is on positive local z. The old renderer used a
# handle-coordinate as ``mouth_world`` and then took the farthest xy cloud point as the handle,
# which swapped the two ends of the pot. These values match tea_gt_keypoints() and the task's
# authored handle/mouth geometry.
_TEA_HANDLE_LOCAL = np.array([0.0, 0.0566, -0.0624])
_TEA_MOUTH_LOCAL = np.array([0.0, 0.0516, 0.0651])
# Half-width of the handle feature the plan asks the fingers to close around. Like the capsule
# lid lip below, the handle is much thinner than its owning body: sizing it from the teapot's
# whole-object extent makes an otherwise pinchable feature look too wide for the gripper.
_TEA_HANDLE_HALF_W = 0.010

# Capsule geometry, in the machine-root frame. Same numbers vlm_dp/grounding/capsule.py and
# vlm_dp/offline_context.py use: the pod bay sits on the machine's vertical axis, and the lid
# travels roughly this far up when it swings open.
_CAPSULE_BAY_LOCAL = np.array([0.0, 0.0, 0.27])
_CAPSULE_LID_OPEN_LIFT = np.array([0.0, 0.0, 0.12])
# How far short of that open lift the lip may stop and still count as OPENED. Authored here,
# beside the lift it is a band on, and rendered into the plan as {open_clear}: the lid's rise
# tolerance is the plan's own geometry and must not be laundered through a runtime primitive.
# clearance_margin() would REFUSE it and rightly so -- the lip is a DECLARED feature on a body
# the robot never carries, so no clearance of it can be derived from that body's extents. 0.05
# of the 0.12 lift means the lip has to reach ~58% of the commanded travel; it is a 5cm band and
# not a 5mm one because a hinge arc does not stop on a millimetre.
_CAPSULE_OPEN_CLEAR = 0.05
# Where the pod comes to rest relative to the bay keypoint. Was a literal np.array([0, 0, 0.02])
# inside the plan's place stage; it is a field now because the descent gate and the stage's
# completion predicate have to name the SAME seat, and two copies of a literal drift apart.
_CAPSULE_BAY_SEAT = np.array([0.0, 0.0, 0.02])
# The closed lid's graspable rim, in the same machine-root frame as the bay. Calibrated from
# the asset (see _capsule_lip_diagnostic, which re-derives and prints it every run); it is the
# lid-body offset vlm_dp/grounding/capsule.py carries as _LID_LIP_LOCAL, resolved through the
# lid body's rest transform so it can be applied to the machine root directly.
_CAPSULE_LIP_LOCAL = np.array([-0.0784, -0.2392, 0.3924])
# The rim is the machine's topmost, far-side corner: a grazing surface the depth camera samples
# thinly, and the width-at-grasp-point probe downstream needs a real neighbourhood to measure a
# pinch in. These mirror rekep's probe (_GRASP_PROBE_R / _MIN_LOCAL_PTS): if the declared rim is
# seen by fewer than this many points, the plan names the nearest point that IS seen instead, so
# it never asks the robot to close on geometry nothing observed.
_CAPSULE_LIP_PROBE_R, _CAPSULE_LIP_MIN_PTS = 0.03, 20
# Half-width of the rim the plan asks the fingers to close on, declared with the point in the same
# way and from the same asset calibration as _CAPSULE_LIP_LOCAL (it is the number
# vlm_dp/grounding/capsule.py carries as _LID_EXTENTS[0]). It travels with the declaration because
# a declared grasp FEATURE and its owner's body are different geometries: without it, every grasp
# term downstream sizes itself to the machine's ~320mm keepout extent instead of a 10mm lip, which
# floors aperture_region, opens the centre dead zone and the close gate to 127mm, and puts a 33cm
# straddle repulsion around the very point the gripper is told to stand on.
_CAPSULE_LIP_HALF_W = 0.010


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
        """Return a metadata line's text, or None when the plan does not carry it."""
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

    # Per-stage CONTACT MODE, a vlm_dp extension the ReKep response format has no place for and
    # every other plan omits. It states, per stage, whether the hand closes AROUND the feature
    # (pinch) or ON it (press). The compiler infers a contact mode from measured geometry, and
    # from plan structure under grounding.contact_criterion="plan"; both are inferences about
    # feasibility, and neither can express "this rim fits the fingers but is to be pressed
    # anyway", which is task knowledge. Absent, the metadata key is absent and the compiler's
    # inference is untouched -- so no existing plan changes behaviour.
    modes = _opt_line("contact_modes")
    if modes is not None:
        metadata["contact_modes"] = [s.strip().strip('"\'')
                                     for s in modes.strip("[]").split(",")]
        named = [(i + 1, m) for i, m in enumerate(metadata["contact_modes"]) if m]
        print(f"[fake-vlm] {task_key}: plan declares contact modes {named}", flush=True)

    # --- write one file per (stage, kind), including the empty ones ---
    # load_stage() reads every stage{N}_path_constraints.txt unconditionally, so a stage with
    # no path constraint still needs the file to exist.
    for idx in range(1, metadata["num_stages"] + 1):
        for kind in ("subgoal", "path"):
            key = f"stage{idx}_{kind}"
            body = "\n\n".join("\n".join(functions[n]) for n in sorted(grouped.get(key, [])))
            with open(os.path.join(out_dir, f"{key}_constraints.txt"), "w", encoding="utf-8") as f:
                f.write(body + "\n" if body else "")

    # --- completion predicates (vlm_dp extension; absent from most plans) ---
    # Same split, different question: a stage<N>_completion block says whether the EVENT the
    # stage names has happened, instead of how far its sub-goal scalar is from zero. Written
    # out per stage so the loader reads them exactly like the constraint files.
    # The resolved {placeholder} values, so the rendered plan can be reproduced exactly offline.
    # A rollout log records what each predicate DECIDED; without these it does not record what
    # the predicate was deciding about, and no re-evaluation against the same episode is possible.
    with open(os.path.join(out_dir, "render_fields.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fields.items()},
                  f, indent=2)

    n_pred = predicates.write_files(output, out_dir, metadata["num_stages"])
    if n_pred:
        print(f"[fake-vlm] {task_key}: {n_pred}/{metadata['num_stages']} stages carry a "
              f"completion predicate", flush=True)
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

    # Lid lip: declared from the machine root frame, exactly the way the bay already is. The
    # rim is a few millimetres of near-tangential surface, so a single depth view samples it
    # sparsely and the front-most point of the top slab lands ~90mm inboard and to the side of
    # the real rim -- far enough that stage 1 never grasped it. The machine is a *calibrated
    # fixture*: the plan already reads its root pose to place the bay and the mid-air open goal,
    # so one more fixed offset in that frame is the same class of information, and unlike the
    # cloud search it does not depend on which facets the camera happened to see this episode.
    lip_world = _observable(root + _quat_rotate_wxyz(quat, _CAPSULE_LIP_LOCAL), pts["capsule"])
    open_world = lip_world + _CAPSULE_LID_OPEN_LIFT
    bay_world = root + _quat_rotate_wxyz(quat, _CAPSULE_BAY_LOCAL)

    pod = _nearest_kp_distinct(keypoints, pts["can"].mean(axis=0), set())
    n = len(keypoints)
    lip, open_goal, bay = n, n + 1, n + 2
    extra = [(lip_world, "capsule", _CAPSULE_LIP_HALF_W), (open_world, None), (bay_world, "capsule")]

    kps = np.concatenate([np.asarray(keypoints, dtype=np.float64),
                          np.stack([lip_world, open_world, bay_world])], axis=0)
    # Lift target: the pod's pick-up position raised _LIFT_HEIGHT, expressed as an offset from the
    # BAY keypoint because that one is on the machine and does not move. Anchoring to the carried
    # pod's own keypoint would be degenerate (the target would track the pod).
    lift_pod = (kps[pod] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - kps[bay]).tolist()
    # Seat and hover offsets from the bay keypoint: the hover point for the transport stage is the
    # seat raised by the carry clearance, so the following stage's descent is straight down.
    off_pod = _CAPSULE_BAY_SEAT.tolist()
    hover_pod = (_CAPSULE_BAY_SEAT + np.array([0.0, 0.0, _CARRY_HOVER])).tolist()
    # Absolute world heights the plan's scalar rules measure against: the LIFT stage's one-sided
    # vertical sub-goal, the transport stage's carry-height floor, and the descent gate's hover
    # height. `lift_pod` is still supplied for any plan wanting the full 3-D lift point.
    lift_z_pod = float(kps[pod][2] + _LIFT_HEIGHT)
    carry_z_pod = float(kps[pod][2] + _LIFT_HEIGHT - _CARRY_SLACK)
    hover_z_pod = float(kps[bay][2] + hover_pod[2])
    metadata = _render("capsule", out_dir, lip=lip, open_goal=open_goal, pod=pod, bay=bay,
                       lift_pod=lift_pod, off_pod=off_pod, hover_pod=hover_pod,
                       lift_z_pod=lift_z_pod, carry_z_pod=carry_z_pod, hover_z_pod=hover_z_pod,
                       open_clear=float(_CAPSULE_OPEN_CLEAR))
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] capsule roles pod=kp{pod} (snapped, {np.round(kps[pod], 3)}) "
          f"lip=kp{lip} (declared, {np.round(lip_world, 3)}) "
          f"open_goal=kp{open_goal} (declared, {np.round(open_world, 3)}) "
          f"bay=kp{bay} (declared, {np.round(bay_world, 3)})", flush=True)
    _capsule_lip_diagnostic(env, lip_world, root, quat)
    return metadata, {"lid": lip, "open_goal": open_goal, "pod": pod, "bay": bay}, extra


def _capsule_lip_diagnostic(env, lip_world, root, quat):
    """Report how far the declared lid lip sits from the privileged one (diagnostic only).

    Also prints the privileged lip re-expressed in the machine-root frame: that is the number
    _CAPSULE_LIP_LOCAL is calibrated from, and printing it every run is how a change to the
    asset would be caught rather than silently drifting the grasp point.
    """
    try:
        from vlm_dp.grounding.capsule import gt_keypoints
        from types import SimpleNamespace

        gt_kps, _ = gt_keypoints(SimpleNamespace(env=env))
        gt_lip = np.asarray(gt_kps[0], dtype=np.float64)
        err = float(np.linalg.norm(gt_lip - lip_world)) * 1e3
        local = _quat_rotate_wxyz(_quat_conj_wxyz(quat), gt_lip - root)
        print(f"[fake-vlm] capsule lip check: declared {np.round(lip_world, 3)} vs privileged "
              f"{np.round(gt_lip, 3)} ({err:.0f}mm) -- diagnostic only, "
              f"the plan uses the declared point", flush=True)
        print(f"[fake-vlm] capsule lip calibration: privileged lip in the machine-root frame is "
              f"{np.round(local, 4).tolist()} (constant _CAPSULE_LIP_LOCAL = "
              f"{np.round(_CAPSULE_LIP_LOCAL, 4).tolist()})", flush=True)
    except Exception as exc:                       # never let a diagnostic break grounding
        print(f"[fake-vlm] capsule lip check unavailable ({exc})", flush=True)


def _quat_rotate_wxyz(quat, vec):
    w, x, y, z = [float(v) for v in quat]
    q = np.array([x, y, z])
    return vec + 2.0 * np.cross(q, np.cross(q, vec) + w * vec)


def _observable(point, pts):
    """Return the declared point, moved the least distance that makes it observable.

    A declared point is only useful if perception can certify what is at it: the compiler
    measures the grip width from the cloud in a small ball around the grasp point, and a point
    nothing was sampled near falls back to the whole object's width and compiles as a press.
    The machine's lid rim is its topmost far-side edge -- grazing geometry a single depth view
    barely samples -- so the plan walks the declared point toward the cloud only until the
    probe has enough support, keeping the grasp as close to the true rim as evidence allows.
    """
    point = np.asarray(point, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    d = np.linalg.norm(pts - point[None], axis=1)
    seen = int((d <= _CAPSULE_LIP_PROBE_R).sum())
    if seen >= _CAPSULE_LIP_MIN_PTS:
        print(f"[fake-vlm] capsule lip observability: {seen} cloud points within "
              f"{_CAPSULE_LIP_PROBE_R * 1e3:.0f}mm -- declared rim kept", flush=True)
        return point
    step = 0.0025
    direction = pts[int(np.argmin(d))] - point
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    for i in range(1, int((float(d.min()) + _CAPSULE_LIP_PROBE_R) / step) + 1):
        moved = point + direction * (i * step)
        n = int((np.linalg.norm(pts - moved[None], axis=1) <= _CAPSULE_LIP_PROBE_R).sum())
        if n >= _CAPSULE_LIP_MIN_PTS:
            print(f"[fake-vlm] capsule lip observability: {seen} cloud points within "
                  f"{_CAPSULE_LIP_PROBE_R * 1e3:.0f}mm of the declared rim (nearest "
                  f"{float(d.min()) * 1e3:.0f}mm); moved {i * step * 1e3:.0f}mm inboard to "
                  f"{np.round(moved, 3)}, where {n} points support the grip probe", flush=True)
            return moved
    snapped = pts[int(np.argmin(d))]
    print(f"[fake-vlm] capsule lip observability: the rim is unobservable ({seen} points within "
          f"{_CAPSULE_LIP_PROBE_R * 1e3:.0f}mm and no inboard point does better); snapping "
          f"{float(d.min()) * 1e3:.0f}mm to {np.round(snapped, 3)}", flush=True)
    return snapped


def _quat_conj_wxyz(quat):
    w, x, y, z = [float(v) for v in quat]
    return np.array([w, -x, -y, -z])


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
    extra = []
    pts = {}
    for name in ("teapot", "teacup"):
        p = _masked_points(grounded, env, name)
        if p is None:
            raise SystemExit(f"[fake-vlm] no masked points for {name}")
        pts[name] = p
    pot = env.scene["teapot"].data
    root = pot.root_pos_w[0].cpu().numpy()
    quat = pot.root_quat_w[0].cpu().numpy()
    handle_anchor = root + _quat_rotate_wxyz(quat, _TEA_HANDLE_LOCAL)
    mouth_anchor = root + _quat_rotate_wxyz(quat, _TEA_MOUTH_LOCAL)
    if grounded.get("gt_meta") is not None:

        roles["teapot"], roles["mouth"], roles["teacup"] = 0, 1, 2
        kps = np.asarray(keypoints, dtype=np.float64)
    else:


        tp = pts["teapot"]
        # Ground each semantic feature from its own calibrated anchor. The handle declaration is
        # kept on observed geometry by taking the closest teapot-cloud sample in full 3-D; the
        # mouth remains a proposed keypoint, selected against its separate positive-z anchor.
        handle_world = tp[int(np.argmin(np.linalg.norm(tp - handle_anchor[None], axis=-1)))]
        taken = set()
        # Reserve the proposal the old renderer used for the handle so changing the handle into
        # a declared feature cannot silently reassign that same proposal to the cup or mouth.
        handle_snap = _nearest_kp_distinct(keypoints, handle_world, taken)
        taken.add(handle_snap)
        roles["teacup"] = _nearest_kp_distinct(keypoints, pts["teacup"].mean(axis=0), taken)
        taken.add(roles["teacup"])
        roles["mouth"] = _nearest_kp_distinct(keypoints, mouth_anchor, taken)
        # A proposed point can name WHERE the handle is, but it carries no feature geometry.
        # Declare the observed handle explicitly so the grounding seam propagates both its owner
        # and its plan-authored half-width to every grasp term and completion predicate.
        roles["teapot"] = len(keypoints)
        extra = [(handle_world, "teapot", _TEA_HANDLE_HALF_W)]
        kps = np.concatenate(
            [np.asarray(keypoints, dtype=np.float64), handle_world.reshape(1, 3)], axis=0)
    cup_top = np.array([pts["teacup"][:, 0].mean(), pts["teacup"][:, 1].mean(),
                        pts["teacup"][:, 2].max()])
    cup_off = (cup_top - kps[roles["teacup"]]).tolist()
    h, m, c = roles["teapot"], roles["mouth"], roles["teacup"]


    lever = float(np.linalg.norm(kps[m] - kps[h]))
    rest_dz = float(kps[m][2] - kps[h][2])
    pour_margin = rest_dz - max(0.03, 0.5 * lever)

    # Lift target: the handle's pick-up position raised _LIFT_HEIGHT, expressed as an offset from
    # the TEACUP keypoint because that one is on a fixture and does not move. Anchoring to the
    # carried teapot's own keypoint would be degenerate (the target would track the teapot).
    lift_teapot = (kps[h] + np.array([0.0, 0.0, _LIFT_HEIGHT]) - kps[c]).tolist()
    # Absolute world heights the plan's scalar rules measure against. The lift stage states its
    # sub-goal as a one-sided vertical shortfall rather than a 3-D distance to `lift_teapot`, so
    # it needs the target height as a scalar; `lift_teapot` is still supplied for any plan
    # wanting the full 3-D lift point. Both are on the HANDLE keypoint -- the grasped feature,
    # whose height the hand controls directly -- not the mouth at the end of the lever arm.
    lift_z_teapot = float(kps[h][2] + _LIFT_HEIGHT)
    carry_z_teapot = float(kps[h][2] + _LIFT_HEIGHT - _CARRY_SLACK)
    # Absolute world height of the pour hover point (the cup rim raised by the carry clearance
    # the stage-3 sub-goal already adds), for stage 3's completion predicate.
    hover_z_mouth = float(kps[c][2] + cup_off[2] + _CARRY_HOVER)
    # How far below that hover height the mouth may sit and still count as "clear above the cup".
    # Authored HERE and rendered into the plan as {mouth_clear} rather than resolved by
    # clearance_margin({m}), because clearance_margin is defined only for a keypoint on a body
    # the plan CARRIES and the mouth's owner is decided at runtime by keypoint registration --
    # the spout tip is a snapped proposed keypoint, and whether it registers to the teapot or to
    # the table under it is a fact about this episode's segmentation, not about the plan. A
    # clearance the plan cannot be sure the primitive may answer is the plan's own to state.
    metadata = _render("tea", out_dir, h=h, m=m, c=c, cup_off=cup_off, pour_margin=pour_margin,
                       lift_teapot=lift_teapot, lift_z_teapot=lift_z_teapot,
                       carry_z_teapot=carry_z_teapot, hover_z_mouth=hover_z_mouth,
                       mouth_clear=float(_TEA_MOUTH_CLEAR),
                       handle_half_width=float(_TEA_HANDLE_HALF_W))
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] tea roles teapot-handle=kp{h} "
          f"({'declared 10mm feature' if extra else 'ground-truth feature'}) "
          f"mouth=kp{m} teacup=kp{c}", flush=True)
    return metadata, roles, extra


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


def _spoon_insertion(out_dir, keypoints, grounded, env, clearance):
    """Insert the spaghetti-spoon handle into the tracked utensil-holder mouth.

    Four feature points are declared from perceived clouds instead of assuming the proposer sampled
    the thin handle end or the hollow mouth: a pinch point near the head, both spoon ends, and the
    holder mouth.  The shared ReKep tracker registers each point to its perceived owner.
    """
    del clearance
    keypoints = np.asarray(keypoints, dtype=np.float64)
    spoon_name, holder_name = "pink_spaghetti_spoon", "utensil_holder"
    spoon = _cloud(grounded, env, spoon_name)
    holder = _cloud(grounded, env, holder_name)
    if spoon is None or spoon.shape[0] < 40:
        raise SystemExit("[fake-vlm] spoon_insertion: no usable perceived spoon cloud")
    if holder is None or holder.shape[0] < 40:
        raise SystemExit("[fake-vlm] spoon_insertion: no usable perceived holder cloud")

    xy = spoon[:, :2]
    centre_xy = np.median(xy, axis=0)
    _, _, vh = np.linalg.svd(xy - centre_xy, full_matrices=False)
    axis = vh[0] / max(float(np.linalg.norm(vh[0])), 1e-9)
    side = np.array([-axis[1], axis[0]])
    along = (xy - centre_xy) @ axis
    lo, hi = np.percentile(along, [4, 96])

    def endpoint(value):
        band = max(0.015, 0.10 * float(hi - lo))
        pts = spoon[np.abs(along - value) <= band]
        middle = np.median(pts[:, :2], axis=0)
        width = float(np.percentile(np.abs((pts[:, :2] - middle) @ side), 90))
        return np.median(pts, axis=0), width

    end_lo, width_lo = endpoint(lo)
    end_hi, width_hi = endpoint(hi)
    handle, head = ((end_lo, end_hi) if width_lo <= width_hi else (end_hi, end_lo))

    # Pinch the neck just behind the broad head, with a local width carried alongside the declared
    # feature so the grasp preflight and cost never size this feature as the whole 33cm utensil.
    grasp_probe = head + 0.25 * (handle - head)
    distance = np.linalg.norm(spoon - grasp_probe, axis=1)
    local = spoon[np.argsort(distance)[:max(40, min(250, len(spoon)))]]
    grasp = np.median(local, axis=0)
    local_mid = np.median(local[:, :2], axis=0)
    grasp_half = float(np.percentile(np.abs((local[:, :2] - local_mid) @ side), 90))
    grasp_half = float(np.clip(grasp_half, 0.006, 0.030))

    mouth = _top(holder)
    outer = _mouth_radius(holder, fallback=0.055)
    mouth_radius = float(np.clip(0.65 * outer, 0.030, 0.055))
    seat_radius = float(min(0.025, 0.75 * mouth_radius))
    insert_depth, hover, capture = 0.075, 0.12, 0.10

    g, h, b, m = len(keypoints), len(keypoints) + 1, len(keypoints) + 2, len(keypoints) + 3
    metadata = _render(
        "spoon_insertion", out_dir, g=g, h=h, b=b, mouth=m,
        lift_g=(grasp + np.array([0.0, 0.0, _LIFT_HEIGHT]) - mouth).tolist(),
        insert_hover=hover, insert_depth=insert_depth, mouth_radius=mouth_radius,
        seat_radius=seat_radius, insert_capture=capture,
    )
    metadata["task_spec"] = {
        "instruction": grounded.get("instruction"), "payload": spoon_name,
        "destination": holder_name, "mode": "handle-first insertion",
    }
    metadata["resolved"] = {
        "grasp_world": np.round(grasp, 6).tolist(),
        "handle_world": np.round(handle, 6).tolist(),
        "head_world": np.round(head, 6).tolist(),
        "mouth_world": np.round(mouth, 6).tolist(),
        "grasp_half_width": round(grasp_half, 6),
        "endpoint_half_widths": [round(width_lo, 6), round(width_hi, 6)],
    }
    metadata, roles = _finish(
        out_dir, metadata,
        {spoon_name: g, "spoon_handle": h, "spoon_head": b, holder_name: m},
        f"spoon_insertion roles grasp=kp{g} handle=kp{h} head=kp{b} mouth=kp{m} "
        f"grasp_half={grasp_half:.3f}m mouth_r={mouth_radius:.3f}m",
    )
    extras = ((grasp, spoon_name, grasp_half), (handle, spoon_name),
              (head, spoon_name), (mouth, holder_name))
    return metadata, roles, extras


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
              "banana_in_bowl": _banana_in_bowl, "mustard_left_bin": _mustard_left_bin,
              "spoon_insertion": _spoon_insertion}


def generate(task_key, out_dir, keypoints, grounded, env, clearance=0.015):
    """Write fake VLM metadata and constraint files for a task.

    Returns ``(metadata, roles, extra_keypoints)``. ``extra_keypoints`` is a list of
    ``(world_point, owner_name_or_None)``, or ``(world_point, owner_name_or_None,
    grasp_half_width)``, the task declared: points the plan needs that no proposed keypoint stands
    for. The caller appends them and registers them for tracking. A declaration that names a GRASP
    feature carries its half-width, which is the geometry every downstream grasp term uses for it
    (rekep.ground propagates it); the owner's whole-body extent describes a different thing.
    """
    if task_key not in _FAKE_VLMS:
        raise SystemExit(f"[fake-vlm] no fake VLM for task {task_key!r}, register one in _FAKE_VLMS "
                         f"(have: {sorted(_FAKE_VLMS)})")
    os.makedirs(out_dir, exist_ok=True)
    out = _FAKE_VLMS[task_key](out_dir, keypoints, grounded, env, clearance)
    return (out[0], out[1], out[2] if len(out) > 2 else ())
