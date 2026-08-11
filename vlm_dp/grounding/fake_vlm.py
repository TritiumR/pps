"""Generate deterministic ReKep constraint artifacts for supported tasks."""
import json
import os

import cv2
import numpy as np

from vlm_dp.grounding import predicates
from vlm_dp.grounding.masks import _masked_points, _nearest_kp

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_vlm_output")

# Height an object is raised before transport. Matches the lift the `template` compiler injects
# (rekep._LIFT_HEIGHT), so a plan carrying its own explicit lift stage is structurally equal to
# the templated one -- with the difference that every stage keeps its constraints under `_vlm`.
_LIFT_HEIGHT = 0.15

# Transport geometry the plan references: how high the hover point sits above the place
# point, and how much slack the carry-height path rule allows below the lift height.
_CARRY_HOVER, _CARRY_SLACK = 0.10, 0.03
# Tea: the band separating "mouth still clear above the cup" from "mouth descended onto it".
# Same 5cm the other plans' place predicates use; authored here because the mouth is not a
# body the plan carries as far as the runtime's ownership rule is concerned (see _tea).
_TEA_MOUTH_CLEAR = 0.05

# Pot-cover handle extraction from the observed cover point cloud. The handle is the small
# raised component above the broad lid surface; these thresholds are deliberately in sensor
# space and do not use the cover's simulator pose or asset geometry.
_POT_HANDLE_MIN_RISE, _POT_HANDLE_MIN_PTS = 0.008, 12
_POT_HANDLE_HALF_W_BOUNDS = (0.004, 0.025)
_POT_HANDLE_Z_INSET = 0.005
# Semantic pre-grasp waypoint. With simple_auth's 0.12m sub-goal tolerance, 0.20m still
# guarantees at least 8cm of vertical clearance when the approach stage advances.
_POT_APPROACH_HEIGHT = 0.20
_POT_LIFT_RESIDUAL_SCALE = 0.03
# Authored clear-above/descended band for the visually declared lid-handle feature.
_POT_LID_CLEAR = 0.05

# Capsule geometry, in the machine-root frame. Same numbers vlm_dp/grounding/capsule.py and
# vlm_dp/offline_context.py use: the pod bay sits on the machine's vertical axis, and the lid
# travels roughly this far up when it swings open.
_CAPSULE_BAY_LOCAL = np.array([0.0, 0.0, 0.27])
_CAPSULE_LID_OPEN_LIFT = np.array([0.0, 0.0, 0.12])
# Closed-lid hinge geometry used only to turn a visually tracked rim point into an angle.
# The hinge lies one lid radius inboard from the front lip in the calibrated machine frame.
_CAPSULE_LID_RADIUS = float(_CAPSULE_LID_OPEN_LIFT[2])
_CAPSULE_HINGE_FROM_LIP_LOCAL = np.array([0.0, _CAPSULE_LID_RADIUS, 0.0])
_CAPSULE_OPEN_ANGLE_DEG = 85.0  # sensor tolerance around the physical 90-degree stop
_CAPSULE_ANGLE_RESIDUAL_SCALE = float(np.deg2rad(5.0))
# Closed-finger contact point below the lid lip.  A separate semantic waypoint sits outside and
# farther below the circular lid, so the approach cannot shortcut through the ring from above.
_CAPSULE_UNDER_LIP_DROP = 0.04
_CAPSULE_UNDER_LIP_INSET = 0.025
_CAPSULE_PRE_UNDER_OUT = 0.08
_CAPSULE_PRE_UNDER_DROP = 0.08
_CAPSULE_PRE_UNDER_RESIDUAL_SCALE = 0.10
_CAPSULE_UNDER_LIP_RESIDUAL_SCALE = 0.03
# Lateral corridor across the visually observed short edge of the coffee-maker body.
_CAPSULE_WIDTH_PERCENTILES = (5.0, 95.0)
_CAPSULE_WIDTH_BODY_CLEAR = 0.035
_CAPSULE_WIDTH_TARGET_ALLOWANCE = 0.005
_CAPSULE_WIDTH_RESIDUAL_SCALE = 0.03
_CAPSULE_TOP_PERCENTILE = 95.0
_CAPSULE_STAGE1_TOP_MARGIN = 0.0
_CAPSULE_TOP_RESIDUAL_SCALE = 0.01
# Height band used by the stage-2 sustained-history diagnostic. The control sub-goal below is
# stricter because it normalizes the remaining push stroke by the full 0.12m travel.
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

    metadata = {"num_stages": int(_line("num_stages")),
                "grasp_keypoints": _int_list(_line("grasp_keypoints")),
                "release_keypoints": _int_list(_line("release_keypoints"))}

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

    # Optional per-stage gripper intent independent of grasp ownership.  This lets a plan keep
    # closed fingers while approaching a press contact without falsely claiming an object grasp.
    gripper_modes = _opt_line("gripper_modes")
    if gripper_modes is not None:
        metadata["gripper_modes"] = [s.strip().strip('"\'')
                                     for s in gripper_modes.strip("[]").split(",")]

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

    Four of the five roles have no proposed keypoint to snap to: the lid lip is a thin rim, the
    under-lip point and open goal are in mid-air, and the bay is a recess inside the machine.
    They are *declared* instead -- returned as extra keypoints the caller appends and registers.

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
    hinge_world = lip_world + _quat_rotate_wxyz(quat, _CAPSULE_HINGE_FROM_LIP_LOCAL)
    inward = hinge_world - lip_world
    inward[2] = 0.0
    inward /= max(float(np.linalg.norm(inward)), 1e-9)
    # The TCP is the fingertip contact point, so put it under the lid surface, slightly
    # inboard of the rim. The preceding waypoint remains outside/below for collision-free entry.
    under_world = (lip_world + _CAPSULE_UNDER_LIP_INSET * inward
                   - np.array([0.0, 0.0, _CAPSULE_UNDER_LIP_DROP]))
    pre_under_world = (lip_world - _CAPSULE_PRE_UNDER_OUT * inward
                       - np.array([0.0, 0.0, _CAPSULE_PRE_UNDER_DROP]))
    # Inward is the machine depth direction; its horizontal normal is the short/width
    # direction.  Measure its bounds from the segmented machine cloud, shrink them by
    # the gripper-body radius, and retain a small reachable band around the contact point.
    width_axis = np.array([-inward[1], inward[0], 0.0], dtype=np.float64)
    width_axis /= max(float(np.linalg.norm(width_axis)), 1e-9)
    machine_pts = np.asarray(pts["capsule"], dtype=np.float64)
    machine_pts = machine_pts[np.isfinite(machine_pts).all(axis=1)]
    width_proj = (machine_pts - root[None, :]) @ width_axis
    width_lo, width_hi = np.percentile(width_proj, _CAPSULE_WIDTH_PERCENTILES)
    # `machine_pts` are depth-deprojected world-frame XYZ samples from the coffee-machine mask.
    # This is a physical world-Z height estimate, not an image-row or projected-camera bound.
    machine_top_z = float(np.percentile(machine_pts[:, 2], _CAPSULE_TOP_PERCENTILE))
    stage1_tcp_ceiling = machine_top_z - _CAPSULE_STAGE1_TOP_MARGIN
    target_width = float((under_world - root) @ width_axis)
    corridor_lo = min(float(width_lo + _CAPSULE_WIDTH_BODY_CLEAR),
                      target_width - _CAPSULE_WIDTH_TARGET_ALLOWANCE)
    corridor_hi = max(float(width_hi - _CAPSULE_WIDTH_BODY_CLEAR),
                      target_width + _CAPSULE_WIDTH_TARGET_ALLOWANCE)
    if corridor_hi <= corridor_lo:
        raise SystemExit("[fake-vlm] observed coffee-machine width is too narrow for a "
                         "reachable under-lid corridor")
    # TCP local +z points from the wrist toward the fingertips. Aim it from the
    # below-lip waypoint toward the hinge: inward into the machine and slightly up.
    tool_z_facing = hinge_world - under_world
    tool_z_facing /= max(float(np.linalg.norm(tool_z_facing)), 1e-9)
    open_world = lip_world + _CAPSULE_LID_OPEN_LIFT
    bay_world = root + _quat_rotate_wxyz(quat, _CAPSULE_BAY_LOCAL)

    pod = _nearest_kp_distinct(keypoints, pts["can"].mean(axis=0), set())
    n = len(keypoints)
    lip, lip_rest, hinge, open_goal, bay, under_lip, pre_under = range(n, n + 7)
    # "lid" is seeded into SensedWorld before CoTracker is primed. The live lip therefore follows
    # the observed lid surface, while lip_rest and hinge stay fixed in the calibrated fixture
    # frame. Runtime completion reads their angle; it never reads the simulator joint.
    extra = [(lip_world, "lid", _CAPSULE_LIP_HALF_W),
             (lip_world, "capsule"),
             (hinge_world, "capsule"),
             (open_world, None),
             (bay_world, "capsule"),
             (under_world, "capsule", _CAPSULE_LIP_HALF_W),
             (pre_under_world, "capsule", _CAPSULE_LIP_HALF_W)]

    kps = np.concatenate([np.asarray(keypoints, dtype=np.float64),
                          np.stack([lip_world, lip_world, hinge_world, open_world,
                                    bay_world, under_world, pre_under_world])], axis=0)
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
    metadata = _render("capsule", out_dir, lip=lip, lip_rest=lip_rest, hinge=hinge,
                       under_lip=under_lip, pre_under=pre_under,
                       under_lip_residual_scale=float(_CAPSULE_UNDER_LIP_RESIDUAL_SCALE),
                       pre_under_residual_scale=float(_CAPSULE_PRE_UNDER_RESIDUAL_SCALE),
                       machine_origin=root.tolist(), width_axis=width_axis.tolist(),
                       width_lo=float(corridor_lo), width_hi=float(corridor_hi),
                       width_residual_scale=float(_CAPSULE_WIDTH_RESIDUAL_SCALE),
                       stage1_tcp_ceiling=stage1_tcp_ceiling,
                       top_residual_scale=float(_CAPSULE_TOP_RESIDUAL_SCALE),
                       open_goal=open_goal, pod=pod, bay=bay,
                       lift_pod=lift_pod, off_pod=off_pod, hover_pod=hover_pod,
                       lift_z_pod=lift_z_pod, carry_z_pod=carry_z_pod, hover_z_pod=hover_z_pod,
                       open_clear=float(_CAPSULE_OPEN_CLEAR),
                       open_lift=float(_CAPSULE_LID_OPEN_LIFT[2]),
                       open_angle_rad=float(np.deg2rad(_CAPSULE_OPEN_ANGLE_DEG)),
                       angle_residual_scale=float(_CAPSULE_ANGLE_RESIDUAL_SCALE))
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    # Video stages 0-1 point local tool +z straight upward in world coordinates.  The push stage
    # then keeps its fixture-facing upward direction while rotating the lid.
    metadata["approach_axes"] = [[0.0, 0.0, 1.0]] * 2 + [tool_z_facing.tolist()] + [None] * 4
    metadata["approach_axis_scales"] = [10.0, 10.0, 1.0] + [1.0] * 4
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fake-vlm] capsule roles pod=kp{pod} (snapped, {np.round(kps[pod], 3)}) "
          f"lip=kp{lip} (declared, {np.round(lip_world, 3)}) "
          f"hinge=kp{hinge} (declared, {np.round(hinge_world, 3)}) "
          f"under_lip=kp{under_lip} (declared, {np.round(under_world, 3)}) "
          f"pre_under=kp{pre_under} (declared, {np.round(pre_under_world, 3)}) "
          f"width_corridor=[{corridor_lo:.3f}, {corridor_hi:.3f}]m "
          f"axis={np.round(width_axis, 3)} "
          f"machine_top={machine_top_z:.3f}m "
          f"stage1_tcp_ceiling={stage1_tcp_ceiling:.3f}m "
          f"open_goal=kp{open_goal} (declared, {np.round(open_world, 3)}) "
          f"bay=kp{bay} (declared, {np.round(bay_world, 3)})", flush=True)
    _capsule_lip_diagnostic(env, lip_world, root, quat)
    return metadata, {"lid": lip, "lid_rest": lip_rest, "hinge": hinge,
                      "under_lip": under_lip, "pre_under": pre_under, "open_goal": open_goal,
                      "pod": pod, "bay": bay}, extra


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


def _raised_handle(grounded, cover_pts):
    """Find the lid handle in raw depth inside the visually detected lid region.

    SAM usually segments only the broad lid surface and cuts the raised handle out of the
    cover mask. The mask still gives a reliable image-space region and lid height, so inspect
    the unmasked depth inside its bounding rectangle. A supported component immediately above
    the lid is the handle; substantially higher components are usually the robot crossing the
    box. This uses only the camera segmentation and depth -- no simulator pose or asset geometry.
    """
    cover_pts = np.asarray(cover_pts, dtype=np.float64)
    cover_pts = cover_pts[np.isfinite(cover_pts).all(axis=1)]
    points = np.asarray(grounded.get("points"))
    labels = np.asarray(grounded.get("masks"))
    if len(cover_pts) < 2 * _POT_HANDLE_MIN_PTS or points.ndim != 3 or labels.ndim != 2:
        return None

    cover_ids = [
        int(obj_id) for obj_id, name in grounded.get("id_to_prim", {}).items()
        if str(name).rsplit("/", 1)[-1] == "cover"
    ]
    if not cover_ids:
        return None
    cover_pixels = np.isin(labels, cover_ids)
    rows, cols = np.nonzero(cover_pixels)
    if len(rows) < 2 * _POT_HANDLE_MIN_PTS:
        return None

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    patch = points[y0:y1, x0:x1]
    finite = np.isfinite(patch).all(axis=-1)
    lid_z = float(np.percentile(cover_pts[:, 2], 60))
    raised = finite & (patch[..., 2] >= lid_z + _POT_HANDLE_MIN_RISE)
    count, component, stats, _ = cv2.connectedComponentsWithStats(
        raised.astype(np.uint8), connectivity=8
    )

    # Scale support with the observed lid region. On the formal 640x360 views this rejects the
    # small depth fringe at the lid edge while retaining the handle by a comfortable margin.
    min_support = max(_POT_HANDLE_MIN_PTS, int(np.ceil(0.01 * raised.size)))
    candidates = []
    for idx in range(1, count):
        support = int(stats[idx, cv2.CC_STAT_AREA])
        if support < min_support:
            continue
        handle = patch[component == idx].astype(np.float64, copy=False)
        rise = float(np.median(handle[:, 2]) - lid_z)
        if rise < _POT_HANDLE_MIN_RISE:
            continue
        # The handle is the first supported surface above the lid. Robot/gripper components
        # crossing this ROI are visibly farther above it and therefore rank later.
        candidates.append((rise, -support, handle))
    if not candidates:
        return None

    rise, neg_support, handle = min(candidates, key=lambda item: (item[0], item[1]))
    support = -neg_support
    centre = np.median(handle, axis=0)
    # Aim slightly into the handle body rather than at its highest visible surface.  This
    # increases finger overlap while retaining a visual lower bound above the broad lid.
    raw_z = float(centre[2])
    centre[2] = max(lid_z + _POT_HANDLE_MIN_RISE,
                    raw_z - _POT_HANDLE_Z_INSET)
    z_inset = raw_z - float(centre[2])
    xy = handle[:, :2] - centre[None, :2]
    cov = xy.T @ xy / max(len(xy) - 1, 1)
    _, axes = np.linalg.eigh(cov)
    projected = xy @ axes
    lo, hi = np.percentile(projected, [5, 95], axis=0)
    spans = hi - lo
    # The visible handle pixels are often denser at one end, so their median sits near an
    # edge. Centre the robust oriented bounding box instead; this remains purely visual.
    box_shift = (0.5 * (lo + hi)) @ axes.T
    centre[:2] += box_shift
    half_width = float(np.clip(0.5 * spans.min(), *_POT_HANDLE_HALF_W_BOUNDS))
    narrow_axis = axes[:, int(np.argmin(spans))]
    narrow_axis = narrow_axis / max(float(np.linalg.norm(narrow_axis)), 1e-9)
    axis = (float(narrow_axis[0]), float(narrow_axis[1]), 0.0)
    return centre, half_width, axis, support, rise, float(np.linalg.norm(box_shift)), z_inset


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
                       mouth_clear=float(_TEA_MOUTH_CLEAR))
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
    extra = ()
    handle_axis = None
    if grounded.get("gt_meta") is not None:

        roles["cover"], roles["egg"], roles["pot"] = 0, 1, 2
    else:


        taken = set()
        roles["pot"] = _nearest_kp_distinct(keypoints, pts["pot"].mean(axis=0), taken)
        taken.add(roles["pot"])
        handle = _raised_handle(grounded, pts["cover"])
        if handle is None:
            raise SystemExit("[fake-vlm] no supported raised handle in the visually detected "
                             "pot-lid region; refusing the unsafe rim-keypoint fallback")
        (handle_world, handle_half_w, handle_axis, support, rise,
         center_shift, z_inset) = handle
        roles["cover"] = len(keypoints)
        extra = ((handle_world, "cover", handle_half_w),)
        keypoints = np.concatenate(
            [np.asarray(keypoints, dtype=np.float64), handle_world[None]], axis=0)
        print(f"[fake-vlm] pot handle=kp{roles['cover']} (declared from {support} observed "
              f"ROI points, rise={rise:.3f}, half_width={handle_half_w:.3f}, "
              f"center_shift={center_shift * 1e3:.0f}mm, "
              f"z_inset={z_inset * 1e3:.0f}mm, "
              f"narrow_axis={np.round(handle_axis, 3)})", flush=True)
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
                       approach_lid=[0.0, 0.0, _POT_APPROACH_HEIGHT],
                       lid_clear=float(_POT_LID_CLEAR),
                       lift_lid=lift_lid, lift_egg=lift_egg,
                       lift_residual_scale=float(_POT_LIFT_RESIDUAL_SCALE),
                       hover_lid=hover["lid"], hover_egg=hover["egg"],
                       lift_z_lid=lift_z["lid"], lift_z_egg=lift_z["egg"],
                       carry_z_lid=carry_z["lid"], carry_z_egg=carry_z["egg"],
                       hover_z_lid=hover_z["lid"], hover_z_egg=hover_z["egg"])
    # vlm_dp extension, not part of the ReKep response format the parser understands.
    metadata["steer_policies"] = ["on_failure"] * metadata["num_stages"]
    if handle_axis is not None:
        metadata["grasp_axes"] = {str(lid): list(handle_axis)}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[fake-vlm] pot roles lid=kp{lid} egg=kp{egg} pot=kp{pot}", flush=True)
    return metadata, roles, extra


_FAKE_VLMS = {"weight": _weight, "capsule": _capsule, "tea": _tea, "pot": _pot}


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
