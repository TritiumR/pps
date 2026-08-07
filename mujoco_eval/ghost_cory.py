"""Faithful port of Cory's phase-endpoint ghost renderer.

Source: sim_infra (branch `mpc`) `scripts/render_pick_ball_phase_endpoint_ghost_wandb.py`,
functions `prepare_ghost_model` / `composite_ghost` and the trail-alpha block.

This is a PARALLEL implementation; `record._render_ghosts` is unchanged and stays the default.
Select this one with `--ghost_style cory`. Two things differ from ours and are the reason to
have it:

  * MASK. He takes an exact geom mask from MuJoCo's segmentation renderer. Ours diffs an
    all-hidden pass against the ghost pass and thresholds at 8/255, which is approximate and
    depends on the ghost contrasting with the background.
  * ALPHA. He normalises the trail so the COMBINED opacity where all poses overlap equals
    `alpha` regardless of how many ghosts are drawn:
        weights = linspace(0.35, 1.0, n);  a_i = 1 - (1 - alpha) ** (w_i / sum(w))
    Ours assigns each ghost an independent alpha, so N ghosts get progressively more opaque as
    N grows. His default alpha is 0.38.

He renders every trail pose in one cyan; we tint the terminal keypose differently. Matching him
means a single colour, so `color` defaults to his (0.0, 0.85, 1.0).
"""

from __future__ import annotations

import numpy as np

from . import viz

GHOST_COLOR = (0.0, 0.85, 1.0)     # his prepare_ghost_model(..., color=(0.0, 0.85, 1.0))
GHOST_ALPHA = 0.38                 # his --alpha default
_WARNED = []                       # one-time notice when the segmentation mask is unavailable


def trail_alphas(count, alpha=GHOST_ALPHA):
    """Per-ghost alphas whose overlap composites to `alpha`; later poses weigh more."""
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    weights = np.linspace(0.35, 1.0, count)
    return 1.0 - np.power(1.0 - float(alpha), weights / weights.sum())


def _segmentation_mask(env, ids, camera, hw):
    """Exact geom mask from the segmentation buffer, mirroring his mask expression.

    He builds a standalone `mujoco.Renderer`. Doing that here creates a second GL context
    alongside robosuite's EGL one; the readback then comes from the wrong framebuffer and
    mujoco decodes a normal colour image as segment ids ("index 7757126 is out of bounds").
    robosuite's own sim.render(segmentation=True) shares the context the rest of the rollout
    already uses, so it is both correct and free of an extra context.

    Returns bool [hw, hw] where the geom is one of `ids` -- his
    `isin(seg[..., 0], ghost_hand_ids) & (seg[..., 1] == mjOBJ_GEOM)`, with robosuite's
    channel order (0 = object type, 1 = object id).
    """
    import mujoco

    ctx = getattr(env.sim, "_render_context_offscreen", None)
    if ctx is None:
        return None
    model = env.sim.model
    cam = int(mujoco.mj_name2id(getattr(model, "_model", model),
                                mujoco.mjtObj.mjOBJ_CAMERA, camera))
    if cam < 0:
        return None
    # robosuite's own read_pixels(segmentation=True) decodes with
    #     rgb[:, :, 1] * (2 ** 8)
    # on a uint8 array, which numpy 2 rejects (NEP 50): "Python integer 256 out of bounds for
    # uint8". So render through its context but decode here, casting to uint32 first.
    ctx.render(width=int(hw), height=int(hw), camera_id=cam, segmentation=True)
    rgb = np.asarray(ctx.read_pixels(int(hw), int(hw), depth=False, segmentation=False),
                     dtype=np.uint32)
    segimage = rgb[..., 0] + rgb[..., 1] * (2 ** 8) + rgb[..., 2] * (2 ** 16)
    scene = ctx.scn
    segimage[segimage >= scene.ngeom + 1] = 0          # robosuite's out-of-range guard
    table = np.full((scene.ngeom + 1, 2), -1, dtype=np.int32)
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        if geom.segid != -1:
            table[geom.segid + 1] = (geom.objtype, geom.objid)
    seg = table[segimage][::-1]                        # read_pixels is bottom-up; env.rgb is not
    mask = (np.isin(seg[..., 1], np.asarray(ids))
            & (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)))
    # Measured on robosuite 1.4.1 + mujoco 2.3.2: mjr_render does not honour the mjRND_SEGMENT /
    # mjRND_IDCOLOR flags this context sets, so the readback is an ordinary colour image (8978
    # distinct colours for a 69-geom scene) and every pixel decodes out of range. Returning the
    # resulting empty mask would drop the ghost silently, which is worse than a coarser mask --
    # so treat an implausible mask as unavailable and let the caller fall back.
    return mask if mask.any() else None


def ghost_layer(env, arm_q, color=GHOST_COLOR, camera="agentview", hw=512):
    """Render the robot alone at `arm_q`; return (rgb, mask) with his segmentation mask.

    Falls back to the two-pass difference mask when segmentation is unavailable, so a missing
    renderer costs precision rather than the whole video.
    """
    model, data = env.sim.model, env.sim.data
    rgba0 = np.array(model.geom_rgba, copy=True)
    qpos0 = np.array(data.qpos, copy=True)
    ids = viz._robot_geom_ids(model)
    try:
        # prepare_ghost_model: everything transparent, then the robot in one flat colour.
        model.geom_rgba[:] = np.zeros(4, dtype=model.geom_rgba.dtype)
        data.qpos[env._robot._ref_joint_pos_indexes] = np.asarray(arm_q, dtype=np.float64)[:7]
        env.sim.forward()
        model.geom_rgba[ids] = np.asarray((*color, 1.0), dtype=model.geom_rgba.dtype)
        ghost = np.asarray(env.rgb(camera, hw=hw), dtype=np.uint8).copy()
        try:
            mask = _segmentation_mask(env, ids, camera, hw)
        except Exception as exc:
            mask = None
            if not _WARNED:
                _WARNED.append(1)
                import traceback
                print(f"[mujoco-eval] ghost_style=cory: segmentation mask unavailable "
                      f"({exc}); using the two-pass mask\n{traceback.format_exc()}", flush=True)
        if mask is None:
            if not _WARNED:
                _WARNED.append(1)
                print("[mujoco-eval] ghost_style=cory: segmentation mask empty "
                      "(mjr_render is ignoring mjRND_SEGMENT in this robosuite/mujoco build); "
                      "using the two-pass mask", flush=True)
            model.geom_rgba[:, 3] = 0.0
            env.sim.forward()
            empty = np.asarray(env.rgb(camera, hw=hw), dtype=np.int16)
            mask = np.abs(ghost.astype(np.int16) - empty).max(axis=-1) > 8
    finally:
        model.geom_rgba[:] = rgba0
        data.qpos[:] = qpos0
        env.sim.forward()
    return ghost, mask


def render_ghosts(env, rows, hw, alpha=GHOST_ALPHA, color=GHOST_COLOR, min_sep=0.08):
    """Layers for one replan's goal rows, in his single colour with his normalised alphas."""
    rows = [np.asarray(q, dtype=np.float64) for q in rows]
    now = np.asarray(env.q0(), dtype=np.float64)[:7]
    keep = [q for q in rows if float(np.abs(q[:7] - now).max()) >= min_sep]
    alphas = trail_alphas(len(keep), alpha)
    layers, labels = [], []
    rgb = tuple(int(255 * c) for c in color)
    for n, (q, a) in enumerate(zip(keep, alphas), start=1):
        ghost, mask = ghost_layer(env, q, color, hw=hw)
        if not np.any(mask):
            continue          # his code raises here; a blank ghost must not kill our video
        layers.append((ghost, mask, float(a)))
        # His AWE waypoint renderer labels every ghost W1..Wn at the centroid of its own mask.
        # The phase-endpoint script this file is otherwise ported from does not, but the thing
        # being drawn here IS the waypoint chain, so labelling matches him rather than departing.
        ys, xs = np.nonzero(mask)
        labels.append((f"W{n}", (int(np.median(xs)), int(np.median(ys))), rgb))
    return {"layers": layers, "labels": labels}
