"""Object-relative grasp/insertion geometry, derived from mesh vertices.

Shared by the probe (rev2_probe_limbs.py) and the scripted policy
(rev2_policy.py) so the numbers reported are exactly the numbers used.

Everything here is expressed in the object's own body frame; the policy
composes it with the privileged world pose. Nothing is hardcoded in world
coordinates.
"""

import numpy as np


def limb_frame(pts, rel=1.6, nbins=16, max_grip_width=0.075,
               bias="middle", inset=0.18):
    """Locate the graspable limb (utensil handle / drill grip) of a point cloud.

    The object's principal axis is binned; each bin's cross-section extent is
    measured perpendicular to that axis. The limb is the contiguous run of bins
    around the *narrowest* one whose extent stays within ``rel`` times the
    minimum. That single rule picks the handle on a spatula (2 cm shaft vs a
    9 cm blade) and the grip on the YCB drill (4.4 cm grip vs a 9-18 cm body
    and battery), so no per-object special-casing is needed.

    ``bias`` selects where along the limb to grasp:
      "middle" -- the limb's midpoint. Needed when the far half of the limb must
                  stay free, i.e. a utensil handle whose end goes into the holder.
      "com"    -- as close as the limb allows to the centroid's projection on the
                  axis, kept ``inset`` of the limb length clear of either
                  junction. A freely hanging object pivots about the pinch under
                  its own weight and the lever arm is what drives it; on the
                  drill this halves the grasp-to-centroid offset.

    Returns a dict in the object body frame.
    """
    pts = np.asarray(pts, dtype=float)
    centre = pts.mean(axis=0)
    centred = pts - centre
    _, _, v = np.linalg.svd(centred, full_matrices=False)
    axis = v[0]
    t = centred @ axis
    perp = centred - np.outer(t, axis)

    # perpendicular basis for measuring cross-sections
    u, w = v[1], v[2]
    edges = np.linspace(t.min(), t.max(), nbins + 1)
    extent, counts = np.full(nbins, np.inf), np.zeros(nbins, dtype=int)
    for i in range(nbins):
        hi = t <= edges[i + 1] if i == nbins - 1 else t < edges[i + 1]
        m = (t >= edges[i]) & hi
        counts[i] = int(m.sum())
        if counts[i] < 8:
            continue
        pu, pw = perp[m] @ u, perp[m] @ w
        extent[i] = max(float(pu.max() - pu.min()), float(pw.max() - pw.min()))

    i0 = int(np.argmin(extent))
    thresh = min(extent[i0] * rel, max_grip_width)
    lo = hi = i0
    while lo - 1 >= 0 and extent[lo - 1] <= thresh:
        lo -= 1
    while hi + 1 < nbins and extent[hi + 1] <= thresh:
        hi += 1

    t_lo, t_hi = float(edges[lo]), float(edges[hi + 1])
    in_limb = (t >= t_lo) & (t <= t_hi)
    limb_pts = pts[in_limb]

    # Where along the limb to close the fingers.
    t_mid = 0.5 * (t_lo + t_hi)
    if bias == "com":
        pad = inset * (t_hi - t_lo)
        t_grasp = float(np.clip(0.0, t_lo + pad, t_hi - pad))
    else:
        t_grasp = t_mid

    # Average a band about that point, clamped to the limb, so the grasp lands
    # on the limb's own centreline rather than in the air beside it.
    half = max(0.012, 0.15 * (t_hi - t_lo))
    b_lo, b_hi = max(t_grasp - half, t_lo), min(t_grasp + half, t_hi)
    band = (t >= b_lo) & (t <= b_hi)
    grasp_local = pts[band].mean(axis=0) if band.sum() >= 8 else limb_pts.mean(axis=0)
    bu = (pts[band] - centre) @ u if band.sum() >= 8 else (limb_pts - centre) @ u
    bw = (pts[band] - centre) @ w if band.sum() >= 8 else (limb_pts - centre) @ w

    # A free end exists when the limb runs out to the object's own extreme
    # (a utensil handle tip); the drill's grip is attached at both ends.
    span = float(t.max() - t.min())
    tol = 0.06 * span
    free_sign = 0
    if t_lo - t.min() < tol:
        free_sign = -1
    elif t.max() - t_hi < tol:
        free_sign = +1

    free_t = t.min() if free_sign < 0 else t.max()
    free_end_local = centre + axis * free_t if free_sign else None
    # unit vector, in body frame, pointing from the grasp toward the free end
    free_dir = axis * float(np.sign(free_t - t_mid)) if free_sign else None

    lu, lw = (limb_pts - centre) @ u, (limb_pts - centre) @ w
    return {
        "centroid": centre,
        "axis": axis,
        "perp_u": u,
        "perp_w": w,
        "bin_extents": extent.tolist(),
        "bin_counts": counts.tolist(),
        "limb_bins": [lo, hi],
        "limb_t": [t_lo, t_hi],
        "limb_len": t_hi - t_lo,
        "grasp_local": grasp_local,
        "grasp_t": t_grasp,
        # width the fingers actually close across, at the grasp band
        "grasp_width_u": float(bu.max() - bu.min()),
        "grasp_width_w": float(bw.max() - bw.min()),
        "limb_width_u": float(lu.max() - lu.min()),
        "limb_width_w": float(lw.max() - lw.min()),
        "grasp_to_centroid_xy": float(np.linalg.norm((grasp_local - centre)[:2])),
        "free_end_local": free_end_local,
        "free_dir": free_dir,
        "obj_t_range": [float(t.min()), float(t.max())],
    }


def _cavity_at(top, zmin, zmax, ncell):
    """One rasterisation attempt; None when the interior region is not bounded."""
    lo = top[:, :2].min(axis=0)
    hi = top[:, :2].max(axis=0)
    step = (hi - lo) / ncell
    step[step <= 0] = 1e-6
    occ = np.zeros((ncell, ncell), dtype=bool)
    idx = np.clip(((top[:, :2] - lo) / step).astype(int), 0, ncell - 1)
    occ[idx[:, 0], idx[:, 1]] = True

    seed = (ncell // 2, ncell // 2)
    if occ[seed]:
        free = np.argwhere(~occ)
        if free.size == 0:
            return None
        seed = tuple(free[np.argmin(np.abs(free - ncell / 2).sum(axis=1))])
    seen = np.zeros_like(occ)
    stack = [seed]
    seen[seed] = True
    while stack:
        a, b = stack.pop()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            na, nb = a + da, b + db
            if 0 <= na < ncell and 0 <= nb < ncell and not seen[na, nb] and not occ[na, nb]:
                seen[na, nb] = True
                stack.append((na, nb))
    cells = np.argwhere(seen)
    if cells.size == 0:
        return None
    # a region reaching the raster border has leaked through the wall
    if (cells[:, 0].min() == 0 or cells[:, 1].min() == 0
            or cells[:, 0].max() == ncell - 1 or cells[:, 1].max() == ncell - 1):
        return None

    mouth_lo = lo + cells.min(axis=0) * step
    mouth_hi = lo + (cells.max(axis=0) + 1) * step
    size = mouth_hi - mouth_lo
    if min(size) <= 0 or float(np.prod(size)) < 0.05 * float(np.prod(hi - lo)):
        return None
    return {
        "rim_z": zmax,
        "floor_z": zmin,
        "mouth_lo": mouth_lo,
        "mouth_hi": mouth_hi,
        "mouth_centre": 0.5 * (mouth_lo + mouth_hi),
        "mouth_size": size,
        "ncell": ncell,
    }


def cavity(pts, top_frac=0.80, ncells=(48, 36, 28, 20, 14)):
    """Interior opening and rim height of an open-top container, from its mesh.

    Occupancy of the topmost slice is rasterised and the mouth is the bounded
    empty region inside it. Returns rim z, the mouth's xy bounds and its centre
    -- what the insertion target is expressed relative to, since the storage
    box's cavity is offset from its prim origin and the prim translate alone
    would be the wrong aim point.

    Resolution is swept coarse-ward: a thin crate wall rasterised too finely
    leaves gaps, the fill leaks out to the border and the cavity is missed.
    """
    pts = np.asarray(pts, dtype=float)
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    top = pts[pts[:, 2] >= zmin + top_frac * (zmax - zmin)]
    if top.shape[0] < 16:
        top = pts
    for ncell in ncells:
        got = _cavity_at(top, zmin, zmax, ncell)
        if got is not None:
            return got
    return None


def yaw_quat(yaw):
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` about +Z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / float(np.dot(q, q))


def quat_from_matrix(R):
    """Quaternion (w, x, y, z) from a 3x3 rotation matrix."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def matrix_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def align_rotation(v_from, v_to, spin_axis=None, spin=0.0):
    """Rotation taking unit ``v_from`` to unit ``v_to`` (minimal rotation).

    ``spin`` optionally adds a rotation about ``v_to`` afterwards, which is the
    free degree of freedom left over when only one axis is constrained.
    """
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        R = np.eye(3)
    elif c < -1 + 1e-9:
        # antiparallel: rotate pi about any axis orthogonal to a
        t = np.array([1.0, 0.0, 0.0])
        if abs(a @ t) > 0.9:
            t = np.array([0.0, 1.0, 0.0])
        k = np.cross(a, t)
        k /= np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + 2 * K @ K
    else:
        k = np.cross(a, b)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + K + K @ K / (1 + c)
    if spin_axis is not None and abs(spin) > 1e-12:
        k = np.asarray(spin_axis, dtype=float)
        k = k / np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = (np.eye(3) + np.sin(spin) * K + (1 - np.cos(spin)) * K @ K) @ R
    return R
