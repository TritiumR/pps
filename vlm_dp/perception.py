"""Text-prompted object perception: one camera frame -> per-object masks and world positions.

GroundingDINO localises each named object, SAM cuts its mask, the scene depth lifts it to world.
Labels are assigned one-to-one (Hungarian) and masks are eroded before back-projection.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import re

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from scipy.optimize import linear_sum_assignment

from rekep.isaaclab_helpers import camera_to_rekep_inputs, workspace_bounds_from_scene
from vlm_dp.sim_helpers import center_from_points

_SIG_TOL = 60.0       # uint8-RGB distance. Same object under lighting drift stays well inside, a
                      # pear-vs-apple swap is far outside. Re-ID after track loss needs appearance
                      # evidence, not just a label assignment.
_BOX_THRESH = 0.15    # keep every plausible box and let the assignment choose, rather than pre-filtering
_TEXT_THRESH = 0.22
_IOU_MERGE = 0.80     # boxes overlapping this much are the same physical object
_MIN_PIXELS = 80      # below this a mask cannot be clustered into keypoints (matches the proposer's floor)
# Support ring: the surface an object rests on, sampled just outside its own footprint since it occludes
# what is directly beneath it. Multiples of the object's own radius, so it scales with object size.
_RING_INNER = 1.2     # start clear of the object's own silhouette and mask bleed
_RING_OUTER = 3.0     # stay local, a distant surface is a different one
_RING_PAD = 0.02      # m, keeps the ring usable for a very small object
_MIN_RING_PTS = 50    # below this the ring is not a surface measurement
_MASK_ESCAPE_TOL = 8.0    # px of average bleed past the prompt box, beyond which the mask left its object
_OBSTACLE_MIN_H = 0.01    # m above the support before leftover cloud counts as an obstacle, not the surface


def _backends():
    """Apply the load-bearing GroundingDINO and transformers compat shim and return the vision-backend
    helper (checkpoint dir and image transform). Self-contained, no moka import."""
    from vlm_dp import vision_backends   # deferred: applies the transformers shim on import
    return vision_backends


@functools.lru_cache(maxsize=1)
def _dino():
    """GroundingDINO, built once. The stock helper rebuilds a 700MB checkpoint on every call."""
    vb = _backends()
    from groundingdino.util.inference import load_model
    return load_model(os.path.join(vb.ckpt_dir(), "config", "grounding_dino.py"),
                      os.path.join(vb.ckpt_dir(), "ckpts", "groundingdino_swint_ogc.pth"))


@functools.lru_cache(maxsize=1)
def _sam(device="cuda"):
    """SAM, built once. The stock helper reloads a 2.5GB checkpoint on every call."""
    vb = _backends()
    from segment_anything import build_sam, SamPredictor
    ckpt = os.path.join(vb.ckpt_dir(), "ckpts", "sam_vit_h_4b8939.pth")
    return SamPredictor(build_sam(checkpoint=ckpt).to(device))


@functools.lru_cache(maxsize=1)
def _sam_auto(device="cuda"):
    """SAM automatic mask generator (class-agnostic regions for sam_vlm), sharing the cached SAM weights."""
    _backends()
    from segment_anything import SamAutomaticMaskGenerator
    return SamAutomaticMaskGenerator(_sam(device).model, points_per_side=16, pred_iou_thresh=0.88,
                                     stability_score_thresh=0.9, min_mask_region_area=200)


def _iou(a, b):
    """IoU of two xyxy boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a, b):
    """IoU of two boolean masks (used to drop SAM's nested/duplicate auto-masks)."""
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else 0.0


def _r(t):
    """Round a 3-tuple of metres for logging."""
    return tuple(round(float(x), 3) for x in t)


def _mask_escape(mask, box) -> float:
    """How far a SAM mask spills outside its prompting box, as an average bleed width in pixels.

    Normalised by the box perimeter, not its area: boundary error is a boundary effect, so it scales
    with perimeter. An area fraction would be scale-dependent and would punish small objects for the
    same few pixels of bleed a large one gets away with (measured: a 3px bleed reads as 20% of a 50px
    box but 2% of a 500px one). In these units the threshold is a property of the segmenter, how many
    pixels SAM's boundary wanders, and carries over to any scene or object size.

    SAM's contract is to segment the object its box indicates, so a correct mask lies inside its box and
    bleeds a few pixels at the boundary, never spilling across the scene. This is a self-consistency
    check between two stages the pipeline already runs, with no ground truth and no size threshold.

    It exists because a bad mask is unrecoverable downstream: object extents, position and keypoint
    clustering all read the same points, so one bled mask corrupts all three at once. Measured on tea, a
    bad teapot mask gave 154k points and a 729mm half-extent against 48mm true, and neither percentile
    trimming nor connected-component selection could repair it (the bad cluster is the mask). Rejecting
    the frame is honest, repairing it is not possible.
    """
    total = int(mask.sum())
    if total == 0:
        return float("inf")
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    h, w = mask.shape[-2:]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, 0), w), min(max(y1, 0), h)
    if x1 <= x0 or y1 <= y0:
        return float("inf")
    outside = total - int(mask[y0:y1, x0:x1].sum())
    perimeter = 2.0 * ((x1 - x0) + (y1 - y0))
    return outside / max(perimeter, 1.0)


def _cluster_blobs(pts, voxel=0.02, limit=12, min_pts=_MIN_RING_PTS):
    """Split a point cloud into spatially separate blobs -> ``[(centre, extents), ...]``, largest first.

    Same occupancy labelling as ``_dominant_cluster``, but keeping every component instead of the
    biggest: unexplained scene geometry is several obstacles, not one. Extents use the same
    ``(narrow, wide, half-height)`` convention as ``object_extents``, so a blob is interchangeable with
    a named object everywhere downstream.
    """
    from scipy import ndimage

    lo = pts.min(axis=0)
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    grid = np.zeros(tuple(idx.max(axis=0) + 1), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return []
    per_point = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    out = []
    for label in range(1, n + 1):
        blob = pts[per_point == label]
        if blob.shape[0] < min_pts:
            continue
        half = (blob.max(axis=0) - blob.min(axis=0)) / 2.0
        centre = (blob.max(axis=0) + blob.min(axis=0)) / 2.0
        out.append((centre, (float(min(half[0], half[1])), float(max(half[0], half[1])), float(half[2])),
                    blob.shape[0]))
    out.sort(key=lambda b: -b[2])
    return [(c, e) for c, e, _ in out[:limit]]


def _dominant_cluster(pts, voxel=0.01, max_cells=4_000_000):
    """Largest connected component of a point cloud. Returns (mask, share of points, its extent).

    Voxel connected components, not distance clustering: a back-projected mask is already a raster, so
    occupancy labelling is the natural operation and it needs no neighbour count or density knob, only a
    voxel size set by the depth image's resolution rather than tuned per object.
    """
    from scipy import ndimage

    lo = pts.min(axis=0)
    span = pts.max(axis=0) - lo
    while np.prod(np.floor(span / voxel) + 3) > max_cells:   # coarsen rather than allocate a huge grid
        voxel *= 2.0
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    grid = np.zeros(tuple(idx.max(axis=0) + 1), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))   # 26-connectivity
    if n <= 1:
        return np.ones(pts.shape[0], dtype=bool), 1.0, tuple(span / 2.0)
    per_point = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    biggest = np.bincount(per_point[per_point > 0]).argmax()
    keep = per_point == biggest
    main = pts[keep]
    return keep, float(keep.mean()), tuple((main.max(axis=0) - main.min(axis=0)) / 2.0)


class Perception:
    """Finds the named objects in the camera frame and reports where they are.

    prompts maps scene object to detector text, for example scale to kitchen scale.
    """

    def __init__(self, prompts: dict, fixtures=(), erode: int = 2, device: str = "cuda",
                 segment: str = "groundedsam", support: str | None = None, support_names=()):
        self.fixtures = tuple(fixtures)   # objects whose geometry is workcell calibration, not perception
        self.support = support            # surface the support_names objects rest on
        self.support_names = frozenset(support_names)
        # Fixtures are excluded from the vocabulary so their labels cannot claim a real object's box.
        self.prompts = {n: t for n, t in prompts.items() if n not in self.fixtures}
        self.names = list(self.prompts)
        self.erode = erode
        self.device = device
        # Segmenter: groundedsam is GroundingDINO text vocab plus SAM (default). sam_vlm is class-agnostic
        # SAM regions named by a VLM from plain names.
        self.segment = segment
        self.masks: dict = {}      # name -> (H,W) bool, from the last look
        self._sig: dict = {}       # name -> mean-RGB signature from the first (settled) look
        self.rgb = None
        self.points = None
        self._bounds = None                 # sam_vlm workspace box, measured once (calibration)
        self._id_centroids: dict = {}       # sam_vlm identity: name -> last world centroid (VLM once, then track)
        self._fixture_masks: dict = {}
        self._fixture_points: dict = {}

    def warmup(self):
        """Load detector + segmenter before anything else initialises torch (import-order clash)."""
        _dino()
        _sam(self.device)

    def calibrate(self, env):
        """Measure the fixtures once. They never move and per-frame masks of them are unreliable."""
        if self.segment == "sam_vlm":   # the box that trims class-agnostic SAM masks to the workcell
            self._bounds = workspace_bounds_from_scene(env.env, margin=0.6)   # raw IsaacLab env (has .scene)
        if not self.fixtures:
            return
        _, points, m, id_to_prim = camera_to_rekep_inputs(env.cam, 0)
        for name in self.fixtures:
            rel = re.sub(r"^/World/envs/env_[^/]*/", "", env.env.scene[name].cfg.prim_path)
            ids = [i for i, prim in id_to_prim.items() if rel and rel in prim]
            mask = np.isin(m, ids) & np.isfinite(points).all(axis=-1)
            if not int(mask.sum()):
                raise SystemExit(f"[perception] cannot calibrate fixture {name!r}: it has no points")
            self._fixture_masks[name] = mask
            self._fixture_points[name] = points[mask]
        print(f"[perception] calibrated fixtures (not perceived): {list(self._fixture_points)}", flush=True)

    def observe(self, env) -> dict:
        """Look through the camera. Returns {name: world position} for every object found."""
        rgb, points, _, _ = camera_to_rekep_inputs(env.cam, 0)   # rgb and depth only, seg channel unused
        return self.observe_frame(rgb, points)

    def observe_frame(self, rgb, points) -> dict:
        """Segment one captured frame. Returns {name: world position} for confidently detected objects."""
        self.rgb, self.points = rgb, points
        self.masks = dict(self._segment(rgb), **self._fixture_masks)   # fixtures are known, not detected
        self._verify_identities()
        out = {}
        for name in self.masks:
            pos = self.position(name)
            if pos is not None:
                out[name] = pos
        return out

    def _verify_identities(self):
        """Re-identification needs appearance evidence. A label whose mask no longer looks like the
        object it was first seen as (a Hungarian pear-apple swap after a drop) is dropped from this frame
        rather than accepted, so the belief stays stale until a matching look.
        """
        for name in list(self.masks):
            if name in self._fixture_masks:
                continue
            mask = self._erode(self.masks[name])
            if int(mask.sum()) < _MIN_PIXELS:
                continue
            sig = self.rgb[mask].reshape(-1, 3).mean(axis=0)
            ref = self._sig.get(name)
            if ref is None:
                self._sig[name] = sig
            elif float(np.linalg.norm(sig - ref)) > _SIG_TOL:
                print(f"[perception] '{name}' mask fails appearance check "
                      f"(d={float(np.linalg.norm(sig - ref)):.0f}); rejected this frame", flush=True)
                del self.masks[name]

    def position(self, name) -> np.ndarray | None:
        """Grasp centre of name: the middle of the visible extent on every axis. Objects on the declared
        support use its surface as the bottom, giving the resting-geometry centre."""
        pts = self.object_points(name)
        if pts is None:
            return None
        support_top = None
        if name in self.support_names:
            if self.support:                              # declared entity wins where a task names one
                spts = self.object_points(self.support)
                if spts is not None:
                    support_top = float(np.percentile(spts[:, 2], 95))
            else:                                         # else measure the surface it stands on
                support_top = self.support_height(name)
        return center_from_points(pts, support_top)

    def support_height(self, name) -> float | None:
        """Height of the surface name rests on, measured from the scene cloud around its footprint.

        A named support entity is a hand-authored role, and only one task ever declared one, so every
        other task lost the resting-geometry grasp centre and took the visible-extent midpoint, which
        measures about 1 to 2 cm too high (an object's top is better seen than its occluded bottom) and
        puts the grasp above the object.

        Derived instead: an object occludes what is directly beneath it, but the same surface just
        outside its footprint is in plain view, so a ring around the object measures what it stands on.
        Works for a board, tray, counter or bare table without naming any of them, and needs no scene
        entity to exist for the surface. The median rejects a neighbouring object clipping the ring, and
        points at or above the object's own top are excluded so a taller neighbour cannot pull it up.
        """
        pts = self.object_points(name)
        if pts is None or self.points is None:
            return None
        centre = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
        r_obj = float(np.max(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0))) / 2.0
        scene = self.points[np.isfinite(self.points).all(axis=-1)].reshape(-1, 3)
        d = np.linalg.norm(scene[:, :2] - centre, axis=1)
        ring = scene[(d > r_obj * _RING_INNER) & (d < r_obj * _RING_OUTER + _RING_PAD)]
        ring = ring[ring[:, 2] < float(np.percentile(pts[:, 2], 5))]   # below the object's own base
        if ring.shape[0] < _MIN_RING_PTS:
            return None
        return float(np.median(ring[:, 2]))

    def unexplained_obstacles(self, support_z=None, limit=12, max_half=0.10) -> list:
        """Scene geometry that belongs to no named object. Returns unnamed [(centre, extents), ...].

        The counterpart to deriving the vocabulary from the instruction. Once only referents are named,
        the distractors stop being scene objects, and an unnamed distractor the arm cannot see is worse
        than a named one it might grasp. But an obstacle never needs identity, only extent: the cost must
        route around a thing, not know what it is.

        So the leftover cloud (everything outside every named mask, above the support plane) is clustered
        into blobs and returned with the same (centre, extents) shape a named object has. The existing
        keepout terms then treat them as obstacles with no change: they exclude by name (grasp_obj,
        payload, place_target, placed), and an anonymous blob matches none of those, so it can never be
        mistaken for a target. Identity for referents, geometry for everything else.
        """
        if self.points is None:
            return []
        finite = np.isfinite(self.points).all(axis=-1)
        claimed = np.zeros(finite.shape, dtype=bool)
        for mask in self.masks.values():
            claimed |= self._dilate(mask)          # dilate: a mask's own boundary is not an obstacle
        pts = self.points[finite & ~claimed].reshape(-1, 3)
        if support_z is not None:
            pts = pts[pts[:, 2] > support_z + _OBSTACLE_MIN_H]   # the support plane is not an obstacle
        if pts.shape[0] < _MIN_RING_PTS:
            return []
        # Only object-scale blobs. Anything wider is structure (a counter edge, a wall, the robot's own
        # column): a keepout cylinder around it would wall off the workspace, which is why clear caps its
        # radius too. Structure is already handled by the table plane (floor and z_table).
        return [(c, e) for c, e in _cluster_blobs(pts, limit=limit * 3) if e[1] <= max_half][:limit]

    def _dilate(self, mask):
        """Grow a mask by the erosion radius, so its own boundary is not read as unexplained."""
        k = np.ones((2 * self.erode + 1, 2 * self.erode + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)

    def object_extents(self, name) -> tuple | None:
        """(grip, keepout, half_height) estimated from the segmented point cloud, no object model."""
        pts = self.object_points(name)
        if pts is None:
            return None
        x_half = float(pts[:, 0].max() - pts[:, 0].min()) / 2.0
        y_half = float(pts[:, 1].max() - pts[:, 1].min()) / 2.0
        # Slot [2] is a half-height, consumed as z_bottom, keepout cylinder and place z. usd_extents
        # returns half too.
        z_half = float(pts[:, 2].max() - pts[:, 2].min()) / 2.0
        return (min(x_half, y_half), max(x_half, y_half), z_half)

    def extent_diagnostics(self, name) -> str | None:
        """Why an extent is what it is: point count, trimmed spans, and cluster structure.

        Answers the one question that decides how to fix a bad extent: is the mask correct with a few
        stray back-projected pixels (percentile trimming repairs it), or has the mask swallowed a large
        region (trimming cannot repair it and the measurement must be rejected instead)? Read frac (share
        of points in the dominant cluster) and main (that cluster's own extent).
        """
        pts = self.object_points(name)
        if pts is None:
            return None
        raw = tuple((pts[:, i].max() - pts[:, i].min()) / 2.0 for i in range(3))
        trims = {}
        for lo, hi in ((1, 99), (5, 95), (10, 90)):
            band = np.percentile(pts, [lo, hi], axis=0)
            trims[f"{lo}-{hi}"] = tuple((band[1, i] - band[0, i]) / 2.0 for i in range(3))
        keep, frac, main = _dominant_cluster(pts)
        return (f"n={pts.shape[0]} raw={_r(raw)} "
                + " ".join(f"p{k}={_r(v)}" for k, v in trims.items())
                + f" | clusters: frac={frac:.2f} n_main={int(keep.sum())} main={_r(main)}")

    def object_points(self, name) -> np.ndarray | None:
        """The object's world points: eroded mask back-projected through depth (fixtures: calibrated)."""
        if name in self._fixture_points:
            return self._fixture_points[name]
        mask = self.masks.get(name)
        if mask is None or self.points is None:
            return None
        mask = self._erode(mask)
        sel = mask & np.isfinite(self.points).all(axis=-1)
        pts = self.points[sel]
        return pts if pts.shape[0] >= 20 else None

    def label_image(self):
        """``(labels (H,W) int32, id_to_prim)`` in the shape the keypoint proposer consumes."""
        labels = np.zeros(self.points.shape[:2], dtype=np.int32)
        id_to_prim = {}
        for i, (name, mask) in enumerate(self.masks.items(), start=1):
            labels[self._erode(mask)] = i
            id_to_prim[i] = name
        return labels, id_to_prim

    def _erode(self, mask):
        if self.erode <= 0:
            return mask
        k = np.ones((2 * self.erode + 1, 2 * self.erode + 1), np.uint8)
        return cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)

    def _segment(self, rgb) -> dict:
        """RGB -> ``{name: binary mask}`` by the selected segmenter (see ``segment``)."""
        if self.segment == "sam_vlm":
            return self._segment_sam_vlm(rgb)
        return self._segment_groundedsam(rgb)

    def _segment_sam_vlm(self, rgb) -> dict:
        """Class-agnostic SAM regions named once by a VLM. Later frames re-identify by position."""
        masks = self._sam_masks(rgb)
        centroids = [self._world_centroid(m) for m in masks]
        keep = [(m, c) for m, c in zip(masks, centroids) if c is not None]
        if not keep:
            return {}
        masks, centroids = [m for m, _ in keep], [c for _, c in keep]
        idx = (self._reassociate(centroids) if self._id_centroids
               else self._vlm_identify(rgb, masks, centroids))
        return {name: masks[i] for name, i in idx.items()}

    def _sam_masks(self, rgb) -> list:
        """Class-agnostic SAM regions, trimmed to the workcell box and de-duplicated (largest first)."""
        anns = _sam_auto(self.device).generate(rgb)
        bounds = self._bounds if self._bounds and self._bounds[0] is not None else None
        within = None
        if bounds is not None:
            bmin, bmax = bounds
            within = (np.isfinite(self.points).all(axis=-1)
                      & (self.points[..., 0] >= bmin[0]) & (self.points[..., 0] <= bmax[0])
                      & (self.points[..., 1] >= bmin[1]) & (self.points[..., 1] <= bmax[1])
                      & (self.points[..., 2] >= bmin[2]) & (self.points[..., 2] <= bmax[2]))
        out: list = []
        for a in sorted(anns, key=lambda a: -a["area"]):
            m = np.asarray(a["segmentation"], dtype=bool)
            if within is not None:
                m = m & within
            if int(m.sum()) < _MIN_PIXELS:
                continue
            if any(_mask_iou(m, o) > 0.7 for o in out):   # SAM auto returns nested masks, keep the larger
                continue
            out.append(m)
            if len(out) >= 20:
                break
        return out

    def _world_centroid(self, mask):
        """Mean world position of a mask's valid depth points, or None if too few."""
        sel = mask & np.isfinite(self.points).all(axis=-1)
        return self.points[sel].mean(axis=0) if int(sel.sum()) >= 20 else None

    def _vlm_identify(self, rgb, masks, centroids) -> dict:
        """Ask GPT-4o which numbered region is which named object, caching each name's centroid."""
        names = list(self.prompts)
        # Build the format example from THIS task's own names: a fixed example (pear/apple) primes the
        # VLM with objects that do not exist in another task's scene.
        example = ", ".join(f'"{n}": {i}' for i, n in enumerate(names[:2])) or '"<name>": 0'
        prompt = (
            f"The image shows a scene; candidate regions are outlined in red and numbered 0 to {len(masks) - 1}. "
            "Some regions are the table, the background, or a part of an object -- ignore those. "
            f"For each object in this list, give the number of the region that IS that object: {names}. "
            "If an object is not visible, use -1. "
            'Reply with only a JSON object mapping each name to its integer region number, '
            f'e.g. {{{example}}}.')
        overlay = self._mask_overlay(rgb, masks)
        try:                                           # keep the exact image the VLM saw, for inspection
            dbg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "vlm_mpc", "vlm_base", "sam_vlm_query.png")
            os.makedirs(os.path.dirname(dbg), exist_ok=True)
            cv2.imwrite(dbg, overlay[..., ::-1])
        except Exception:
            pass
        try:
            mapping = json.loads(self._chat(overlay, prompt))
        except Exception as exc:                       # a bad reply must not crash the run
            print(f"[perception:sam_vlm] VLM identify failed ({exc}); no objects named this frame", flush=True)
            return {}
        out = {}
        for name in names:
            i = mapping.get(name)
            if isinstance(i, int) and 0 <= i < len(masks):
                out[name] = i
                self._id_centroids[name] = centroids[i]
        print(f"[perception:sam_vlm] VLM named {out} from {len(masks)} regions", flush=True)
        return out

    def _reassociate(self, centroids) -> dict:
        """After the VLM has named the objects once, re-identify them by nearest world centroid."""
        out, used = {}, set()
        for name, c0 in self._id_centroids.items():
            best, best_d = None, np.inf
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d = float(np.linalg.norm(c - c0))
                if d < best_d:
                    best, best_d = i, d
            if best is not None and best_d < 0.5:     # an object does not jump half a metre between looks
                out[name] = best
                used.add(best)
                self._id_centroids[name] = centroids[best]
        return out

    def _mask_overlay(self, rgb, masks):
        """Draw each candidate region's outline and its index on the RGB, for the VLM to read."""
        img = np.ascontiguousarray(rgb.copy())
        for i, m in enumerate(masks):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
            ys, xs = np.where(m)
            cv2.putText(img, str(i), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def _chat(self, rgb, prompt) -> str:
        """One GPT-4o vision call returning a JSON string (same client/encoding as ConstraintGenerator)."""
        _, buf = cv2.imencode(".png", rgb[..., ::-1])   # cv2 expects BGR
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        reply = client.chat.completions.create(
            model="gpt-4o", temperature=0.0, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        return reply.choices[0].message.content

    def _segment_groundedsam(self, rgb) -> dict:
        """RGB -> ``{name: binary mask}``, with the labels competing for objects one-to-one."""
        # Deferred, not hoistable: the vision-backend import order is load-bearing (see _backends).
        from vlm_dp.vision_backends import load_pil_image

        _, image_torch = load_pil_image(Image.fromarray(rgb).convert("RGB"))
        H, W = rgb.shape[:2]
        boxes = self._assign(self._detect(image_torch, W, H))
        if not boxes:
            return {}

        order = list(boxes)
        sam = _sam(self.device)
        sam.set_image(rgb)
        b = torch.as_tensor(np.stack([boxes[n] for n in order]), dtype=torch.float32)
        with torch.no_grad():
            masks, _, _ = sam.predict_torch(
                point_coords=None, point_labels=None,
                boxes=sam.transform.apply_boxes_torch(b, (H, W)).to(self.device),
                multimask_output=False)
        out = {}
        for i, name in enumerate(order):
            m = masks[i, 0].cpu().numpy()
            if int(m.sum()) < _MIN_PIXELS:
                continue
            escaped = _mask_escape(m, boxes[name])
            if escaped > _MASK_ESCAPE_TOL:               # see _mask_escape: a mask that left its own box
                print(f"[perception] '{name}' mask escaped its detection box "
                      f"({escaped:.0f}px average bleed); rejected this frame", flush=True)
                continue
            out[name] = m
        return out

    def _detect(self, image_torch, W, H) -> dict:
        """Every plausible box per label, rather than each label's single best one."""
        # Deferred, not hoistable: the vision-backend import order is load-bearing (see _backends).
        from groundingdino.util import box_ops
        from groundingdino.util.inference import predict

        model = _dino()
        dets = {}
        for name, text in self.prompts.items():
            b, logits, _ = predict(model=model, image=image_torch, caption=text,
                                   box_threshold=_BOX_THRESH, text_threshold=_TEXT_THRESH)
            if len(logits) == 0:
                dets[name] = []
                continue
            xyxy = box_ops.box_cxcywh_to_xyxy(b) * torch.Tensor([W, H, W, H])
            dets[name] = [(xyxy[i].numpy(), float(logits[i])) for i in range(len(logits))]
        return dets

    def _assign(self, dets) -> dict:
        """One box per label, one label per box (Hungarian over detection scores)."""
        cands = []                                     # the union of every label's boxes, deduplicated
        for name in self.names:
            for box, _ in dets.get(name, []):
                if not any(_iou(box, c) > _IOU_MERGE for c in cands):
                    cands.append(box)
        if not cands:
            return {}
        score = np.zeros((len(self.names), len(cands)))   # zero where a label never proposed that box
        for i, name in enumerate(self.names):
            for box, s in dets.get(name, []):
                for j, c in enumerate(cands):
                    if _iou(box, c) > _IOU_MERGE:
                        score[i, j] = max(score[i, j], s)
        rows, cols = linear_sum_assignment(-score)
        return {self.names[i]: cands[j] for i, j in zip(rows, cols) if score[i, j] > 0}
