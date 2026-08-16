"""Detect named objects, segment them, and lift their masks into world space."""
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
from vlm_dp.sim_helpers import center_from_points, usd_extents

_SIG_TOL = 60.0


_BOX_THRESH = 0.15
_TEXT_THRESH = 0.22
_IOU_MERGE = 0.80
_MIN_PIXELS = 80


_RING_INNER = 1.2
_RING_OUTER = 3.0
_RING_PAD = 0.02
_SUPPORT_MAX_HALF = 0.10
_MIN_RING_PTS = 50
_MASK_ESCAPE_TOL = 8.0
# Fraction of a detection box whose depth must fall inside the workspace for it to be a candidate.
_BOX_IN_WORKSPACE = 0.5
_OBSTACLE_MIN_H = 0.01

# --- size prior on the mask -> object assignment -------------------------------------------
# A caption alone cannot tell two objects of the same colour apart: on the tea scene both the
# teapot and the teacup are pale-green jade, and GroundingDINO happily reads the (larger,
# nearer) teapot as a "teacup". Depth can tell them apart -- the task declares how big each
# object is, and a candidate mask's cloud says how big the thing under it actually is.
#
# What ranks candidates is ONE-SIDED: only growth counts against a pairing. A cloud wider than
# the object it is named for is a mask that has left the object, and there is no innocent
# reading of it. Shrinkage has several: a single depth view sees one side, the mask is eroded,
# grazing pixels drop out, and a half-occluded mango reads 18mm against a 47mm USD half-width
# while being perfectly correct. Penalising shrinkage symmetrically is not merely
# over-cautious, it actively misassigns -- it moved the mango's label onto the cabbage next to
# it, because the cabbage's fuller cloud "fitted" the mango's declared size better than the
# mango's own occluded one did.
_SIZE_OVER = 0.45
# Shrinkage still means something when there is no caption at all to go on, so the rescue path
# (and only the rescue path) reads a two-sided agreement, and demands a strong one.
_SIZE_UNDER = 1.10
_SIZE_ACCEPT = 0.60
# Upper bound on candidate boxes carried into the (batched) mask pass, for cost.
_MAX_CANDIDATES = 40


def _backends():
    """Load the local vision backend compatibility helpers."""
    from vlm_dp import vision_backends
    return vision_backends


@functools.lru_cache(maxsize=1)
def _dino():
    """Load and cache the GroundingDINO model."""
    vb = _backends()
    from groundingdino.util.inference import load_model
    return load_model(os.path.join(vb.ckpt_dir(), "config", "grounding_dino.py"),
                      os.path.join(vb.ckpt_dir(), "ckpts", "groundingdino_swint_ogc.pth"))


@functools.lru_cache(maxsize=1)
def _sam(device="cuda"):
    """Load and cache the SAM predictor."""
    vb = _backends()
    from segment_anything import build_sam, SamPredictor
    ckpt = os.path.join(vb.ckpt_dir(), "ckpts", "sam_vit_h_4b8939.pth")
    return SamPredictor(build_sam(checkpoint=ckpt).to(device))


@functools.lru_cache(maxsize=1)
def _sam_auto(device="cuda"):
    """Load and cache the SAM automatic mask generator."""
    _backends()
    from segment_anything import SamAutomaticMaskGenerator
    return SamAutomaticMaskGenerator(_sam(device).model, points_per_side=16, pred_iou_thresh=0.88,
                                     stability_score_thresh=0.9, min_mask_region_area=200)


def _iou(a, b):
    """Return the intersection-over-union of two boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a, b):
    """Return the intersection-over-union of two masks."""
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else 0.0


def _r(t):
    """Round a three-dimensional metric tuple for logging."""
    return tuple(round(float(x), 3) for x in t)


def _mask_escape(mask, box) -> float:
    """Measure average mask spill beyond its prompt box."""
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


def _log_ratio(ext, ref):
    """Per-axis log size ratio of a measured half-extent triple against a declared one."""
    if ext is None or ref is None:
        return None
    e = np.maximum(np.asarray(ext, dtype=float), 1e-3)
    r = np.maximum(np.asarray(ref, dtype=float), 1e-3)
    return np.log(e / r)


def _size_penalty(ext, ref) -> float:
    """Discount in (0, 1] for a candidate that is too BIG to be the named object.

    Scale-free: it works on log ratios, so it carries no task-specific length. Returns 1.0 --
    no opinion -- when either side is missing, or whenever the candidate is no larger than
    declared, so an occluded object is never punished for the half of it the camera cannot see.
    """
    ratio = _log_ratio(ext, ref)
    if ratio is None:
        return 1.0
    d = float(np.mean(np.maximum(ratio, 0.0))) / _SIZE_OVER
    return float(np.exp(-d * d))


def _size_agreement(ext, ref) -> float:
    """Two-sided score in (0, 1] of how well a measured extent matches a declared one."""
    ratio = _log_ratio(ext, ref)
    if ratio is None:
        return 0.0
    d = float(np.mean(np.maximum(ratio, 0.0) / _SIZE_OVER + np.maximum(-ratio, 0.0) / _SIZE_UNDER))
    return float(np.exp(-d * d))


def _cluster_blobs(pts, voxel=0.02, limit=12, min_pts=_MIN_RING_PTS):
    """Split a point cloud into connected obstacle blobs."""
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
    """Return the largest connected component of a point cloud."""
    from scipy import ndimage

    lo = pts.min(axis=0)
    span = pts.max(axis=0) - lo
    while np.prod(np.floor(span / voxel) + 3) > max_cells:
        voxel *= 2.0
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    grid = np.zeros(tuple(idx.max(axis=0) + 1), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return np.ones(pts.shape[0], dtype=bool), 1.0, tuple(span / 2.0)
    per_point = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    biggest = np.bincount(per_point[per_point > 0]).argmax()
    keep = per_point == biggest
    main = pts[keep]
    return keep, float(keep.mean()), tuple((main.max(axis=0) - main.min(axis=0)) / 2.0)


class Perception:
    """Detect named objects and estimate their world geometry."""

    def __init__(self, prompts: dict, fixtures=(), erode: int = 2, device: str = "cuda",
                 segment: str = "groundedsam", support: str | None = None, support_names=(),
                 support_extents: bool = False):
        self.fixtures = tuple(fixtures)
        self.support = support
        self.support_names = frozenset(support_names)


        self.support_extents = bool(support_extents)


        self.distrust: set = set()

        self.prompts = {n: t for n, t in prompts.items() if n not in self.fixtures}
        self.names = list(self.prompts)
        self.erode = erode
        self.device = device


        self.segment = segment
        self.masks: dict = {}
        self._sig: dict = {}
        self.rgb = None
        self.points = None
        self._bounds = None
        self._id_centroids: dict = {}
        self._fixture_masks: dict = {}
        self._fixture_points: dict = {}
        # Declared half-extents per object, filled by calibrate() from whatever the task's
        # assets say. Empty (no size prior at all) until then, and for any object the task
        # does not describe.
        self.expected_extents: dict = {}
        # Per-frame memo for object_points(), whose dominant-component reduction is the one
        # genuinely non-trivial computation on a path several callers hit repeatedly.
        self._cloud_cache: dict = {}

    def warmup(self):
        """Load detector and segmenter models."""
        _dino()
        _sam(self.device)

    def calibrate(self, env):
        """Measure static fixture masks and points."""
        self._bounds = workspace_bounds_from_scene(env.env, margin=0.6)
        if self._bounds[0] is not None:
            print(f"[perception] workspace bounds {np.round(self._bounds[0], 2)} .. "
                  f"{np.round(self._bounds[1], 2)}", flush=True)
        # The size prior's reference side. usd_extents() is the same reader rekep grounding uses
        # one layer up for its gross-extent preflight, so the prior is measured against exactly
        # the quantity that would later refuse the plan. Objects with no USD body (a table that
        # is part of the room, a distractor absent from this scene) simply get no prior.
        self.expected_extents = {n: e for n, e in usd_extents(env, self.names).items()}
        if self.expected_extents:
            print(f"[perception] size prior over {sorted(self.expected_extents)} "
                  f"(no declared extent for {sorted(set(self.names) - set(self.expected_extents))})",
                  flush=True)
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
        """Capture and process one camera frame."""
        rgb, points, _, _ = camera_to_rekep_inputs(env.cam, 0)
        return self.observe_frame(rgb, points)

    def observe_frame(self, rgb, points) -> dict:
        """Segment a frame and return detected object positions."""
        self.rgb, self.points = rgb, points
        self.masks = dict(self._segment(rgb), **self._fixture_masks)
        self._cloud_cache = {}
        self._verify_identities()
        out = {}
        for name in self.masks:
            pos = self.position(name)
            if pos is not None:
                out[name] = pos
        return out

    def _verify_identities(self):
        """Reject detections that fail appearance consistency."""
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
            elif name in self.distrust:


                self._sig[name] = sig
            elif float(np.linalg.norm(sig - ref)) > _SIG_TOL:
                print(f"[perception] '{name}' mask fails appearance check "
                      f"(d={float(np.linalg.norm(sig - ref)):.0f}); rejected this frame", flush=True)
                del self.masks[name]

    def position(self, name) -> np.ndarray | None:
        """Estimate an object's world-space grasp center."""
        pts = self.object_points(name)
        if pts is None:
            return None
        support_top = None
        if name in self.support_names and self.support:
            spts = self.object_points(self.support)
            if spts is not None:
                support_top = float(np.percentile(spts[:, 2], 95))


        half = (pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)).max() / 2.0
        if support_top is None and float(half) <= _SUPPORT_MAX_HALF:


            support_top = self.support_height(name)
        return center_from_points(pts, support_top)

    def support_height(self, name) -> float | None:
        """Estimate the supporting surface height around an object."""
        pts = self.object_points(name)
        if pts is None or self.points is None:
            return None
        centre = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
        r_obj = float(np.max(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0))) / 2.0
        scene = self.points[np.isfinite(self.points).all(axis=-1)].reshape(-1, 3)
        d = np.linalg.norm(scene[:, :2] - centre, axis=1)
        ring = scene[(d > r_obj * _RING_INNER) & (d < r_obj * _RING_OUTER + _RING_PAD)]
        ring = ring[ring[:, 2] < float(np.percentile(pts[:, 2], 5))]
        if ring.shape[0] < _MIN_RING_PTS:
            return None
        return float(np.median(ring[:, 2]))

    def unexplained_obstacles(self, support_z=None, limit=12, max_half=0.10) -> list:
        """Return clustered scene geometry not claimed by named objects."""
        if self.points is None:
            return []
        finite = np.isfinite(self.points).all(axis=-1)
        claimed = np.zeros(finite.shape, dtype=bool)
        for mask in self.masks.values():
            claimed |= self._dilate(mask)
        pts = self.points[finite & ~claimed].reshape(-1, 3)
        if support_z is not None:
            pts = pts[pts[:, 2] > support_z + _OBSTACLE_MIN_H]
        if pts.shape[0] < _MIN_RING_PTS:
            return []


        return [(c, e) for c, e in _cluster_blobs(pts, limit=limit * 3) if e[1] <= max_half][:limit]

    def _dilate(self, mask):
        """Dilate a binary mask by the configured radius."""
        k = np.ones((2 * self.erode + 1, 2 * self.erode + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)

    def object_extents(self, name) -> tuple | None:
        """Estimate grasp, keepout, and height extents from object points."""
        pts = self.object_points(name)
        if pts is None:
            return None
        x_half = float(pts[:, 0].max() - pts[:, 0].min()) / 2.0
        y_half = float(pts[:, 1].max() - pts[:, 1].min()) / 2.0


        z_half = float(pts[:, 2].max() - pts[:, 2].min()) / 2.0
        if self.support_extents:


            sup = self.support_height(name)
            if sup is not None:


                z_half = max(z_half, (float(np.percentile(pts[:, 2], 95)) - sup) / 2.0)
        return (min(x_half, y_half), max(x_half, y_half), z_half)

    def extent_diagnostics(self, name) -> str | None:
        """Summarize point-cloud extent and cluster diagnostics."""
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
        """Return valid world points inside an object's eroded mask.

        Reduced to the cloud's dominant connected component. A mask that is correct to the eye
        still leaks a handful of pixels at its border onto whatever lies behind the object --
        the far wall, the floor -- and depth turns each of those into a point metres away. One
        such point is enough to report a 96mm teapot as 770mm wide, which is how a perfectly
        centred mask (centre error 43mm) came to be refused as "run off the object". The
        component the mask is actually sitting on is what every caller means by the object.
        """
        if name in self._fixture_points:
            return self._fixture_points[name]
        mask = self.masks.get(name)
        if mask is None or self.points is None:
            return None
        if name in self._cloud_cache:
            return self._cloud_cache[name]
        sel = self._erode(mask) & np.isfinite(self.points).all(axis=-1)
        pts = self.points[sel]
        if pts.shape[0] < 20:
            return None
        keep, frac, _ = _dominant_cluster(pts)
        if int(keep.sum()) >= 20 and frac < 1.0:
            print(f"[perception] '{name}': dropped {int((~keep).sum())} of {pts.shape[0]} "
                  f"cloud points outside the dominant component", flush=True)
            pts = pts[keep]
        self._cloud_cache[name] = pts
        return pts

    def label_image(self):
        """Return integer labels and their object-name mapping."""
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
        """Segment named objects with the configured backend."""
        if self.segment == "sam_vlm":
            return self._segment_sam_vlm(rgb)
        return self._segment_groundedsam(rgb)

    def _segment_sam_vlm(self, rgb) -> dict:
        """Name class-agnostic SAM regions with a VLM."""
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
        """Generate filtered, deduplicated SAM masks."""
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
            if any(_mask_iou(m, o) > 0.7 for o in out):
                continue
            out.append(m)
            if len(out) >= 20:
                break
        return out

    def _world_centroid(self, mask):
        """Return a mask's world-space centroid when depth is sufficient."""
        sel = mask & np.isfinite(self.points).all(axis=-1)
        return self.points[sel].mean(axis=0) if int(sel.sum()) >= 20 else None

    def _vlm_identify(self, rgb, masks, centroids) -> dict:
        """Assign object names to numbered regions with GPT-4o."""
        names = list(self.prompts)


        example = ", ".join(f'"{n}": {i}' for i, n in enumerate(names[:2])) or '"<name>": 0'
        prompt = (
            f"The image shows a scene; candidate regions are outlined in red and numbered 0 to {len(masks) - 1}. "
            "Some regions are the table, the background, or a part of an object -- ignore those. "
            f"For each object in this list, give the number of the region that IS that object: {names}. "
            "If an object is not visible, use -1. "
            'Reply with only a JSON object mapping each name to its integer region number, '
            f'e.g. {{{example}}}.')
        overlay = self._mask_overlay(rgb, masks)
        try:
            dbg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "vlm_mpc", "vlm_base", "sam_vlm_query.png")
            os.makedirs(os.path.dirname(dbg), exist_ok=True)
            cv2.imwrite(dbg, overlay[..., ::-1])
        except Exception:
            pass
        try:
            mapping = json.loads(self._chat(overlay, prompt))
        except Exception as exc:
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
        """Reassociate named objects by nearest world centroid."""
        out, used = {}, set()
        for name, c0 in self._id_centroids.items():
            best, best_d = None, np.inf
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d = float(np.linalg.norm(c - c0))
                if d < best_d:
                    best, best_d = i, d
            if best is not None and best_d < 0.5:
                out[name] = best
                used.add(best)
                self._id_centroids[name] = centroids[best]
        return out

    def _mask_overlay(self, rgb, masks):
        """Draw numbered region outlines for VLM identification."""
        img = np.ascontiguousarray(rgb.copy())
        for i, m in enumerate(masks):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
            ys, xs = np.where(m)
            cv2.putText(img, str(i), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def _chat(self, rgb, prompt) -> str:
        """Request a JSON region mapping from GPT-4o vision."""
        _, buf = cv2.imencode(".png", rgb[..., ::-1])
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
        """Segment named objects with GroundingDINO and SAM.

        The mask pass runs over *every* candidate box, not only the ones the caption scores
        would have picked, so the assignment can be decided on what is actually under each box
        (its depth-cloud size) rather than on the caption alone. SAM's per-box prediction is
        independent given the image embedding, so the masks of the boxes that would have been
        chosen anyway are bit-for-bit what the single-pass version produced.
        """

        from vlm_dp.vision_backends import load_pil_image

        _, image_torch = load_pil_image(Image.fromarray(rgb).convert("RGB"))
        H, W = rgb.shape[:2]
        cands, score = self._candidates(self._in_workspace(self._detect(image_torch, W, H)))
        if not cands:
            return {}
        cand_masks = self._masks_for(rgb, cands)
        chosen = self._assign(cands, score, cand_masks)
        if not chosen:
            return {}
        boxes = {name: cands[j] for name, j in chosen.items()}

        out = {}
        for name, j in chosen.items():
            m = cand_masks[j]
            if int(m.sum()) < _MIN_PIXELS:
                continue
            escaped = _mask_escape(m, boxes[name])
            if escaped > _MASK_ESCAPE_TOL:
                print(f"[perception] '{name}' mask escaped its detection box "
                      f"({escaped:.0f}px average bleed); rejected this frame", flush=True)
                continue
            inside = self._workspace_fraction(m)
            if inside is not None and inside < _BOX_IN_WORKSPACE:
                # The box survived the workspace test but the mask SAM grew inside it did not:
                # the detector boxed a piece of the room, not a task object.
                print(f"[perception] '{name}' mask lies {(1 - inside) * 100:.0f}% outside the "
                      f"workspace; rejected this frame", flush=True)
                continue
            out[name] = m
        if os.environ.get("VLMDP_SEG_DEBUG"):
            self._dump_segmentation(rgb, out, boxes)
        return out

    def _dump_segmentation(self, rgb, masks, boxes):
        """Write an annotated segmentation frame for offline inspection."""
        img = np.ascontiguousarray(rgb.copy())
        for i, (name, m) in enumerate(masks.items()):
            colour = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                      (0, 255, 255)][i % 6]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, colour, 2)
            x0, y0, x1, y1 = (int(round(float(v))) for v in boxes[name])
            cv2.rectangle(img, (x0, y0), (x1, y1), colour, 1)
            cv2.putText(img, name, (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colour, 2, cv2.LINE_AA)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "vlm_mpc", "vlm_base",
                            f"seg_debug_p{os.getpid()}_{len(self._sig)}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img[..., ::-1])
        print(f"[perception] segmentation debug frame -> {path}", flush=True)

    def _detect(self, image_torch, W, H) -> dict:
        """Return plausible detection boxes for each prompt."""

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

    def _workspace_mask(self):
        """Return the per-pixel in-workspace test, or None when there is no workspace box."""
        if self._bounds is None or self._bounds[0] is None or self.points is None:
            return None
        lo, hi = self._bounds
        pts = self.points
        return (np.isfinite(pts).all(axis=-1)
                & (pts[..., 0] >= lo[0]) & (pts[..., 0] <= hi[0])
                & (pts[..., 1] >= lo[1]) & (pts[..., 1] <= hi[1])
                & (pts[..., 2] >= lo[2]) & (pts[..., 2] <= hi[2]))

    def _workspace_fraction(self, mask):
        """Fraction of a mask's depth-carrying pixels that fall inside the workspace."""
        within = self._workspace_mask()
        if within is None:
            return None
        finite = mask & np.isfinite(self.points).all(axis=-1)
        n = int(finite.sum())
        return None if n < _MIN_PIXELS else float((finite & within).sum()) / n

    def _in_workspace(self, dets) -> dict:
        """Drop detections that mostly lie outside the task workspace.

        An open-vocabulary detector asked for a "teapot" in a furnished room will happily return
        the bookcase across the room at a respectable score, and the Hungarian assignment has no
        way to prefer the real one. Depth does: the task's objects are, by construction, inside
        the workspace box, and a box whose points mostly are not is not one of them.
        """
        within = self._workspace_mask()
        if within is None:
            return dets
        h, w = within.shape
        out = {}
        for name, boxes in dets.items():
            kept = []
            for box, score in boxes:
                x0, y0, x1, y1 = (int(round(float(v))) for v in box)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(max(x1, 0), w), min(max(y1, 0), h)
                if x1 <= x0 or y1 <= y0:
                    continue
                patch = within[y0:y1, x0:x1]
                if float(patch.mean()) >= _BOX_IN_WORKSPACE:
                    kept.append((box, score))
            if len(kept) != len(boxes):
                print(f"[perception] '{name}': dropped {len(boxes) - len(kept)} of {len(boxes)} "
                      f"detections lying outside the workspace", flush=True)
            out[name] = kept
        return out

    def _candidates(self, dets):
        """Merge every caption's detections into unique boxes and score them by caption."""
        cands = []
        for name in self.names:
            for box, _ in dets.get(name, []):
                if not any(_iou(box, c) > _IOU_MERGE for c in cands):
                    cands.append(box)
        cands = cands[:_MAX_CANDIDATES]
        score = np.zeros((len(self.names), len(cands)))
        for i, name in enumerate(self.names):
            for box, s in dets.get(name, []):
                for j, c in enumerate(cands):
                    if _iou(box, c) > _IOU_MERGE:
                        score[i, j] = max(score[i, j], s)
        return cands, score

    def _masks_for(self, rgb, boxes) -> list:
        """Run one batched SAM pass over candidate boxes."""
        H, W = rgb.shape[:2]
        sam = _sam(self.device)
        sam.set_image(rgb)
        b = torch.as_tensor(np.stack(boxes), dtype=torch.float32)
        with torch.no_grad():
            masks, _, _ = sam.predict_torch(
                point_coords=None, point_labels=None,
                boxes=sam.transform.apply_boxes_torch(b, (H, W)).to(self.device),
                multimask_output=False)
        return [masks[i, 0].cpu().numpy() for i in range(len(boxes))]

    def _extent_under(self, mask, valid) -> tuple | None:
        """Half-extents of the depth cloud under a mask, ordered like the declared extents.

        Trimmed at the 1st/99th percentile rather than taken raw: border pixels that leaked
        onto the background would otherwise make every candidate look enormous, which is the
        one error the prior must not make -- it reads growth as evidence of a bad mask. This
        is the cheap stand-in for the dominant-component reduction object_points() does, which
        would be too costly to run on every candidate of every frame.
        """
        pts = self.points[self._erode(mask) & valid]
        if pts.shape[0] < 20:
            return None
        lo, hi = np.percentile(pts, [1, 99], axis=0)
        half = (hi - lo) / 2.0
        return (float(min(half[0], half[1])), float(max(half[0], half[1])), float(half[2]))

    def _assign(self, cands, score, masks) -> dict:
        """Assign one unique candidate mask to each label, on caption score and size prior."""
        if not cands:
            return {}
        weight, agree = np.array(score, dtype=float), None
        if self.expected_extents and self.points is not None:
            valid = np.isfinite(self.points).all(axis=-1)
            within = self._workspace_mask()
            if within is not None:
                valid = valid & within
            ext = [self._extent_under(m, valid) for m in masks]
            ref = [self.expected_extents.get(name) for name in self.names]
            weight = weight * np.array([[_size_penalty(e, r) for e in ext] for r in ref])
            agree = np.array([[_size_agreement(e, r) for e in ext] for r in ref])
        rows, cols = linear_sum_assignment(-weight)
        out = {self.names[i]: int(j) for i, j in zip(rows, cols) if weight[i, j] > 0}
        if agree is not None:
            self._rescue(out, agree)
            self._report_prior(score, out)
        return out

    def _rescue(self, out, agree):
        """Give a name no caption placed a candidate the size prior is confident about.

        Kept strictly separate from the scored assignment above rather than folded in as a
        floor score. A size-only pairing that competes on the same scale can outbid a real
        detection -- it did, moving a distractor's label onto its neighbour's mask -- and no
        prior should ever take a candidate away from the caption that actually fired on it.
        This only fills names the scored pass left empty, from candidates it left unused.
        """
        used = set(out.values())
        for i, name in enumerate(self.names):
            if name in out:
                continue
            free = [j for j in range(agree.shape[1])
                    if j not in used and agree[i, j] >= _SIZE_ACCEPT]
            if not free:
                continue
            j = max(free, key=lambda j: agree[i, j])
            print(f"[perception] '{name}': no caption placed it; taking unclaimed candidate {j} "
                  f"on size alone (agreement {agree[i, j]:.2f})", flush=True)
            out[name] = j
            used.add(j)

    def _report_prior(self, score, chosen):
        """Log when the size prior overrules the caption-only assignment."""
        rows, cols = linear_sum_assignment(-score)
        plain = {self.names[i]: int(j) for i, j in zip(rows, cols) if score[i, j] > 0}
        if plain != chosen:
            print(f"[perception] size prior changed the assignment: caption-only {plain} -> "
                  f"{dict(sorted(chosen.items()))}", flush=True)
