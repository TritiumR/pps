"""Text-prompted object perception: one camera frame -> per-object masks and world positions.

Replaces the simulator's instance segmentation, which handed the controller pixel-exact masks already
keyed by object. Here the objects have to be found and named from the image, using GroundingDINO to
localise each name and SAM to cut its mask, then the scene's own depth to lift it into the world.

Two things this does differently from the usual GroundedSAM call, each because the naive version was
measured failing on this scene:

**Labels compete for objects, instead of each taking its own best box.** Prompting one name at a time and
taking that name's top-scoring box lets a distractor win a label outright: on the weight scene, "apple"
landed on the *mango* and the real apple went undetected. Prompting every object in the scene and solving
a one-to-one assignment makes the mango's own label claim it, which frees "apple" for the apple.

**Masks are eroded before they are back-projected.** The grasp centre is the mid-point of the mask's
horizontal extent, which is a min/max over the masked points -- so a single pixel of silhouette bleed,
where the mask edge overhangs the object and the depth behind it belongs to the table, lands a point
centimetres away and drags the centre with it. A ground-truth instance mask is pixel-exact and has no such
rim; a predicted one has one no matter how well it scores on IoU.
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

_BOX_THRESH = 0.15    # keep every plausible box and let the assignment choose, rather than pre-filtering
_TEXT_THRESH = 0.22
_IOU_MERGE = 0.80     # boxes overlapping this much are the same physical object
_MIN_PIXELS = 80      # below this a mask cannot be clustered into keypoints (matches the proposer's floor)


def _backends():
    """Import the vision backends through moka.vision.segmentation, and only through it.

    That module imports segment_anything before groundingdino and patches a transformers incompatibility
    on the way in. Importing groundingdino directly skips both and dies inside timm, which drags in
    torch._dynamo and hits a typing_extensions clash. The import order is load-bearing, so it lives in one
    place and everything goes through it.
    """
    import moka.vision.segmentation as seg   # noqa: F401  -- imported for its order + compat shim
    return seg


@functools.lru_cache(maxsize=1)
def _dino():
    """GroundingDINO, built once. The stock helper rebuilds a 700MB checkpoint on every call."""
    seg = _backends()
    from groundingdino.util.inference import load_model
    return load_model(os.path.join(seg._MOKA_DIR, "config", "grounding_dino.py"),
                      os.path.join(seg._MOKA_DIR, "ckpts", "groundingdino_swint_ogc.pth"))


@functools.lru_cache(maxsize=1)
def _sam(device="cuda"):
    """SAM, built once. The stock helper reloads a 2.5GB checkpoint on every call."""
    seg = _backends()
    from segment_anything import build_sam, SamPredictor
    ckpt = os.path.join(seg._MOKA_DIR, "ckpts", "sam_vit_h_4b8939.pth")
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


class Perception:
    """Finds the named objects in the camera frame and reports where they are.

    ``prompts`` maps each scene object to the text a detector is asked to find ("scale" -> "kitchen
    scale"). The names are semantic content of the image, the sort of thing a VLM reads off it; they carry
    no pose, size, or identity information from the simulator.
    """

    def __init__(self, prompts: dict, fixtures=(), erode: int = 2, device: str = "cuda",
                 segment: str = "groundedsam"):
        self.fixtures = tuple(fixtures)   # objects whose geometry is workcell calibration, not perception
        # A fixture is not looked for at all. Leaving its name in the vocabulary would let it take part in
        # the assignment and win a box that belongs to a real object -- the "kitchen scale" label happily
        # claims the pear once the scale itself is out of frame.
        self.prompts = {n: t for n, t in prompts.items() if n not in self.fixtures}
        self.names = list(self.prompts)
        self.erode = erode
        self.device = device
        # Which segmenter names the objects. groundedsam (default): GroundingDINO on a per-object text vocab
        # (the ``prompts`` values, e.g. "yellow pear") + SAM. sam_vlm: class-agnostic SAM regions named by a
        # VLM from the plain object names alone -- the ReKep-faithful "semantics live in the VLM", with no
        # hand-written colour table. Only the keys of ``prompts`` (the names) are used under sam_vlm.
        self.segment = segment
        self.masks: dict = {}      # name -> (H,W) bool, from the last look
        self.rgb = None
        self.points = None
        self._bounds = None                 # sam_vlm workspace box, measured once (calibration)
        self._id_centroids: dict = {}       # sam_vlm identity: name -> last world centroid (VLM once, then track)
        self._fixture_masks: dict = {}
        self._fixture_points: dict = {}

    def warmup(self):
        """Load the detector and the segmenter now, before anything else initialises torch.

        Import order matters here and it is not optional: GroundingDINO pulls in timm, which pulls in
        torch._dynamo, and doing that after the policy stack has already set torch up trips a
        typing_extensions clash inside timm that kills the process. Loading them first is also what the
        probes do, which is why they run and a lazy import does not.
        """
        _dino()
        _sam(self.device)

    def calibrate(self, env):
        """Measure the fixtures once. They are furniture: bolted down, and they never move.

        The scale is the reason this exists. Its rigid body is only the weighing platform, but a segmenter
        prompted for it returns the whole appliance -- housing, display and base -- and on this camera it
        also runs off the edge of the frame. So the top of its mask is the top of the display, and an
        object placed there would be placed on the display. Estimating it per-frame would inject a large
        error into something that does not move, which is what calibration is for.

        Perceiving the objects you manipulate and calibrating the furniture you do not is what a real cell
        does. It is recorded here rather than assumed, so it can be read off the run.
        """
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
        """Look through the camera: return ``{name: world position}`` for every object found."""
        rgb, points, _, _ = camera_to_rekep_inputs(env.cam, 0)   # rgb + depth only; the seg channel is unused
        return self.observe_frame(rgb, points)

    def observe_frame(self, rgb, points) -> dict:
        """Segment one already-captured frame: ``{name: world position}`` for every object found.

        Objects that were not confidently detected are simply absent. The caller keeps its previous belief
        for those rather than trusting a low-confidence mask, which on an occluded object is worse than no
        mask at all: the visible sliver's centroid can be centimetres from the object.
        """
        self.rgb, self.points = rgb, points
        self.masks = dict(self._segment(rgb), **self._fixture_masks)   # fixtures are known, not detected
        out = {}
        for name in self.masks:
            pos = self.position(name)
            if pos is not None:
                out[name] = pos
        return out

    def position(self, name) -> np.ndarray | None:
        """Grasp centre of ``name``: the middle of its horizontal extent, at the height of its silhouette.

        A single view sees an object's top surface down to its silhouette edge, which is its widest visible
        cross-section and about where a gripper should close. The mean of the visible points sits high on
        the top shell instead, so a low percentile of z is used for the height.
        """
        pts = self.object_points(name)
        if pts is None:
            return None
        xy = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2.0
        return np.array([xy[0], xy[1], float(np.percentile(pts[:, 2], 15))], dtype=np.float64)

    def object_extents(self, name) -> tuple | None:
        """``(grip, keepout, half_height)`` estimated from the object's own point cloud, no object model.

        The controller needs a grasp radius (narrow horizontal half-extent), a collision radius (the wide
        one), and a half-height. All three come from the segmented silhouette so the base runs on image and
        instruction alone, the way a general policy would. A top-down-ish view sees the full horizontal
        silhouette, so the two horizontal half-extents are reliable; it sees only the top of the object, so
        the visible z-span reads about the object's half-height for a rounded object and is used as such.
        """
        pts = self.object_points(name)
        if pts is None:
            return None
        x_half = float(pts[:, 0].max() - pts[:, 0].min()) / 2.0
        y_half = float(pts[:, 1].max() - pts[:, 1].min()) / 2.0
        z_span = float(pts[:, 2].max() - pts[:, 2].min())
        return (min(x_half, y_half), max(x_half, y_half), z_span)

    def object_points(self, name) -> np.ndarray | None:
        """The object's world points: its eroded mask back-projected through the depth image.

        A fixture's points come from calibration instead, and never change (see ``calibrate``).
        """
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
        """``(labels (H,W) int32, id_to_prim)`` in the shape the keypoint proposer already consumes.

        The proposer only needs a set of binary object masks -- it re-derives them as ``labels == uid`` and
        never looks at which object a uid is. The prim-path map is kept because the rest of the front-end
        joins names to pixels through it; here it is synthesised from the detections, so it carries the
        names we asked for rather than the simulator's scene graph.
        """
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
        """SAM regions named by a VLM (no colour vocabulary): the ReKep-faithful segmentation.

        Class-agnostic SAM masks in place of GroundingDINO's per-name boxes, and the VLM reads the numbered
        regions and says which is which -- the semantics live in the VLM, as in ReKep, not in a hand-written
        colour table. The VLM is asked once (the first look); later frames re-identify by nearest position,
        since objects do not teleport between looks, so re-perception during a rollout adds no VLM calls.
        """
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
            if any(_mask_iou(m, o) > 0.7 for o in out):   # SAM auto returns nested masks; keep the larger
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
        """Ask GPT-4o which numbered region is which named object; cache each name's centroid."""
        names = list(self.prompts)
        prompt = (
            f"The image shows a scene; candidate regions are outlined in red and numbered 0 to {len(masks) - 1}. "
            "Some regions are the table, the background, or a part of an object -- ignore those. "
            f"For each object in this list, give the number of the region that IS that object: {names}. "
            "If an object is not visible, use -1. "
            'Reply with only a JSON object mapping each name to its integer region number, '
            'e.g. {"pear": 3, "apple": 7}.')
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
        from moka.vision.segmentation import load_pil_image

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
            if int(m.sum()) >= _MIN_PIXELS:
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
        """One box per label, one label per box (Hungarian over the detection scores).

        This is the fix for the distractor stealing a label. Independently arg-maxing lets "apple" and
        "mango" both point at the mango; forcing a one-to-one assignment makes them compete, and the mango
        wins its own label by a wide margin, so "apple" falls to the apple.
        """
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
