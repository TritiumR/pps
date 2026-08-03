"""ReKep keypoint proposer: DINOv2 patch features -> per-mask KMeans in (PCA feature + xyz)
space -> workspace-bounds filter -> MeanShift merge -> numbered keypoint overlay. Returns the
3D keypoints and the annotated image.

Ported from upstream ReKep; PPS tweaks are a config-selectable DINOv2 backbone and small
epsilon guards for the degenerate tiny masks IsaacLab instance-seg can produce.
"""

import os

import cv2
import numpy as np
import torch
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift
from torch.nn.functional import interpolate

from rekep.utils import filter_points_by_bounds


class KeypointProposer:
    def __init__(self, config, feature_fn=None):
        self.config = config
        self.device = torch.device(self.config["device"])
        # Optional injected feature source: (transformed_rgb, shape_info) -> [H*W, D] on device.
        # None = the deployed DINOv2 path (default, unchanged).
        self.feature_fn = feature_fn
        if feature_fn is not None:
            self.dinov2 = None
        else:
            # Load DINOv2 from the local torch.hub cache when present, so grounding never hits github (its
            # ref check runs on every seed and a flaky response there crashed mid-run). Download once if absent.
            dino_model = self.config.get("dino_model", "dinov2_vits14")
            hub_local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
            if os.path.isdir(hub_local):
                self.dinov2 = torch.hub.load(hub_local, dino_model, source="local").eval().to(self.device)
            else:
                self.dinov2 = torch.hub.load("facebookresearch/dinov2", dino_model).eval().to(self.device)
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])
        # n_jobs=1: the parallel reduction makes cluster count and ordering non-deterministic, so
        # the same scene yielded different keypoint INDICES every run and no rollout replayed. At a
        # few hundred points the workers bought nothing.
        self.mean_shift = MeanShift(bandwidth=self.config["min_dist_bt_keypoints"], bin_seeding=True, n_jobs=1)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed(self.config["seed"])

    def get_keypoints(self, rgb, points, masks):
        # Re-seed here, not only in __init__: _cluster_features calls the randomized
        # torch.pca_lowrank, and by now the global RNG has been advanced by the env reset, the
        # sampler and DINOv2's forward. Seeding at the call site pins the proposal to
        # (image, config).
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed_all(self.config["seed"])
        torch.backends.cudnn.deterministic = True   # DINOv2's forward otherwise perturbs the PCA input
        torch.backends.cudnn.benchmark = False      # autotuning picks different kernels run-to-run
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        features_flat = (self.feature_fn(transformed_rgb, shape_info) if self.feature_fn is not None
                         else self._get_features(transformed_rgb, shape_info))
        # cluster each mask's features into candidate keypoints
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(
            points, features_flat, masks
        )
        if candidate_keypoints.size == 0:
            # Every mask was skipped (too large, or fewer pixels than num_candidates_per_mask). Say so
            # here: the downstream bounds filter would raise an opaque IndexError on the empty array.
            sizes = sorted((int(m.sum()) for m in masks), reverse=True)[:8]
            raise ValueError(
                f"keypoint proposal produced no candidates from {len(masks)} masks "
                f"(largest pixel counts {sizes}, max_mask_ratio={self.config['max_mask_ratio']}, "
                f"num_candidates_per_mask={self.config['num_candidates_per_mask']}): the scene has no "
                "clusterable object masks in view"
            )
        # drop candidates outside the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]
        # merge nearby candidates (MeanShift in xyz)
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]
        # sort by pixel location
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        # annotate the image with numbered keypoints
        projected = self._project_keypoints_to_img(
            rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat
        )
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        # one binary mask per instance id
        masks = [masks == uid for uid in np.unique(masks)]
        # resize so H, W are whole multiples of the dinov2 patch size
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        shape_info = {
            "img_h": H,
            "img_w": W,
            "patch_h": patch_h,
            "patch_w": patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        projected = rgb.copy()
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # white box with a black outline
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(
                projected,
                (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                (pixel[1] + box_width // 2, pixel[0] + box_height // 2),
                (255, 255, 255),
                -1,
            )
            cv2.rectangle(
                projected,
                (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                (pixel[1] + box_width // 2, pixel[0] + box_height // 2),
                (0, 0, 0),
                2,
            )
            # the index, in red
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    def _get_features(   # NO autocast: fp16 is not bitwise-reproducible
self, transformed_rgb, shape_info):
        img_h = shape_info["img_h"]
        img_w = shape_info["img_w"]
        patch_h = shape_info["patch_h"]
        patch_w = shape_info["patch_w"]
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict["x_norm_patchtokens"]  # [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # [num_cams, patch_h, patch_w, feature_dim]
        # bilinearly upsample the patch features to per-pixel features
        interpolated_feature_grid = (
            interpolate(
                raw_feature_grid.permute(0, 3, 1, 2),  # [num_cams, feature_dim, patch_h, patch_w]
                size=(img_h, img_w),
                mode="bilinear",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )  # [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        for rigid_group_id, binary_mask in enumerate(masks):
            # skip masks that are too large
            if np.mean(binary_mask) > self.config["max_mask_ratio"]:
                continue
            # keep only this mask's (foreground) features/points/pixels
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # Drop non-finite depth points: a room-scale scene's masks include invalid-depth pixels
            # (windows, far plane), and one NaN row makes every kmeans centre-shift NaN -- the loop
            # then never meets tol and spins forever (measured: 400k+ iterations on the tea scene).
            finite = np.isfinite(feature_points).all(axis=-1)
            if not finite.all():
                obj_features_flat = obj_features_flat[torch.as_tensor(finite, device=obj_features_flat.device)] \
                    if torch.is_tensor(obj_features_flat) else obj_features_flat[finite]
                feature_pixels = feature_pixels[finite]
                feature_points = feature_points[finite]
            # skip masks too small to form num_candidates clusters
            if obj_features_flat.shape[0] < self.config["num_candidates_per_mask"]:
                continue
            # PCA the features to 3 dims (less sensitive to noise/texture)
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            # +eps guards degenerate (constant) dims that would make kmeans diverge on NaN
            # (IsaacLab instance-seg produces many such tiny masks).
            features_pca = (features_pca - features_pca.min(0)[0]) / (
                features_pca.max(0)[0] - features_pca.min(0)[0] + 1e-6
            )
            X = features_pca
            # append normalized xyz as extra clustering dims
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0] + 1e-6
            )
            X = torch.cat([X, feature_points_torch], dim=-1)
            # kmeans over the (feature + xyz) space into candidate regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config["num_candidates_per_mask"],
                distance="euclidean",
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config["num_candidates_per_mask"]):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices
