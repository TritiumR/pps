"""Self-contained GroundingDINO and SAM backend glue for vlm_dp perception.

Vendors the pieces vlm_dp used from moka.vision.segmentation so moka is never imported:
  the transformers get_head_mask compatibility shim, load_pil_image (the GroundingDINO image
  transform), and ckpt_dir (where the checkpoints and config live).

Checkpoints are not vendored (multi-GB, never committed). ckpt_dir returns $VLM_DP_CKPTS if set,
else the in-repo moka dir, and must contain config/grounding_dino.py and
ckpts/{groundingdino_swint_ogc.pth, sam_vit_h_4b8939.pth}.
"""
import os

# GroundingDINO's BertModelWarper expects BertModel.get_head_mask, which transformers >= 5 removed.
# GroundingDINO only calls it with head_mask=None, which maps to [None] * num_layers, so restore that
# standard behavior at import (before any DINO model is built) instead of pinning transformers lower.
import transformers.modeling_utils as _tmu

if not hasattr(_tmu.ModuleUtilsMixin, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    _tmu.ModuleUtilsMixin.get_head_mask = _get_head_mask


def ckpt_dir() -> str:
    """Directory holding config/grounding_dino.py and ckpts/*.pth.

    $VLM_DP_CKPTS overrides. Default is the in-repo moka dir where the weights live (asset path only,
    the moka package is not imported).
    """
    env = os.environ.get("VLM_DP_CKPTS")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "moka")


def load_pil_image(image_pil):
    """GroundingDINO preprocessing. Returns (np_rgb, transformed_tensor)."""
    # Deferred: heavy, and must run after the import-time shim above.
    import groundingdino.datasets.transforms as T
    import numpy as np
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None)
    return image, image_transformed
