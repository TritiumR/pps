"""GroundingDINO and SAM integration helpers for vlm_dp perception."""

import os

import transformers.modeling_utils as _tmu


# Restore compatibility with transformers versions that removed get_head_mask.
if not hasattr(_tmu.ModuleUtilsMixin, "get_head_mask"):

    def _get_head_mask(
        self,
        head_mask,
        num_hidden_layers,
        is_attention_chunked=False,
    ):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(
                head_mask,
                num_hidden_layers,
            )
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    _tmu.ModuleUtilsMixin.get_head_mask = _get_head_mask


def ckpt_dir() -> str:
    """Return the directory containing model configs and checkpoints."""
    env = os.environ.get("VLM_DP_CKPTS")
    if env:
        return env

    repo = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(repo, "moka")


def load_pil_image(image_pil):
    """Return the RGB image and GroundingDINO input tensor."""
    import groundingdino.datasets.transforms as T
    import numpy as np

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    image = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None)
    return image, image_transformed