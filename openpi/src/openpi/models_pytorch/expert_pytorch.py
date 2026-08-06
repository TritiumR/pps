from typing import Literal
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor, GemmaForCausalLM
from transformers.cache_utils import Cache
from transformers.models.auto import CONFIG_MAPPING


class DINOExpertModel(nn.Module):
    def __init__(
        self,
        dino_model_name,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "float32",
        freeze_dino_encoder: bool = False,
        language_vocab_size: int | None = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # Resolve dinov3 directory path relative to this file's location
        # This file is at: openpi/src/openpi/models_pytorch/expert_pytorch.py
        # dinov3 is at: openpi/dinov3/
        current_file = Path(__file__)
        openpi_dir = current_file.parent.parent.parent.parent
        dinov3_dir = openpi_dir / "dinov3"

        model_name = str(
            dino_model_name.split("/")[-1].split("-")[0]
            + "_"
            + dino_model_name.split("/")[-1].split("-")[1]
        )

        checkpoint_name = model_name + ".pth"

        # Convert to absolute path string for torch.hub.load
        dinov3_path = str(dinov3_dir.resolve())
        weights_path = str((dinov3_dir / "checkpoints" / checkpoint_name).resolve())

        # Initialize DINOv3 model for vision encoding
        self.dino_model = torch.hub.load(
            dinov3_path,
            model_name,
            source="local",
            weights=weights_path,
        )

        # Freeze DINO encoder parameters if requested
        if freeze_dino_encoder:
            for param in self.dino_model.parameters():
                param.requires_grad = False

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            # vocab_size=257152,
            vocab_size=1,  # Set to 1 since we don't use language decoding (lm_head is unused)
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # Continuous image/state/action embeddings bypass Gemma's input table.
        self.gemma_expert.model.embed_tokens = None  # type: ignore
        self.language_embedding = (
            nn.Embedding(language_vocab_size, action_expert_config.width)
            if language_vocab_size is not None
            else None
        )
        if self.language_embedding is not None:
            nn.init.normal_(self.language_embedding.weight, mean=0.0, std=0.02)
        # The proxy consumes decoder hidden states directly and never calls the
        # language-modeling head. Keep it in the state dict for checkpoint
        # compatibility, but exclude it from gradient reduction under DDP.
        self.gemma_expert.lm_head.requires_grad_(False)

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """Embed images using DINOv3 model.

        Args:
            image: Image tensor of shape (batch_size, channels, height, width)
                  Expected to be in [0, 1] range or already normalized

        Returns:
            Image features from DINOv3 model of shape (batch_size, num_patches, hidden_size)
        """
        from torchvision.transforms import v2

        # print("image.shape", image.shape)
        # print("image.dtype", image.dtype)
        # print("image.max()", image.max())
        # print("image.min()", image.min())
        # print("image.mean()", image.mean())

        # Validate input shape: must be (batch_size, channels, height, width)
        if image.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (batch, channels, height, width), "
                f"got {image.dim()}D tensor with shape {image.shape}"
            )

        batch_size, channels, height, width = image.shape

        # Convert from [-1, 1] to [0, 1] range
        # Check if image is in [-1, 1] range (min < 0)
        if image.min() < 0:
            image = (image + 1.0) / 2.0

        # Resize to 224x224 if needed (DINOv3 standard size)
        if height != 224 or width != 224:
            resize = v2.Resize((224, 224), antialias=True)
            image = resize(image)

        # Apply ImageNet normalization
        # Normalize expects input in [0, 1] range
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        pixel_values = normalize(image)

        # Get features from DINOv3
        outputs = self.dino_model(pixel_values, patch_embedding=True)
        # print(outputs.shape)
        # Return last_hidden_state which contains patch features
        return outputs

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.language_embedding is None:
            raise RuntimeError("Language embeddings are disabled for this DINO expert.")
        return self.language_embedding(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        adarms_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Cache | None]:
        """Forward pass through the Gemma action expert.

        This wrapper keeps the interface small and focused: it takes a single
        sequence of embeddings (typically image + state/action tokens
        concatenated) and returns the last hidden states from Gemma together
        with optional past key values.
        """
        if inputs_embeds is None:
            raise ValueError("inputs_embeds cannot be None")

        output = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )

        return output.last_hidden_state, output.past_key_values
