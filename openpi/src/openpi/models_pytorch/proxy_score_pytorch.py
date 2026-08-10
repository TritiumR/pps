import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """pi0 / big_vision block mask: token i sees j iff cumsum(att)[j] <= cumsum(att)[i].

    att_masks is per token: 1 starts a new block, 0 continues the previous one. So a run of
    zeros is one MUTUALLY VISIBLE block, and blocks are causal with respect to each other.
    The model already declares this -- embed_prefix emits [0]*num_img_embs (the DINO patches are
    one block) and embed_suffix emits [1] for the state then [1] + [0]*(H-1) for the action chunk
    -- and both call sites were discarding it, which is what left the expert fully causal.
    """
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d.to(torch.bool)


def _to_additive_4d(att_2d: Tensor, dtype: torch.dtype) -> Tensor:
    """[B, Q, K] bool -> [B, 1, Q, K] additive mask (0 keep, -inf drop)."""
    out = torch.zeros(att_2d.shape[0], 1, att_2d.shape[1], att_2d.shape[2],
                      dtype=dtype, device=att_2d.device)
    return out.masked_fill(~att_2d[:, None], torch.finfo(dtype).min)


def _two_block_mask(pad_masks: Tensor, n_suffix: int, dtype: torch.dtype,
                    n_query: int | None = None) -> Tensor:
    """Build the [B, 1, Q, K] additive mask the diffusion head is supposed to use.

    Block 1 -- prefix queries: causal within the prefix, padding respected.
    Block 2 -- suffix (diffusion) queries: may see all valid prefix tokens AND every valid suffix
               token, in both directions. That is what makes p(a, w | o) a joint distribution
               rather than p(a | o) p(w | a, o); without it the goal rows are invisible to the
               action rows and key-pose steering cannot reach the actions through the network.

    Queries are the last `n_query` positions of the key axis (equal to K when there is no cached
    prefix, and n_suffix when the prefix is cached).
    """
    b, k = pad_masks.shape
    device = pad_masks.device
    q = int(k if n_query is None else n_query)
    # With a cached prefix only the suffix tokens are fed, so there are q queries against k keys.
    # Query j sits at absolute position k - q + j.
    q_abs = torch.arange(k - q, k, device=device)             # [Q]
    k_abs = torch.arange(k, device=device)                    # [K]
    is_suffix_k = k_abs >= (k - n_suffix)
    is_suffix_q = q_abs >= (k - n_suffix)
    causal = q_abs[:, None] >= k_abs[None, :]                 # [Q, K]
    allow = causal | (is_suffix_q[:, None] & is_suffix_k[None, :])
    allow = allow[None] & pad_masks.to(torch.bool)[:, None, :]   # [B, Q, K]
    out = torch.zeros(b, 1, q, k, dtype=dtype, device=device)
    return out.masked_fill(~allow[:, None], torch.finfo(dtype).min)

import openpi.models.gemma as _gemma
from openpi.models_pytorch.expert_pytorch import DINOExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
)


def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch_size].")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def squaredcos_cap_v2_alpha_bar(t: torch.Tensor | float) -> torch.Tensor | float:
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2


_ALPHAS_CUMPROD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _alphas_cumprod_key(num_train_timesteps, device, dtype):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return int(num_train_timesteps), device, dtype


def ddim_alphas_cumprod(num_train_timesteps: int, *, device, dtype) -> torch.Tensor:
    """Cosine alpha_bar schedule, memoized per (steps, device, dtype).

    The build is a Python loop of ~8 * num_train_timesteps tiny device-side ops; on a shared
    GPU that is ~350 ms, and inference calls it twice per DDIM level. The schedule is a pure
    function of the key and every caller only reads/indexes it, so it is cached (treat the
    returned tensor as read-only). Values are bitwise identical to rebuilding.
    """
    if num_train_timesteps <= 1:
        raise ValueError("num_train_timesteps must be greater than 1.")

    key = _alphas_cumprod_key(num_train_timesteps, device, dtype)
    cached = _ALPHAS_CUMPROD_CACHE.get(key)
    if cached is not None:
        return cached

    alpha_cumprod = torch.ones((), device=device, dtype=dtype)
    alphas = []
    for idx in range(int(num_train_timesteps)):
        t1 = torch.as_tensor(idx / float(num_train_timesteps), device=device, dtype=dtype)
        t2 = torch.as_tensor((idx + 1) / float(num_train_timesteps), device=device, dtype=dtype)
        beta = 1.0 - squaredcos_cap_v2_alpha_bar(t2) / squaredcos_cap_v2_alpha_bar(t1)
        beta = torch.clamp(beta, max=0.999)
        alpha_cumprod = alpha_cumprod * (1.0 - beta)
        alphas.append(alpha_cumprod)
    out = torch.stack(alphas)
    _ALPHAS_CUMPROD_CACHE[key] = out
    return out


def ddim_iteration_alphas(
    *,
    iteration: int,
    num_iterations: int,
    num_train_timesteps: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if iteration < 0 or iteration >= num_iterations:
        raise ValueError(f"iteration must be in [0, {num_iterations}), got {iteration}.")
    if num_iterations > num_train_timesteps:
        raise ValueError("num_iterations must be <= num_train_timesteps.")

    alphas = ddim_alphas_cumprod(num_train_timesteps, device=device, dtype=dtype)
    step_ratio = int(num_train_timesteps) // int(num_iterations)
    if step_ratio <= 0:
        raise ValueError("num_iterations must be <= num_train_timesteps.")

    timestep = int((int(num_iterations) - 1 - int(iteration)) * step_ratio)
    prev_timestep = timestep - step_ratio
    alpha_t = alphas[timestep]
    alpha_prev = (
        alphas[prev_timestep]
        if prev_timestep >= 0
        else torch.ones((), device=device, dtype=dtype)
    )
    time_cond = torch.as_tensor(
        timestep / max(float(num_train_timesteps - 1), 1.0),
        device=device,
        dtype=dtype,
    )
    return alpha_t, alpha_prev, time_cond


def make_att_2d_masks(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    """Build pi0-style block attention from padding and block-boundary masks."""
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be rank 2, got {pad_masks.ndim}.")
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be rank 2, got {att_masks.ndim}.")
    if pad_masks.shape != att_masks.shape:
        raise ValueError(
            "pad_masks and att_masks must have the same shape, got "
            f"{tuple(pad_masks.shape)} and {tuple(att_masks.shape)}."
        )

    block_ids = torch.cumsum(att_masks.to(torch.int32), dim=1)
    block_mask = block_ids[:, None, :] <= block_ids[:, :, None]
    valid = pad_masks.to(torch.bool)
    return block_mask & valid[:, None, :] & valid[:, :, None]


def _expand_prefix_kv(cache: DynamicCache, bsize: int) -> DynamicCache:
    """Batch-1 prefix cache -> a fresh view-backed cache for `bsize` rows (never mutated)."""
    legacy = cache.to_legacy_cache()
    if legacy[0][0].shape[0] != 1:
        raise ValueError("prefix_kv must be built at batch 1.")
    return DynamicCache.from_legacy_cache(
        tuple(
            (k.expand(bsize, -1, -1, -1), v.expand(bsize, -1, -1, -1))
            for k, v in legacy
        )
    )


class ProxyScorePytorch(nn.Module):
    """Image/state-conditioned score proxy for DDIM/MBD score-space PPS steering."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # F1. Two-block attention among the diffusion tokens (actions may see waypoints/keypose).
        # DEFAULT OFF, deliberately: every existing checkpoint was TRAINED causally, and serving
        # one bidirectionally feeds it an attention pattern it never saw -- silently degrading
        # every proxy we already have. New training sets MG_PROXY_BIDIR_SUFFIX=1, which is also
        # written into action_norm_stats.json so the server can adopt the checkpoint's own mask
        # instead of a global default.
        self.bidirectional_suffix = os.environ.get("MG_PROXY_BIDIR_SUFFIX", "0") == "1"
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.expert_model = DINOExpertModel(
            dino_model_name=config.dino_model_name,
            action_expert_config=action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
            freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
        )
        self.bidirectional_attention = getattr(config, "bidirectional_attention", False)

        action_dim = config.action_dim
        self.action_in_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, action_dim)
        self.state_proj = nn.Linear(action_dim, action_expert_config.width)
        # CFG-style conditioning. The model consumes images+state only -- the prompt is tokenized
        # for transform parity and never reaches the network -- so "task vs null" cannot be
        # signalled through language. Widening state_proj would break every existing checkpoint,
        # so add a separate embedding instead: index 1 = task-conditioned, 0 = null.
        # ZERO-INITIALISED, so at load time both branches are identical and a checkpoint trained
        # before this existed behaves exactly as before. Training is what separates them.
        self.cond_emb = nn.Embedding(2, action_expert_config.width)
        # Goal context token. Built only when the config asks for one, so a goal_dim=0
        # checkpoint has exactly the parameters it had before.
        self.goal_proj = (
            nn.Linear(config.goal_dim, action_expert_config.width)
            if getattr(config, "goal_dim", 0) > 0
            else None
        )
        # Joint action/goal denoising. A DEDICATED encoder/decoder pair for the goal row, so the
        # row is goal_row_dim wide end to end rather than borrowing the action chunk's feature
        # axis -- which is what removes the "keep dims 3..action_dim inert" problem entirely
        # instead of managing it. Built only when asked, so goal_row=False checkpoints are
        # parameter-identical to before.
        if getattr(config, "goal_row", False):
            self.goal_row_in_proj = nn.Linear(config.goal_row_dim, action_expert_config.width)
            self.goal_row_out_proj = nn.Linear(action_expert_config.width, config.goal_row_dim)
        else:
            self.goal_row_in_proj = None
            self.goal_row_out_proj = None
        nn.init.zeros_(self.cond_emb.weight)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        torch.set_float32_matmul_precision("high")
        if (
            getattr(config, "compile_sample_actions", False)
            and os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower()
            not in ("1", "true", "yes")
        ):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, image_keys=IMAGE_KEYS, train=train
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def _sample_train_alpha(self, bsize: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        alphas = ddim_alphas_cumprod(
            self.config.ddim_num_train_timesteps,
            device=device,
            dtype=dtype,
        )
        idx = torch.randint(0, alphas.shape[0], (bsize,), device=device)
        alpha = alphas[idx]
        time = idx.to(dtype=dtype) / max(float(alphas.shape[0] - 1), 1.0)
        return alpha, time

    def _alpha_from_time(self, time: torch.Tensor, device, dtype) -> torch.Tensor:
        alphas = ddim_alphas_cumprod(
            self.config.ddim_num_train_timesteps,
            device=device,
            dtype=dtype,
        )
        idx = torch.round(
            torch.clamp(time.to(device=device, dtype=dtype), 0.0, 1.0)
            * max(float(alphas.shape[0] - 1), 1.0)
        ).to(dtype=torch.long)
        return alphas[idx]

    def embed_prefix(
        self, images, img_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.expert_model.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        # Stashed because both call sites discard the returned value; _run_diffusion_head needs it
        # to build the block mask, and threading it through three signatures would touch the
        # serving path for no benefit.
        self._last_prefix_att = att_masks
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, cond=None, goal=None,
                     noisy_goal_row=None):
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        if cond is not None:
            # cond: [B] long in {0, 1}. Added to the state token so both branches share every
            # weight -- which is what makes s(c=1) - s(c=0) an exact density ratio rather than a
            # difference between two independently trained models.
            c = torch.as_tensor(cond, device=state_emb.device).reshape(-1).long()
            if c.numel() == 1 and state_emb.shape[0] > 1:
                c = c.expand(state_emb.shape[0])
            state_emb = state_emb + self.cond_emb(c).to(state_emb.dtype)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [1]

        if self.goal_proj is not None:
            if goal is None:
                raise ValueError(
                    "goal_dim > 0 but no goal was passed; a goal-conditioned checkpoint must "
                    "never be run with the goal silently dropped.")
            goal = torch.as_tensor(goal, device=device)
            goal = goal.to(self.goal_proj.weight.dtype).reshape(bsize, -1)
            goal_emb = self.goal_proj(goal).to(state_emb.dtype)
            embs.append(goal_emb[:, None, :])
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            # att_mask 0 puts the goal in the STATE's attention block, so the block structure
            # the action tokens see is unchanged and the goal is pure context: attended to by
            # the action rows, never denoised, never scored.
            att_masks += [0]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        pad_masks.append(
            torch.ones(
                bsize,
                action_time_emb.shape[1],
                dtype=torch.bool,
                device=timestep.device,
            )
        )
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # getattr, not attribute access: subclasses in the tests build a bare nn.Module
        # without running this class's __init__, and a diffusion path that raises there
        # would be a regression in code that has nothing to do with the goal row.
        if getattr(self, "goal_row_in_proj", None) is not None:
            if noisy_goal_row is None:
                raise ValueError(
                    "goal_row is enabled but no noisy goal row was passed; the row is denoised "
                    "jointly with the actions and cannot be silently dropped.")
            gr = noisy_goal_row.to(self.goal_row_in_proj.weight.dtype).reshape(bsize, 1, -1)
            gr_emb = self.goal_row_in_proj(gr)
            # Same time conditioning and the same MLP as the action rows, so the goal row rides
            # the identical noise schedule and adds no parameters beyond its own projections.
            gr_time = time_emb[:, :1, :].expand_as(gr_emb)
            gr_emb = self.action_time_mlp_out(
                F.silu(self.action_time_mlp_in(torch.cat([gr_emb, gr_time], dim=2))))
            embs.append(gr_emb)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            # att_mask 0 keeps the goal row inside the ACTION block, so with a bidirectional
            # suffix the action rows attend to it and it attends to them -- the joint denoising
            # this model exists to test.
            att_masks += [0]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        self._last_suffix_att = att_masks
        return embs, pad_masks, att_masks, None

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def build_attention_mask(
        self,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
        *,
        n_query: int | None = None,
    ) -> torch.Tensor:
        """Use the legacy causal mask or the checkpoint-compatible block mask."""
        if not self.bidirectional_attention:
            return pad_masks
        attention_mask = self._prepare_attention_masks_4d(
            make_att_2d_masks(pad_masks, att_masks)
        )
        if n_query is not None:
            attention_mask = attention_mask[:, :, -int(n_query) :, :]
        return attention_mask

    def build_expert_masks(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        *,
        n_query: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat(
            [prefix_att_masks.to(torch.int32), suffix_att_masks.to(torch.int32)],
            dim=1,
        )
        return (
            self.build_attention_mask(pad_masks, att_masks, n_query=n_query),
            pad_masks,
        )

    @torch.no_grad()
    def prefix_kv_cache(self, prefix_embs, prefix_pad_masks, *, dtype=None) -> DynamicCache:
        """Per-layer K/V of the image prefix, using the configured attention mask.

        Inference reuses one prefix across every DDIM level and every candidate, so caching it
        turns each later call into a 16-token forward. Build at batch 1 and expand. `dtype`
        must match the precision the consumer runs at, or the concat with the suffix K/V
        promotes the whole attention back to the wider type.
        """
        position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).to(dtype=torch.long)
        attention_mask = self.build_attention_mask(
            prefix_pad_masks,
            torch.zeros_like(prefix_pad_masks),
        )
        cache = DynamicCache()
        self.expert_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=prefix_embs,
            use_cache=True,
            adarms_cond=None,
        )
        if dtype is None:
            return cache
        return DynamicCache.from_legacy_cache(
            tuple((k.to(dtype), v.to(dtype)) for k, v in cache.to_legacy_cache())
        )

    def _run_diffusion_head(
        self,
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
        prefix_kv: DynamicCache | None = None,
        prefix_att_masks=None,
        suffix_att_masks=None,
    ) -> torch.Tensor:
        if prefix_kv is None:
            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            position_ids = position_ids.to(dtype=torch.long)
            past_key_values = None
        else:
            # Cached prefix: only the suffix tokens are fed; the mask still spans past+current
            # and the positions are the suffix slice of the full-sequence positions.
            embs = suffix_embs
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            position_ids = position_ids[:, -suffix_embs.shape[1] :].to(dtype=torch.long)
            past_key_values = _expand_prefix_kv(prefix_kv, suffix_embs.shape[0])
        # F1, CRITICAL. Passing a 2-D padding mask lets Gemma build its default CAUSAL mask, so a
        # diffusion token at row i cannot attend to rows > i. The chunk is laid out
        # [actions | AWE waypoints | keypose], so under causal attention the ACTION rows can never
        # see the goal rows -- measured bitwise: perturbing rows 15-20 moved the goal outputs by up
        # to 52.6 and the action outputs by exactly 0.0 at every level. That makes the whole
        # premise of joint action/goal denoising inoperative: steering the key pose cannot reach
        # the executed actions through the network.
        # The intended mask is two-block: prefix stays causal-with-padding, and every diffusion
        # token sees every other diffusion token. self.bidirectional_suffix=False restores the old
        # behaviour exactly, for checkpoints trained under it.
        attn = pad_masks
        if prefix_att_masks is None and self.bidirectional_attention:
            prefix_att_masks = torch.zeros_like(prefix_pad_masks)
        if prefix_att_masks is not None and suffix_att_masks is not None:
            attn, pad_masks = self.build_expert_masks(
                prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks,
                n_query=embs.shape[1] if prefix_kv is not None else None)
        elif getattr(self, "bidirectional_suffix", False):
            pa = getattr(self, "_last_prefix_att", None)
            sa = getattr(self, "_last_suffix_att", None)
            if (pa is not None and sa is not None
                    and pa.shape[1] + sa.shape[1] == pad_masks.shape[1]):
                att = torch.cat([pa.expand(pad_masks.shape[0], -1).to(torch.int32),
                                 sa.expand(pad_masks.shape[0], -1).to(torch.int32)], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att)
                off = pad_masks.shape[1] - embs.shape[1]
                if off:
                    att_2d = att_2d[:, off:, :]
                attn = _to_additive_4d(att_2d, pad_masks.dtype if pad_masks.is_floating_point()
                                       else torch.float32)
            else:
                attn = _two_block_mask(pad_masks, suffix_embs.shape[1], embs.dtype,
                                       n_query=embs.shape[1])
        hidden_states, _ = self.expert_model.forward(
            attention_mask=attn,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        if getattr(self, "goal_row_out_proj", None) is not None:
            # Suffix tail is [action rows | goal row]; decode each with its own head.
            rows = hidden_states[:, -(self.config.action_horizon + 1) :].to(dtype=torch.float32)
            return (self.action_out_proj(rows[:, : self.config.action_horizon]),
                    self.goal_row_out_proj(rows[:, self.config.action_horizon :]))
        suffix_out = hidden_states[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out), None

    @staticmethod
    def _goal_kwargs(goal):
        """Forward `goal` only when there is one.

        predict_score_from_prefix already did this; the two direct call sites did not, so a
        subclass that overrides _predict_model_output_from_prefix without a `goal` parameter --
        which the unit tests do -- got a TypeError on every unconditional path.
        """
        return {} if goal is None else {"goal": goal}

    def _predict_model_output_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
        prefix_kv: DynamicCache | None = None,
        cond=None,
        goal=None,
        noisy_goal_row=None,
        return_goal_row=False,
        prefix_att_masks=None,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            time_cond,
            cond=cond,
            goal=goal,
            noisy_goal_row=noisy_goal_row,
        )
        action_out, goal_out = self._run_diffusion_head(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            adarms_cond,
            prefix_kv=prefix_kv,
            prefix_att_masks=prefix_att_masks,
            suffix_att_masks=suffix_att_masks,
        )
        return (action_out, goal_out) if return_goal_row else action_out

    def predict_score_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
        prefix_kv: DynamicCache | None = None,
        cond=None,
        goal=None,
        prefix_att_masks=None,
    ) -> torch.Tensor:
        # Forwarded only when set, so subclasses overriding the hook keep their signature.
        extra = {"prefix_kv": prefix_kv} if prefix_kv is not None else {}
        if cond is not None:
            extra["cond"] = cond
        if goal is not None:
            extra["goal"] = goal
        if prefix_att_masks is not None:
            extra["prefix_att_masks"] = prefix_att_masks
        output = self._predict_model_output_from_prefix(
            state,
            prefix_embs,
            prefix_pad_masks,
            x_t,
            time_cond,
            **extra,
        )
        if self.config.prediction_type == "score":
            return output
        if self.config.prediction_type == "regress":
            raise ValueError(
                "prediction_type 'regress' has no score: the head is not noise-conditioned."
            )

        alpha = self._alpha_from_time(time_cond, x_t.device, x_t.dtype)
        if self.config.prediction_type == "x0":
            # score s.t. Tweedie x0 = (x_t + beta * score) / sqrt(alpha) recovers the output.
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            return (sqrt_alpha[:, None, None] * output - x_t) / beta[:, None, None]
        sqrt_beta = torch.sqrt(torch.clamp(1.0 - alpha, min=1e-6))
        return -output / sqrt_beta[:, None, None]

    def forward(
        self,
        observation,
        actions=None,
        noise=None,
        time=None,
        score_target=None,
        goal=None,
        goal_target=None,
        *,
        mode="train",
        return_parts=False,
        **_,
    ) -> Tensor:
        if mode != "train":
            raise ValueError(f"Unsupported forward mode for ProxyScorePytorch: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for score training.")

        actions = actions[..., : self.config.action_dim]
        images, img_masks, state = self._preprocess_observation(observation, train=True)
        loss_weight = None
        direct_score_target = score_target is not None

        if score_target is not None:
            if time is None:
                raise ValueError("time must be provided when training from direct score targets.")
            x_t = actions
            target = score_target[..., : self.config.action_dim].to(
                device=actions.device,
                dtype=actions.dtype,
            )
            time = time.to(device=actions.device, dtype=actions.dtype)
        elif self.config.prediction_type == "regress":
            # Plain chunked regression: L2 on the clean chunk, no noising. The action/time
            # tokens are zeroed, so the chunk is predicted from the image/state prefix alone.
            x_t = torch.zeros_like(actions)
            time = torch.zeros(
                actions.shape[0], device=actions.device, dtype=actions.dtype
            )
            target = actions
        else:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            else:
                noise = noise[:, :, : self.config.action_dim]
            if time is None:
                alpha, time = self._sample_train_alpha(
                    actions.shape[0],
                    actions.device,
                    actions.dtype,
                )
            else:
                alpha = self._alpha_from_time(time, actions.device, actions.dtype)

            alpha = alpha.to(device=actions.device, dtype=actions.dtype)
            time = time.to(device=actions.device, dtype=actions.dtype)
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            sqrt_beta = torch.sqrt(beta)

            x_t = sqrt_alpha[:, None, None] * actions + sqrt_beta[:, None, None] * noise
            if self.config.prediction_type == "epsilon":
                target = noise
            elif self.config.prediction_type == "x0":
                target = actions
            else:
                target = -noise / sqrt_beta[:, None, None]
                loss_weight = beta[:, None, None]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )
        grouped_labels = x_t.ndim == 4
        if grouped_labels:
            if target.ndim != 4 or time.ndim != 2:
                raise ValueError(
                    "Grouped score targets require x_t/score shape [B, L, H, D] "
                    "and time shape [B, L]."
                )
            batch_size, labels_per_observation = x_t.shape[:2]
            if prefix_embs.shape[0] != batch_size:
                raise ValueError("Observation batch does not match grouped score targets.")
            # Repeat the once-computed visual prefix: preserves gradients from every score label
            # without rerunning DINO per reverse-diffusion state.
            prefix_embs = prefix_embs.repeat_interleave(labels_per_observation, dim=0)
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(
                labels_per_observation, dim=0
            )
            prefix_att_masks = prefix_att_masks.repeat_interleave(
                labels_per_observation, dim=0
            )
            state = state.repeat_interleave(labels_per_observation, dim=0)
            if goal is not None:
                goal = goal.repeat_interleave(labels_per_observation, dim=0)
            x_t = x_t.flatten(0, 1)
            target = target.flatten(0, 1)
            time = time.flatten(0, 1)
        if getattr(self, "goal_row_in_proj", None) is not None:
            # Joint denoising: the goal row is noised with the SAME alpha/time as the action
            # chunk, so the two are recovered on one schedule and the model has to use whatever
            # it can infer about the goal while denoising the actions.
            if goal_target is None:
                raise ValueError("goal_row is enabled but no goal_target was passed.")
            if grouped_labels:
                raise ValueError("goal_row does not support grouped score labels.")
            if direct_score_target or self.config.prediction_type != "x0":
                raise ValueError(
                    "goal_row training requires the x0 diffusion path; direct score targets "
                    "and 'regress' never build the alpha schedule the goal row rides on.")
            g_clean = (torch.as_tensor(goal_target, device=actions.device, dtype=actions.dtype)
                       .reshape(actions.shape[0], 1, self.config.goal_row_dim)
                       * self.config.goal_row_gain)
            g_noise = self.sample_noise(g_clean.shape, g_clean.device).to(g_clean.dtype)
            g_t = sqrt_alpha[:, None, None] * g_clean + sqrt_beta[:, None, None] * g_noise
            pred, pred_goal = self._predict_model_output_from_prefix(
                state, prefix_embs, prefix_pad_masks, x_t, time, goal=goal,
                noisy_goal_row=g_t, return_goal_row=True)
            loss = F.mse_loss(pred, target, reduction="none")
            if loss_weight is not None:
                loss = loss * loss_weight
            goal_loss = F.mse_loss(pred_goal, g_clean, reduction="none")
            if not return_parts:
                raise ValueError(
                    "goal_row training must be called with return_parts=True; the two losses "
                    "carry different element counts and must not be silently pooled.")
            return {"action": loss, "goal": goal_loss}

        predict = (
            self.predict_score_from_prefix
            if direct_score_target or self.config.prediction_type == "score"
            else self._predict_model_output_from_prefix
        )
        pred = predict(state, prefix_embs, prefix_pad_masks, x_t, time,
                       prefix_att_masks=prefix_att_masks,
                       **self._goal_kwargs(goal))
        loss = F.mse_loss(pred, target, reduction="none")
        if loss_weight is not None:
            loss = loss * loss_weight
        if grouped_labels:
            loss = loss.unflatten(0, (batch_size, labels_per_observation))
        return {"action": loss, "goal": None} if return_parts else loss

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        start_time=1.0,
        goal=None,
        return_trace=False,
        goal_row_noise=None,
        goal_row_clamp=None,
        goal_row_clamp_from=0,
        return_goal_row=False,
    ) -> Tensor:
        """Reverse diffusion; with goal_row, the goal is denoised jointly with the actions.

        Args:
            goal_row_clamp: [B, goal_row_dim] in the shared interface units. When given, the
                goal row is OVERWRITTEN at every level with that value forward-noised to the
                level's alpha, instead of being denoised freely. That makes the row a commanded
                input rather than a prediction, which is exactly the implicit-transmission probe:
                whatever the actions do differently is caused by the clamped row.
            goal_row_clamp_from: first denoising ITERATION at which the clamp applies (0 = all).
                The clamp is forward-noised, so at early levels it is mostly noise and carries
                little of the commanded goal -- unlike a context token, which is clean at every
                level. Clamping only the late, high-alpha levels isolates that asymmetry instead
                of leaving it as an unmeasured confound in the A-vs-B comparison.
            return_goal_row: also return the model's own final goal estimate, in interface units.
        """
        del start_time
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        joint_goal = getattr(self, "goal_row_in_proj", None) is not None
        if joint_goal:
            g_shape = (bsize, 1, self.config.goal_row_dim)
            if goal_row_noise is None:
                goal_row_noise = self.sample_noise(g_shape, device)
            goal_row_noise = goal_row_noise.reshape(g_shape).to(noise.dtype)
            g_t = goal_row_noise
            g_clamp = None
            if goal_row_clamp is not None:
                g_clamp = (torch.as_tensor(goal_row_clamp, device=device, dtype=noise.dtype)
                           .reshape(g_shape) * self.config.goal_row_gain)
        elif goal_row_clamp is not None or return_goal_row:
            raise ValueError("goal_row_clamp/return_goal_row need a goal_row model.")

        images, img_masks, state = self._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )
        if self.config.prediction_type == "regress":
            # One forward: the head emits the chunk directly (no reverse chain).
            return self._predict_model_output_from_prefix(
                state,
                prefix_embs,
                prefix_pad_masks,
                torch.zeros_like(noise),
                torch.zeros(bsize, device=device, dtype=noise.dtype),
                **self._goal_kwargs(goal),
                prefix_att_masks=prefix_att_masks,
            )
        x_t = noise
        trace = []
        for iteration in range(int(num_steps)):
            alpha, alpha_prev, time_cond = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=int(num_steps),
                num_train_timesteps=self.config.ddim_num_train_timesteps,
                device=device,
                dtype=x_t.dtype,
            )
            expanded_time = time_cond.expand(bsize)
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            sqrt_beta = torch.sqrt(beta)
            if joint_goal:
                if g_clamp is not None and iteration >= int(goal_row_clamp_from):
                    # Replacement conditioning: the row carries the COMMANDED goal at this
                    # level's noise scale, re-imposed every level so it cannot drift back to
                    # whatever the model would have inferred.
                    g_t = sqrt_alpha * g_clamp + sqrt_beta * goal_row_noise
                output, g_out = self._predict_model_output_from_prefix(
                    state, prefix_embs, prefix_pad_masks, x_t, expanded_time, goal=goal,
                    noisy_goal_row=g_t, return_goal_row=True,
                    prefix_att_masks=prefix_att_masks)
            else:
                output = self._predict_model_output_from_prefix(
                    state,
                    prefix_embs,
                    prefix_pad_masks,
                    x_t,
                    expanded_time,
                    **self._goal_kwargs(goal),
                    prefix_att_masks=prefix_att_masks,
                )
            if self.config.prediction_type == "epsilon":
                eps_hat = output
                x0_hat = (x_t - sqrt_beta * eps_hat) / sqrt_alpha
            elif self.config.prediction_type == "x0":
                x0_hat = output
                eps_hat = (x_t - sqrt_alpha * x0_hat) / sqrt_beta
            else:
                score = output
                x0_hat = (x_t + beta * score) / sqrt_alpha
                eps_hat = -sqrt_beta * score
            if return_trace:
                # The clean-chunk estimate at this level: what the denoising analysis plots.
                trace.append(x0_hat.detach().clone())
            x_t = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps_hat
            )
            if joint_goal:
                g0_hat = g_out
                g_eps_hat = (g_t - sqrt_alpha * g0_hat) / sqrt_beta
                g_t = (torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * g0_hat
                       + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * g_eps_hat)

        if torch.is_grad_enabled():
            logging.warning("ProxyScorePytorch.sample_actions is expected to run under no_grad.")
        out = (x_t,)
        if return_trace:
            out = out + (torch.stack(trace, dim=1),)
        if return_goal_row:
            # Back to the shared interface units the caller speaks.
            out = out + (g_t.reshape(bsize, self.config.goal_row_dim)
                         / self.config.goal_row_gain,)
        return out[0] if len(out) == 1 else out
