import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames
import copy
import torch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.proxy_pytorch import (
    build_proxy_expert_masks,
    build_proxy_prefix_mask,
)

logger = logging.getLogger(__name__)


def _resolve_steer_target(args, action_horizon: int) -> tuple[str, int | None]:
    target = getattr(args, "steer_chunk_target", "all")
    if not isinstance(target, str):
        target = target.value if hasattr(target, "value") else str(target)
    target = str(target).lower()

    if target == "first":
        return target, 0
    if target == "last":
        return target, action_horizon - 1
    return "all", None


def _apply_proxy_steering(
    base_v_t: torch.Tensor,
    steer_v_t: torch.Tensor,
    mimic_v_t: torch.Tensor,
    proxy_action_dim: int,
    steer_scale: float,
    steer_target_idx: int | None,
    only_steer: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_t = base_v_t.clone()
    applied_delta = torch.zeros_like(base_v_t)
    steer_mask = torch.zeros(
        (base_v_t.shape[0], base_v_t.shape[1], 1),
        device=base_v_t.device,
        dtype=base_v_t.dtype,
    )

    if only_steer:
        if steer_target_idx is None:
            v_t[:, :, :proxy_action_dim] = steer_v_t
            applied_delta[:, :, :proxy_action_dim] = (
                steer_v_t - base_v_t[:, :, :proxy_action_dim]
            )
            steer_mask[:, :, 0] = 1.0
        else:
            v_t[:, steer_target_idx, :proxy_action_dim] = steer_v_t[
                :, steer_target_idx, :
            ]
            applied_delta[:, steer_target_idx, :proxy_action_dim] = (
                steer_v_t[:, steer_target_idx, :]
                - base_v_t[:, steer_target_idx, :proxy_action_dim]
            )
            steer_mask[:, steer_target_idx, 0] = 1.0
        return v_t, applied_delta, steer_mask

    steer_delta = steer_scale * (steer_v_t - mimic_v_t)
    if steer_target_idx is None:
        v_t[:, :, :proxy_action_dim] += steer_delta
        applied_delta[:, :, :proxy_action_dim] = steer_delta
        steer_mask[:, :, 0] = 1.0
    else:
        v_t[:, steer_target_idx, :proxy_action_dim] += steer_delta[
            :, steer_target_idx, :
        ]
        applied_delta[:, steer_target_idx, :proxy_action_dim] = steer_delta[
            :, steer_target_idx, :
        ]
        steer_mask[:, steer_target_idx, 0] = 1.0

    return v_t, applied_delta, steer_mask


def _steer_forward_all(
    base_model,
    steer_model,
    mimic_model,
    images_0, images_1, img_masks_0, img_masks_1,
    lang_tokens, lang_masks,
    state,
    x_t,
    num_steps: int,
    proxy_action_dim: int,
    steer_scale: torch.Tensor,
    share_proxy_dino: bool,
):
    """Full prefix + denoising loop for 3-model steering, suitable for torch.compile.

    Covers the same work as pi0's compiled sample_actions: prefix encoding,
    KV cache construction, and the iterative denoising loop. Only obs_to_input
    (CPU/NumPy transforms) and output_to_actions remain outside.
    """
    images = [images_0, images_1]
    img_masks = [img_masks_0, img_masks_1]

    # --- Base prefix ---
    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(base_prefix_pad_masks, base_prefix_att_masks)
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(base_prefix_att_2d_masks)

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    # --- Proxy prefix (DINO) ---
    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = steer_model.embed_prefix(images, img_masks)
    if share_proxy_dino:
        mimic_prefix_embs = steer_prefix_embs
        mimic_prefix_pad_masks = steer_prefix_pad_masks
        mimic_prefix_att_masks = steer_prefix_att_masks
    else:
        mimic_prefix_embs, mimic_prefix_pad_masks, mimic_prefix_att_masks = mimic_model.embed_prefix(images, img_masks)

    # --- Denoising loop ---
    bsize = x_t.shape[0]
    device = x_t.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    for _ in range(num_steps):
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state, base_prefix_pad_masks, base_past_key_values, x_t, expanded_time,
        )

        steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
            steer_model.embed_suffix(
                state[:, :proxy_action_dim], x_t[:, :, :proxy_action_dim], expanded_time,
            )
        )
        mimic_suffix_embs, mimic_suffix_pad_masks, mimic_suffix_att_masks, mimic_adarms_cond = (
            mimic_model.embed_suffix(
                state[:, :proxy_action_dim], x_t[:, :, :proxy_action_dim], expanded_time,
            )
        )

        steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
        mimic_embs = torch.cat([mimic_prefix_embs, mimic_suffix_embs], dim=1)
        steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
            steer_model, steer_prefix_pad_masks, steer_prefix_att_masks,
            steer_suffix_pad_masks, steer_suffix_att_masks,
        )
        mimic_attention_mask, mimic_pad_masks = build_proxy_expert_masks(
            mimic_model, mimic_prefix_pad_masks, mimic_prefix_att_masks,
            mimic_suffix_pad_masks, mimic_suffix_att_masks,
        )

        steer_position_ids = (torch.cumsum(steer_pad_masks, dim=1) - 1).to(dtype=torch.long)
        mimic_position_ids = (torch.cumsum(mimic_pad_masks, dim=1) - 1).to(dtype=torch.long)

        steer_hidden_states, _ = steer_model.expert_model.forward(
            attention_mask=steer_attention_mask,
            position_ids=steer_position_ids,
            past_key_values=None,
            inputs_embeds=steer_embs,
            use_cache=False,
            adarms_cond=steer_adarms_cond,
        )
        mimic_hidden_states, _ = mimic_model.expert_model.forward(
            attention_mask=mimic_attention_mask,
            position_ids=mimic_position_ids,
            past_key_values=None,
            inputs_embeds=mimic_embs,
            use_cache=False,
            adarms_cond=mimic_adarms_cond,
        )

        steer_v_t = steer_model.action_out_proj(
            steer_hidden_states[:, -steer_model.config.action_horizon:].to(dtype=torch.float32)
        )
        mimic_v_t = mimic_model.action_out_proj(
            mimic_hidden_states[:, -mimic_model.config.action_horizon:].to(dtype=torch.float32)
        )

        v_t = base_v_t.clone()
        v_t[:, :, :proxy_action_dim] += steer_scale * (steer_v_t - mimic_v_t)

        x_t = x_t + dt * v_t
        denoise_time = denoise_time + dt

    return x_t


_compiled_steer_forward_all = None


def _get_compiled_steer_forward():
    global _compiled_steer_forward_all
    if _compiled_steer_forward_all is None:
        import os
        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes"):
            _compiled_steer_forward_all = _steer_forward_all
            logger.info("Steer forward: using eager (torch.compile disabled)")
        else:
            _compiled_steer_forward_all = torch.compile(
                _steer_forward_all, mode="max-autotune",
            )
            logger.info("Steer forward: compiled with max-autotune")
    return _compiled_steer_forward_all


def infer_actions_compiled(base_policy, steer_policy, mimic_policy, raw_obs, args, noise=None):
    """Inference with prefix + denoising compiled via torch.compile.

    Only obs_to_input (CPU/NumPy transforms) and output_to_actions remain
    outside the compiled region -- everything on GPU is compiled, matching
    how pi0_pytorch.py compiles sample_actions.
    """
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model
    mimic_model = mimic_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (bsize, base_model.config.action_horizon, base_action_dim)
    if noise is None:
        noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    share_proxy_dino = (
        steer_model.config.freeze_dino_encoder
        and mimic_model.config.freeze_dino_encoder
        and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
    )
    steer_scale = torch.as_tensor(args.steer_scale, dtype=torch.float32, device=device)

    forward_fn = _get_compiled_steer_forward()
    x_t = forward_fn(
        base_model, steer_model, mimic_model,
        images[0], images[1], img_masks[0], img_masks[1],
        lang_tokens, lang_masks,
        state, noise,
        args.num_steps, proxy_action_dim, steer_scale,
        share_proxy_dino,
    )

    actions = base_policy.output_to_actions(inputs, x_t)
    return {"actions": actions, "visualize_vectors_step": {}}


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                print(f"infer_time: {infer_time * 1000} ms")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def infer_actions_fast(
    base_policy, steer_policy, mimic_policy, raw_obs, args, noise=None, skip_viz=False,
    concurrent_proxies=False,
):
    """Optimized inference with KV cache for all models (base + proxy).

    This version caches prefix KV for proxy models to avoid recomputing
    prefix attention at every denoising step.

    Args:
        noise: Optional pre-generated noise tensor. If None, noise is sampled.
               Pass the same noise to compare with infer_actions.
        skip_viz: If True, skip .cpu().numpy() visualization copies for lower latency.
        concurrent_proxies: If True, run steer and mimic expert forwards on separate
            CUDA streams for concurrent execution. Only beneficial when the expert
            models don't fully saturate the GPU.
    """
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model
    mimic_model = mimic_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    if noise is None:
        noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    # ========== Base model prefix caching ==========
    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    # ========== Proxy models prefix embedding and KV caching ==========
    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    # If the DINO encoder is frozen and the model names are the same,
    # use the same prefix embeddings for mimic
    if (
        steer_model.config.freeze_dino_encoder
        and mimic_model.config.freeze_dino_encoder
        and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
    ):
        mimic_prefix_embs = steer_prefix_embs
        mimic_prefix_pad_masks = steer_prefix_pad_masks
        mimic_prefix_att_masks = steer_prefix_att_masks
    else:
        mimic_prefix_embs, mimic_prefix_pad_masks, mimic_prefix_att_masks = (
            mimic_model.embed_prefix(images, img_masks)
        )

    # Cache KV for steer proxy model
    steer_prefix_position_ids = torch.cumsum(steer_prefix_pad_masks, dim=1) - 1
    steer_prefix_position_ids = steer_prefix_position_ids.to(dtype=torch.long)

    logger.debug(f"Proxy prefix seq len: {steer_prefix_embs.shape[1]}")

    _, steer_past_key_values = steer_model.expert_model.forward(
        attention_mask=build_proxy_prefix_mask(
            steer_model, steer_prefix_pad_masks, steer_prefix_att_masks
        ),
        position_ids=steer_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=steer_prefix_embs,
        use_cache=True,
        adarms_cond=None,
    )

    # Cache KV for mimic proxy model
    mimic_prefix_position_ids = torch.cumsum(mimic_prefix_pad_masks, dim=1) - 1
    mimic_prefix_position_ids = mimic_prefix_position_ids.to(dtype=torch.long)

    _, mimic_past_key_values = mimic_model.expert_model.forward(
        attention_mask=build_proxy_prefix_mask(
            mimic_model, mimic_prefix_pad_masks, mimic_prefix_att_masks
        ),
        position_ids=mimic_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=mimic_prefix_embs,
        use_cache=True,
        adarms_cond=None,
    )

    # ========== Denoising loop ==========
    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    visualize_vectors_step = dict()
    steer_target, steer_target_idx = _resolve_steer_target(
        args, base_model.config.action_horizon
    )

    if not skip_viz:
        cpu_inputs = dict()
        cpu_inputs["state"] = inputs["state"].cpu().numpy()
        visualize_vectors_step["inputs"] = cpu_inputs
        visualize_vectors_step["num_steps"] = args.num_steps
        visualize_vectors_step["vectors"] = []
        visualize_vectors_step["base_vectors"] = []
        visualize_vectors_step["steer_vectors"] = []
        visualize_vectors_step["mimic_vectors"] = []
        visualize_vectors_step["steer_target"] = steer_target
        visualize_vectors_step["steer_target_idx"] = steer_target_idx
        visualize_vectors_step["steer_masks"] = []
        visualize_vectors_step["steer_delta_vectors"] = []

    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        if not skip_viz:
            visualize_vectors_step["vectors"].append(x_t.cpu().numpy())

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        if denoise_time >= args.steer_step:
            steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
                steer_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            mimic_suffix_embs, mimic_suffix_pad_masks, mimic_suffix_att_masks, mimic_adarms_cond = (
                mimic_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            steer_prefix_offsets = torch.sum(steer_prefix_pad_masks, dim=-1)[:, None]
            steer_suffix_position_ids = (
                steer_prefix_offsets + torch.cumsum(steer_suffix_pad_masks, dim=1) - 1
            )
            steer_suffix_position_ids = steer_suffix_position_ids.to(dtype=torch.long)

            mimic_prefix_offsets = torch.sum(mimic_prefix_pad_masks, dim=-1)[:, None]
            mimic_suffix_position_ids = (
                mimic_prefix_offsets + torch.cumsum(mimic_suffix_pad_masks, dim=1) - 1
            )
            mimic_suffix_position_ids = mimic_suffix_position_ids.to(dtype=torch.long)

            # The prefix is already in the KV cache, so only the suffix tokens
            # are queried -> pass the prefix length as the query offset.
            steer_attention_mask, _ = build_proxy_expert_masks(
                steer_model,
                steer_prefix_pad_masks,
                steer_prefix_att_masks,
                steer_suffix_pad_masks,
                steer_suffix_att_masks,
                query_offset=steer_prefix_pad_masks.shape[1],
            )
            mimic_attention_mask, _ = build_proxy_expert_masks(
                mimic_model,
                mimic_prefix_pad_masks,
                mimic_prefix_att_masks,
                mimic_suffix_pad_masks,
                mimic_suffix_att_masks,
                query_offset=mimic_prefix_pad_masks.shape[1],
            )

            if concurrent_proxies:
                steer_stream = torch.cuda.Stream()
                mimic_stream = torch.cuda.Stream()
                with torch.cuda.stream(steer_stream):
                    steer_hidden_states, _ = steer_model.expert_model.forward(
                        attention_mask=steer_attention_mask,
                        position_ids=steer_suffix_position_ids,
                        past_key_values=steer_past_key_values,
                        inputs_embeds=steer_suffix_embs,
                        use_cache=False,
                        adarms_cond=steer_adarms_cond,
                    )
                with torch.cuda.stream(mimic_stream):
                    mimic_hidden_states, _ = mimic_model.expert_model.forward(
                        attention_mask=mimic_attention_mask,
                        position_ids=mimic_suffix_position_ids,
                        past_key_values=mimic_past_key_values,
                        inputs_embeds=mimic_suffix_embs,
                        use_cache=False,
                        adarms_cond=mimic_adarms_cond,
                    )
                torch.cuda.current_stream().wait_stream(steer_stream)
                torch.cuda.current_stream().wait_stream(mimic_stream)
            else:
                steer_hidden_states, _ = steer_model.expert_model.forward(
                    attention_mask=steer_attention_mask,
                    position_ids=steer_suffix_position_ids,
                    past_key_values=steer_past_key_values,
                    inputs_embeds=steer_suffix_embs,
                    use_cache=False,
                    adarms_cond=steer_adarms_cond,
                )

                mimic_hidden_states, _ = mimic_model.expert_model.forward(
                    attention_mask=mimic_attention_mask,
                    position_ids=mimic_suffix_position_ids,
                    past_key_values=mimic_past_key_values,
                    inputs_embeds=mimic_suffix_embs,
                    use_cache=False,
                    adarms_cond=mimic_adarms_cond,
                )

            steer_suffix_out = steer_hidden_states[
                :, -steer_model.config.action_horizon :
            ]
            steer_suffix_out = steer_suffix_out.to(dtype=torch.float32)
            mimic_suffix_out = mimic_hidden_states[
                :, -mimic_model.config.action_horizon :
            ]
            mimic_suffix_out = mimic_suffix_out.to(dtype=torch.float32)

            steer_v_t = steer_model.action_out_proj(steer_suffix_out)
            mimic_v_t = mimic_model.action_out_proj(mimic_suffix_out)

            if not skip_viz:
                visualize_vectors_step["steer_vectors"].append(steer_v_t.cpu().numpy())
                visualize_vectors_step["mimic_vectors"].append(mimic_v_t.cpu().numpy())

            if args.use_decreasing_steer_scale:
                steer_scale = args.steer_scale * denoise_time.item()
            elif args.use_increasing_steer_scale:
                steer_scale = args.steer_scale * (1 - denoise_time.item())
            else:
                steer_scale = args.steer_scale
            v_t, applied_delta, steer_mask = _apply_proxy_steering(
                base_v_t=base_v_t,
                steer_v_t=steer_v_t,
                mimic_v_t=mimic_v_t,
                proxy_action_dim=proxy_action_dim,
                steer_scale=steer_scale,
                steer_target_idx=steer_target_idx,
                only_steer=args.only_steer,
            )
            if not skip_viz:
                visualize_vectors_step["steer_masks"].append(steer_mask.cpu().numpy())
                visualize_vectors_step["steer_delta_vectors"].append(
                    applied_delta.cpu().numpy()
                )
        else:
            v_t = base_v_t
            if not skip_viz:
                visualize_vectors_step["steer_masks"].append(
                    torch.zeros(
                        (base_v_t.shape[0], base_v_t.shape[1], 1),
                        dtype=base_v_t.dtype,
                        device=base_v_t.device,
                    )
                    .cpu()
                    .numpy()
                )
                visualize_vectors_step["steer_delta_vectors"].append(
                    torch.zeros_like(base_v_t).cpu().numpy()
                )

        if not skip_viz:
            visualize_vectors_step["base_vectors"].append(base_v_t.cpu().numpy())

        x_t = x_t + dt * v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
        "visualize_vectors_step": visualize_vectors_step,
    }
    return results


def infer_actions(base_policy, steer_policy, mimic_policy, raw_obs, args, noise=None, skip_viz=False):
    """Original inference without KV cache for proxy models.

    This version recomputes prefix attention at every denoising step for proxy models.
    Kept for comparison and debugging purposes.

    Args:
        noise: Optional pre-generated noise tensor. If None, noise is sampled.
               Pass the same noise to compare with infer_actions_fast.
        skip_viz: If True, skip .cpu().numpy() visualization copies for lower latency.
    """
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model
    mimic_model = mimic_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    if noise is None:
        noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    if (
        steer_model.config.freeze_dino_encoder
        and mimic_model.config.freeze_dino_encoder
        and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
    ):
        mimic_prefix_embs = steer_prefix_embs
        mimic_prefix_pad_masks = steer_prefix_pad_masks
        mimic_prefix_att_masks = steer_prefix_att_masks
    else:
        mimic_prefix_embs, mimic_prefix_pad_masks, mimic_prefix_att_masks = (
            mimic_model.embed_prefix(images, img_masks)
        )

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    visualize_vectors_step = dict()
    steer_target, steer_target_idx = _resolve_steer_target(
        args, base_model.config.action_horizon
    )

    if not skip_viz:
        cpu_inputs = dict()
        cpu_inputs["state"] = inputs["state"].cpu().numpy()
        visualize_vectors_step["inputs"] = cpu_inputs
        visualize_vectors_step["num_steps"] = args.num_steps
        visualize_vectors_step["vectors"] = []
        visualize_vectors_step["base_vectors"] = []
        visualize_vectors_step["steer_vectors"] = []
        visualize_vectors_step["mimic_vectors"] = []
        visualize_vectors_step["steer_target"] = steer_target
        visualize_vectors_step["steer_target_idx"] = steer_target_idx
        visualize_vectors_step["steer_masks"] = []
        visualize_vectors_step["steer_delta_vectors"] = []

    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        if not skip_viz:
            visualize_vectors_step["vectors"].append(x_t.cpu().numpy())

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        if denoise_time >= args.steer_step:
            steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
                steer_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            mimic_suffix_embs, mimic_suffix_pad_masks, mimic_suffix_att_masks, mimic_adarms_cond = (
                mimic_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
            mimic_embs = torch.cat([mimic_prefix_embs, mimic_suffix_embs], dim=1)
            steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
                steer_model,
                steer_prefix_pad_masks,
                steer_prefix_att_masks,
                steer_suffix_pad_masks,
                steer_suffix_att_masks,
            )
            mimic_attention_mask, mimic_pad_masks = build_proxy_expert_masks(
                mimic_model,
                mimic_prefix_pad_masks,
                mimic_prefix_att_masks,
                mimic_suffix_pad_masks,
                mimic_suffix_att_masks,
            )
            steer_position_ids = torch.cumsum(steer_pad_masks, dim=1) - 1
            mimic_position_ids = torch.cumsum(mimic_pad_masks, dim=1) - 1
            steer_position_ids = steer_position_ids.to(dtype=torch.long)
            mimic_position_ids = mimic_position_ids.to(dtype=torch.long)

            steer_hidden_states, _ = steer_model.expert_model.forward(
                attention_mask=steer_attention_mask,
                position_ids=steer_position_ids,
                past_key_values=None,
                inputs_embeds=steer_embs,
                use_cache=False,
                adarms_cond=steer_adarms_cond,
            )

            mimic_hidden_states, _ = mimic_model.expert_model.forward(
                attention_mask=mimic_attention_mask,
                position_ids=mimic_position_ids,
                past_key_values=None,
                inputs_embeds=mimic_embs,
                use_cache=False,
                adarms_cond=mimic_adarms_cond,
            )

            steer_suffix_out = steer_hidden_states[
                :, -steer_model.config.action_horizon :
            ]
            steer_suffix_out = steer_suffix_out.to(dtype=torch.float32)
            mimic_suffix_out = mimic_hidden_states[
                :, -mimic_model.config.action_horizon :
            ]
            mimic_suffix_out = mimic_suffix_out.to(dtype=torch.float32)

            steer_v_t = steer_model.action_out_proj(steer_suffix_out)
            mimic_v_t = mimic_model.action_out_proj(mimic_suffix_out)

            if not skip_viz:
                visualize_vectors_step["steer_vectors"].append(steer_v_t.cpu().numpy())
                visualize_vectors_step["mimic_vectors"].append(mimic_v_t.cpu().numpy())

            if args.use_decreasing_steer_scale:
                steer_scale = args.steer_scale * denoise_time.item()
            elif args.use_increasing_steer_scale:
                steer_scale = args.steer_scale * (1 - denoise_time.item())
            else:
                steer_scale = args.steer_scale
            v_t, applied_delta, steer_mask = _apply_proxy_steering(
                base_v_t=base_v_t,
                steer_v_t=steer_v_t,
                mimic_v_t=mimic_v_t,
                proxy_action_dim=proxy_action_dim,
                steer_scale=steer_scale,
                steer_target_idx=steer_target_idx,
                only_steer=args.only_steer,
            )
            if not skip_viz:
                visualize_vectors_step["steer_masks"].append(steer_mask.cpu().numpy())
                visualize_vectors_step["steer_delta_vectors"].append(
                    applied_delta.cpu().numpy()
                )
        else:
            v_t = base_v_t
            if not skip_viz:
                visualize_vectors_step["steer_masks"].append(
                    torch.zeros(
                        (base_v_t.shape[0], base_v_t.shape[1], 1),
                        dtype=base_v_t.dtype,
                        device=base_v_t.device,
                    )
                    .cpu()
                    .numpy()
                )
                visualize_vectors_step["steer_delta_vectors"].append(
                    torch.zeros_like(base_v_t).cpu().numpy()
                )

        if not skip_viz:
            visualize_vectors_step["base_vectors"].append(base_v_t.cpu().numpy())

        x_t = x_t + dt * v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
        "visualize_vectors_step": visualize_vectors_step,
    }
    return results


class WebsocketSteerServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        steer_policy: _base_policy.BasePolicy,
        mimic_policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._steer_policy = steer_policy
        self._mimic_policy = mimic_policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                step = obs["step"]
                compare_mode = obs.get(
                    "compare", False
                )  # Test mode: run both with same noise
                args = copy.deepcopy(self._args)
                if (
                    args.base_take_over_interval != -1
                    and step % args.base_take_over_interval == 0
                ):
                    args.steer_step = 1.1
                obs.pop("step", None)
                obs.pop("compare", None)
                visualize_mode = obs.pop("visualize", False)

                infer_time = time.monotonic()
                with torch.no_grad():
                    if compare_mode:
                        # Test mode: generate noise once and use for both methods
                        # This ensures fair comparison between slow and fast methods
                        temp_obs, _ = self._base_policy.obs_to_input(obs)
                        bsize = temp_obs.state.shape[0]
                        device = temp_obs.state.device
                        base_model = self._base_policy._model
                        actions_shape = (
                            bsize,
                            base_model.config.action_horizon,
                            base_model.config.action_dim,
                        )
                        shared_noise = base_model.sample_noise(actions_shape, device)

                        # Warm-up run (compile CUDA kernels, allocate memory)
                        _ = infer_actions_fast(
                            self._base_policy,
                            self._steer_policy,
                            self._mimic_policy,
                            obs,
                            args,
                            noise=shared_noise.clone(),
                        )

                        # Synchronize GPU before timing
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        start_time = time.monotonic()
                        slow_action = infer_actions(
                            self._base_policy,
                            self._steer_policy,
                            self._mimic_policy,
                            obs,
                            args,
                            noise=shared_noise.clone(),
                        )
                        # Synchronize GPU after slow
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        slow_time = time.monotonic() - start_time

                        # Synchronize GPU before fast timing
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        start_time = time.monotonic()
                        fast_action = infer_actions_fast(
                            self._base_policy,
                            self._steer_policy,
                            self._mimic_policy,
                            obs,
                            args,
                            noise=shared_noise.clone(),
                        )
                        # Synchronize GPU after fast
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        fast_time = time.monotonic() - start_time
                        # Debug: compare intermediate vectors between slow and fast
                        slow_base = slow_action["visualize_vectors_step"][
                            "base_vectors"
                        ]
                        fast_base = fast_action["visualize_vectors_step"][
                            "base_vectors"
                        ]
                        slow_steer = slow_action["visualize_vectors_step"][
                            "steer_vectors"
                        ]
                        fast_steer = fast_action["visualize_vectors_step"][
                            "steer_vectors"
                        ]
                        slow_mimic = slow_action["visualize_vectors_step"][
                            "mimic_vectors"
                        ]
                        fast_mimic = fast_action["visualize_vectors_step"][
                            "mimic_vectors"
                        ]

                        import numpy as np

                        base_diffs = []
                        steer_diffs = []
                        mimic_diffs = []
                        for i, (sb, fb) in enumerate(zip(slow_base, fast_base)):
                            diff = np.abs(sb - fb).max()
                            base_diffs.append(diff)
                        for i, (ss, fs) in enumerate(zip(slow_steer, fast_steer)):
                            diff = np.abs(ss - fs).max()
                            steer_diffs.append(diff)
                        for i, (sm, fm) in enumerate(zip(slow_mimic, fast_mimic)):
                            diff = np.abs(sm - fm).max()
                            mimic_diffs.append(diff)

                        logger.info(f"Base vector diffs per step: {base_diffs}")
                        logger.info(f"Steer vector diffs per step: {steer_diffs}")
                        logger.info(f"Mimic vector diffs per step: {mimic_diffs}")
                        logger.info(
                            f"Max base diff: {max(base_diffs) if base_diffs else 0}"
                        )
                        logger.info(
                            f"Max steer diff: {max(steer_diffs) if steer_diffs else 0}"
                        )
                        logger.info(
                            f"Max mimic diff: {max(mimic_diffs) if mimic_diffs else 0}"
                        )

                        action = {
                            "slow_actions": slow_action["actions"],
                            "fast_actions": fast_action["actions"],
                            "slow_time": slow_time,
                            "fast_time": fast_time,
                            "actions": fast_action["actions"],  # Default to fast
                            "visualize_vectors_step": fast_action[
                                "visualize_vectors_step"
                            ],
                            "debug": {
                                "base_diffs": base_diffs,
                                "steer_diffs": steer_diffs,
                                "mimic_diffs": mimic_diffs,
                            },
                        }
                    elif visualize_mode:
                        action = infer_actions_fast(
                            self._base_policy,
                            self._steer_policy,
                            self._mimic_policy,
                            obs,
                            args,
                            skip_viz=False,
                        )
                    else:
                        action = infer_actions_compiled(
                            self._base_policy,
                            self._steer_policy,
                            self._mimic_policy,
                            obs,
                            args,
                        )
                infer_time = time.monotonic() - infer_time

                print(f"Inference time: {infer_time * 1000} ms")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def guide_actions(base_policy, steer_policy, raw_obs, args):
    # time_start = time.time()
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    # print(f"Preprocess observation latency: {time.time() - time_start}")
    # time_start = time.time()

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    # print(f"Embed prefix latency: {time.time() - time_start}")
    # time_start = time.time()

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    while denoise_time >= -dt / 2:
        # time_denoise_start = time.time()
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        # print(f"Base denoise latency: {time.time() - time_denoise_start}")
        # time_denoise_start = time.time()

        if denoise_time >= args.steer_step:
            steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
                steer_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
            steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
                steer_model,
                steer_prefix_pad_masks,
                steer_prefix_att_masks,
                steer_suffix_pad_masks,
                steer_suffix_att_masks,
            )
            steer_position_ids = torch.cumsum(steer_pad_masks, dim=1) - 1
            steer_position_ids = steer_position_ids.to(dtype=torch.long)

            steer_hidden_states, _ = steer_model.expert_model.forward(
                attention_mask=steer_attention_mask,
                position_ids=steer_position_ids,
                past_key_values=None,
                inputs_embeds=steer_embs,
                use_cache=False,
                adarms_cond=steer_adarms_cond,
            )

            steer_suffix_out = steer_hidden_states[
                :, -steer_model.config.action_horizon :
            ]
            steer_suffix_out = steer_suffix_out.to(dtype=torch.float32)
            steer_v_t = steer_model.action_out_proj(steer_suffix_out)

            v_t = base_v_t
            v_t[:, :, :proxy_action_dim] = steer_v_t

            # print(f"Proxy denoise latency: {time.time() - time_denoise_start}")
        else:
            v_t = base_v_t

        x_t = x_t + dt * v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
    }
    return results


class WebsocketGuideServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        steer_policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._steer_policy = steer_policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                with torch.no_grad():
                    action = guide_actions(
                        self._base_policy,
                        self._steer_policy,
                        obs,
                        self._args,
                    )
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# Steer policy overwrite base policy in the first few steps, then only use base policy
def consecutive_actions(base_policy, steer_policy, raw_obs, args):
    # time_start = time.time()
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    # print(f"Preprocess observation latency: {time.time() - time_start}")
    # time_start = time.time()

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    # print(f"Embed prefix latency: {time.time() - time_start}")
    # time_start = time.time()

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise.clone()
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    visualize_vectors_step = dict()

    cup_inputs = dict()
    cup_inputs["state"] = inputs["state"].cpu().numpy()

    visualize_vectors_step["inputs"] = cup_inputs
    visualize_vectors_step["num_steps"] = args.num_steps
    visualize_vectors_step["vectors"] = []
    visualize_vectors_step["base_vectors"] = []
    visualize_vectors_step["steer_vectors"] = []

    while denoise_time >= -dt / 2:
        visualize_vectors_step["vectors"].append(x_t.cpu().numpy())
        # time_denoise_start = time.time()
        expanded_time = denoise_time.expand(bsize)

        steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
            steer_model.embed_suffix(
                state[:, :proxy_action_dim],
                x_t[:, :, :proxy_action_dim],
                expanded_time,
            )
        )

        steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
        steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
            steer_model,
            steer_prefix_pad_masks,
            steer_prefix_att_masks,
            steer_suffix_pad_masks,
            steer_suffix_att_masks,
        )
        steer_position_ids = torch.cumsum(steer_pad_masks, dim=1) - 1
        steer_position_ids = steer_position_ids.to(dtype=torch.long)

        steer_hidden_states, _ = steer_model.expert_model.forward(
            attention_mask=steer_attention_mask,
            position_ids=steer_position_ids,
            past_key_values=None,
            inputs_embeds=steer_embs,
            use_cache=False,
            adarms_cond=steer_adarms_cond,
        )

        steer_suffix_out = steer_hidden_states[:, -steer_model.config.action_horizon :]
        steer_suffix_out = steer_suffix_out.to(dtype=torch.float32)
        steer_v_t = steer_model.action_out_proj(steer_suffix_out)

        visualize_vectors_step["steer_vectors"].append(steer_v_t.cpu().numpy())
        visualize_vectors_step["base_vectors"].append(None)

        v_t = steer_v_t

        x_t[:, :, :proxy_action_dim] = x_t[:, :, :proxy_action_dim] + dt * v_t
        denoise_time += dt

    # linear interpolate between the x_t denoised by steer policy and initial noise
    x_t = x_t * (1 - args.base_start) + noise * args.base_start

    denoise_time = torch.tensor(args.base_start, dtype=torch.float32, device=device)

    while denoise_time >= -dt / 2:
        visualize_vectors_step["vectors"].append(x_t.cpu().numpy())
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        visualize_vectors_step["base_vectors"].append(base_v_t.cpu().numpy())
        visualize_vectors_step["steer_vectors"].append(None)

        v_t = base_v_t

        x_t = x_t + dt * v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
        "visualize_vectors_step": visualize_vectors_step,
    }
    return results


class WebsocketConsecutiveServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        steer_policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._steer_policy = steer_policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                with torch.no_grad():
                    action = consecutive_actions(
                        self._base_policy,
                        self._steer_policy,
                        obs,
                        self._args,
                    )
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def classifier_free_actions(base_policy, steer_policy, raw_obs, args):
    # time_start = time.time()
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    visualize_vectors_step = dict()

    cup_inputs = dict()
    cup_inputs["state"] = inputs["state"].cpu().numpy()

    visualize_vectors_step["inputs"] = cup_inputs
    visualize_vectors_step["num_steps"] = args.num_steps
    visualize_vectors_step["vectors"] = []
    visualize_vectors_step["base_vectors"] = []
    visualize_vectors_step["steer_vectors"] = []
    visualize_vectors_step["mimic_vectors"] = []

    while denoise_time >= -dt / 2:
        # time_denoise_start = time.time()
        expanded_time = denoise_time.expand(bsize)

        visualize_vectors_step["vectors"].append(x_t.cpu().numpy())

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        if denoise_time >= args.steer_step:
            steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
                steer_model.embed_suffix(
                    state[:, :proxy_action_dim],
                    x_t[:, :, :proxy_action_dim],
                    expanded_time,
                )
            )

            steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
            steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
                steer_model,
                steer_prefix_pad_masks,
                steer_prefix_att_masks,
                steer_suffix_pad_masks,
                steer_suffix_att_masks,
            )
            steer_position_ids = torch.cumsum(steer_pad_masks, dim=1) - 1
            steer_position_ids = steer_position_ids.to(dtype=torch.long)

            steer_hidden_states, _ = steer_model.expert_model.forward(
                attention_mask=steer_attention_mask,
                position_ids=steer_position_ids,
                past_key_values=None,
                inputs_embeds=steer_embs,
                use_cache=False,
                adarms_cond=steer_adarms_cond,
            )

            steer_suffix_out = steer_hidden_states[
                :, -steer_model.config.action_horizon :
            ]
            steer_suffix_out = steer_suffix_out.to(dtype=torch.float32)
            steer_v_t = steer_model.action_out_proj(steer_suffix_out)

            visualize_vectors_step["steer_vectors"].append(steer_v_t.cpu().numpy())
            visualize_vectors_step["mimic_vectors"].append(None)

            v_t = base_v_t.clone()  # Clone to avoid modifying base_v_t in-place
            if args.use_decreasing_steer_scale:
                steer_scale = args.steer_scale * denoise_time.item()
            elif args.use_increasing_steer_scale:
                steer_scale = args.steer_scale * (1 - denoise_time.item())
            else:
                steer_scale = args.steer_scale
            # # Do not steer the gripper dimension
            # v_t[:, :, :proxy_action_dim - 1] += steer_scale * (
            #     steer_v_t[:, :, :proxy_action_dim - 1] - base_v_t[:, :, :proxy_action_dim - 1]
            # )
            # Steer the gripper dimension
            v_t[:, :, :proxy_action_dim] += steer_scale * (
                steer_v_t - base_v_t[:, :, :proxy_action_dim]
            )
        else:
            v_t = base_v_t

        visualize_vectors_step["base_vectors"].append(base_v_t.cpu().numpy())

        x_t = x_t + dt * v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
        "visualize_vectors_step": visualize_vectors_step,
    }
    return results


def _classifier_free_forward_all(
    base_model,
    steer_model,
    images_0, images_1, img_masks_0, img_masks_1,
    lang_tokens, lang_masks,
    state,
    x_t,
    num_steps: int,
    proxy_action_dim: int,
    steer_step: torch.Tensor,
    steer_scale: torch.Tensor,
    use_decreasing_steer_scale: bool,
    use_increasing_steer_scale: bool,
):
    """Full classifier-free prefix + denoising loop, suitable for torch.compile."""
    images = [images_0, images_1]
    img_masks = [img_masks_0, img_masks_1]

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = (
        steer_model.embed_prefix(images, img_masks)
    )

    bsize = x_t.shape[0]
    device = x_t.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    for _ in range(num_steps):
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
            steer_model.embed_suffix(
                state[:, :proxy_action_dim],
                x_t[:, :, :proxy_action_dim],
                expanded_time,
            )
        )

        steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
        steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
            steer_model,
            steer_prefix_pad_masks,
            steer_prefix_att_masks,
            steer_suffix_pad_masks,
            steer_suffix_att_masks,
        )
        steer_position_ids = (torch.cumsum(steer_pad_masks, dim=1) - 1).to(
            dtype=torch.long
        )

        steer_hidden_states, _ = steer_model.expert_model.forward(
            attention_mask=steer_attention_mask,
            position_ids=steer_position_ids,
            past_key_values=None,
            inputs_embeds=steer_embs,
            use_cache=False,
            adarms_cond=steer_adarms_cond,
        )

        steer_v_t = steer_model.action_out_proj(
            steer_hidden_states[:, -steer_model.config.action_horizon :].to(
                dtype=torch.float32
            )
        )

        if use_decreasing_steer_scale:
            step_scale = steer_scale * denoise_time
        elif use_increasing_steer_scale:
            step_scale = steer_scale * (1.0 - denoise_time)
        else:
            step_scale = steer_scale

        steer_mask = (denoise_time >= steer_step).to(dtype=base_v_t.dtype)
        steer_delta = step_scale * steer_mask * (
            steer_v_t - base_v_t[:, :, :proxy_action_dim]
        )
        v_t = base_v_t.clone()
        v_t[:, :, :proxy_action_dim] += steer_delta

        x_t = x_t + dt * v_t
        denoise_time = denoise_time + dt

    return x_t


_compiled_classifier_free_forward_all = None


def _get_compiled_classifier_free_forward():
    global _compiled_classifier_free_forward_all
    if _compiled_classifier_free_forward_all is None:
        import os

        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            _compiled_classifier_free_forward_all = _classifier_free_forward_all
            logger.info("Classifier-free forward: using eager (torch.compile disabled)")
        else:
            _compiled_classifier_free_forward_all = torch.compile(
                _classifier_free_forward_all, mode="max-autotune",
            )
            logger.info("Classifier-free forward: compiled with max-autotune")
    return _compiled_classifier_free_forward_all


def classifier_free_actions_compiled(base_policy, steer_policy, raw_obs, args, noise=None):
    """Classifier-free inference with prefix + denoising compiled via torch.compile."""
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    if noise is None:
        noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )
    steer_step = torch.as_tensor(args.steer_step, dtype=torch.float32, device=device)
    steer_scale = torch.as_tensor(args.steer_scale, dtype=torch.float32, device=device)

    forward_fn = _get_compiled_classifier_free_forward()
    x_t = forward_fn(
        base_model,
        steer_model,
        images[0],
        images[1],
        img_masks[0],
        img_masks[1],
        lang_tokens,
        lang_masks,
        state,
        noise,
        args.num_steps,
        proxy_action_dim,
        steer_step,
        steer_scale,
        args.use_decreasing_steer_scale,
        args.use_increasing_steer_scale,
    )

    actions = base_policy.output_to_actions(inputs, x_t)
    return {
        "actions": actions,
        "visualize_vectors_step": {},
    }


class WebsocketClassifierFreeServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        steer_policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._steer_policy = steer_policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # step = obs["step"]
                args = copy.deepcopy(self._args)
                # if (
                #     args.base_take_over_interval != -1
                #     and step % args.base_take_over_interval == 0
                # ):
                #     args.steer_step = 1.1
                # obs.pop("step")

                infer_time = time.monotonic()
                with torch.no_grad():
                    action = classifier_free_actions_compiled(
                        self._base_policy,
                        self._steer_policy,
                        obs,
                        args,
                    )
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def multi_prompt_cfg_actions(policy, raw_obs, args):
    """
    Classifier-free guidance using the SAME model backbone with different prompts.

    This implements the CFG formula:
        v_guided = v_negative + guidance_scale * (v_positive - v_negative)

    When guidance_scale=0: output follows negative prompt entirely
    When guidance_scale=1: output follows positive prompt entirely
    When guidance_scale>1: output is pushed further towards positive prompt
    """
    model = policy._model

    # --- Step 1: Process observation with POSITIVE prompt ---
    obs_positive = copy.deepcopy(raw_obs)
    obs_positive["prompt"] = args.prompt_positive
    obs_pos, inputs_pos = policy.obs_to_input(obs_positive)

    # --- Step 2: Process observation with NEGATIVE prompt ---
    obs_negative = copy.deepcopy(raw_obs)
    obs_negative["prompt"] = args.prompt_negative
    obs_neg, inputs_neg = policy.obs_to_input(obs_negative)

    bsize = obs_pos.state.shape[0]
    device = obs_pos.state.device

    action_dim = model.config.action_dim
    actions_shape = (bsize, model.config.action_horizon, action_dim)
    noise = model.sample_noise(actions_shape, device)

    # --- Step 3: Preprocess both observations ---
    # Positive prompt observation
    images_pos, img_masks_pos, lang_tokens_pos, lang_masks_pos, state = (
        model._preprocess_observation(obs_pos, train=False)
    )
    # Negative prompt observation (only lang tokens differ, images and state are the same)
    _, _, lang_tokens_neg, lang_masks_neg, _ = model._preprocess_observation(
        obs_neg, train=False
    )

    # --- Step 4: Compute prefix embeddings for POSITIVE prompt ---
    pos_prefix_embs, pos_prefix_pad_masks, pos_prefix_att_masks = model.embed_prefix(
        images_pos, img_masks_pos, lang_tokens_pos, lang_masks_pos
    )
    pos_prefix_att_2d_masks = make_att_2d_masks(
        pos_prefix_pad_masks, pos_prefix_att_masks
    )
    pos_prefix_position_ids = torch.cumsum(pos_prefix_pad_masks, dim=1) - 1

    pos_prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(
        pos_prefix_att_2d_masks
    )
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )

    # Cache key-values for positive prompt prefix
    _, pos_past_key_values = model.paligemma_with_expert.forward(
        attention_mask=pos_prefix_att_2d_masks_4d,
        position_ids=pos_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[pos_prefix_embs, None],
        use_cache=True,
    )

    # --- Step 5: Compute prefix embeddings for NEGATIVE prompt ---
    neg_prefix_embs, neg_prefix_pad_masks, neg_prefix_att_masks = model.embed_prefix(
        images_pos, img_masks_pos, lang_tokens_neg, lang_masks_neg
    )
    neg_prefix_att_2d_masks = make_att_2d_masks(
        neg_prefix_pad_masks, neg_prefix_att_masks
    )
    neg_prefix_position_ids = torch.cumsum(neg_prefix_pad_masks, dim=1) - 1

    neg_prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(
        neg_prefix_att_2d_masks
    )

    # Cache key-values for negative prompt prefix
    _, neg_past_key_values = model.paligemma_with_expert.forward(
        attention_mask=neg_prefix_att_2d_masks_4d,
        position_ids=neg_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[neg_prefix_embs, None],
        use_cache=True,
    )

    # --- Step 6: Denoising loop with CFG ---
    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    # For visualization/debugging
    visualize_data = {
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "prompt_positive": args.prompt_positive,
        "prompt_negative": args.prompt_negative,
        "vectors": [],
        "positive_vectors": [],
        "negative_vectors": [],
    }

    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        visualize_data["vectors"].append(x_t.cpu().numpy())

        # Compute velocity from POSITIVE prompt
        v_positive = model.denoise_step(
            state,
            pos_prefix_pad_masks,
            pos_past_key_values,
            x_t,
            expanded_time,
        )

        # Compute velocity from NEGATIVE prompt
        v_negative = model.denoise_step(
            state,
            neg_prefix_pad_masks,
            neg_past_key_values,
            x_t,
            expanded_time,
        )

        visualize_data["positive_vectors"].append(v_positive.cpu().numpy())
        visualize_data["negative_vectors"].append(v_negative.cpu().numpy())

        # Apply classifier-free guidance
        if denoise_time >= args.guidance_start_step:
            # Compute dynamic guidance scale if enabled
            if args.use_decreasing_guidance:
                guidance_scale = args.guidance_scale * denoise_time.item()
            elif args.use_increasing_guidance:
                guidance_scale = args.guidance_scale * (1 - denoise_time.item())
            else:
                guidance_scale = args.guidance_scale

            # CFG formula: v = v_neg + scale * (v_pos - v_neg)
            # Equivalent to: v = (1 - scale) * v_neg + scale * v_pos
            v_t = v_negative + guidance_scale * (v_positive - v_negative)
        else:
            # Before guidance_start_step, just use positive prompt velocity
            v_t = v_positive

        x_t = x_t + dt * v_t
        denoise_time += dt

    # Convert final denoised actions to output format
    actions = policy.output_to_actions(inputs_pos, x_t)

    results = {
        "actions": actions,
        "visualize_data": visualize_data,
    }
    return results


class WebsocketMultiPromptCFGServer:
    """
    Serves a policy using classifier-free guidance with multiple prompts on the SAME model backbone.

    This is memory-efficient compared to loading two separate models, as model weights are shared.
    The guidance formula used is:
        v_guided = v_negative + guidance_scale * (v_positive - v_negative)
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata to client
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Allow dynamic override of prompts from client
                args = copy.deepcopy(self._args)
                if "prompt_positive" in obs:
                    args.prompt_positive = obs.pop("prompt_positive")
                if "prompt_negative" in obs:
                    args.prompt_negative = obs.pop("prompt_negative")
                if "guidance_scale" in obs:
                    args.guidance_scale = obs.pop("guidance_scale")

                infer_time = time.monotonic()
                with torch.no_grad():
                    result = multi_prompt_cfg_actions(
                        self._policy,
                        obs,
                        args,
                    )
                infer_time = time.monotonic() - infer_time

                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def midpoint_actions(base_policy, raw_obs, args):
    """
    Sample actions using the Midpoint method (2nd order Runge-Kutta) instead of Euler.

    Midpoint method:
        v_t = denoise_step(x_t, t)
        x_mid = x_t + (dt/2) * v_t
        v_mid = denoise_step(x_mid, t + dt/2)
        x_t = x_t + dt * v_mid

    Note: num_steps is the number of integration steps. Each step requires 2 denoise calls,
    so total denoise calls = 2 * num_steps.
    """
    obs, inputs = base_policy.obs_to_input(raw_obs)

    bsize = obs.state.shape[0]
    device = obs.state.device

    base_model = base_policy._model

    base_action_dim = base_model.config.action_dim

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs, train=False)
    )

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    dt = -1.0 / args.num_steps
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    visualize_vectors_step = dict()

    cpu_inputs = dict()
    cpu_inputs["state"] = inputs["state"].cpu().numpy()

    visualize_vectors_step["inputs"] = cpu_inputs
    visualize_vectors_step["num_steps"] = args.num_steps
    visualize_vectors_step["vectors"] = []
    visualize_vectors_step["base_vectors"] = []
    visualize_vectors_step["midpoint_vectors"] = []

    while denoise_time >= -dt_tensor / 2:
        expanded_time = denoise_time.expand(bsize)

        visualize_vectors_step["vectors"].append(x_t.cpu().numpy())

        # Step 1: Get velocity at current point (x_t, t)
        v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )

        visualize_vectors_step["base_vectors"].append(v_t.cpu().numpy())

        # Step 2: Compute midpoint
        x_mid = x_t + (dt_tensor / 2) * v_t
        t_mid = denoise_time + dt_tensor / 2
        expanded_t_mid = t_mid.expand(bsize)

        # Step 3: Get velocity at midpoint (x_mid, t_mid)
        v_mid = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_mid,
            expanded_t_mid,
        )

        visualize_vectors_step["midpoint_vectors"].append(v_mid.cpu().numpy())

        # Step 4: Full step using midpoint velocity
        x_t = x_t + dt_tensor * v_mid
        denoise_time += dt_tensor

    actions = base_policy.output_to_actions(inputs, x_t)

    results = {
        "actions": actions,
        "visualize_vectors_step": visualize_vectors_step,
    }
    return results


class WebsocketMidpointServer:
    """Serves a policy using the Midpoint method (2nd order Runge-Kutta) for denoising.

    See websocket_client_policy.py for a client implementation.

    Note: num_steps is the number of integration steps. Each step requires 2 denoise calls,
    so total denoise calls = 2 * num_steps.
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        args,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._args = args
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                with torch.no_grad():
                    action = midpoint_actions(
                        self._base_policy,
                        obs,
                        self._args,
                    )
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class WebsocketResidualServer:
    """Serves a VLA + Residual policy combination using the websocket protocol.

    The residual policy predicts corrections to the VLA policy's actions.
    Final action = VLA action + Residual action (in normalized space),
    then output transform is applied.
    """

    def __init__(
        self,
        vla_policy: _base_policy.BasePolicy,
        residual_policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        num_vla_steps: int = 10,
        num_residual_steps: int = 10,
    ) -> None:
        self._vla_policy = vla_policy
        self._residual_policy = residual_policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._num_vla_steps = num_vla_steps
        self._num_residual_steps = num_residual_steps
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                with torch.no_grad():
                    action = self._infer_residual(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _infer_residual(self, raw_obs: dict) -> dict:
        """Run VLA + Residual inference.

        1. Preprocess observation
        2. Run VLA model to get normalized vla_action
        3. Run residual model conditioned on vla_action to get normalized residual
        4. Compute final_action = vla_action + residual (normalized)
        5. Apply output transform to get actual action
        """
        import numpy as np

        # Get observation and inputs
        vla_obs, vla_inputs = self._vla_policy.obs_to_input(raw_obs)
        residual_obs, residual_inputs = self._residual_policy.obs_to_input(raw_obs)

        # Get device from VLA model
        vla_model = self._vla_policy._model
        device = self._vla_policy._pytorch_device

        # Step 1: Run VLA model to get normalized action
        vla_action = vla_model.sample_actions(
            device, vla_obs, num_steps=self._num_vla_steps
        ).clone()

        # Step 2: Prepare observation for residual model
        residual_model = self._residual_policy._model

        # Truncate vla_action to match residual model's action_dim if needed
        residual_action_dim = residual_model.config.action_dim
        vla_action_for_residual = vla_action[:, :, :residual_action_dim]

        # Step 3: Run residual model conditioned on VLA action
        residual_action = residual_model.sample_actions(
            device,
            residual_obs,
            vla_action_for_residual,
            num_steps=self._num_residual_steps,
        )

        # Step 4: Compute final action (in normalized space)
        # final_action = vla_action + residual_action
        final_action = vla_action.clone()
        final_action[:, :, :residual_action_dim] = (
            vla_action_for_residual + residual_action
        )

        # Step 5: Apply output transform
        # Convert to numpy and apply output transform
        # (assuming both policies use same normalization)
        actions = self._vla_policy.output_to_actions(vla_inputs, final_action)

        results = {
            "actions": actions,
        }

        return results


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
