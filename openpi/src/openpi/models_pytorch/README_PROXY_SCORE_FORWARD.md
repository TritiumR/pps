# ProxyScore task-policy forward paths

This file archives the two conditioning paths used by the PyTorch ProxyScore
task policy. It is documentation only; the live implementation remains in
`proxy_score_pytorch.py` and `expert_pytorch.py`.

## Source and checkpoint map

| Variant | Source commit | Checkpoint experiment | Conditioning |
| --- | --- | --- | --- |
| Image-only (current) | `594d021` | `task_eps_bidir_openpi_image_only` | DINO image tokens + robot state |
| Language-conditioned (archived) | `bdf4420` | `task_eps_bidir_openpi` | DINO image tokens + prompt tokens + robot state |

The relevant live files are:

- `proxy_score_pytorch.py`: observation preprocessing, prefix/suffix creation,
  training `forward()`, and DDIM `sample_actions()`.
- `expert_pytorch.py`: DINO image encoder and Gemma action expert. The archived
  language version also owns the prompt-token embedding table here.
- `../models/proxy_score_config.py`: model flags and dimensions.
- `../training/config.py`: the four `score_task_*` training configs.

Both variants use epsilon prediction, bidirectional attention, the OpenPI Gemma
replacement, and no legacy Gemma input scaling. The only intended difference
is whether prompt tokens are part of the prefix.

The common forward flow is:

```text
observation
  -> preprocess images (and optionally prompt tokens)
  -> DINO image embeddings [+ learned prompt-token embeddings]
  -> prefix embeddings

state + noisy action x_t + diffusion time t
  -> state/action/time projections
  -> suffix embeddings

prefix + suffix
  -> bidirectional 4-D attention mask
  -> Gemma action expert
  -> last action_horizon hidden states
  -> action_out_proj (linear layer)
  -> predicted epsilon
```

## Image-only forward path (current, `594d021`)

The current model deliberately does not construct a language embedding table.
`prompt_from_task=True` may still make the data pipeline tokenize the prompt,
but this model never reads those fields.

### Expert construction

```python
self.expert_model = DINOExpertModel(
    dino_model_name=config.dino_model_name,
    action_expert_config=action_expert_config,
    use_adarms=[False, False],
    precision=config.dtype,
    freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
)
```

Inside `DINOExpertModel`, Gemma's token table is removed because all inputs are
already continuous embeddings:

```python
self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
self.gemma_expert.model.embed_tokens = None
```

### Observation preprocessing and prefix embedding

```python
def _preprocess_observation(self, observation, *, train=True):
    observation = _preprocessing.preprocess_observation_pytorch(
        observation, image_keys=IMAGE_KEYS, train=train
    )
    return (
        list(observation.images.values()),
        list(observation.image_masks.values()),
        observation.state,
    )

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
    return embs, pad_masks, att_masks
```

### Training `forward()` conditioning connection

The noising, epsilon target, grouped-label handling, Gemma call, and MSE loss
are shared by both variants. These are the image-only conditioning lines inside
`forward()`:

```python
actions = actions[..., : self.config.action_dim]
images, img_masks, state = self._preprocess_observation(observation, train=True)

# x_t, time, and target are prepared here. With prediction_type="epsilon",
# target is either epsilon_target or sampled noise.

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images, img_masks
)

pred = self._predict_model_output_from_prefix(
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t,
    time,
    prefix_att_masks,
)
loss = F.mse_loss(pred, target, reduction="none")
```

The live `forward()` selects `predict_score_from_prefix` instead when training
from a direct score target or when `prediction_type="score"`; the four clean
task-policy configs use `prediction_type="epsilon"`.

### Inference `sample_actions()` conditioning connection

```python
images, img_masks, state = self._preprocess_observation(
    observation, train=False
)
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images, img_masks
)
```

The prefix is computed once, then reused at every DDIM denoising iteration.

## Language-conditioned forward path (archived, `bdf4420`)

This is the exact conditioning implementation used to train checkpoints under
`task_eps_bidir_openpi`. It must be restored as a complete set: config fields,
the Expert embedding table, observation reads, and both training/inference
prefix calls. Restoring only one part either fails checkpoint loading or silently
ignores the prompt.

### Required config fields

```python
use_language_tokens: bool = False
language_vocab_size: int = 257152
```

Each language-conditioned `score_task_{weight,tea,pot,capsule}` config set:

```python
use_language_tokens=True
```

The data config also used `DataConfig(prompt_from_task=True)`, which produces
`tokenized_prompt` and `tokenized_prompt_mask`.

### Expert construction and prompt embedding table

`ProxyScorePytorch.__init__` passed the vocabulary size only when language was
enabled:

```python
self.expert_model = DINOExpertModel(
    dino_model_name=config.dino_model_name,
    action_expert_config=action_expert_config,
    use_adarms=[False, False],
    precision=config.dtype,
    freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
    language_vocab_size=(
        config.language_vocab_size
        if getattr(config, "use_language_tokens", False)
        else None
    ),
)
```

`DINOExpertModel.__init__` accepted the optional size and created a separate
embedding table. Gemma's own input table still remained disabled:

```python
def __init__(
    self,
    dino_model_name: str,
    action_expert_config,
    use_adarms=None,
    precision: Literal["bfloat16", "float32"] = "float32",
    freeze_dino_encoder: bool = False,
    language_vocab_size: int | None = None,
):
    # ... DINO and Gemma construction ...
    self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
    self.gemma_expert.model.embed_tokens = None
    self.language_embedding = (
        nn.Embedding(language_vocab_size, action_expert_config.width)
        if language_vocab_size is not None
        else None
    )
    if self.language_embedding is not None:
        nn.init.normal_(self.language_embedding.weight, mean=0.0, std=0.02)

def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
    if self.language_embedding is None:
        raise RuntimeError("Language embeddings are disabled for this DINO expert.")
    return self.language_embedding(tokens)
```

This extra embedding table is why a language checkpoint has additional state
dict keys and is not shape/key compatible with the image-only model.

### Observation preprocessing and prefix embedding

```python
def _preprocess_observation(self, observation, *, train=True):
    observation = _preprocessing.preprocess_observation_pytorch(
        observation, image_keys=IMAGE_KEYS, train=train
    )
    return (
        list(observation.images.values()),
        list(observation.image_masks.values()),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.state,
    )

def embed_prefix(
    self, images, img_masks, lang_tokens=None, lang_masks=None
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

    if getattr(self.config, "use_language_tokens", False):
        if lang_tokens is None or lang_masks is None:
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask are required when "
                "use_language_tokens=True."
            )
        lang_emb = self.expert_model.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks.to(torch.bool))
        att_masks += [0] * lang_emb.shape[1]

    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
    att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
    return embs, pad_masks, att_masks
```

The `sqrt(hidden_width)` multiplication applies only to the learned prompt
embedding. It is separate from `legacy_gemma_input_scale`, which would scale
the entire continuous input sequence and remained disabled.

### Training `forward()` conditioning connection

```python
actions = actions[..., : self.config.action_dim]
images, img_masks, lang_tokens, lang_masks, state = (
    self._preprocess_observation(observation, train=True)
)

# x_t, time, and epsilon target preparation are identical to image-only.

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images, img_masks, lang_tokens, lang_masks
)

pred = self._predict_model_output_from_prefix(
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t,
    time,
    prefix_att_masks,
)
loss = F.mse_loss(pred, target, reduction="none")
```

### Inference `sample_actions()` conditioning connection

```python
images, img_masks, lang_tokens, lang_masks, state = (
    self._preprocess_observation(observation, train=False)
)
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images, img_masks, lang_tokens, lang_masks
)
```

As in training, the prompt affects the model only because its embeddings are
concatenated into `prefix_embs`. Tokenization by itself is not conditioning.

## Shared suffix, Gemma, and linear output head

The two variants share this downstream computation. `embed_suffix()` projects
the robot state, noisy action, and diffusion timestep. The action expert then
returns hidden states, and the final linear layer predicts one value per action
dimension:

```python
def _predict_model_output_from_prefix(
    self,
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t,
    time_cond,
    prefix_att_masks=None,
) -> torch.Tensor:
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
        state,
        x_t,
        time_cond,
    )
    return self._run_diffusion_head(
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
        prefix_att_masks,
        suffix_att_masks,
    )

def _run_diffusion_head(...):
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    # Build the bidirectional 4-D mask, then run Gemma.
    hidden_states, _ = self.expert_model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -self.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)
```

The actual output head is initialized as:

```python
self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
```

For the clean task policies, this linear layer predicts epsilon directly. Score
is derived only when a score-space consumer requests it:

```python
score = -epsilon / sqrt(1 - alpha)
```

## Recovering exact historical sources

The snippets above preserve the conditioning-specific code in readable form.
The exact full files can always be recovered from Git without changing the
working tree:

```bash
git show bdf4420:openpi/src/openpi/models_pytorch/proxy_score_pytorch.py
git show bdf4420:openpi/src/openpi/models_pytorch/expert_pytorch.py
git show bdf4420:openpi/src/openpi/models/proxy_score_config.py

git show 594d021:openpi/src/openpi/models_pytorch/proxy_score_pytorch.py
git show 594d021:openpi/src/openpi/models_pytorch/expert_pytorch.py
git show 594d021:openpi/src/openpi/models/proxy_score_config.py
```

Do not load a `task_eps_bidir_openpi` checkpoint with the image-only class or a
`task_eps_bidir_openpi_image_only` checkpoint with the language class. Their
conditioning semantics and state dictionaries intentionally differ.
