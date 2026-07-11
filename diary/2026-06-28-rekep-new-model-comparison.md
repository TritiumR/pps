# ReKep constraint-VLM: gpt-4o vs newer models — comparison plan

**2026-06-28** — Investigation note (no implementation). ReKep's constraint generator currently uses
**gpt-4o** (upstream's choice; faithful). Question raised: would a newer/stronger VLM (e.g.
**gpt-5.4**) author better constraints, and how would we compare them? Findings + a proposed
comparison methodology below. **Not built** — recorded for later.

Context: `rekep/constraint_generation.py` calls `client.chat.completions.create(model=<config>, ...)`
with the model read from `rekep/config.yaml: constraint_generator.model` (currently `gpt-4o`,
`temperature: 0.0`, `max_tokens: 2048`, streamed). The VLM turns the keypoint-annotated image +
instruction into per-stage relational constraint code + metadata (`num_stages`, `grasp_keypoints`,
`release_keypoints`).

---

## Finding 1 — model availability
The account reaches the full **GPT-5.x** family: `gpt-5`, `gpt-5.1`, `gpt-5.2`, **`gpt-5.4`**,
`gpt-5.5`, plus `-mini`/`-nano`/`-pro` variants, and the older `o1`/`o3`/`o4-mini` reasoning models.
So gpt-4o vs gpt-5.4 (and optionally 5.4-mini / 5.5 / 5.4-pro) is directly testable.

## Finding 2 — it is NOT a pure config swap (call-site adaptation needed)
Probed gpt-5.4 with ReKep's exact call shape:
- `max_tokens` + `temperature=0.0` + `stream` → **400 BadRequest**:
  *"'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."*
- `max_completion_tokens` + default temperature → **works**.

GPT-5.x are reasoning models, so the call site needs a small conditional (faithful gpt-4o path
untouched, ~10 lines):
- `max_tokens` → `max_completion_tokens`;
- omit `temperature` (or set `1`) — reasoning models reject `temperature≠1`;
- **bump the token budget** — reasoning tokens are spent before the visible answer, so 2048 can
  starve the constraint output (return empty);
- model stays a config field; possibly expose `reasoning_effort`.

Keep **gpt-4o as the faithful default**; treat any 5.x model as an ablation knob so the reproducible
ReKep path is never touched.

## Proposed comparison methodology
**Isolate the VLM.** Keypoint proposal is deterministic (DINOv2 + fixed seed), so fix the scene →
cache one annotated image + keypoint set → feed the **same** `(image, instruction)` to each model →
diff their constraint programs. Removes perception/controller noise so only the VLM varies. Pure VLM
calls over cached images — **no Isaac boot, ~cents per call.**

Objective metrics:
| Metric | Measures | How |
|---|---|---|
| Keypoint-selection accuracy | picked the right keypoint? | distance from selected `grasp_keypoints` to the **GT target** (e.g. handle) |
| Stage-decomposition correctness | right plan structure? | `num_stages` vs the task's true stage count |
| Constraint validity | does the code work? | parses + loads through the np→torch shim + cost decreases monotonically toward target |
| Stability | how consistent? | K repeats → variance in selection / stage count |

- Run **across the 4 PPS tasks** (tea / pot / weight / capsule), not just the mug — multi-stage tasks
  discriminate stage-decomposition quality far better than a single-stage grasp.
- Cached inputs already (mostly) exist: `results/rekep/<task>/keypoints.png` from the ReKep rollouts,
  plus the mug `results/vlm_mpc/rekep/lift_mug_vlm_sideny/keypoints.png`.
- **Optional downstream check:** run each model's constraints through DIAL with everything else fixed
  (same keypoints, same seed) → compare `grasp_tcp_err` / `held`. Secondary — reintroduces controller
  noise, so it's a tie-breaker, not the headline.
- Optional LLM-as-judge for constraint *semantics*, but the objective metrics above come first.

Harness shape (when built): a VLM-only loop over `models × tasks × K repeats` reading cached annotated
images, tabulating the four metrics. Cheap; can run without a GPU.

## Status
Deferred — noted only. The faithful gpt-4o path stays as-is; this is a forward-looking ablation that
pairs naturally with the [[2026-06-28-rekep-implementation-outline]] "potential additions" list.
