# Characterizing base-policy steerability for PPS

**Question (the advisor's).** For Proxy Policy Steering, when are we *satisfied* with the base policy?
Task success is the wrong metric: a base can have 0 % success and be excellent (it just needs to be
steerable toward success), or have a task-specific success rate and be a *bad* base (too committed to be
steered elsewhere). We need principled criteria for a good base, independent of task success.

**Claim.** Task success is an *outcome* of steering a good base, not a property of the base. So we should
judge the base by the **precondition that makes steering work**, not by what steering eventually produces.

---

## 1. What steering actually is

PPS steering is guidance in the base's denoising flow, restricted to the proxy dims and gated by `steer_step`:

```
v_t[:, :, :proxy_action_dim] += steer_scale · (task_v − ref_v)
```

The in-repo paper (`Proxy Policy Steering.pdf`) formalizes this as a **relative density correction** —
`π̃ ∝ π_base·(π_task/π_ref)^γ` (Eq 2) — whose flow-matching realization is the **additive velocity residual**
`v_PPS = v_base + γ(v_task − v_ref)` (Eq 4), the flow generalization of LLM proxy-tuning (expert − anti-expert
logits). `steer_scale = γ`; optimal γ ∈ (0.4, 0.6); the reference proxy is on-policy-distilled from the base
(Eq 5) and the task proxy is initialized from it (Eq 6).

Two facts make this precise:

- **Diffusion ≡ flow matching.** velocity `v = α_t ε̂ − σ_t x̂` is interconvertible with the score
  `∇log p_t`; **DDIM = flow-matching Euler**; stochasticity is a tunable *churn* η (ODE→SDE). So PPS's
  velocity injection **is** a score composition `∇log p += λ(∇log p_task − ∇log p_ref)` — a Bayesian tilt —
  *valid only within a shared noise schedule and action normalization.*
- **Two interfaces, one property.** A flow policy can be steered by (i) **choosing/optimizing the initial
  noise** (DSRL — RL in latent noise space, black-box) or (ii) **adding a score/velocity term** (PPS, CFG).
  Both are gated by the **same** base property: the base must express behavioural variation as a **navigable
  noise→action map / score field**, not collapse to a point. So characterizing the base serves *both*.

**Unifying principle.**
> Steerability = **reachability**(good modes) × **controllability**(the reshaping) × **fidelity**(stays
> on-manifold), delivered through a flow process whose noise→action map is rich, smooth, and whose field is
> a true score composable with the steering term. A good base is **central and shallow** in behaviour
> space: close (in steering-budget terms) to many task optima, committed to none.

---

## 2. The base-policy scorecard

Each criterion is a measurement, and success is deliberately absent.

| # | criterion | probe | good base |
|---|---|---|---|
| 1 | **coverage** — mass near good actions | good-action recall @ε vs demos, over the **image of the noise prior** | recall > 0 on *all* tasks, even at 0 % own success |
| 2 | **structured uncertainty** | action entropy / #modes across the rollout | high at task-decision points, low on task-agnostic competence |
| 3 | **susceptibility / dose–response** | `steer_scale` sweep → realized shift, alignment | smooth, monotone, saturating; aligned with commanded direction |
| 4 | **on-manifold fidelity** | jerk, joint limits, base-likelihood vs `steer_scale` | stays feasible up to a usable budget λ* |
| 5 | **anisotropy** | eigen-spectrum of the response (cost Hessian / flow Jacobian) | soft directions = task DOF, stiff = collision/smoothness |
| 6 | **synergy** | base-only vs proxy-only vs base+steer on a *dense* metric | base+steer strictly dominates both |

The **deep version** collapses 3–5: *steerability is anisotropic susceptibility.* For our DIAL base the
soft-weighted MPPI update is a Boltzmann distribution `p ∝ exp(−cost/temp)`, so susceptibility = flatness of
the cost basin — measurable one-line via the softmax **effective sample size** `ESS = 1/Σwᵢ²`.

---

## 3. Literature grounding

- **DSRL** (*Steering Your Diffusion Policy with Latent-Space RL*) — steers by RL in the **noise space**;
  the steering handle is the **noise→action map**. If that map is degenerate (optimizer), there is nothing
  to steer. ⇒ criterion 1 measured as the *image of the noise prior*.
- **PDP** (*Parameterized Diffusion Policies*) — a **smooth, semantic behaviour manifold** turns diffusion
  "from stochastic diversity into a precise, optimizable steering tool," with interpolation and novel-behaviour
  synthesis. ⇒ criterion 2 upgraded: entropy must be *navigable*, not just present.
- **Diffusion↔Flow matching** — velocity ≡ score; DDIM = flow Euler; churn η controls mode re-opening.
  ⇒ criterion 3/4 grounded; the sampler-vs-optimizer axis is a *continuous knob*; composability is a hard
  prerequisite.
- **Diffusion Policy** (Chi et al.) — the archetype base: conditional denoising over action chunks, chosen
  *because* a **trained true score** preserves multimodality without collapse. ⇒ the gold standard our DIAL
  base approximates at inference.

---

## 4. The diagnostic we run first (no checkpoints needed)

The literature says the first thing to measure is not coverage-in-general but the **richness of the
noise→action map**, because it gates both steering interfaces. Three probes at one frozen state
(`vlm_mpc/tasks/probe_steerability.py`, `--task probe_steerability`):

- **P1 output diversity** — spread of *independent plans* from a fixed state. Terminal-EE spread
  `< 5 mm` = optimizer (collapsed); `> 20 mm` = sampler (multimodal). *(DSRL precondition.)*
- **P2 ESS vs denoise step** — effective sample size + plan change across the reverse loop. Does the
  high-entropy window *contain* `steer_step`, or does it commit early? *(churn / commitment timing.)*
- **P3 latent interpolation** — terminal EE as the initial noise is slerp-interpolated. A rich base traces a
  smooth behaviour span; a flat span means the latent map is un-navigable. *(PDP.)*

Each is swept across the collapse knobs the references identify: reverse-update mode
(`score_space` vs `ddim` = flow-Euler) × DIAL temperature. The money question: **does the flow-sampler path
+ higher temperature restore a rich, navigable map?** Output → `results/vlm_mpc/probe/`.

---

## 5. What the diagnostic actually found (P1–P3)

**Prediction (wrong).** I expected the base to be an *optimizer* — MPPI mean-seeking collapsing to the cost
min regardless of the seed, giving a near-degenerate noise→action map.

**Result** (`probe1`, one far-from-target state, fresh noise) **falsifies that.** Every config clears the
sampler bar: terminal-EE spread across independent plans is **27–115 mm**, and the latent span (P3, with the
MC noise averaged out) is **56–279 mm** — so the **initial noise genuinely drives behaviour**, meeting the
DSRL precondition. ESS *rises* (17 %→74–99 %) rather than pinning at 1/512: explore-early / consensus-late,
with each seed committing to a *different* mode. And `ddim` (flow-Euler) is **~2–4× richer** than
`score_space`, exactly as the diffusion↔flow equivalence predicts. So at the approach state the base is a
genuine, navigable sampler, not a collapsed optimizer.

**Regime map** (`probe2`, 3 regimes × {`score_space`, `ddim`}; `results/vlm_mpc/probe/probe2.*`).

| regime / update | d (m) | P1 term | P2 ESS 0→T | P3 span | verdict |
|---|---|---|---|---|---|
| far_fresh / score_space | 0.46 | 25 mm | 20→77 % | 97 mm | sampler |
| far_fresh / ddim | 0.46 | 113 mm | 12→87 % | 207 mm | sampler |
| near_fresh / score_space | 0.06 | 18 mm | 80→73 % | 87 mm | partial |
| near_fresh / ddim | 0.06 | 70 mm | 15→97 % | 258 mm | sampler |
| near_warm / score_space | 0.06 | 41 mm | **0→0 %** | 209 mm | sampler |
| near_warm / ddim | 0.06 | 84 mm | **0→2 %** | 190 mm | sampler |

1. **No collapse anywhere.** The base stays a sampler in every regime (P1 ≥ 18 mm even at d = 0.06 m); the
   "locks up when grabbing" hypothesis is *not* supported.
2. **Warm-start is the commitment knob, not proximity.** Both `near_warm` rows pin within-plan **ESS at the
   floor (~0 %)** across every reverse step, vs 70–97 % for fresh. That — not the near-target state — is the
   real "executes-the-mean" signature (warm-start is what the base normally runs).
3. **ESS ↔ diversity inversion.** Committed ≠ non-diverse: `near_warm` has ESS ≈ 0 yet the *highest* across-seed
   spread. High ESS (fresh) **averages** many samples → each run regresses to a similar mean → low spread;
   low ESS (warm) **latches** one seed-dependent candidate → high spread. So ESS and output-diversity move
   *opposite*, and **ESS alone is not a clean steerability scalar.**
4. `ddim` richer in every regime; the latent map is roughest at the approach-to-grab transition
   (`near_fresh` smoothness 0.38–0.49 vs 0.74 elsewhere).

**The Pareto tension, now measured.** Warm-start (added for temporal smoothness) **trades off local
steerability** — it collapses within-plan ESS to the floor. So the two interfaces diverge in the warm operating
regime: **DSRL latent steering still works** (the map stays navigable everywhere), but **PPS score-guidance has
the least within-step spread to tilt** exactly there. *Actionable:* steer where ESS is high — prefer `ddim`, and
place `steer_step` in a high-ESS phase / relax warm commitment during the steered window. Caveat: one object,
one near state (pre-grasp, d = 0.06 m); the ESS↔diversity inversion needs a second object to confirm.

**Residual limitations (independent of the above).** The field is still the score of our *geometric cost*,
not of the *data* distribution, and its velocity scale is not calibrated to a trained flow's — so
composability (#4) with pi0.5-trained proxies still needs alignment.

**Strategic fork** for "swap pi0.5 → VLM-DP as a steerable base":

- **A. pi0.5's trained flow is the base** (canonical PPS). Has the flow properties by construction; the
  scorecard just *characterizes* it. Low risk.
- **B. VLM-DP-native flow is the base.** Must be *given* the properties — either **run the MPC as a sampler**
  (DDIM/flow-Euler, higher churn/temperature, defer commitment, calibrate the field to the proxy space) or
  **distill** the MPC behaviour into a trained flow (data-grounded, natively steerable, optionally with a
  PDP-style behaviour parameter). Higher effort; the "proper" answer.

Note the collaborator's `sim_free_mpc` already uses the DIAL score to **steer a pi0 flow** — evidence that
the **flow field is the natural locus of steering**, and that a VLM-DP *base* should emit a flow field, not
an argmax plan.

---

## 6. The satisfaction bar

We are satisfied with the base when it clears the scorecard — **with success off the list**:

1. good-action recall > threshold on **all** tasks (breadth, not depth),
2. entropy concentrated and navigable at decision points,
3. monotone steering response with a non-trivial on-manifold budget λ*,
4. soft directions aligned with task DOF, stiff with safety DOF,
5. base+steer strictly dominates base-alone and proxy-alone on a dense metric.

This rules out the advisor's two traps: a 0 %-but-useless base fails #1; a task-specific base fails
#1-breadth and #2. The one honest tension is **susceptibility vs stability** — maximal steerability trades
off against staying on-manifold unsteered, so "good base" is a **Pareto frontier**, not a scalar.

---

## 7. Now vs. after checkpoints

- **Now (no checkpoints):** P1–P3 on the DIAL base + ESS/cost-curvature susceptibility. Answers "is our base
  even the right *kind of object* to steer," and whether the DDIM/flow path fixes it.
- **After base + proxies land:** the full CFG-scale dose–response (#3/#4), synergy (#6), and DSRL-style
  latent coverage (#1) with the real steer vector — the principled version of "just try steering."

---

## 8. Positioning against the PPS paper

The in-repo paper judges the base by *post-steering success* (+53 % avg on π₀.₅; π₀ 5→55 %), but its own
**Limitation 3** concedes the base's *intrinsic* quality is what decides whether PPS helps:

> "When the base model is substantially weaker than the proxy policy, this complementarity may diminish,
> and PPS may offer less benefit than training a specialist from scratch."

It gives the conceptual answer (the base must supply complementary broad priors — robustness, recovery,
on-manifold coverage) but **no intrinsic metric**. The scorecard (§2) and probes (§4) *are* that metric: they
characterize the precondition PPS assumes but never measures. The paper's failure analysis is direct evidence
for the split — steering cuts **incorrect-mode** failures −69 % (the proxy selects the mode) and **OOD-state**
failures −44 % (the base prior keeps the sampler in-support) — i.e. coverage/on-manifold (#1, #4) +
mode-selection = synergy (#6).

---

## 9. Score-space (diffusion) PPS — relaxing the flow-matching requirement

The paper's Limitation 2 notes PPS extends to diffusion via the score↔velocity correspondence. It does, and
the correction is *natively* a score difference: `∇log(π_task/π_ref)^γ = γ(s_task − s_ref)`. Since velocity and
score are linearly related per noise level, `v = a_t x_t + b_t s_t`, and the model-independent drift `a_t x_t`
**cancels in the difference**:
```
v_task − v_ref = b_t (s_task − s_ref).
```
So for a Gaussian flow/diffusion base **of the same family** score space is a *reparameterization*, not a
relaxation (DDIM = flow-Euler → identical trajectory).

**Genuine relaxations it buys:**
- **Diffusion-policy bases become eligible** (DDPM/ε-prediction, not just flow VLAs): `s_base = −ε̂_base/σ_t`.
- **Mismatched but known schedules** — score is the schedule-invariant currency; compose after **SNR alignment**
  of noise levels, so base and proxies need not share the flow schedule.
- **Stochastic samplers** (DDPM/ancestral, churn η>0) — the ODE/velocity form is the η=0 special case.

**What does NOT relax:** shared action representation + normalization; SNR/noise-level alignment (this *replaces*
schedule identity); the shared-approximation-error condition C2 (init task-from-ref); and re-distilling the
reference against a *new* base's score (C1, Eq 5) whenever the base is swapped.

**Recipe:**
```
s_base = −ε̂_base(x_t, t, o, l) / σ_t
t'     = SNR_match(t)                                   # align proxy noise level
s_PPS  = s_base + γ · ( s_task(x_t, t', o) − s_ref(x_t, t', o) )
x_{t-1} = DDIM_or_DDPM_step(x_t, s_PPS, t)              # DDIM recovers vanilla PPS
# train: ref by score-distillation from base (Eq 5); task init-from-ref, diffusion/flow-match on demos (Eq 6)
```

**Why it matters for VLM-DP-as-base:** our DIAL base already emits score-style updates
(`step_score_space` / `step_mbd_score`), so score-space PPS is its natural interface — we need not force the MPC
into a flow velocity, only expose its per-noise-level score with an SNR map (a *more achievable* form of
composability, #4). Residual gap: our score is the score of `exp(−cost/temp)` (a geometric cost), not of the
data distribution, so swapping VLM-DP in still requires **re-distilling the reference proxy against VLM-DP's own
score** for C1 to hold.
