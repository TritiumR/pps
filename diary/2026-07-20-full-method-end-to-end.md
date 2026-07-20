# VLM-DP / PPS — end-to-end method (Config B)

**One line.** A VLM turns an image + instruction into a geometric cost; a Model-Based-Diffusion (MBD)
sampling-MPC denoises action chunks against that cost as its "base policy"; two small geometry-conditioned
proxies steer the base's denoising in score space toward demonstrated behavior (Proxy Policy Steering),
without retraining the base.

**Provenance tags** used below: **[FE]** = ReKep-style VLM front-end (produces the task + cost; inherited, not
our contribution). **[BASE]** = MBD sampling-MPC base. **[PPS]** = our steering contribution. **[LOOP]** =
closed-loop control.

## Notation
- Action chunk `a ∈ R^(H×8)`: H horizon steps × (7 joint targets + 1 gripper channel).
- Reverse-diffusion schedule: steps k = K→0, cumulative noise ᾱ_k, β_k = 1 − ᾱ_k; x_k = the noisy chunk.
- **Observation `o`** — the state the policy conditions on. In vanilla PPS `o` is an RGB image; **here `o` is a
  geometric state**: object positions + extents, the active subtask's target + phase, payload, place target,
  robot joints + end-effector pose. (The base score is a function of `o`, invariant to pixels, so the proxies
  condition on `o` too.)
- `J(a; o)` = geometric cost, lower = better.

## Pipeline (input → output)

**1 · Task specification — image + instruction → subtasks + cost. [FE]**
A ReKep-style VLM reads the RGB image + instruction and emits a *grounding*: the objects (as keypoints/regions)
and an **ordered sequence of subtasks** (grasp obj_i → lift → place on obj_j), each with a geometric target and
optionally a relational constraint. The subtasks compile into a phase-gated cost `J(a; o)`: reach-to-target (or
ReKep relational constraints), grasp geometry (center / aperture / straddle / fingertip), carry-hold +
place-descent, and feasibility regularizers (collision / floor / orientation / smoothness / consistency). There
is no learned reward — J changes only through `o` (phase, targets) as the task advances.

**2 · Base policy π_base — MBD sampling-MPC (no learned network). [BASE]**
The base denoises an action chunk in which the score is a cost-weighted Monte-Carlo estimate. At step k, from x_k:
  1. propose N candidates  a_i = x_k + σ_k ε_i   (x_k-centered; σ_k annealed to the noise level),
  2. weight               w_i = softmax_i( −J(a_i; o) / λ ),
  3. posterior-mean clean action  x̂0 = Σ_i w_i a_i        (the MBD denoiser),
  4. score               s_base(x_k) = (√ᾱ_k · x̂0 − x_k) / β_k,
  5. reverse step         x_(k−1) = step(x_k, s_base).
After K steps, x_0 is the plan. An inner `iterations` refinement of x̂0 trades distribution for competence:
1 = the faithful posterior mean (distributional, steerable); large = a mode-seeking hill-climb (competent but
collapsed to a delta).

**3 · Proxies — small geometry-conditioned nets, trained once offline. [PPS]**
Two nets predict the clean action x0 from (o, x_k, k). We predict x0, not the raw score, because a score target
diverges as β_k→0; the score is recovered as s = (√ᾱ_k · x0 − x_k) / β_k.
  - Reference v_ref: distilled *on-policy* to match the base's own score along the base's denoising trajectory —
    L_ref = E ‖ s_ref(x_k,k) − s_base(x_k,k) ‖² .   ("What the base does.")
  - Task v_task: *initialized from v_ref*, then fine-tuned on demonstrations —  L_task = E ‖ x0_task − a_demo ‖² .
    ("How task supervision shifts behavior.")
Init-from-ref makes the two share architecture / initialization / approximation error, so their difference
isolates the task-induced change (the reason classifier-free guidance uses one network for cond/uncond).

**4 · PPS steering — combine at every denoise step. [PPS]**
        s = s_base + γ · ( s_task − s_ref ).
γ=0 recovers the base; larger γ tilts the denoising trajectory toward the demonstrated modes while shared base
behavior cancels. With x0-parameterization the correction is bounded —
        s_task − s_ref = (√ᾱ_k / β_k) ( x0_task − x0_ref ),
so the step targets x̂0_base + γ (x0_task − x0_ref) — no low-noise blow-up. Applied over a mid-noise interval. It
changes the sample only if the base is (i) distributional — mass to re-weight — and (ii) steerable — the
per-step tilt persists rather than being erased by the next step's re-optimization. Both are set by the sampler
(iterations, proposal annealing), not by the cost.

**5 · Control loop — receding-horizon replanning. [LOOP]**
One control chunk = one full reverse-diffusion denoise (K steps) producing a horizon-H action chunk x_0; only
the first k steps are executed, then the loop replans. Per chunk: update `o` (§6) and the active subtask (§7);
build J and bind the proxies to `o`; initialize x_K — **warm-start** (SDEdit: re-noise the shifted previous
plan and denoise only its last few steps) while carrying a payload, else fresh noise; run the K-step denoise
with the PPS-corrected score; decode x_0, clip per-step motion, execute k steps; carry the shifted plan
(warm-start seed) + a consistency reference into the next chunk.

**Diffusion at a replan (key point):** the reverse process is **memoryless across chunks** — each chunk denoises
from x_K anew, and s_base is re-estimated from fresh proposals against the *current* `o`; the previous chunk's
score/trajectory is discarded. Only *positional* information carries over (the warm-start seed and the
consistency reference), not diffusion state. So there are two nested restarts: within a chunk the score is
re-optimized at every one of the K steps; across chunks the whole denoise restarts. Steering tilts therefore
accumulate only *within* a chunk's K steps.

**6 · State estimation — how `o` stays current. [FE / LOOP]**
`o` is rebuilt every chunk so the base and proxies always condition on the current scene (closed loop).
Privileged setting (`--state gt`): object poses read straight from the simulator — a trivial upper bound. Real
setting (`--state real`): perceive at t=0; **while an object is held, propagate it rigidly with the gripper by
forward kinematics** (exact — and vision is worst then, since the hand occludes it); re-perceive on loss or
periodically; grasp status from a gripper-aperture sensor (proprioception), not a contact oracle. `o` feeds
**both** the cost J and the proxy conditioning, so a state change moves s_base and the proxy scores together —
and a perception error misleads both consistently.

**7 · Subtask sequencing with backtracking. [FE]**
Control walks the VLM's ordered subtask list, advancing when a subtask's constraint is satisfied (a committed
grasp, or a place release) and **backtracking to re-grasp when the payload is lost** — the ReKep-style
sequential-constraint mechanism. (Implementation detail: advances/regressions are debounced over a few
consecutive chunks; the held object's gripper-local offset is captured at grasp so the carry terms ride it
rigidly.) A subtask change is a **discrete edit to `o` between chunks** — new target, new phase, new payload —
which flips which cost terms are active (e.g. reach off / place-descent on once a payload is set) and re-points
the proxy conditioning. So at a subtask boundary the base's objective **and** the steering direction both change
discretely, and the next fresh (warm-started) denoise re-aims.

## How the parts couple
- The VLM sets the cost; the cost defines the base distribution; the base distribution is what the proxies
  distill and steer. Change the instruction → change subtasks → change `o` → change J → change π_base.
- The proxies never touch the base weights or the cost; they add only a per-step score residual. PPS is thus
  *training-free* adaptation: the base's manipulation priors are preserved and demonstrations are injected as a
  score residual.
- Proxies condition on geometry (`o`), not pixels, because the base score is geometry-driven.

## Distinctive vs vanilla PPS (steering pi0 / pi0.5)
1. The base is a VLM-cost-driven MBD *sampler*, not a learned VLA — we steer a sampling process. **[BASE]**
2. The observation `o` is geometric, not RGB. **[PPS]**
3. x0-prediction (not a score target) for stability at low noise. **[PPS]**

## Diagram (boxes → arrows)
    [RGB + instruction] → [VLM / ReKep front-end] → [grounding: objects + ORDERED SUBTASKS + keypoints]      [FE]
    [state estimation: gt poses  OR  perception + FK-while-held + re-perceive]  ──per chunk──→ [observation o] [FE/LOOP]
    [subtask sequencing: advance on constraint satisfied, backtrack on payload loss] ──sets phase/target──→ [o] [FE]
    [o: object poses, phase, target, payload, eef, joints] ─┬─→ [cost  J(a; o)]
                                                            └─→ [proxies  s_ref , s_task]                     [PPS]
    [cost J] → [MBD base: K-step reverse denoise, cost-weighted MC score s_base] (fresh each chunk; warm-started) [BASE]
    [proxies] ──(s_task − s_ref)──(× γ)──⊕──→ [corrected score s]   @ each denoise step k                     [PPS]
    [steered x_0] → [decode + clip → execute k of H steps in IsaacLab] → [update state estimate] ──loop──→ [o] [LOOP]
    (carried across chunks: warm-start seed + consistency reference — positional only, NOT diffusion state)
    Offline: [base rollouts] ──distill──→ [reference proxy] ──init + fine-tune on demos──→ [task proxy]        [PPS]

## Current status (honest footer, not part of the Methods)
Front-end + cost + MBD base + both proxies are built and individually validated. Faithful PPS on a
mode-collapsed base (iters=8) is null — the base erases the additive steer. The open experiment is faithful PPS
on a *steerable* base (iters=1 + annealed proposals): re-distil the reference on it, init the task from that
reference, and test whether steering moves success 0 → >0. Base task-competence (grasp/place) is a separate
axis, capped by a deliberately soft gripper drive; it is NOT a prerequisite for PPS.
