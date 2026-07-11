# Region, selector, steering, and affordance distributions — VLM-DP cost design discussion

**2026-06-28** — A long Socratic working session on the *conceptual* design of the VLM-DP grasp
cost: why a region needs (or doesn't need) a selector, how cost actually drives MPC action
selection, where semantics live, how the region is supplied, and how all of this generalizes to
multi-object steering and the affordance-distribution view. Companion to
[[2026-06-28-grasp-diagnosis-and-steerability]] (which covers the empirical grasp diagnosis and the
two-senses-of-steerable correction). This entry is the design reasoning, not code.

Running examples used throughout: the **mug handle** (connected convex bar), the **spoon** (two ends,
only one task-valid — GOAL.md's tested caveat), **pour-tea** (relational/multi-stage), the **weight**
task (multi-object pick-and-place).

---

## 1. How cost actually selects an action in DIAL-MPC (the mechanics)
Each control step: propose ~512 candidate **trajectories** (joint-accel chunks, H knots) → FK each →
the cost maps each candidate to **one scalar** (a keypoint enters only as a distance term, summed over
the horizon) → weights `w = softmax(−(J−J_min)/temperature)` → new mean = **cost-weighted average** of
candidates → anneal the sampling covariance, repeat → execute **one knot**, replan (receding horizon).
- "Minimize cost" = the mean is repeatedly pulled toward wherever the low-cost mass is. **It is an
  average, not an argmin.**
- Terms interact only by **weighted summation** into the scalar; the weights set the trade-offs.
- **ESS** (`1/Σwᵢ²`) = how many candidates meaningfully contribute. A single-keypoint spike collapses
  ESS→1 (degenerate); a region keeps a fat low-cost cluster (healthy ESS). This is the mechanical face
  of "narrow distribution."

## 2. Why a region needs a selector — and when it doesn't
- A **region cost is flat by design** (distance-to-segment = 0 along the bar). Flat = **indifferent** =
  *declines to pick a point*. That's GOAL.md's "optionality, not a decision." A point cost *does*
  determine the grasp — but it's the brittle, un-steerable spike we're fleeing.
- **Key nuance (MPPI averages, diffusion samples):** a diffusion policy commits to a mode by
  *sampling*; MPPI/DIAL commits by *averaging*. Consequence:
  - **Connected convex region (mug bar):** the average of valid points stays on the region → MPC lands
    a valid grasp (≈centroid, nudged by the smoothness term). **No explicit selector needed** — and the
    centroid is mid-bar, exactly off the top-stub spike that broke us. So the region alone may fix the
    mug.
  - **Disconnected / non-convex / task-mixed (spoon, or two objects):** averaging two modes lands in
    the **gap** (grasp of nothing / wrong end). **Selector mandatory** — not because the cost is flat,
    but because *MPPI can't average across separated modes.*
- **The selector can be a cost term iff the preference is geometric** (mug: "mid/low bar"). It **must be
  external** when the preference is **semantic** (spoon-end depends on task intent — no geometric term
  can decide it). So: region = *where a grasp is valid*; selector = *which valid grasp the task wants*,
  often not a geometric fact.

## 3. Multi-stage coupling, and the backtracking correction
- Grasp choice is genuinely **interdependent** with downstream stages (the right teapot grasp is the
  one the *pour* is feasible from). In a fully **joint** cost, selection would emerge from downstream
  value — "the cost is enough." But ReKep/DIAL **decompose** into stages + **finite horizon**, which
  severs that forward coupling per stage.
- **Correction (caught mid-discussion):** decomposition doesn't fully sever it — GOAL.md keeps ReKep's
  **backtracking**. But backtracking is **backward + reactive** (re-establish a *violated* path
  constraint, e.g. re-grasp on slip), **not forward anticipation**. It *recovers from* a bad grasp; it
  doesn't make the grasp stage *pour-aware*. And on backtrack ReKep re-solves the **same** grasp
  objective → same grasp (unless paired with a region → reactive trial-and-error). **Our port doesn't
  even implement backtracking** (`run_rekep_rollout.py:214` is a forward-only loop) — upstream's, not
  yet ported.
- So **backtracking and selector are complementary**: backtracking *recovers from* bad grasps; a
  forward selector / feasibility term *prevents* them. Much downstream-aware selection is **already**
  done by the VLM at authoring time (it picks "handle" for "pour" because it reads the whole task).

## 4. Where the semantics live (VLM vs keypoints)
- **Semantics live in the VLM, not the keypoints.** ReKep's keypoint proposal (DINOv2 → cluster) is
  **instruction-blind** — it clusters visually-distinct regions, not semantic parts. The VLM does *all*
  the semantic reasoning (stages, which keypoint = handle), post-hoc on the annotated image.
- Pattern = **perception proposes geometric candidates → VLM selects by commonsense.** MOKA pushes
  *object*-level semantics into perception (GroundingDINO open-vocab) but the grasp *point* is still a
  geometric candidate (contour FPS + centroid) the VLM picks via marked VQA.
- **The ceiling:** the VLM can only select a part perception **covered**. VLMs are unreliable at raw
  coordinates (SoM 25.7→86.4; MOKA: "multiple-choice > continuous locations"), which is *why* keypoints
  exist. So our grasp failure was **perception placement** (keypoint on the top stub) + **recall**
  (visibility) — **not** VLM reasoning (GPT correctly picked the handle).

## 5. How the region is supplied (the design space)
A region = **which part (VLM/semantic) ⊗ where in 3D (perception/geometry)**. The VLM must *not* supply
metric geometry. Three concrete options:
- **A. Points + VLM composes** — VLM selects the *set* of handle keypoints; driver builds the segment
  between them; cost = point-to-segment. Minimal; keypoints we already have bracket the bar. **Keypoint
  still necessary as the anchor.**
- **C. Points + VLM authors the region formula** — VLM writes the region as code over keypoints
  (ReKep's existing "compose from multiple keypoints" idiom). Slightly more faithful, more brittle.
- **B. Perception emits regions/masks** — open-vocab/part segmentation returns the handle *as a region*;
  VLM selects which mask. The deep, MOKA-flavored change; the only thing that fixes **recall**.
- **Proposal change is conditional:** *not* needed for **placement** (compose from existing points), *is*
  needed for **recall** (part not covered). Recommendation: test the cost+region path first (placement),
  reach for proposal changes only when recall bites — and the highest-leverage version is "propose
  regions, not points," which also removes the reconstruct-region-from-points step.

## 6. Multi-object steering and the affordance distribution
- **Steering only redistributes mass the base already has** (disjoint support → no effect). So if the
  base committed to one keypoint (object A), you **cannot steer to object B** — no mass there to amplify.
- For cross-object steerability, **B must be in the base's support** — the cost must place mass on *all*
  candidate grasps: an **affordance distribution over candidates** (high-to-least likely). This is the
  multi-object generalization of the single-object region (spread over a manifold → spread over the
  candidate set).
- **MPPI-averaging twist:** objects are *separated* modes → averaging lands in the gap (grasp of
  nothing). So a multimodal base buys *steerability* but is unsafe to execute by averaging → **steer
  early to commit to one mode, average late within the winner.**
- **Architectural contrast:** **PPS** steers a **VLA** base — multimodality is *free* (a learned policy
  already covers many objects). **VLM-DP** steers a **cost-induced** base — multimodality is *not* free;
  it must be **built into the cost** (affordance over candidates), else it's a spike and un-steerable.
- **Discrete vs continuous:** a wrong-*object* error is discrete → cheapest fix is **re-selection**
  (re-query VLM/selector), no multimodal base needed. **Steering** is for continuous/learned corrections
  *among modes the base already covers*.

## 7. Affordance distribution = VoxPoser — and why VoxPoser loses to ReKep
- The affordance distribution **is** VoxPoser's value map (GOAL.md borrows "the region idea" from
  VoxPoser, *as an analytic formula, not a dense grid, not its planner*). VoxPoser **greedy-descends**
  (commits to the nearest peak — sidesteps averaging but risks local minima); our DIAL **averages**.
- **Why VoxPoser fails vs ReKep — the density↔precision trade-off:** a value field is a *soft position
  attractor*; it **cannot express precise relations between keypoints**. The spread that makes it
  steerable makes it imprecise/non-relational. Pour-tea: VoxPoser gets the teapot *above* the cup but
  can't encode *"spout-to-handle vector tilted at angle θ"* (orientation is a separate coarse map) →
  spills. ReKep writes that as a relational keypoint constraint → pours. Plus coarse 100³ grid + greedy
  planner. ReKep's wins are exactly on **relational / multi-stage / precise-orientation** tasks.
- **The synthesis (and the answer to "do we inherit VoxPoser's failure?"):** take VoxPoser's region
  **spread** but in ReKep's **relational, analytic** form — a region *defined by keypoint relations*
  (e.g. `min` distance to keypoint-defined segments), not a dense field. Spread (steerable) **without**
  surrendering relational precision. You only inherit VoxPoser's failure if you adopt its dense,
  position-only field.

---

## Takeaways
1. **Region first, selector only when geometry can't decide.** For the connected-convex mug bar, MPC's
   averaging likely lands a valid mid-bar grasp with *no* selector — the cheapest test of the whole idea.
2. **The selector's true job is committing across *separated* modes** (spoon ends, objects) — because
   MPPI averages and can't blend separated modes. Within a connected region it's often unnecessary.
3. **Semantics = VLM, geometry = perception.** Never let the VLM emit coordinates; it names/selects,
   perception grounds. The region is built on keypoints/masks, not instead of them.
4. **Cross-object steerability requires the affordance over candidates in the cost** (VLM-DP doesn't get
   VLA-style free multimodality). For discrete wrong-object, prefer re-selection over steering.
5. **Keep the region relational + analytic** to get VoxPoser's spread without VoxPoser's imprecision.

## Open design choices (for when we build)
- **Selector for v1:** geometric place-relation (in-cost) vs MOKA VQA vs the demo residual (PPS).
- **Region source:** geometric construction from VLM-selected keypoints (no prompt change) vs VLM-
  authored region formula (prompt change) vs perception-emitted region mask (proposal change).
- **Proposal:** unchanged (placement) until recall fails; then multi-view or region/part proposal.
- **Next concrete experiment:** region-only on the mug (point-to-segment over the two handle keypoints,
  nothing else changed) — does DIAL's average land mid-bar and hold?
