# Mug-handle grasp diagnosis + the two senses of "steerable"

**2026-06-28** — Write-up of (1) why the rung-3 VLM-DP mug grasp reaches the handle but never *holds*,
established by a tolerance sweep, and (2) a conceptual clarification of "steerable" that the
debugging surfaced — convergence (mode-finding) vs. PPS-style distribution-reshaping. The second
point reframes why GOAL.md's **region + selector** is the fix, and corrects a conflation I made.

Context: `vlm_mpc/_rekep_vlm_pick.py` (rung 3) = camera keypoints → GPT-4o constraint → np→torch shim
→ DIAL-MPC, grasping the nucleus mug by the handle (side_ny camera, yaw=180). See
[[2026-06-28-rekep-implementation-outline]].

---

## 1. Why the grasp misses — it's the grasp *target*, not perception/reach/orientation

Isolated by elimination, all on the **same** cached keypoints + GPT constraint + seed:

- **Not perception** — camera + DINOv2 + GPT-4o correctly find and select the handle (`grasp_keypoints=[0]`).
- **Not reach distance** — tolerance sweep on DESCEND (`--tol`, the early-exit threshold):

  | `tol` | reached | what happened | held |
  |---|---|---|---|
  | 0.03 (3 cm) | 2.95 cm | handle *outside* the finger span | ❌ |
  | 0.012 (1.2 cm) | 0.79 cm | good pre-grasp, **momentary** grip | ❌ |
  | 0.0 | 0.24 cm | textbook pre-grasp, but **shoved the mug** ~14 cm | ❌ |

  Reaching closer is conclusively NOT the lever: stop short → miss; reach all the way → disturb the
  object. `tol=0` even displaced the mug (grasp keypoint moved `[0.513,−0.065,0.098]→[0.408,−0.14,0.013]`).
- **Not orientation** — the new grasp-yaw term (`make_rekep_grasp_cost`) drove the closing axis radial
  (`|fy·r|` 0.42→0.93 in the A/B), yet `held` stayed False either way.
- **It IS the grasp target (keypoint height).** The VLM keypoint sits at the **top of the handle**
  (z≈0.098), where it merges into the rim — only a short stub of bar there, so the closed fingers
  can't *retain* it through the lift (grip → momentary lift → slip). Contrast **rung 2**, which HELD
  at the same ~3 cm reach error because its keypoint was on the **open mid-bar** (z=0.046).

Net: the binding constraint is **where on the handle the grasp targets**, not how close or how oriented.

## 2. The grasp-yaw + straddle cost term (built, validated, kept)

`vlm_mpc/costs.py: make_rekep_grasp_cost` adds two FIXED `J_feas` terms (VLM `J_task` untouched),
parameterized by the VLM-selected `grasp_idx`: a **yaw** term (closing axis radial = along horizontal
handle→center, since the bar is ~vertical so "⊥ bar" is degenerate) and a **straddle** term (fingertips
flank the bar). Validated: shape/finite/degenerate (`agent_tests/_grasp_cost_check.py`), and the A/B
confirms it reorients the gripper (0.42→0.93). It does NOT flip `held` here — because the target is the
top stub, and a side note: the straddle term is **trivially satisfied** when reach can't center the TCP
on the bar (handle outside the finger span passes "keep fingers off the bar"). Keep the term; it's a
legitimate generalization of the cube `make_grasp_cost`, just not the lever for *this* failure.

## 3. The clarification: two different "steerable"s

While debugging I claimed "the single-keypoint reach cost *was steerable*" because DIAL drove the TCP
from ~10 cm to 0.24 cm. That's true but it's the **wrong sense** of the word for GOAL.md / PPS:

- **Convergence / mode-finding** (what I showed): can the annealed sampler descend `J` to its optimum?
  A property of cost *shape* (density). ReKep's **norm** constraint `‖ee−kp‖` is a smooth dense bowl →
  yes, DIAL descends it. (Good: a small piece of GOAL.md's second gating risk is retired — the reach
  term is dense/steerable-to-descend, *not* the flat-then-spike worst case.)
- **PPS-style steerability** (the intended sense): the base distribution `p(a) ∝ exp(−J/λ)` must have
  **spread/multimodality** for a steering residual `s = s_base + α(s_task − s_ref)` to *reshape* it.
  You can't steer a spike — a bounded residual barely moves a near-delta.

**A single keypoint gives a unimodal, near-delta `p(a)`**: one target → all good actions pile on the
same place → no *task-meaningful* optionality (the only spread is incidental kinematic redundancy, not
"grasp here or there"). So in the PPS sense a single keypoint is exactly the **un-steerable** case —
which is precisely what GOAL.md line 44 means: *"a single-point objective makes p(a) sharply peaked and
therefore un-steerable."* My descent reading was the wrong lens for that sentence.

### Why this reframes the region + selector fix (GOAL.md)
- A single point → spike → no mass to steer. **(un-steerable)**
- A **region** (e.g. the bar segment between the two handle keypoints) → a **ridge** of near-equal-cost
  actions → `p(a)` now has width → the residual/selector has mass to redistribute. **(steerable)**
- A bare region is "optionality, not a decision" (GOAL.md's tested **spoon** caveat): without a
  **selector** it drifts to the wrong end → the **region must be paired with a selector** (a place
  relation now, MOKA-style VQA, or the demo residual later).

Our grasp failure *is* this picture: the single keypoint gave one mode (top stub), DIAL converged to
it, and there was **no other mass to steer toward** the good mid-bar grasp. Region + selector supplies
both halves (spread to steer + the decision of where on the ridge to land).

## 4. Standing recommendation / next step
Stop tuning reach (conclusively not the lever). The principled fix is **region + selector**, not a bare
bar-segment (which would risk reproducing the wrong-end drift). MVP-cost read so far (GOAL.md line 65):
**reach over a region + a selector + orient-down** is the core; **yaw** is a refinement, **straddle**
was inert at the distances reached. Contact-rich *hold* may ultimately be the residual's job per GOAL.md.
