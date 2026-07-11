# Why the robot goes crazy — diagnosis + fix (for the sim_free_mpc side)

**TL;DR:** it's **not your cost**. It's that the per-step **"push" toward the MPC target is unbounded**, and the
**decoded joint targets aren't clamped**. Bound the push **or** clamp the joints and it behaves.

## What we did
We ran **your exact cost + weights** (reach 25 / terminal 40 / smooth 0.03 / delta 0.005 / orient 0.25) through
**our** DIAL controller (execute-the-mean, in IsaacLab), and toggled two guards. Same cost, same seed, same pear
reach — only the guards change:

| run | per-step push | joint clamp | result |
|---|---|---|---|
| 1 | bounded | clamped | **sane** — reaches 0.5 cm |
| 2 | bounded | **none** | **sane** — 0.76 cm |
| 3 | **unbounded** | none | **crazy** — TCP 33→74→123→95→138 cm, never settles |

→ The cost is identical in all three. It only goes crazy when the push is **unbounded *and* unclamped**.
Either guard alone keeps it sane.

## What "the push" is
The MPC says the joints should be at **ā**; right now they're at **x_t**. Your conversion
`v_vlm = −(ā − x_t)/dt` is the push: *"cover the whole gap in one tick."* If **ā is far**, the gap is big →
the push is big → it overshoots and flings the arm. Switching velocity → score doesn't help: that's just a
different way to write the **same push**, so the magnitude problem carries over (and score actually grows
~1/σ_t near the end of denoising, so it can be *worse*).

## The fix (representation-agnostic)
1. **Bound / scale the push** — cap the step size, or scale `(ā − x_t)/dt` down to a sane magnitude
   (in score space: weight by the noise level σ_t). This is the **root fix**.
2. **Clamp the decoded joint targets** to `[q_lo, q_hi]` — a backstop at the action level that catches it
   regardless of velocity vs. score. (This is exactly what run 3 was missing.)

## Where these live in our code (reference)
- `vlm_mpc/sampler.py` → `make_accel_sampler.integrate()`:
  - **bound the push:** `eps = torch.clamp(eps, -accel_clip, accel_clip)` (caps the per-step drive),
  - **clamp the joints:** `q = torch.clamp(q + qd*dt, q_lo, q_hi)` (`q_lo/q_hi` read from the robot model).
- `vlm_mpc/_droid_collab_cost.py` → the reproduction: your cost+weights, with `--no-clamp` and `--accel_clip`
  flags that produced the three runs above.

## Fastest check on your side
Log the **decoded joint-target chunk** each step: if values go **out of range** or **jump**, it's the
push/clamp (not the cost). Try, in order: clamp the decoded targets; then rescale `v_vlm` (or σ_t-weight the
score). Set the steering term to 0 to isolate. FK is fine (your 0.171574 offset matches ours).
