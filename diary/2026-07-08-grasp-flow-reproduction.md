# Reproducing the grasp_flow success (spi + B-spline + seeding)

**Goal.** Reproduce the collaborator's working `grasp_flow` weight-task rollout inside our
checkpoint-free `sim_free_mbd` diagnostic, and understand *why hers grasps-and-holds and ours
didn't*. Her reference video:
`1_...legacy_score_cost-grasp_flow_space-action_g1_n0.5_t0.7_clip0.15_success.mp4`.

---

## Starting point: the machinery was already matched

Two code audits established that our reproduction runs her *literal imported code*:

- **Engine** (`SimFreeMPC.step_score_space`, `_optimize_chunk`, the DIAL sampler, `PandaFK`) — the
  same functions, imported unchanged from `sim_free_mpc/`.
- **Decode** — a pure quantile un-normalize + delta→absolute reconstruction + truncate. No flow
  integration, no learned decoder, no gripper special-casing. In `--vlm_base` the paligemma
  transformer is never run (`state=None`); the base checkpoint supplies only the norm-stats.
- **Cost** — her exact `GraspFlowStateCost` (`cost_style=grasp_flow`).
- **Params** — `g1 n0.5 t0.7 clip0.15`, `legacy_score`, action space, 11 denoise steps, init noise —
  all read from her filename.

And yet ours stalled: the gripper hovered at ~0.35 and never closed, so the grasp never held.

## The chase: what was actually different

**1. The gripper is smoothed like a joint.** `_optimize_chunk` reduces the 15-step chunk to a few
control points and B-spline-expands them back, applied to *all 8 channels including the gripper*. A
grasp needs a sharp 0→1 snap at the seating instant; a smoothed gripper averages out to ~0.35 and
cannot commit, no matter how perfect the arm positioning (`gate=1.00`, reach 1mm). Proof: with
smoothing off, the gripper snapped to 1.0 and `grasp_pear` fired — but the *arm* then jittered and
diverged (no smoothing = unstable, 94cm scene disturbance).

So: smoothing gives a smooth arm but a dead gripper; no smoothing gives a live gripper but a
thrashing arm. Neither extreme works alone.

**2. `spi=2`.** Found from her video overlay (not her code, which was a methodology miss). Her rollout
re-plans every **2** env steps (`steps_per_inference=2`); we ran `exec_knot=8`. Tight 2-step feedback
catches the pear the instant it starts to slip; 8-step open-loop drops it before the next replan.
This is the piece that lets the arm stay stable *with* smoothing: the fast feedback does the
stabilizing, so the B-spline can smooth the arm without the gripper needing to be sharp within one
plan.

**3. Seeding.** Found by finally *reading* her rollout loop (`eval_steering.py:2714-2717`). She seeds
`random` + `np.random` + `torch` *before* `env.reset()`; our `sim_free_mbd` seeded only `torch`, and
`DroidEnv.reset()` runs in `__init__`. Result: the pear spawned at a *different pose every run* (the
non-determinism seen throughout), so we grasped a random, often harder, pear rather than her seed-1
pear. Fixed by seeding all three RNGs before `DroidEnv` construction.

## The answer

`spi=2` + engine-native B-spline + all-three-RNG seeding + her params:

```
sim_free_mbd.py --cost_style grasp_flow --real_stats --init noise --mode denoise --update score_space \
  --score_scale 1.0 --noise 0.5 --temperature 0.7 --joint_delta_clip 0.15 \
  --denoise_iters 11 --exec_knot 2 --interpolate --basis linear --knots 0 --guard --seed 1
```

`--interpolate --basis linear --knots 0` selects the engine's *native* B-spline (matching her config
path), not our `apply_horizon_basis` fixed-4-knot monkeypatch. The native knot count comes from
`interpolate_frequency/control_frequency` (her 5/40 defaults).

**Result** (`results/vlm_mpc/sim_free_mbd/grasp_flow_spi2_bspline.mp4`): `grasp_pear` 34/150,
**`pear_on_scale` 2/150 (first successful place)**, smoothness 0.54, jerk 66e-3,
**scene_disturbance 1.1cm** (vs 94cm with no smoothing). It grasps and places the pear cleanly,
dropping a few times mid-carry.

## Regime map (weight task, her params, seed 1)

| smoothing | spi | arm | gripper | outcome |
|---|---|---|---|---|
| B-spline (4/8 knot) | 8 | smooth | stuck ~0.35 | no grasp |
| none | 8 | jerky/diverges | commits 1.0 | grasps, drops |
| none | 2 | jerky (0.23) | commits | grasps 48x, no place, 94cm |
| **native B-spline** | **2** | **smooth (0.54)** | **commits** | **grasps + places, 1.1cm** |

## Methodology lesson

Most of this was grep-driven searching plus agent audits, not reading her implementation end-to-end.
That is why `spi` came from the video and seeding was hand-waved until the rollout loop was read
properly. Both real divergences were plainly visible in her rollout code.

## Next

- **Firmer grip** — scale the `GraspFlowCostWeights` gripper terms (`close_gripper` 2,
  `lift_gripper`/`place_gripper` 40) so it stops dropping mid-carry.
- **Longer episode** — raise `max_chunks` (rollout length) so it has time to place both objects
  (pear then apple).

## Aside: the pipeline (the actual downstream goal)

Separately this session the **ReKep VLM + MPC pipeline** (`vlm_base/main.py --ground rekep_fake|gt`)
reached a working grasp on GT grounding: a **lift stage** (grasp→lift→place, gripper held closed,
gated on a confirmed object rise) plus a **seating latch** (close the gripper on proximity, not the
lenient cost gate) made GT pick the pear (`grasped=Y, dz +3.9cm`, clean). The ReKep grasp is
perception-precision-limited: the VLM keypoint sits high on the object, and a bbox-center +
base-height grasp-center estimate helps but the masked point cloud is top-biased. The utensil task
(a fifth task: utensils into a holder) can reuse the existing `holder` task + `pen holder001` crock
as a template.
