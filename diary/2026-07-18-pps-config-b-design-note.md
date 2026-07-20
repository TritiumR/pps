# PPS Config B — score-space steering of the ReKep+MBD base (design sync note)

**Date:** 2026-07-18 · **For:** Cory / Yixuan (owners of the score-space PPS pipeline)
**Goal:** steer **our ReKep+MBD base** (VLM ReKep keypoints + MBD over a geometric cost) with PPS proxies,
**in score space** — i.e. `s_PPS = s_base + γ(s_task − s_ref)`, `s_base` computed live from ReKep+MBD.

This is the paper's **diffusion extension** (score↔velocity correspondence, §6), *not* the flow-matching case
the paper actually validates. So there's no paper-blessed recipe — hence this note.

---

## 1. What's already in place (robot_dev4)

- Base ckpt `openpi/checkpoints/pytorch/pi05_droid_jointpos/` (+ `assets/droid/norm_stats.json`).
- Demos `data/weight/generated_dataset.hdf5`: 38 demos ×335 steps, all `success=True`. Per frame:
  `table_cam`/`wrist_cam` RGB (720×1280×3), eef pose, gripper, joints, **GT object poses** for all 6 objects.
  **No depth / point cloud.**
- Score-space eval seam `eval_steering.py:1144`: `combined = gamma_base·base_score + steer_scale·(task − ref)`,
  driven by `step_from_score`. Base score from `estimate_mbd_score(_action_prox)` (`sim_free_mpc/planner.py`).
- Proxy trainer `openpi/scripts/train_mpc_proxy_score_pytorch.py`; config `proxy_score_mpc_weight_jointpos`
  (ProxyScoreConfig, action_dim=8, horizon=15, DINOv3, shared norm_stats with the base ckpt).

## 2. The paper's recipe (Algorithm 1), for grounding

- `v_PPS = v_base + γ(v_task − v_ref)` (Eq. 4); product-of-experts relative correction (Eq. 2). Optimal γ ≈ 0.4–0.6.
- **Reference** `v_ref`: **on-policy distillation of the frozen base** — roll out the *base sampler* on task
  obs, match `v_base` **along the denoising trajectory** (Eq. 5). Ablation C1: endpoint-only supervision
  (`w/o vel`) drops 64→50%.
- **Task** `v_task`: **init from the reference**, finetune on **demonstration actions** (Eq. 6). Ablation C2:
  from-scratch task (`w/o tune`) drops to 48%.
- §4.2: proxies may condition on modalities the base doesn't (point clouds); only the **action representation
  + flow schedule** must be shared.

## 3. The gap — the current trainer does NOT match the paper for our base

1. Both proxies are trained on **MBD-score labels** (`estimate_mbd_score`), differing only by
   `--subtask_mode {heuristic, empty}` (heuristic = phase flags `grasp_pear`/`pear_on_scale`/`grasp_apple`
   set from object positions; empty = neutral). So the "task" proxy learns the **subtask-conditioned base
   score, not the demo actions** — it can't inject behavior the base lacks. (Paper: task = demo actions.)
2. Those labels use `cost_style=grasp_flow` → the **collaborator base**, not our ReKep cost. **There is no
   `rekep` cost_style** in `sim_free_mpc` (only grasp_flow/priority/ref_style/explore).
3. Labels are generated **offline over demo windows**, not from **on-policy base-sampler rollouts** (paper's
   reference is on-policy).
4. Our ReKep cost needs **keypoints** (RGB+depth front-end); demos have RGB but **no depth**. And our
   composite cost is **stage-driven** (needs a driver), vs the trainer's context-only per-frame call.

## 4. Proposed design (score-space instantiation of Algorithm 1)

Agreed direction (Jeremy): reference should distill our **real** base; task should be finetuned on **demo
actions**. Concretely:

- **Reference `s_ref`:** on-policy distill our **real ReKep+MBD base**. Roll out our base (vlm_base) sampler
  on the task observations, collect intermediate denoising states `x_ki`, train `s_ref` to match our base's
  score `s_base` **along the trajectory**. Needs running vlm_base (real keypoints + stage machine).
- **Task `s_task`:** init from `s_ref`, finetune on the **demo actions** (flow-matching, Eq. 6).
- **Eval:** `s_PPS = s_base + γ(s_task − s_ref)`, `s_base` live from ReKep+MBD.

## 5. Questions for you (you own this pipeline)

1. **Reference:** agreed it must be an **on-policy distillation of our real ReKep+MBD base** (not grasp_flow)?
   Is there existing machinery to roll out the vlm_base base and record `(x_ki, s_base)` along the denoise
   trajectory into the trainer's format, or does that need building?
2. **Task:** agreed it should be **finetuned on demo actions** (Eq. 6)? Is there a task-on-demo-actions path
   (`train_proxy_score_pytorch.py`?) we should use instead of `train_mpc_proxy_score`'s `--subtask_mode`?
3. **What is `--subtask_mode` for**, then — a deliberate score-space variant (phase-conditioning as a
   stand-in for demo-action task), an earlier experiment, or superseded?
4. **Score-space extension:** have you scoped/derived the score↔velocity instantiation for a **geometric-MBD
   base** (the paper leaves it as future work)? Any gotchas re the shared-schedule/action-rep requirement (§4.2)?
5. **Keypoints for labels:** demos have RGB but **no depth**, so the ReKep front-end can't run offline. For
   on-policy base rollouts we'd run the live base (which has depth). Is that the intended label path, or is
   there a plan for depth-free demo labeling?

## 6. Next steps (pending answers)

- Build the on-policy **reference-label generator** that runs our vlm_base base over the task observations and
  records `(x_ki, s_base)` along the denoise trajectory.
- Wire the **task-on-demo-actions** finetune (init from reference).
- **Smoke test** (few labels / few steps) end-to-end before the multi-hour full run.
