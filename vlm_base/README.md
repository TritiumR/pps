# vlm_base

Training-free grounded manipulation on IsaacLab. A swappable grounding front-end decides *what/where*
(which object to grasp, where to place, the per-stage objective); a flow-matching sampling-MPC
(`sim_free_mpc`, used unchanged) solves *how*. This is the base policy that PPS steers; the native DIAL
variant is in `dial_mpc/`, shared IsaacLab infrastructure in `sim_common/`.

## Running

Requires Isaac Sim + IsaacLab (the script launches `AppLauncher`). From the repo root:

```bash
# --task   scene:     weight | pot | tea | capsule   (-> Isaac-<Task>-Droid-Visuomotor-v0)
# --ground front-end: gt | rekep_fake | rekep_real
python -m vlm_base.main --task weight --ground gt
```

```bash
--task weight --ground gt            # GT oracle pick-and-place (pear -> scale)
--task weight --ground rekep_fake    # ReKep grounding, canned VLM (no LLM call)
--task tea    --ground rekep_real    # ReKep grounding, real VLM
--task weight --ground gt --exp_name demo --seed 3
```

Output: `results/vlm_mpc/vlm_base/<task>/<exp_name>.mp4` + printed metrics (`exp_name` defaults to
`<task>_<ground>`). All parameters live in `configs/base.yaml`; the CLI adds `--exp_name` / `--seed` /
`--joint_delta_clip` overrides. Object identity (grasp/place objects) is task metadata in `task_prompts.json`.

> **jeremy**: `cd docker && docker compose exec pps bash`, then `unset DISPLAY` (headless EGL — the container's `DISPLAY=:12` makes Isaac try GLX and fail with `GLXBadFBConfig`), then run the commands (`python` is aliased to Isaac Sim's interpreter).

## Layout

```
vlm_base/
├── main.py              # base runner: --task scene, --ground front-end -> rollout + video
├── base_driver.py       # grounding-agnostic driver: runs a Grounding on the MPC, records video + metrics
├── base_cost.py         # CompositeCost: config-selected weighted sum of cost terms
├── cost_terms.py        # atomic cost-term registry (TERMS); each term self-gates on the stage
├── sim_free_core.py     # adapter to sim_free_mpc (checkpoint-free decode, MPC build, smoother, warm-start)
├── metrics.py           # rollout metrics: motion smoothness, scene disturbance, reach
├── configs/
│   └── base.yaml        # all tunable parameters (run / grounding / engine / sampler / cost)
└── diagnostics/         # standalone scripts (run by path, not --task)
    ├── sim_free_mbd.py       # run the SimFreeMPC engine end to end on the weight task
    └── probe_steerability.py # measure the base's steerability
```

## Cost terms

`CompositeCost` sums `weight * cost_terms.TERMS[name]` for each `{name: weight}` in `cost.terms`. Terms
self-gate — one that does not apply to the current stage returns zero. Add a term: write `def my_term(I)`
in `cost_terms.py`, decorate with `@register("my_term")`, add `my_term: <weight>` under `cost.terms`.

| Term | Role |
|---|---|
| `reach`, `terminal_reach` | squared TCP→target distance (non-constraint stages) |
| `rekep_subgoal`, `rekep_path` | ReKep relational subgoal / running path constraints (constraint stages) |
| `smooth`, `joint_delta`, `consistency` | motion smoothness, trust region, stay near the previous plan |
| `orientation` | keep the tool axis pointing down |
| `straddle`, `collision`, `floor` | fingertips bracket the grasp point, keepout from objects, stay above the table |
