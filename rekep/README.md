# rekep

ReKep (Relational Keypoint Constraints) on IsaacLab. A VLM (GPT-4o) proposes keypoints on the scene and
writes per-stage relational constraint functions over them; a solver turns each stage's constraints into
the next end-effector subgoal pose, which the arm reaches via IK-Rel control (grasp/release per the
generated metadata). The keypoint proposer + constraint generator are ported from [upstream ReKep](https://github.com/huangwl18/ReKep); the
IsaacLab glue (camera, tracking, visualization) is native. Also serves as the grounding front-end for
`vlm_base` / `dial_mpc` via `sim_common.grounding`.

## Running

Requires Isaac Sim + IsaacLab, and an OpenAI API key for GPT-4o — constraint generation reads it from the
`OPENAI_API_KEY` environment variable ([get a key](https://platform.openai.com/api-keys); add it to your
shell profile to persist):

```bash
export OPENAI_API_KEY=sk-...
```

From the repo root:

```bash
# --task     IsaacLab task id (IK-Rel control):  Isaac-<Weight|Pot|Tea|Capsule>-Droid-Visuomotor-IK-Rel-v0
# --task_key scene key: weight | pot | tea | capsule   (-> instruction from task_prompts.json)
python rekep/scripts/run_rekep.py --task Isaac-Tea-Droid-Visuomotor-IK-Rel-v0 --task_key tea
```

```bash
--plan-only    # stop after grounding (keypoints + constraints); skip the solve + rollout
--use_cached   # reuse keypoints/constraints from a prior run (no LLM call)
```

```bash
# keypoint proposal + tracking overlay only (no LLM, no rollout):
python rekep/scripts/visualize_keypoints.py --task Isaac-Tea-Droid-Visuomotor-v0 --task_key tea
```

Output: `results/rekep/<exp_name>/` — proposed-keypoint images, per-stage constraint `.txt` files,
`metadata.json`, and `<exp_name>_rekep_rollout.mp4` (`exp_name` defaults to `--task_key`).

> **jeremy**: enter the container first — `cd docker && docker compose exec pps bash` — then run the commands as-is (inside, `python` is aliased to Isaac Sim's interpreter).

## Layout

```
rekep/
├── grounding.py             # front-end: propose keypoints from one camera frame + workspace bounds
├── keypoint_proposal.py     # keypoint proposer: DINOv2 patch features -> per-mask KMeans
├── constraint_generation.py # GPT-4o writes the per-stage subgoal/path constraint functions
├── keypoint_tracking.py     # KeypointTracker: registers keypoints to rigid bodies, tracks via GT poses
├── solvers.py               # SubgoalSolver: constraints -> next end-effector subgoal pose
├── isaaclab_helpers.py      # IsaacLab glue: camera depth+seg readout, workspace bounds from scene
├── rekep_viz.py             # project 3D keypoints to pixels, draw the overlay
├── video.py                 # H.264 mp4 writer
├── utils.py                 # config loading, constraint-fn loading, grasping cost fn
├── configs/
│   └── default.yaml         # keypoint proposer / constraint generator (VLM) / solver params
├── prompts/
│   └── prompt_template.txt  # the GPT-4o constraint-writing prompt
└── scripts/                 # standalone entry points
    ├── run_rekep.py             # ground + solve + roll out (--plan-only, --use_cached)
    └── visualize_keypoints.py   # keypoint proposal + tracking overlay (no LLM)
```
