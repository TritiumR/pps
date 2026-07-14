# sim_free_mpc handoff

This directory contains the FK/cost-only MPC used as the MBD base score for
score-space PPS.  For the current path, focus on:

- `planner.py`: `SimFreeMPC`, including action-proximity MBD score methods.
- `action_space.py`: decodes normalized model actions to real robot joint actions.
- `ddim.py`: DDIM alpha schedule used by MPC and score policies.
- `dial_sampler.py`: MPPI/DIAL optimizer.
- `costs_*.py`: geometric task costs.  For weight, `costs_grasp_flow.py` is the newest staged cost.

## Main API

```python
from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig

planner = SimFreeMPC(
    base_policy,
    SimFreeMPCConfig(
        task_name="weight",
        num_samples=512,
        iterations=8,
        noise=0.5,
        temperature=0.2,
        action_dims=8,
        cost_style="grasp_flow",
        optimize_space="action",
        ddim_num_train_timesteps=100,
    ),
)

score, info = planner.estimate_mbd_score_action_prox(
    x_t,           # [1, horizon, action_dim], normalized action space
    policy_inputs, # base_policy.obs_to_input(raw_obs)[1]
    context,       # scene/task state for the cost
    iteration=i,
    num_iterations=num_steps + 1,
)
```

`planner.step_mbd_score_action_prox(...)` is the direct-step variant if you want it to
return the next `x_t` instead of only the score.

`mbd_score_action_warm` keeps `mbd_score_action_prox` unchanged, but initializes
each replan after the first from the previous normalized action trajectory. The
trajectory is shifted by the number of executed actions and padded by repeating
its final action. Its arm deltas are then rebased from the previous joint state
to the current joint state so the absolute joint targets stay unchanged; the
first replan in each rollout still starts from Gaussian noise.

## What the module does

`mbd_score_action_prox` samples clean action candidates around the current noisy
action `x_t`, evaluates each candidate with FK + geometric cost, forms a weighted
clean estimate `x0_hat`, then converts it to a DDIM score:

```text
proposal_std = mpc_noise * sqrt(1 - alpha_bar_t)
score = (sqrt(alpha_bar_t) * x0_hat - x_t) / (1 - alpha_bar_t)
```

The cost itself is not the score.  The cost only chooses `x0_hat`.

## Required inputs

`policy_inputs` should come from the same policy used by the planner:

```python
_, policy_inputs = base_policy.obs_to_input(raw_obs)
```

`context` is a dict with scene state.  Useful keys:

- `task`
- `subtasks`
- `joint_pos`, `joint_vel`, `eef_pos`, `eef_quat`, `gripper_pos`
- `robot_root_pos`, `robot_root_quat`
- `objects`, e.g. `{"pear": {"pos": ..., "quat": ...}}`

See `eval_steering.py::build_mpc_context` for the live IsaacLab version.

## Constraints

- Batch size is currently 1.
- `optimize_space` must be `"action"` for `mbd_score_action_prox` and
  `mbd_score_action_warm`.
- MPC, task score policy, and ref score policy must share the same norm stats.
- `ddim_num_train_timesteps` must match between MPC and score policies.

## Reference eval command

No-steer MPC-only rollout with the action-proximity MBD score path:

```bash
python eval_steering.py \
    --task Isaac-Weight-Droid-Visuomotor-v0 \
    --prompt "put pear and apple on the scale" \
    --vlm_base \
    --no_steer \
    --mpc_update mbd_score_action_prox \
    --mpc_cost grasp_flow \
    --mpc_optimize_space action \
    --gamma_base 1 \
    --num_steps 10 \
    --mpc_ddim_train_timesteps 100 \
    --mpc_num_samples 512 \
    --mpc_iterations 8 \
    --mpc_noise 0.5 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.15 \
    --task_num_steps 800 \
    --task_debug \
    --mpc_debug \
    --seed_start 1 \
    --seed_end 2 \
    --steps_per_inference 4 \
    --interpolate
```

Related external files:

- `eval_steering.py`: runtime wiring for `--mpc_update mbd_score_action_prox`.
- `openpi/scripts/train_mpc_proxy_score_pytorch.py`: generates ref score labels from these MPC methods.
