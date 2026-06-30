# Sim-Free MPC Cost Function

This documents the hand-written cost currently used by `sim_free_mpc/costs.py`.
The policy still decodes OpenPI model-space action chunks before cost evaluation,
so the cost below is applied to real environment actions.

## Variables

```text
B        = number of sampled action chunks
H        = action horizon
q[b,t]   = first 7 decoded arm joint actions for sample b at horizon step t
g[b,t]   = decoded gripper command for sample b at horizon step t
p[b,t]   = FK end-effector position in world/env frame for sample b at step t
target   = current stage target position in world/env frame
q_now    = current observed 7 arm joint positions
```

For Droid gripper commands, `0` means open and `1` means close.

## Total Cost

```text
cost = reach_cost + smooth_cost + current_delta_cost + gripper_cost
       + orientation_cost
```

The current weights are:

```text
w_reach          = 25.0
w_terminal_reach = 40.0
w_smooth         = 0.03
w_joint_delta    = 0.005
w_gripper        = 0.1
w_orientation    = 0.25
```

## Reach Cost

```text
dist2[b,t] = ||p[b,t] - target||^2

reach_cost[b] =
    w_reach * mean_t(dist2[b,t])
    + w_terminal_reach * dist2[b,H-1]
```

Reason: this is still the main task term. The mean term makes the whole chunk
move toward the current subtask target. The terminal term gives extra pressure
for the end of the chunk to actually arrive near the target.

## Smooth Cost

```text
smooth_cost[b] =
    w_smooth * sum_t=1..H-1 ||q[b,t] - q[b,t-1]||^2
```

Reason: the score-space MPPI update can choose aggressive neighboring samples.
This term discourages jitter and sweeping motions across consecutive predicted
joint targets, which should reduce accidental hits on objects such as the pear.

This term only uses arm joints, not the gripper.

## Current Delta Cost

```text
current_delta_cost[b] =
    w_joint_delta * sum_t=0..H-1 ||q[b,t] - q_now||^2
```

Reason: this keeps sampled chunks local to the current robot state. It is a weak
trust-region term, not a task objective. Without it, a low-cost reach sample can
still make a large arm jump that moves the gripper through nearby objects.

This term only uses arm joints, not the gripper.

## Gripper Cost

First compute:

```text
d[b,t] = ||p[b,t] - target||
near[b,t] = 1 if d[b,t] < 0.07 else 0
```

For pickup stages:

```text
desired_gripper[b,t] = near[b,t]
```

This means open while approaching, close only when near the object.

For placement stages:

```text
desired_gripper[b,t] = 1 - near[b,t]
```

This means stay closed while carrying, open when near the placement target.

The cost is:

```text
gripper_cost[b] =
    w_gripper * mean_t((g[b,t] - desired_gripper[b,t])^2)
```

Reason: this is deliberately weak. It only biases the sampled chunks toward a
reasonable open/close phase. It should not overpower reach or decide contact
physics by itself.

## Orientation Cost

Let `R[b,t]` be the FK end-effector orientation. Define the tool axis as the
end-effector local +Z axis:

```text
tool_axis_world[b,t] = R[b,t] * [0, 0, 1]
desired_axis_world   = [0, 0, -1]
alignment[b,t]       = dot(tool_axis_world[b,t], desired_axis_world)
```

The cost is:

```text
orientation_cost[b] =
    w_orientation * mean_t(1 - alignment[b,t])
```

Reason: this weakly keeps the gripper pointing downward, but it does not
constrain yaw around the vertical axis. That is intentional: the robot should
still be free to rotate around the approach direction while avoiding unusual
sideways/upward wrist postures.

## Weight Task Target Stages

For `Isaac-Weight-Droid-Visuomotor-v0`, the target order is:

```text
if not grasp_pear:
    target = pear position + [0, 0, 0.03]
elif not pear_on_scale:
    target = scale placement target for pear
elif not grasp_apple:
    target = apple position + [0, 0, 0.03]
else:
    target = scale placement target for apple
```

The scale placement target uses measured scene geometry:

```text
scale_center = scale_root + [-0.0470425, 0.0, 0.0272255]

scale_place_z =
    scale_top_offset_z
    + object_half_height
    + placement_clearance_z

target = scale_center + [0, 0, scale_place_z]
```

Measured constants:

```text
scale_top_offset_z  = 0.0523800
apple_half_height   = 0.0376650
pear_half_height    = 0.0620635
placement_clearance = 0.0150000
```
