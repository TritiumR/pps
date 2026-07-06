"""Receding-horizon execution helpers shared by the DIAL-MPC drivers.

The drivers all run the same inner loop: plan -> execute one knot -> record -> shift the warm start,
optionally exiting on a tolerance. ``receding`` captures that; ``hold_pose`` holds a static pose while
toggling the gripper (the grasp/release close loops). Genuinely task-specific control (the weight task's
plan-once mode, the cost+gripper proximity latch) stays in the task modules.
"""
import torch


def tt(v, device="cuda:0"):
    """numpy array / list -> float32 tensor on ``device``."""
    return torch.tensor(v, device=device, dtype=torch.float32)


def hold_pose(env, n, grip_open, record_fn):
    """Hold the current arm pose for ``n`` steps (used for the gripper close/release), recording each."""
    q = env.q0()
    for _ in range(n):
        env.apply_arm(q, grip_open=grip_open)
        record_fn()


def receding(plan, env, ctx, grip_open, max_steps, record_fn, *, exec_knot, H, gen,
             done_fn=None, device="cuda:0"):
    """Common receding-horizon loop: re-plan each step, execute one knot, record, shift the warm start.

    Args:
        plan: a sampler ``plan(mean_a, q0, qd0, ctx, gen) -> (mean_a, q_traj, score, ess)``.
        env: the task env (provides ``q0()`` and ``apply_arm(q_arm, grip_open=...)``).
        ctx: the cost context, either a fixed value or a zero-arg callable re-read each step.
        grip_open: bool gripper command held over the loop.
        max_steps: loop cap.
        record_fn: ``record_fn(step, q_traj, score, ess)`` -- capture a frame / log.
        exec_knot: which planned knot to execute each step.
        H: horizon (for the warm-start shift and the knot clamp).
        gen: the torch RNG.
        done_fn: optional ``done_fn(step) -> bool`` early exit (e.g. tolerance reached).

    Returns:
        ``(last_step, reached)`` -- the final step index and whether ``done_fn`` fired.
    """
    qd0 = torch.zeros(7, device=device)
    mean_a = torch.zeros(H, 7, device=device)
    reached = False
    step = -1
    for step in range(max_steps):
        c = ctx() if callable(ctx) else ctx
        mean_a, q_traj, score, ess = plan(mean_a, env.q0(), qd0, c, gen)
        env.apply_arm(q_traj[min(exec_knot, H - 1)], grip_open=grip_open)
        record_fn(step, q_traj, score, ess)
        mean_a = torch.cat([mean_a[1:], torch.zeros(1, 7, device=device)], dim=0)
        if done_fn is not None and done_fn(step):
            reached = True
            break
    return step, reached
