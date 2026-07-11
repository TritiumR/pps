"""Report ground-truth task success for a rollout.

Uses each IsaacLab task's own ``task_done_<task_key>`` termination function, so the verdict
matches what the eval harness counts as success (placement / pour / insert goal met) rather than
a re-derived heuristic. Call at the end of a rollout (after the final release) to log whether the
full task completed and where the manipulated objects ended up.
"""
import importlib

import numpy as np


def report_task_success(env, task_key):
    """Print + return whether the task is currently in its success state. None on lookup failure."""
    try:
        mod = importlib.import_module(
            f"isaaclab_tasks.manager_based.manipulation.{task_key}.mdp.terminations")
        done_fn = getattr(mod, f"task_done_{task_key}")
        success = bool(done_fn(env).reshape(-1)[0].item())
    except Exception as exc:  # noqa: BLE001 - report, don't crash the rollout
        print(f"[TASK_SUCCESS task={task_key}] check failed: {exc}", flush=True)
        return None
    print(f"[TASK_SUCCESS task={task_key}] success={success}", flush=True)
    for name, obj in (getattr(env.scene, "rigid_objects", {}) or {}).items():
        pos = obj.data.root_link_pos_w[0].detach().cpu().numpy()
        print(f"    {name:12s} {np.round(pos, 3)}", flush=True)
    return success
