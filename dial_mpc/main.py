"""Single entry point for the dial_mpc task drivers (our native DIAL sampling-MPC).

    /isaac-sim/python.sh -m dial_mpc.main --task droid_weight --vlm real --exp_name ...

``--task`` selects a module under ``dial_mpc/tasks/``; that module contributes its own CLI args
(``add_args``) and does the work (``run``) after the Isaac app boots. To add a task: implement
``tasks/<name>.py`` with ``add_args(parser)`` + ``run(args)`` and register it in ``TASKS`` below.
"""
import argparse
import importlib
import os
import sys

# Repo root on sys.path before importing submodules (robust to being run as a script path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_common import runtime

TASKS = {
    "droid_weight": "dial_mpc.tasks.droid_weight",
    "rekep_dial": "dial_mpc.tasks.rekep_dial",
    "droid_weight_free": "dial_mpc.tasks.droid_weight_free",
    "droid_grasp": "dial_mpc.tasks.droid_grasp",
    "franka_lift": "dial_mpc.tasks.franka_lift",
    "fk_sanity": "dial_mpc.tasks.fk_sanity",
    "rekep_pick": "dial_mpc.tasks.rekep_pick",
    "rekep_frontend": "dial_mpc.tasks.rekep_frontend",
    "voxposer_pick": "dial_mpc.tasks.voxposer_pick",
    "voxposer_frontend": "dial_mpc.tasks.voxposer_frontend",
}


def main():
    runtime.bootstrap_syspath()  # repo root + bundled IsaacLab packages (idempotent)

    # First pass: read --task so we can ask just that task for its CLI args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=sorted(TASKS))
    known, _ = pre.parse_known_args()
    task = importlib.import_module(TASKS[known.task])  # light import (heavy work is in run())

    parser = argparse.ArgumentParser(description=f"dial_mpc task: {known.task}")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    task.add_args(parser)
    runtime.add_launcher_args(parser)
    args = parser.parse_args()

    app = runtime.boot(args)
    ok = False
    try:
        task.run(args)
        ok = True
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        try:
            app.close()
        except Exception:
            pass
        # Isaac's close() can spin/hang after teardown, leaving a GPU-hogging zombie on the shared
        # machine. Force-exit so the process actually dies and frees the GPU (exit 1 on failure).
        os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
