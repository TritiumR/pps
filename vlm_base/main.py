"""Single entry point for the vlm_base task drivers (the VLM-DP base on the sim_free engine).

    /isaac-sim/python.sh -m vlm_base.main --task minimal_base --config vlm_base/configs/base.yaml

``--task`` selects a module under ``vlm_base/tasks/``; that module contributes its own CLI args
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
    "minimal_base": "vlm_base.tasks.minimal_base",
    "sim_free_mbd": "vlm_base.tasks.sim_free_mbd",
    "probe_steerability": "vlm_base.tasks.probe_steerability",
}


def main():
    runtime.bootstrap_syspath()  # repo root + bundled IsaacLab packages (idempotent)

    # First pass: read --task so we can ask just that task for its CLI args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=sorted(TASKS))
    known, _ = pre.parse_known_args()
    task = importlib.import_module(TASKS[known.task])  # light import (heavy work is in run())

    parser = argparse.ArgumentParser(description=f"vlm_base task: {known.task}")
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
