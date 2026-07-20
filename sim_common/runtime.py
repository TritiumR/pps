"""Shared launch scaffolding for the standalone driver scripts.

Before a driver can touch IsaacLab it needs the bundled IsaacLab packages on ``sys.path`` and a booted
Isaac Sim app. ``run_standalone`` does that dance -- ``bootstrap_syspath`` -> ``add_launcher_args`` -> ``boot``
-> run -> force-exit -- so each entrypoint's ``__main__`` stays a few lines. The Isaac imports live inside the
functions because ``sys.path`` is only set up by ``bootstrap_syspath``.
"""
import argparse
import os
import sys
import traceback


def bootstrap_syspath():
    """Prepend the repo root and the bundled IsaacLab source packages to ``sys.path``.

    Idempotent (each path is added only once). Returns the repo root.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    src = os.path.join(repo, "IsaacLab", "source")
    for pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        sp = os.path.join(src, pkg)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return repo


def add_launcher_args(parser):
    """Add Isaac ``AppLauncher`` CLI args plus the headless + cameras defaults every driver uses."""
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)


def boot(args):
    """Boot the Isaac Sim app and return the ``simulation_app``.

    Note: ``run_standalone`` never calls ``.close()`` on it -- Kit's teardown is slow and frequently
    hangs (orphaning the process while it still holds the GPU), so drivers hard-exit instead.
    """
    import pinocchio  # noqa: F401  -- import before Isaac Sim (load-order requirement)
    from isaaclab.app import AppLauncher
    return AppLauncher(args).app


def run_standalone(add_args, run, description=None):
    """Boot Isaac and run one ``(add_args, run)`` entrypoint as a standalone script.

    The boilerplate every entrypoint's ``__main__`` needs -- bootstrap sys.path, parse its args + the launcher
    args, boot, run, and force-exit to free the GPU on the shared machine -- kept here so scripts don't
    copy-paste it. Used by the base runner (``vlm_base/main.py``) and the diagnostics alike.

    Exit is a hard ``os._exit`` with NO ``app.close()``: by then ``run`` has written its outputs, and Isaac
    Sim's Kit teardown is slow and frequently hangs -- leaving the process alive holding the GPU long after
    the work is done. Skipping it exits immediately; the OS reclaims the GPU on process death.
    """
    bootstrap_syspath()
    parser = argparse.ArgumentParser(description=description)
    add_args(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    boot(args)   # boots Isaac Sim; the returned simulation_app is intentionally never closed (see docstring)
    ok = False
    try:
        run(args)
        ok = True
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if ok else 1)   # skip Kit teardown -> no post-DONE hang; GPU reclaimed on process death
