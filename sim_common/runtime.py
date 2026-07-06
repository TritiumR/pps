"""Shared launch scaffolding for the vlm_mpc task drivers.

Every driver needs the same two things before it can touch IsaacLab: the bundled IsaacLab packages on
``sys.path``, and a booted Isaac Sim app. ``main.py`` calls these in order
(``bootstrap_syspath`` -> ``add_launcher_args`` -> ``boot``); task modules then do their heavy imports
inside ``run()``, after the app is live. This replaces the ~15-line bootstrap header that used to be
copy-pasted into every standalone driver.
"""
import os
import sys


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
    """Boot the Isaac Sim app and return the ``simulation_app`` (close it in a ``finally``)."""
    import pinocchio  # noqa: F401  -- imported before Isaac Sim (load-order quirk the drivers relied on)
    from isaaclab.app import AppLauncher
    return AppLauncher(args).app
