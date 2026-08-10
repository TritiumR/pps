"""Launch the Isaac Sim application, which every other robolab_eval import depends on.

Nothing that touches `robolab`, `isaaclab` or `omni` may be imported before `AppLauncher` has
run, so this module holds the preamble and every entry point calls it first. Two ordering rules
that are load-bearing and not obvious:

* `import cv2` must precede the isaaclab imports (RoboLab's own scripts all do this); the reverse
  order crashes inside the OpenCV/Omniverse shared-library load.
* `enable_cameras` must be True before the app is built, or the tiled cameras produce nothing and
  every recorded frame is blank.
"""

from __future__ import annotations

import argparse


def launch(extra_args=None, argv=None):
    """Parse the launcher arguments, start the simulation app, and return (app, args)."""
    import cv2                                            # noqa: F401  (ordering, see docstring)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    for add in (extra_args or ()):
        add(parser)
    parser.add_argument("--num_envs", type=int, default=1,
                        help="RoboLab stepping is ~0.2 s/step at 1; batching is a later concern")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("-h", "--help", action="help")
    args = parser.parse_args(argv)
    args.enable_cameras = True
    app = AppLauncher(args).app
    return app, args
