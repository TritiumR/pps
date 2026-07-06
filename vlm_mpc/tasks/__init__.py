"""vlm_mpc task drivers.

Each module here is a minimal task driver exposing ``add_args(parser)`` and ``run(args)``, dispatched by
``vlm_mpc.main`` via ``--task``. Top-level imports must stay light (stdlib only); anything that touches
IsaacLab / torch goes inside ``run()``, which is called after the Isaac app has booted.
"""
