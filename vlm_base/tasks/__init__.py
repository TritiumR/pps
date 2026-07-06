"""vlm_base task drivers.

Each module exposes ``add_args(parser)`` + ``run(args)``, dispatched by ``vlm_base.main`` via
``--task``. Top-level imports stay light (stdlib only); IsaacLab / torch imports go inside ``run()``,
which is called after the Isaac app has booted.
"""
