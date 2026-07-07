"""Standalone diagnostics for the base -- run directly, not through the runner.

``sim_free_mbd`` runs the SimFreeMPC engine end to end on the IsaacLab weight task; ``probe_steerability``
measures the base's steerability. Each has a ``__main__`` block (``runtime.run_standalone``):

    python vlm_base/diagnostics/probe_steerability.py --grasp_obj pear ...
"""
