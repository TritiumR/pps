"""The VLM-DP base: grounded manipulation on the collaborator's sim_free flow-MPC engine.

Runs a swappable grounding (``sim_common.grounding``) through ``sim_free_mpc``'s velocity/flow MPC --
adapted in ``sim_free_core.py`` + ``minimal_base_cost.py`` -- via the shared ``base_driver``. This is
the base we converge on and steer (PPS); the native DIAL alternative lives in ``dial_mpc/``.
``sim_free_mpc`` is used as-is; ``sim_free_core.py`` is the sole adapter.
"""
