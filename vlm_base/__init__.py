"""Grounded manipulation on the sim_free flow-matching MPC engine.

Runs a swappable grounding (``sim_common.grounding``) through ``sim_free_mpc``'s velocity-space MPC via the
shared ``base_driver``, scored by a config-driven ``base_cost.CompositeCost`` (terms in ``cost_terms``).
This is the base policy that PPS steers; the native DIAL variant lives in ``dial_mpc/``. ``sim_free_mpc`` is
used unchanged -- ``sim_free_core.py`` is the sole adapter. ``main.py`` is the runner (``--task`` scene,
``--ground`` front-end); ``diagnostics/`` holds standalone diagnostic scripts.
"""
