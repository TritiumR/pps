"""PyTorch sampling-MPC controller stack for the VLM-DP work, in IsaacLab.

Mirrors hydrax/vlm_mpc but built on Isaac/PyTorch (see the project north star in pps/GOAL.md
and the architecture decision in memory `vlmdp-controller-architecture`). Step 1 = FK that
provably matches Isaac (fk.py + _fk_sanity.py).
"""
