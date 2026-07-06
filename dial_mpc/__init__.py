"""Native PyTorch DIAL sampling-MPC over a geometric cost (our controller, kept as an option).

A sampling-based MPC (``sampler.py``) that ranks candidate joint trajectories by a hand-built or
VLM-authored geometric cost (``costs.py``, ``voxposer_bridge.py``) evaluated purely from forward
kinematics -- no simulation rollout. Tasks under ``tasks/`` are dispatched by ``dial_mpc.main``. The
alternative to the sim_free base in ``vlm_base/``; both share IsaacLab infra in ``sim_common/``.
"""
