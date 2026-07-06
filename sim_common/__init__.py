"""Shared IsaacLab infrastructure for the base (``vlm_base``) and native (``dial_mpc``) stacks.

Depends on neither stack (imports only rekep / isaaclab / torch), so both build on it without
coupling. Holds the env wrappers (``droid_env``, ``isaac_env``), forward kinematics (``fk``), the
launch harness (``runtime``), frame overlay (``overlay``), the ReKep numpy->torch constraint bridge
(``np_shim``), the grounding contract + sources (``grounding/``), scene geometry (``scene_extents``),
and the fake-VLM helpers (``weight_fake_vlm``).
"""
