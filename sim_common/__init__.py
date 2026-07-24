"""Shared IsaacLab infrastructure for the base (``vlm_base``) and native (``dial_mpc``) stacks.

Depends on neither stack (imports only rekep / isaaclab / torch), so both build on it without coupling.
Holds the env wrappers (``envs/``: ``DroidEnv`` / ``LiftEnv`` on a shared ``IsaacLabEnv`` base), forward
kinematics (``fk``: ``FrankaFK`` + ``WorldFK``), scene/pose geometry (``geometry``), the launch harness
(``runtime``), frame overlay (``overlay``), the ReKep numpy->torch constraint bridge (``constraints``), and
the grounding layer (``grounding/``: contract + GT/ReKep sources + mask perception + fake-VLM stub).
"""
