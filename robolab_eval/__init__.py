"""Run the ReKep-MBD base on RoboLab manipulation tasks.

The RoboLab twin of `mujoco_eval`: same planner (sim_free_mpc SimFreeMPC), same cost
(vlm_dp CompositeCost), same grounding contract (vlm_dp.grounding.Grounding) and the same
stage machinery (vlm_dp.bridge.VlmDpBridge). Only the environment differs, so everything
task-agnostic is imported rather than re-implemented.
"""
