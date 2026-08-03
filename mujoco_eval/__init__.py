"""MuJoCo/MimicGen evaluation harness for the MBD sampling-MPC base.

Deliberately imports nothing: `eval` sets MUJOCO_GL and the EGL vendor before the first
mujoco import, which only works if importing the package has no side effects.

Entry points:
    python -m mujoco_eval.eval        one rollout
    python -m mujoco_eval.parallel    N seeds x W workers
    python -m mujoco_eval.bench.bench the G1 demo-compatibility bench
"""
