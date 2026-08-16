"""IsaacLab single-env wrappers on a shared base.

``base.IsaacLabEnv`` factors out the common plumbing; ``droid.DroidEnv`` (weight task, Robotiq) and
``lift.LiftEnv`` (lift cube/mug, Franka) are the concrete envs. Import the concrete env directly
(``from sim_common.envs.droid import DroidEnv``) so a driver only pulls the IsaacLab task it needs.
"""
