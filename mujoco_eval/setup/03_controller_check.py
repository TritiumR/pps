"""Check JOINT_POSITION command semantics and validate an absolute-target shim."""

import inspect
import json

import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.controllers import joint_pos

JOINT = 2


def make_env(ctrl_cfg):
    """Create the robosuite environment for controller tests."""
    return robosuite.make(
        "Stack",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )


def qpos_of(env):
    """Return the current arm joint positions."""
    robot = env.robots[0]
    return np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes])


def main():
    """Inspect and test the JOINT_POSITION controller."""
    print("robosuite:", robosuite.__version__)
    ctrl_cfg = load_controller_config(default_controller="JOINT_POSITION")
    print("--- shipped JOINT_POSITION config ---")
    print(json.dumps(ctrl_cfg, indent=2))
    print("--- JointPositionController.set_goal source ---")
    print(inspect.getsource(joint_pos.JointPositionController.set_goal))
    has_delta_flag = "control_delta" in inspect.getsource(
        joint_pos.JointPositionController
    )
    print("has 'control_delta' anywhere in JointPositionController:", has_delta_flag)

    env = make_env(ctrl_cfg)
    env.reset()
    adim = env.action_dim
    low, high = env.action_spec
    print(f"action_dim={adim}  low={low}  high={high}")
    ctrl = env.robots[0].controller
    print("controller output range:", ctrl.output_min, ctrl.output_max)

    q0 = qpos_of(env)
    for _ in range(40):
        env.step(np.zeros(adim))
    drift = np.abs(qpos_of(env) - q0).max()
    print(
        f"[T1] zero-action drift over 40 steps: {drift:.5f} rad "
        f"({'HOLD -> delta-consistent' if drift < 0.05 else 'MOVED -> not a hold'})"
    )

    env.reset()
    action = np.zeros(adim)
    action[JOINT] = 0.5
    trace = []
    for _ in range(150):
        env.step(action)
        trace.append(qpos_of(env)[JOINT])
    trace = np.array(trace)
    print(
        f"[T2] constant action 0.5 on joint {JOINT}: "
        f"q[10]={trace[10]:.4f} q[50]={trace[50]:.4f} "
        f"q[100]={trace[100]:.4f} q[149]={trace[149]:.4f}"
    )
    late_slope = (trace[-1] - trace[-50]) / 50
    print(
        f"     late slope {late_slope:.5f} rad/step -> "
        f"{'RAMP (delta/integrating)' if abs(late_slope) > 1e-4 else 'CONVERGED (absolute-like or joint limit)'}"
    )

    env.reset()
    q_target = qpos_of(env).copy()
    q_target[JOINT] += 0.4
    errs = []
    for _ in range(100):
        q = qpos_of(env)
        delta = q_target - q
        a = np.zeros(adim)
        a[: len(delta)] = np.clip(delta / ctrl.output_max, -1.0, 1.0)
        env.step(a)
        errs.append(np.abs(qpos_of(env) - q_target).max())
    print(
        f"[T3] shim to absolute target: err[0]={errs[0]:.4f} "
        f"err[25]={errs[25]:.4f} err[99]={errs[99]:.4f} rad"
    )
    print(f"     converged: {errs[-1] < 0.01}")

    print("--- rate facts ---")
    print("control_freq (Hz):", env.control_freq)
    print("sim timestep (s):", env.sim.model.opt.timestep)
    print(
        "model timestep from robosuite:",
        env.model.mujoco_model.opt.timestep
        if hasattr(env.model, "mujoco_model")
        else "n/a",
    )
    env.close()


if __name__ == "__main__":
    main()