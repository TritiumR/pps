"""FK sanity check: our FrankaFK must match IsaacLab's panda_hand pose to ~mm across the workspace.

The MPC cost plans on this FK and executes on Isaac, so the two must be the same kinematics. Isolated
check -- no sampler, no cost, no gripper. Prints FK_OK / FK_FAIL.

    python -m dial_mpc.main --task fk_sanity --n 64
"""

NAME = "fk_sanity"


def add_args(ap):
    ap.add_argument("--n", type=int, default=64, help="number of random configs to test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=2, help="settle steps after write_joint_state")


def run(args):
    import numpy as np
    import torch
    import gymnasium as gym
    from scipy.spatial.transform import Rotation as Rot

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
    from sim_common.fk import FrankaFK

    DEV = "cuda:0"
    TASK = "Isaac-Lift-Cube-Franka-v0"

    def quat_wxyz_to_R(qw):
        return Rot.from_quat(np.concatenate([qw[1:], qw[:1]])).as_matrix()

    def geodesic_deg(Ra, Rb):
        c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    cfg = parse_env_cfg(TASK, device=DEV, num_envs=1)
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.episode_length_s = 1.0e4
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()

    jn = list(robot.data.joint_names)
    arm_ids = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
    ph = list(robot.data.body_names).index("panda_hand")
    fk = FrankaFK(device=DEV)
    print(f"[fk] arm joint ids {arm_ids} | panda_hand body id {ph} | fk joints {fk.joint_names}", flush=True)

    lim_t = getattr(robot.data, "joint_pos_limits", None)
    if lim_t is None:
        lim_t = robot.data.soft_joint_pos_limits
    lim = lim_t[0, arm_ids].detach().cpu().numpy()
    lo, hi = lim[:, 0], lim[:, 1]
    margin = 0.10 * (hi - lo)
    rng = np.random.default_rng(args.seed)
    home = robot.data.default_joint_pos[0, arm_ids].detach().cpu().numpy()
    configs = [home] + [lo + margin + (hi - lo - 2 * margin) * rng.random(7) for _ in range(args.n - 1)]

    try:
        dt = float(env.sim.get_physics_dt())
    except Exception:
        dt = 1.0 / 60.0

    pos_errs, rot_errs = [], []
    for k, q in enumerate(configs):
        qt = torch.tensor(q, device=DEV, dtype=torch.float32)
        full_pos = robot.data.joint_pos.clone()
        full_pos[0, arm_ids] = qt
        robot.write_joint_state_to_sim(full_pos, torch.zeros_like(full_pos))
        robot.set_joint_position_target(full_pos)
        robot.write_data_to_sim()
        for _ in range(args.settle):
            env.sim.step(render=False)
            env.scene.update(dt)

        # Compare FK against the ACTUAL settled config (physics may push infeasible random configs).
        q_act = robot.data.joint_pos[0, arm_ids].detach()
        ph_pos = robot.data.body_pos_w[0, ph].detach().cpu().numpy()
        R_isaac = quat_wxyz_to_R(robot.data.body_quat_w[0, ph].detach().cpu().numpy())
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        R_root = quat_wxyz_to_R(robot.data.root_quat_w[0].detach().cpu().numpy())

        fk_pos_b, fk_R_b = fk.fk(q_act[None])
        fk_pos_w = root_pos + R_root @ fk_pos_b[0].detach().cpu().numpy()
        fk_R_w = R_root @ fk_R_b[0].detach().cpu().numpy()

        pe = float(np.linalg.norm(fk_pos_w - ph_pos))
        re = geodesic_deg(fk_R_w, R_isaac)
        pos_errs.append(pe)
        rot_errs.append(re)
        if k < 3 or pe > 0.005:
            print(f"[fk] cfg{k}: pos_err={pe*1000:.2f}mm rot_err={re:.2f}deg "
                  f"isaac={np.round(ph_pos,4)} fk={np.round(fk_pos_w,4)}", flush=True)

    pe = np.array(pos_errs) * 1000.0
    print(f"[fk] N={len(configs)} max_pos_err={pe.max():.2f}mm mean_pos_err={pe.mean():.2f}mm "
          f"max_rot_err={max(rot_errs):.2f}deg", flush=True)
    print("FK_OK" if (pe.max() < 2.0 and max(rot_errs) < 1.0) else "FK_FAIL", flush=True)
