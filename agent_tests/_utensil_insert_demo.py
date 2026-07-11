"""Simple motion-planning demo: insert the spatula into the holder (GT poses, scripted IK-Rel phases).

No VLM front-end -- privileged GT object poses drive a hand-scripted pick-reorient-insert sequence on the
IK-Rel utensil task, reusing moka.isaac_control's proven drive_to_pose/grasp_at primitives:

  grasp spatula (top-down, contact-aware) -> lift -> reorient vertical (handle down) -> carry over the
  holder -> lower the handle into the crock -> release -> retract.

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh \
        agent_tests/_utensil_insert_demo.py --exp_name spatula_insert
"""
import os
import sys


def add_args(ap):
    ap.add_argument("--task", type=str, default="Isaac-Utensil-Droid-Visuomotor-IK-Rel-v0")
    ap.add_argument("--exp_name", type=str, default="spatula_insert")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--pos_tol", type=float, default=0.02)
    ap.add_argument("--grasp_handle_off", type=float, default=0.05, help="grasp point offset from center toward the handle tip (m)")
    ap.add_argument("--lift_steps", type=int, default=8, help="number of small increments for the gentle lift")
    ap.add_argument("--grasp_yaw_off", type=float, default=0.0, help="extra grasp yaw beyond closing across the handle (deg)")
    ap.add_argument("--grasp_z", type=float, default=0.0, help="z offset added to the spatula center for the grasp (m)")
    ap.add_argument("--lift_h", type=float, default=0.18, help="post-grasp lift (m)")
    ap.add_argument("--reorient_z", type=float, default=0.55, help="EE height while reorienting to vertical (m)")
    ap.add_argument("--hover_z", type=float, default=0.60, help="EE height carrying over the holder (m)")
    ap.add_argument("--insert_z", type=float, default=0.28, help="hard floor for the gentle insert descent (EE z, m)")


def run(args):
    import numpy as np
    import torch
    import gymnasium as gym
    from scipy.spatial.transform import Rotation as Rot

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from sim_common import overlay
    from moka.isaac_control import (ee_pose7, drive_to_pose, hold_gripper, descend_to_contact,
                                    GRIPPER_OPEN, GRIPPER_CLOSE)
    from rekep.video import write_video_h264

    def obj_pose(name):
        st = env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float64)
        qw = st[3:7]
        return st[:3], Rot.from_quat([qw[1], qw[2], qw[3], qw[0]])

    def yawed_top_down(quat, yaw):
        return (Rot.from_rotvec([0.0, 0.0, float(yaw)]) * Rot.from_quat(quat)).as_quat()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    if getattr(env_cfg.terminations, "success", None) is not None:
        env_cfg.terminations.success = None
    if getattr(env_cfg.terminations, "knife_dropping", None) is not None:
        env_cfg.terminations.knife_dropping = None

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(args.settle):
        env.step(hold)

    spat_pos, spat_R = obj_pose("spatula")
    hold_pos, _ = obj_pose("holder")
    spat_yaw = float(np.arctan2(spat_R.as_matrix()[1, 0], spat_R.as_matrix()[0, 0]))
    handle_dir = spat_R.apply([1.0, 0.0, 0.0])  # local +X = handle end
    print(f"[demo] spatula pos={spat_pos.round(3)} yaw={np.degrees(spat_yaw):.0f}deg handle_dir={handle_dir.round(2)}", flush=True)
    print(f"[demo] holder  pos={hold_pos.round(3)}", flush=True)

    cam = env.scene["table_cam"]
    frames = []

    def record(label):
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        frames.append(overlay.plain_frame(rgb, f"insert spatula -> holder | {label}"))

    ee0 = ee_pose7(env)
    top_down = ee0[3:]
    # Read the actual finger-closing axis (world) so we can align the grasp to the spatula instead of
    # guessing. Fingers close along (rfinger - lfinger); at the top-down reset a world-z yaw rotates it 1:1.
    fp = env.scene["ee_frame"].data.target_pos_w[0].detach().cpu().numpy()  # [ee, rfinger, lfinger]
    a0 = float(np.arctan2(fp[1][1] - fp[2][1], fp[1][0] - fp[2][0]))
    length_ang = float(np.arctan2(handle_dir[1], handle_dir[0]))
    # Grasp the SOLID handle -- the head is a slotted turner, so top-down fingers fall through its gaps.
    # Close ACROSS the handle (perpendicular to its length): the natural, secure rod grasp.
    yaw_cmd = length_ang + np.pi / 2 - a0 + np.radians(args.grasp_yaw_off)
    yaw_cmd = (yaw_cmd + np.pi / 2) % np.pi - np.pi / 2  # wrap to [-90,90] (parallel gripper is 180-deg symmetric)
    grasp_quat = yawed_top_down(top_down, yaw_cmd)
    print(f"[demo] finger_axis0_ang={np.degrees(a0):.0f} length_ang={np.degrees(length_ang):.0f} "
          f"yaw_cmd={np.degrees(yaw_cmd):.0f}", flush=True)
    # grasp on the handle; after reorienting, the handle tip (below the grasp) drops into the crock
    grasp_xyz = spat_pos + args.grasp_handle_off * handle_dir
    grasp_xyz[2] = spat_pos[2] + args.grasp_z

    # 1) grasp: hover -> descend to contact (open) -> close & settle -> GENTLE stepped lift so the grip
    # isn't jerked loose (a fast single lift shook the thin handle out of the fingers).
    gx, gy, gz = float(grasp_xyz[0]), float(grasp_xyz[1]), float(grasp_xyz[2])
    drive_to_pose(env, np.array([gx, gy, gz + args.lift_h, *grasp_quat]),
                  GRIPPER_OPEN, lambda: record("hover"), 150, args.pos_tol, 1.0)
    descend_to_contact(env, lambda: record("descend"), gripper_cmd=GRIPPER_OPEN, min_z=gz - 0.05)
    hold_gripper(env, GRIPPER_CLOSE, lambda: record("close"), 90)
    for dz in np.linspace(gz + 0.03, gz + args.lift_h, args.lift_steps):
        drive_to_pose(env, np.array([gx, gy, float(dz), *grasp_quat]),
                      GRIPPER_CLOSE, lambda: record("lift"), 80, args.pos_tol, 1.0)

    # 2) reorient to vertical, handle down: rotate -90 deg about world x (spatula length +y -> -z)
    vert_quat = (Rot.from_rotvec([-np.pi / 2, 0.0, 0.0]) * Rot.from_quat(grasp_quat)).as_quat()
    cur = ee_pose7(env)
    drive_to_pose(env, np.concatenate([[cur[0], cur[1], args.reorient_z], vert_quat]),
                  GRIPPER_CLOSE, lambda: record("reorient"), 150, args.pos_tol, 1.0)

    # 3) carry over the holder. After the -90 deg reorient about world x, the handle tip sits offset from
    # the gripper by (handle_dir_x * d_tip) in world x; aim so the tip centers on the crock opening.
    d_tip = 0.135 - args.grasp_handle_off  # grasp point -> handle tip (half-length minus the grasp offset)
    tgt_x = float(hold_pos[0]) - float(handle_dir[0]) * d_tip
    tgt_y = float(hold_pos[1])
    drive_to_pose(env, np.array([tgt_x, tgt_y, args.hover_z, *vert_quat]),
                  GRIPPER_CLOSE, lambda: record("carry"), 200, args.pos_tol, 1.0)

    # 4) lower the handle GENTLY into the crock until the tip contacts (crock bottom), then release --
    # contact-stopping avoids slamming the light spatula/holder when the tip meets resistance.
    descend_to_contact(env, lambda: record("insert"), gripper_cmd=GRIPPER_CLOSE, min_z=args.insert_z)
    hold_gripper(env, GRIPPER_OPEN, lambda: record("release"), 70)  # open fully + let the spatula settle

    # 5) retract gently, straight up, so the open fingers clear the spatula instead of dragging it back out
    cur = ee_pose7(env)
    for dz in np.linspace(cur[2] + 0.04, args.hover_z, 5):
        drive_to_pose(env, np.array([cur[0], cur[1], float(dz), *vert_quat]),
                      GRIPPER_OPEN, lambda: record("retract"), 60, args.pos_tol, 0.5)

    out_dir = os.path.join(repo, "results", "utensil_demo")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.exp_name}.mp4")
    write_video_h264(frames, out, args.fps)
    print(f"[demo] DONE -> {out} ({len(frames)} frames)", flush=True)
    print(f"[demo] final spatula pos={obj_pose('spatula')[0].round(3)}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim_common import runtime
    runtime.run_standalone(add_args, run, "spatula insertion motion-planning demo")
