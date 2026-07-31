"""Self-contained demo: pick up the spatula and insert it into the holder.

A single standalone script with no imports from the project's own packages (moka / sim_common / rekep) --
the helpers they'd provide (IK-Rel control, a text overlay, an H.264 writer, the Isaac boot) are inlined
below. It drives the IK-Rel utensil task from privileged ground-truth object poses through a scripted
sequence:

    grasp the spatula handle -> gentle lift -> reorient vertical (handle down) ->
    carry over the holder -> lower the handle into the crock -> release -> retract

Requires Isaac Sim + IsaacLab + a GPU and the task's assets (the spatula ships with the task; the
kitchen / knife / holder USDs are those the capsule / knife / holder tasks already use). Run:

    docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh utensil_insert_demo.py --exp_name demo

Output: results/utensil_demo/<exp_name>.mp4
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def bootstrap_syspath():
    """Put the repo root + the bundled IsaacLab source packages on sys.path; return the repo root."""
    repo = os.path.dirname(os.path.abspath(__file__))
    packages = ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic")
    for path in [repo, *(os.path.join(repo, "IsaacLab", "source", p) for p in packages)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo


# --- IK-Rel control (inlined from moka.isaac_control) ---------------------------------------------
# The IK controller acts in the robot BASE frame, so world position/orientation errors are rotated into
# that frame and issued as a 6-DoF arm delta plus a 1-DoF binary gripper: [dx,dy,dz,drx,dry,drz,grip].

GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 1.0


def ee_pose7(env, i=0):
    """End-effector pose as [x, y, z, qx, qy, qz, qw] in the world frame."""
    ee = env.scene["ee_frame"]
    pos = ee.data.target_pos_w[i, 0].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = ee.data.target_quat_w[i, 0].detach().cpu().numpy().astype(np.float64)
    return np.concatenate([pos, np.roll(quat_wxyz, -1)])  # wxyz -> xyzw


def robot_base(env, i=0):
    """Robot base as (world position, scipy Rotation)."""
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[i].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = robot.data.root_quat_w[i].detach().cpu().numpy().astype(np.float64)
    return pos, Rot.from_quat(np.roll(quat_wxyz, -1))


def _apply(env, arm_delta6, gripper_cmd, record_fn):
    """Issue one [arm delta (6), gripper (1)] action, step the sim, and record a frame."""
    action = np.concatenate([arm_delta6, [gripper_cmd]]).astype(np.float32)
    env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
    record_fn()


def drive_to_pose(env, target_pose7, gripper_cmd, record_fn, max_steps, pos_tol, rot_gain):
    """Servo the EE toward an absolute target pose (position-primary), issuing base-frame deltas."""
    for _ in range(max_steps):
        cur = ee_pose7(env)
        pos_err = target_pose7[:3] - cur[:3]
        if np.linalg.norm(pos_err) < pos_tol:
            break
        _, base = robot_base(env)
        arm = np.zeros(6)
        arm[:3] = np.clip(base.inv().apply(pos_err) / 0.5, -0.2, 0.2)
        if rot_gain > 0:
            rotvec = (Rot.from_quat(target_pose7[3:]) * Rot.from_quat(cur[3:]).inv()).as_rotvec()
            arm[3:] = np.clip(rot_gain * base.inv().apply(rotvec) / 0.5, -0.2, 0.2)
        _apply(env, arm, gripper_cmd, record_fn)


def hold_gripper(env, gripper_cmd, record_fn, steps):
    """Hold the EE still (zero arm delta) while opening/closing the gripper."""
    for _ in range(steps):
        _apply(env, np.zeros(6), gripper_cmd, record_fn)


def descend_to_contact(env, record_fn, gripper_cmd=GRIPPER_OPEN, max_steps=400,
                       probe=0.04, stall_eps=0.0025, stall_n=6, min_z=None):
    """Lower straight down (fixed xy + orientation) until the EE stops descending (contact) or hits min_z."""
    quat, xy = ee_pose7(env)[3:], ee_pose7(env)[:2]
    stalled = 0
    for _ in range(max_steps):
        cur = ee_pose7(env)
        _, base = robot_base(env)
        arm = np.zeros(6)
        arm[:3] = np.clip(base.inv().apply(np.array([xy[0], xy[1], cur[2] - probe]) - cur[:3]) / 0.5, -0.2, 0.2)
        rotvec = (Rot.from_quat(quat) * Rot.from_quat(cur[3:]).inv()).as_rotvec()
        arm[3:] = np.clip(base.inv().apply(rotvec) / 0.5, -0.2, 0.2)
        _apply(env, arm, gripper_cmd, record_fn)
        new_z = float(ee_pose7(env)[2])
        if min_z is not None and new_z <= min_z:
            break
        stalled = stalled + 1 if (cur[2] - new_z) < stall_eps else 0
        if stalled >= stall_n:
            break
    return ee_pose7(env)


# --- Overlay + video (inlined from sim_common.overlay + rekep.video) ------------------------------
def label_frame(rgb, text):
    """Camera RGB -> BGR with a green text label burned in (for the video)."""
    img = np.ascontiguousarray(rgb)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def write_video_h264(frames, out_path, fps):
    """Write BGR frames to an H.264 mp4 (temp mp4v -> ffmpeg transcode; keep the raw mp4v on failure)."""
    if not frames:
        return
    tmp = out_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    try:
        subprocess.run(cmd, check=True)
        os.remove(tmp)
    except Exception as exc:  # noqa: BLE001
        print(f"[video] ffmpeg transcode failed ({exc}); keeping {tmp}")
        os.replace(tmp, out_path)


# --- Demo -----------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--task", type=str, default="Isaac-Utensil-Droid-Visuomotor-IK-Rel-v0")
    ap.add_argument("--exp_name", type=str, default="spatula_insert")
    ap.add_argument("--settle", type=int, default=15, help="sim steps to let the scene settle after reset")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--pos_tol", type=float, default=0.02, help="EE position tolerance for drive_to_pose (m)")
    ap.add_argument("--grasp_handle_off", type=float, default=0.05, help="grasp offset from spatula center toward the handle (m)")
    ap.add_argument("--grasp_yaw_off", type=float, default=0.0, help="extra grasp yaw beyond closing across the handle (deg)")
    ap.add_argument("--grasp_z", type=float, default=0.0, help="z offset added to the grasp point (m)")
    ap.add_argument("--lift_h", type=float, default=0.18, help="post-grasp lift height (m)")
    ap.add_argument("--lift_steps", type=int, default=8, help="increments for the gentle lift")
    ap.add_argument("--reorient_z", type=float, default=0.55, help="EE height while reorienting to vertical (m)")
    ap.add_argument("--hover_z", type=float, default=0.60, help="EE height carrying over the holder (m)")
    ap.add_argument("--insert_z", type=float, default=0.28, help="hard floor for the gentle insert descent (EE z, m)")


def run(args, repo):
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  -- registers the utensil task
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    def obj_pose(name):
        """Ground-truth (world position, scipy Rotation) of a scene object."""
        st = env.scene[name].data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float64)
        return st[:3], Rot.from_quat(np.roll(st[3:7], -1))

    def yawed_top_down(quat, yaw):
        """Top-down gripper orientation rotated by `yaw` about world z."""
        return (Rot.from_rotvec([0.0, 0.0, float(yaw)]) * Rot.from_quat(quat)).as_quat()

    # Build the env; drop the success/drop terminations so the demo runs to completion.
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    for term in ("success", "knife_dropping"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    env.reset()
    hold = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for _ in range(args.settle):
        env.step(hold)

    spat_pos, spat_R = obj_pose("spatula")
    holder_pos, _ = obj_pose("holder")
    handle_dir = spat_R.apply([1.0, 0.0, 0.0])  # spatula local +X points along the handle
    length_ang = float(np.arctan2(handle_dir[1], handle_dir[0]))
    print(f"[demo] spatula pos={spat_pos.round(3)} handle_dir={handle_dir.round(2)}  "
          f"holder pos={holder_pos.round(3)}", flush=True)

    cam = env.scene["table_cam"]
    frames = []

    def record(label):
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        frames.append(label_frame(rgb, f"insert spatula -> holder | {label}"))

    # Grasp orientation: read the gripper's actual finger-closing axis (rfinger - lfinger) and yaw it to
    # close ACROSS the handle. The head is a slotted turner (fingers fall through it), so we grasp the
    # solid handle. Wrap to [-90, 90] since a parallel gripper is 180-deg symmetric (avoids a wrist-limit
    # tilt on large yaws).
    top_down = ee_pose7(env)[3:]
    fp = env.scene["ee_frame"].data.target_pos_w[0].detach().cpu().numpy()  # [ee, rfinger, lfinger]
    finger_ang = float(np.arctan2(fp[1][1] - fp[2][1], fp[1][0] - fp[2][0]))
    yaw = length_ang + np.pi / 2 - finger_ang + np.radians(args.grasp_yaw_off)
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    grasp_quat = yawed_top_down(top_down, yaw)
    grasp_xyz = spat_pos + args.grasp_handle_off * handle_dir
    grasp_xyz[2] = spat_pos[2] + args.grasp_z
    gx, gy, gz = float(grasp_xyz[0]), float(grasp_xyz[1]), float(grasp_xyz[2])

    # 1) Grasp: hover -> descend to contact -> close & settle -> gentle stepped lift. The stepped lift
    # matters: a single fast lift jerks the thin handle out of the fingers.
    drive_to_pose(env, np.array([gx, gy, gz + args.lift_h, *grasp_quat]),
                  GRIPPER_OPEN, lambda: record("hover"), 150, args.pos_tol, 1.0)
    descend_to_contact(env, lambda: record("descend"), gripper_cmd=GRIPPER_OPEN, min_z=gz - 0.05)
    hold_gripper(env, GRIPPER_CLOSE, lambda: record("close"), 90)
    for z in np.linspace(gz + 0.03, gz + args.lift_h, args.lift_steps):
        drive_to_pose(env, np.array([gx, gy, float(z), *grasp_quat]),
                      GRIPPER_CLOSE, lambda: record("lift"), 80, args.pos_tol, 1.0)

    # 2) Reorient to vertical, handle down: rotate -90 deg about world x (spatula length +y -> world -z).
    vert_quat = (Rot.from_rotvec([-np.pi / 2, 0.0, 0.0]) * Rot.from_quat(grasp_quat)).as_quat()
    cur = ee_pose7(env)
    drive_to_pose(env, np.concatenate([[cur[0], cur[1], args.reorient_z], vert_quat]),
                  GRIPPER_CLOSE, lambda: record("reorient"), 150, args.pos_tol, 1.0)

    # 3) Carry over the holder. After the reorient the handle tip hangs offset from the gripper by
    # handle_dir_x * d_tip in world x, so aim upstream of that to center the tip on the crock opening.
    d_tip = 0.135 - args.grasp_handle_off  # grasp point -> handle tip (spatula half-length minus grasp offset)
    tgt = [float(holder_pos[0]) - float(handle_dir[0]) * d_tip, float(holder_pos[1]), args.hover_z]
    drive_to_pose(env, np.array([*tgt, *vert_quat]),
                  GRIPPER_CLOSE, lambda: record("carry"), 200, args.pos_tol, 1.0)

    # 4) Insert: lower gently until the tip contacts (contact-stopping avoids slamming the light spatula
    # and knocking the holder), then release.
    descend_to_contact(env, lambda: record("insert"), gripper_cmd=GRIPPER_CLOSE, min_z=args.insert_z)
    hold_gripper(env, GRIPPER_OPEN, lambda: record("release"), 70)

    # 5) Retract gently, straight up, so the open fingers clear the spatula instead of dragging it out.
    cur = ee_pose7(env)
    for z in np.linspace(cur[2] + 0.04, args.hover_z, 5):
        drive_to_pose(env, np.array([cur[0], cur[1], float(z), *vert_quat]),
                      GRIPPER_OPEN, lambda: record("retract"), 60, args.pos_tol, 0.5)

    out = os.path.join(repo, "results", "utensil_demo", f"{args.exp_name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video_h264(frames, out, args.fps)
    print(f"[demo] DONE -> {out} ({len(frames)} frames); "
          f"final spatula pos={obj_pose('spatula')[0].round(3)}", flush=True)


def main():
    repo = bootstrap_syspath()
    import pinocchio  # noqa: F401  -- must be imported before Isaac Sim (load-order requirement)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="spatula insertion motion-planning demo")
    add_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    args = parser.parse_args()

    app = AppLauncher(args).app
    ok = False
    try:
        run(args, repo)
        ok = True
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    finally:
        try:
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0 if ok else 1)  # force-exit to free the GPU on the shared machine


if __name__ == "__main__":
    main()
