"""Throwaway: render a Droid task's settled scene to a video (see the layout)."""
import os
import sys


def add_args(ap):
    ap.add_argument("--task", type=str, default="Isaac-Holder-Droid-Visuomotor-v0")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--exp_name", type=str, default="holder_scene")


def run(args):
    import numpy as np
    from sim_common.envs.droid import DroidEnv
    from sim_common import overlay
    from rekep.video import write_video_h264

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    E = DroidEnv(device="cuda:0", task=args.task)
    scene_objects = list(getattr(E.env.scene, "rigid_objects", {}) or {})
    print(f"[scene] task={args.task} objects={scene_objects}", flush=True)
    for n in scene_objects:
        try:
            pos, R = E.object_pose(n)  # (pos[3], R[3,3])
            yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            print(f"[scene]   {n}: pos={np.round(pos, 3)} yaw={yaw_deg:.0f}deg", flush=True)
        except Exception as e:
            print(f"[scene]   {n}: pose-read failed ({e})", flush=True)

    q0 = E.q0().detach()
    frames = []
    for i in range(args.frames):
        E.apply_arm(q0[:7], grip_open=True)   # hold the reset pose, gripper open
        frames.append(overlay.plain_frame(E.rgb(), f"{args.task}  objects={scene_objects}"))

    out_dir = os.path.join(repo, "results", "vlm_mpc", "scene")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.exp_name}.mp4")
    write_video_h264(frames, out, 20)
    print(f"[scene] DONE -> {out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim_common import runtime
    runtime.run_standalone(add_args, run, "render a Droid task scene to a video")
