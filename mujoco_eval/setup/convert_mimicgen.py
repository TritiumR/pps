"""Convert MimicGen demonstrations to the proxy-training HDF5 schema."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
)
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import argparse
import json
import pathlib
import time

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
OPEN_APERTURE = 0.080
TASK_BODIES = {
    "stack_d0": ("cubeA", "cubeB"),
    "stack_three_d0": ("cubeA", "cubeB", "cubeC"),
    "square_d0": ("SquareNut",),
    # MuJoCo body names, resolved through _body_name ("<name>_main" then "<name>"). These are the
    # model's own names, which differ from the aliases gt.py's coffee grounding uses: the machine
    # body is coffee_machine_body_main and the holder is coffee_machine_pod_holder_holder_main.
    "coffee_d0": (
        "coffee_pod",
        "coffee_machine_body",
        "coffee_machine_lid",
        "coffee_machine_pod_holder_holder",
    ),
    # Two-bin can sorting. The bins are arena bodies, so their poses are constant within a
    # scene and are carried in the per-demo layout attr rather than a states table.
    "sort_can_d0": ("Can",),
    # Continuous-goal tray variant: same single can, same arena bodies.
    "sort_can_tray_d0": ("Can",),
}

# Per-demo attrs copied through from the source demo. The goal block is what makes the
# dataset goal-conditioned, so it has to survive the 224 conversion.
CARRY_ATTRS = (
    "scene_id",
    "goal_colour",
    "prompt",
    "target_quadrant",
    "layout",
    "pair_complete",
    "retries",
    "release_idx",
    "g_task_xyz",
    "g_task_quat_wxyz",
    "g_demo_joint8",
    "g_demo_xyz",
    "g_demo_quat_wxyz",
    # sort_can_tray adds the continuous-goal provenance.
    "split",
    "goal_index",
    "collection_order",
    "scene_complete",
    "goal_error_m",
    "g_task_local_xy",
)


def _make_env(hdf5):
    import mimicgen
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    try:
        from ..envs import SortCanTwoBin  # noqa: F401  registers this repo's own envs
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(_HERE.parent.parent))
        from mujoco_eval.envs import SortCanTwoBin  # noqa: F401

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "low_dim": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_joint_pos",
                "object",
            ]
        }
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    return env


def _visual_only(env):
    """Disable collision geometry in the offscreen renderer."""
    ctx = getattr(env.env.sim, "_render_context_offscreen", None)
    if ctx is not None and ctx.vopt.geomgroup[0]:
        ctx.vopt.geomgroup[0] = 0


def _body_name(sim, name):
    names = list(sim.model.body_names)
    return next(c for c in (f"{name}_main", name) if c in names)


def _base_pose(fk_fit):
    with open(fk_fit) as fh:
        fit = json.load(fh)
    return np.concatenate(
        [fit["base_pos"], fit["base_quat_wxyz"]]
    ).astype(np.float32)


def convert(args):
    root = pathlib.Path(args.data_dir) if args.data_dir else _HERE
    src = root / args.task / "demo.hdf5"
    out = root / args.task / (
        f"demo_{args.size}_{args.part}.hdf5"
        if args.part
        else f"demo_{args.size}.hdf5"
    )
    env = _make_env(str(src))
    bodies = TASK_BODIES[args.task]
    robot_pose = _base_pose(args.fk_fit)[None]
    S = args.size

    with h5py.File(src, "r") as f, h5py.File(out, "w") as g:
        names = sorted(
            f["data"].keys(),
            key=lambda s: int(s.split("_")[1]),
        )
        names = names[args.demo_start:args.demo_end]
        data = g.create_group("data")

        for k, v in f["data"].attrs.items():
            data.attrs[k] = v

        n_frames, t0 = 0, time.time()

        for di, name in enumerate(names):
            d = f[f"data/{name}"]
            act = np.asarray(d["actions"], dtype=np.float32)
            q = np.asarray(
                d["obs/robot0_joint_pos"],
                dtype=np.float32,
            )
            states = np.asarray(d["states"])
            T = act.shape[0] - 1
            xml = d.attrs["model_file"]
            env.reset_to({"states": states[0], "model": xml})

            table = np.empty((T, S, S, 3), dtype=np.uint8)
            wrist = np.empty((T, S, S, 3), dtype=np.uint8)
            body_pose = {
                b: np.empty((T, 7), dtype=np.float32)
                for b in bodies
            }

            for t in range(T):
                env.reset_to({"states": states[t]})
                _visual_only(env)
                table[t] = env.render(
                    mode="rgb_array",
                    height=S,
                    width=S,
                    camera_name="agentview",
                )
                wrist[t] = env.render(
                    mode="rgb_array",
                    height=S,
                    width=S,
                    camera_name="robot0_eye_in_hand",
                )
                sim = env.env.sim

                for b in bodies:
                    bn = _body_name(sim, b)
                    body_pose[b][t, :3] = sim.data.get_body_xpos(bn)
                    body_pose[b][t, 3:] = sim.data.get_body_xquat(bn)

            n_frames += T

            ap = np.asarray(
                d["obs/robot0_gripper_qpos"][:T],
                dtype=np.float32,
            )
            aperture = ap[:, 0] - ap[:, 1]
            closure = np.clip(
                (OPEN_APERTURE - aperture) / OPEN_APERTURE,
                0.0,
                1.0,
            )
            eefq_xyzw = np.asarray(
                d["obs/robot0_eef_quat"][:T],
                dtype=np.float32,
            )
            joint_actions = np.concatenate(
                [
                    q[1:T + 1],
                    ((act[:T, 6:7] + 1.0) * 0.5),
                ],
                axis=-1,
            ).astype(np.float32)

            o = data.create_group(name)
            o.attrs["model_file"] = xml
            o.attrs["num_samples"] = T
            for key in CARRY_ATTRS:
                if key in d.attrs:
                    o.attrs[key] = d.attrs[key]
            obs = o.create_group("obs")
            obs.create_dataset(
                "table_cam",
                data=table,
                compression="gzip",
                compression_opts=1,
                chunks=(1, S, S, 3),
            )
            obs.create_dataset(
                "wrist_cam",
                data=wrist,
                compression="gzip",
                compression_opts=1,
                chunks=(1, S, S, 3),
            )
            obs.create_dataset("joint_pos", data=q[:T])
            obs.create_dataset(
                "joint_vel",
                data=np.asarray(
                    d["obs/robot0_joint_vel"][:T],
                    dtype=np.float32,
                ),
            )
            obs.create_dataset(
                "gripper_pos",
                data=closure[:, None].astype(np.float32),
            )
            obs.create_dataset(
                "eef_pos",
                data=np.asarray(
                    d["obs/robot0_eef_pos"][:T],
                    dtype=np.float32,
                ),
            )
            obs.create_dataset(
                "eef_quat",
                data=np.roll(eefq_xyzw, 1, axis=-1),
            )
            obs.create_dataset(
                "joint_actions",
                data=joint_actions,
            )
            obs.create_dataset(
                "object",
                data=np.asarray(
                    d["obs/object"][:T],
                    dtype=np.float32,
                ),
            )
            obs.create_dataset(
                "robot0_gripper_qpos",
                data=ap,
            )
            o.create_dataset("actions", data=act[:T])

            st = o.create_group("states")
            st.create_dataset("mujoco", data=states[:T])

            for b in bodies:
                st.create_dataset(
                    f"rigid_object/{b}/root_pose",
                    data=body_pose[b],
                )

            st.create_dataset(
                "articulation/robot/root_pose",
                data=np.repeat(robot_pose, T, axis=0),
            )

            # The source's own "total" was copied above with the rest of data.attrs; it counts
            # SOURCE frames, and a part counts only its own converted ones. Overwrite it as we
            # go so a part is self-describing and merge() can just sum.
            data.attrs["total"] = n_frames
            rate = n_frames / (time.time() - t0)
            print(
                f"[{args.part or 'all'}] {name} "
                f"({di + 1}/{len(names)}) T={T} "
                f"cum {n_frames} frames @ {rate:.2f} frames/s",
                flush=True,
            )

    print(f"wrote {out} ({n_frames} frames)", flush=True)


def merge(args):
    root = pathlib.Path(args.data_dir) if args.data_dir else _HERE
    out = root / args.task / f"demo_{args.size}.hdf5"
    parts = [
        root / args.task / f"demo_{args.size}_{p}.hdf5"
        for p in args.merge
    ]

    with h5py.File(out, "w") as g:
        data = g.create_group("data")
        first = True
        frames = 0

        for p in parts:
            with h5py.File(p, "r") as f:
                if first:
                    for k, v in f["data"].attrs.items():
                        data.attrs[k] = v
                    first = False

                # "total" is a frame count, so it has to be summed across parts rather than
                # inherited from the first one.
                frames += int(f["data"].attrs.get("total", 0))

                for name in f["data"]:
                    f.copy(f"data/{name}", data, name=name)

        data.attrs["total"] = frames
        total = len(data.keys())

    print(f"merged {len(parts)} parts -> {out} ({total} demos)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="stack_d0",
        choices=sorted(TASK_BODIES),
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--data_dir",
        default=None,
        help="root holding <task>/demo.hdf5; default is this script's own directory",
    )
    parser.add_argument("--demo_start", type=int, default=0)
    parser.add_argument("--demo_end", type=int, default=200)
    parser.add_argument(
        "--part",
        default=None,
        help="suffix for a shard file",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help="part suffixes to merge",
    )
    parser.add_argument(
        "--fk_fit",
        default=str(
            _HERE.parent
            / "bench/fk_fits/fk_fit_stack_d0.json"
        ),
    )
    args = parser.parse_args()

    if args.merge:
        merge(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()