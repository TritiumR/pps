"""Filter and convert native RoboLab spoon recordings to the proxy HDF5 schema."""

import argparse
import json
import os

import h5py
import numpy as np


TABLE = "obs/image_obs/front_wide_camera"
WRIST = "obs/image_obs/wrist_cam"
Q = "obs/proprio_obs/arm_joint_pos"
GRIP = "obs/proprio_obs/gripper_pos"
EEF_POS = "obs/proprio_obs/eef_pos"
EEF_QUAT = "obs/proprio_obs/eef_quat"
IMG = 224
PROMPT = "Insert the spaghetti spoon into the utensil holder."
TASK = "InsertSpaghettiSpoonTask"


def _copy_tree(src, dst):
    for key, value in src.items():
        if isinstance(value, h5py.Group):
            _copy_tree(value, dst.create_group(key))
        else:
            dst.create_dataset(key, data=np.asarray(value))
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def convert(src, dst, name, result):
    actions = np.asarray(src["actions"], dtype=np.float32)
    q = np.asarray(src[Q], dtype=np.float32)
    length = min(len(actions), len(q)) - 1
    if length < 16:
        raise ValueError(f"{src.name}: only {length} usable frames")
    if actions.shape[1] != 8 or q.shape[1] != 7:
        raise ValueError(f"{src.name}: actions={actions.shape}, q={q.shape}")

    demo = dst.create_group(name)
    obs = demo.create_group("obs")
    for out_key, in_key in (("table_cam", TABLE), ("wrist_cam", WRIST)):
        images = np.asarray(src[in_key][:length])
        if images.shape[1:] != (IMG, IMG, 3) or images.dtype != np.uint8:
            raise ValueError(f"{src.name}/{in_key}: {images.shape} {images.dtype}")
        obs.create_dataset(out_key, data=images, chunks=(1, IMG, IMG, 3),
                           compression="gzip", compression_opts=1)
    obs.create_dataset("joint_pos", data=q[:length])
    gripper = np.asarray(src[GRIP], dtype=np.float32)[:length].reshape(length, 1)
    obs.create_dataset("gripper_pos", data=gripper)
    # Same achieved-action relabel used by convert_mimicgen and p1b_convert:
    # arm action t is achieved q[t+1].  The eighth channel stays the original
    # continuous command because this task intentionally uses a continuous gripper.
    joint_actions = np.concatenate([q[1 : length + 1], actions[:length, 7:8]], axis=-1)
    obs.create_dataset("joint_actions", data=joint_actions.astype(np.float32))
    obs.create_dataset("eef_pos", data=np.asarray(src[EEF_POS], dtype=np.float32)[:length])
    obs.create_dataset("eef_quat", data=np.asarray(src[EEF_QUAT], dtype=np.float32)[:length])

    states = demo.create_group("states")
    rigid = states.create_group("rigid_object")
    for obj in src["states/rigid_object"]:
        pose = np.asarray(src[f"states/rigid_object/{obj}/root_pose"], dtype=np.float32)[:length]
        rigid.create_group(obj).create_dataset("root_pose", data=pose)
    robot_pose = np.asarray(src["states/articulation/robot/root_pose"], dtype=np.float32)[:length]
    states.create_group("articulation").create_group("robot").create_dataset(
        "root_pose", data=robot_pose
    )
    if "initial_state" in src:
        _copy_tree(src["initial_state"], demo.create_group("initial_state"))

    demo.attrs["num_samples"] = length
    demo.attrs["success"] = True
    demo.attrs["source"] = src.name
    demo.attrs["source_attempt"] = int(result["attempt"])
    demo.attrs["task"] = TASK
    demo.attrs["object"] = result["object"]
    demo.attrs["container"] = result["container"]
    demo.attrs["other_utensil"] = result["other_utensil"]
    demo.attrs["other_utensil_retained"] = bool(result["other_utensil_retained"])
    demo.attrs["insertion_events"] = int(result["insertion_events"])
    demo.attrs["release_events"] = int(result["release_events"])
    demo.attrs["phase_order"] = json.dumps(result["phase_order"])
    demo.attrs["randomization"] = json.dumps(result["requested_randomization"])
    demo.attrs["realized_initial_state"] = json.dumps(result["realized_initial_state"])
    return length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=50)
    args = ap.parse_args()

    with open(args.results) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    accepted = [r for r in rows if r.get("accepted")]
    if len(accepted) < args.count:
        raise SystemExit(f"only {len(accepted)} accepted rows; need {args.count}")
    accepted = accepted[: args.count]
    by_attempt = {int(r["attempt"]): r for r in accepted}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    kept, total = [], 0
    with h5py.File(args.src, "r") as source, h5py.File(tmp, "w") as target:
        data = target.create_group("data")
        for attempt, row in sorted(by_attempt.items()):
            source_name = f"demo_{attempt}"
            group = source["data"][source_name]
            if not bool(group.attrs.get("success", False)):
                raise ValueError(f"{source_name}: accepted sidecar but source success attr false")
            out_name = f"demo_{len(kept)}"
            length = convert(group, data, out_name, row)
            kept.append({"demo": out_name, "source": source_name, "attempt": attempt,
                         "num_samples": length})
            total += length
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(
            {"env_name": TASK, "type": "robolab", "prompt": PROMPT,
             "dataset": "single_spoon_insertion_50"}
        )
        data.attrs["prompt"] = PROMPT
        data.attrs["task"] = TASK
        data.attrs["object"] = "pink_spaghetti_spoon"
        data.attrs["container"] = "utensil_holder"
        data.attrs["table_camera_source"] = "front_wide_camera"
        data.attrs["wrist_camera_source"] = "wrist_cam"
        data.attrs["image_size"] = IMG
        data.attrs["action_semantics"] = "achieved_next_joint_pos_7_plus_continuous_gripper_command"
    os.replace(tmp, args.out)

    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump({"dataset": args.out, "source": args.src, "task": TASK,
                   "prompt": PROMPT, "count": len(kept), "total_samples": total,
                   "kept": kept, "filter": {
                       "task_success": True, "other_utensil_retained": False,
                       "insertion_events": 1, "release_events": 1,
                       "finite_actions": True, "minimum_frames": 16,
                   }}, fh, indent=2)
    print(json.dumps({"out": args.out, "manifest": manifest_path,
                      "demos": len(kept), "total_samples": total,
                      "size_bytes": os.path.getsize(args.out)}, indent=2))


if __name__ == "__main__":
    main()
