"""Deep validation receipt for the 50-demo single-spoon proxy dataset."""

import argparse
import json
import os
import sys

import h5py
import numpy as np


REQUIRED = (
    "obs/table_cam", "obs/wrist_cam", "obs/joint_pos", "obs/gripper_pos",
    "obs/joint_actions", "obs/eef_pos", "obs/eef_quat",
    "states/articulation/robot/root_pose",
    "states/rigid_object/pink_spaghetti_spoon/root_pose",
    "states/rigid_object/utensil_holder/root_pose",
    "states/rigid_object/spatula/root_pose",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--expected", type=int, default=50)
    args = ap.parse_args()

    failures, lengths, action_min, action_max = [], [], [], []
    initial, randomization = [], []
    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        names = sorted(data, key=lambda x: int(x.split("_")[-1]))
        if names != [f"demo_{i}" for i in range(args.expected)]:
            failures.append(f"names/count mismatch: {names[:3]}..{names[-3:]} ({len(names)})")
        for name in names:
            d = data[name]
            missing = [key for key in REQUIRED if key not in d]
            if missing:
                failures.append(f"{name}: missing {missing}")
                continue
            length = int(d["obs/joint_actions"].shape[0])
            lengths.append(length)
            if int(d.attrs.get("num_samples", -1)) != length:
                failures.append(f"{name}: num_samples mismatch")
            if not bool(d.attrs.get("success", False)):
                failures.append(f"{name}: success attr false")
            if int(d.attrs.get("insertion_events", -1)) != 1:
                failures.append(f"{name}: insertion_events != 1")
            if int(d.attrs.get("release_events", -1)) != 1:
                failures.append(f"{name}: release_events != 1")
            if bool(d.attrs.get("other_utensil_retained", True)):
                failures.append(f"{name}: distractor utensil retained")
            phases = json.loads(d.attrs["phase_order"])
            if phases.count("INSERT") != 1 or phases.count("RELEASE") != 1:
                failures.append(f"{name}: phase order {phases}")
            for key in REQUIRED:
                if d[key].shape[0] != length and not key.startswith("initial_state/"):
                    failures.append(f"{name}/{key}: length {d[key].shape[0]} != {length}")
            if d["obs/table_cam"].shape[1:] != (224, 224, 3) or d["obs/table_cam"].dtype != np.uint8:
                failures.append(f"{name}: table image format")
            if d["obs/wrist_cam"].shape[1:] != (224, 224, 3) or d["obs/wrist_cam"].dtype != np.uint8:
                failures.append(f"{name}: wrist image format")
            if d["obs/joint_pos"].shape[1] != 7 or d["obs/joint_actions"].shape[1] != 8:
                failures.append(f"{name}: state/action dimensions")
            q = np.asarray(d["obs/joint_pos"], dtype=np.float32)
            a = np.asarray(d["obs/joint_actions"], dtype=np.float32)
            if not np.isfinite(q).all() or not np.isfinite(a).all():
                failures.append(f"{name}: nonfinite q/actions")
            if not np.array_equal(a[:-1, :7], q[1:, :7]):
                failures.append(f"{name}: achieved-next-action invariant")
            if float(a[:, 7].min()) < 0 or float(a[:, 7].max()) > 1:
                failures.append(f"{name}: gripper command outside [0,1]")
            quat = np.asarray(d["obs/eef_quat"], dtype=np.float32)
            if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3):
                failures.append(f"{name}: nonunit eef quaternion")
            action_min.append(a.min(axis=0)); action_max.append(a.max(axis=0))
            initial.append(json.loads(d.attrs["realized_initial_state"]))
            randomization.append(json.loads(d.attrs["randomization"]))

        # Exercise the exact sample reader used by the task-proxy/BC pipeline.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        proxy_scripts = os.path.join(repo_root, "openpi", "scripts")
        sys.path.insert(0, proxy_scripts)
        from train_mpc_proxy_score_pytorch import _demo_sample
        sample = _demo_sample(data[names[0]], 0, action_horizon=15,
                              prompt=str(data.attrs["prompt"]), action_offset=1)
        proxy_sample = {
            "table_shape": list(np.asarray(sample["exterior_image_1_left"]).shape),
            "wrist_shape": list(np.asarray(sample["wrist_image_left"]).shape),
            "joint_shape": list(np.asarray(sample["joint_position"]).shape),
            "gripper_shape": list(np.asarray(sample["gripper_position"]).shape),
            "actions_shape": list(np.asarray(sample["actions"]).shape),
            "prompt": str(sample["prompt"]),
        }
        if proxy_sample["actions_shape"] != [15, 8]:
            failures.append(f"proxy sample actions {proxy_sample['actions_shape']}")

    arr_s = np.asarray([x["spoon_pose_wxyz"] for x in initial], dtype=float)
    arr_h = np.asarray([x["holder_pose_wxyz"] for x in initial], dtype=float)
    arr_q = np.asarray([x["robot_joint_pos"][:7] for x in initial], dtype=float)
    def scalar_range(key):
        values = np.asarray([x[key] for x in randomization], dtype=float)
        return [float(values.min()), float(values.max())]

    policy_keys = ("grasp_shift_m", "grasp_yaw_deg", "approach_h", "lift_h",
                   "transit_bow_m", "over_dz", "retreat_dz", "lateral_offset")
    realized_draw_ranges = {
        key: scalar_range(key) for key in (
            "spoon_dx_m", "spoon_dy_m", "spoon_yaw_deg", "holder_dx_m",
            "holder_dy_m", "holder_yaw_deg"
        )
    }
    robot_delta = np.asarray([x["robot_joint_delta_rad"] for x in randomization], dtype=float)
    realized_draw_ranges["robot_joint_delta_rad"] = [
        robot_delta.min(0).tolist(), robot_delta.max(0).tolist()
    ]
    realized_draw_ranges["policy_force"] = {
        key: [float(min(x["policy_force"][key] for x in randomization)),
              float(max(x["policy_force"][key] for x in randomization))]
        for key in policy_keys
    }
    receipt = {
        "passed": not failures,
        "dataset": args.hdf5,
        "file_size_bytes": os.path.getsize(args.hdf5),
        "demo_count": len(lengths),
        "all_success": not any("success attr" in x for x in failures),
        "all_single_insertion": not any("insertion_events" in x or "phase order" in x for x in failures),
        "no_second_utensil": not any("distractor utensil" in x for x in failures),
        "observation_dimensions": {"table_cam": [224, 224, 3], "wrist_cam": [224, 224, 3],
                                   "joint_pos": 7, "gripper_pos": 1, "eef_pos": 3,
                                   "eef_quat": 4},
        "action_dimension": 8,
        "action_semantics": "achieved next arm joint position (7) + continuous gripper command (1)",
        "length_stats": {"min": int(np.min(lengths)), "max": int(np.max(lengths)),
                         "mean": float(np.mean(lengths)), "median": float(np.median(lengths)),
                         "total": int(np.sum(lengths))},
        "realized_initial_ranges": {
            "spoon_position_xyz": [arr_s[:, :3].min(0).tolist(), arr_s[:, :3].max(0).tolist()],
            "holder_position_xyz": [arr_h[:, :3].min(0).tolist(), arr_h[:, :3].max(0).tolist()],
            "robot_arm_joint_pos": [arr_q.min(0).tolist(), arr_q.max(0).tolist()],
        },
        "realized_requested_randomization_ranges": realized_draw_ranges,
        "proxy_loader_sample": proxy_sample,
        "proxy_loader_confirmed": proxy_sample["actions_shape"] == [15, 8],
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.receipt), exist_ok=True)
    with open(args.receipt, "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
