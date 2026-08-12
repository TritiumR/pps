"""Render representative table/wrist previews from the spoon proxy dataset."""

import argparse
import json
import os

import cv2
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--demos", nargs="+", default=["demo_0", "demo_25", "demo_49"])
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    entries = []
    with h5py.File(args.hdf5, "r") as dataset:
        data = dataset["data"]
        for name in args.demos:
            demo = data[name]
            table = demo["obs/table_cam"]
            wrist = demo["obs/wrist_cam"]
            if table.shape != wrist.shape or table.shape[1:] != (224, 224, 3):
                raise ValueError(f"{name}: incompatible image shapes {table.shape}, {wrist.shape}")

            source_attempt = int(demo.attrs["source_attempt"])
            randomization = json.loads(demo.attrs["randomization"])
            output_name = f"{name}_attempt_{source_attempt}_table_wrist.mp4"
            output_path = os.path.join(args.out, output_name)
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (448, 224),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer for {output_path}")
            try:
                for index in range(table.shape[0]):
                    # Stored observations are RGB; OpenCV's writer consumes BGR.
                    frame = np.concatenate([table[index], wrist[index]], axis=1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"{name}  table", (5, 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(frame, f"wrist  frame {index + 1}/{table.shape[0]}",
                                (229, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    writer.write(frame)
            finally:
                writer.release()

            entries.append({
                "video": output_name,
                "demo": name,
                "source_demo": str(demo.attrs["source"]),
                "source_attempt": source_attempt,
                "seed": int(randomization["seed"]),
                "frames": int(table.shape[0]),
                "fps": args.fps,
                "duration_seconds": float(table.shape[0] / args.fps),
                "randomization": randomization,
                "realized_initial_state": json.loads(demo.attrs["realized_initial_state"]),
            })

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({
            "dataset": args.hdf5,
            "layout": "left=table_cam, right=wrist_cam",
            "selection": "first, middle, and last final accepted demos",
            "entries": entries,
        }, handle, indent=2)
    print(json.dumps({"manifest": manifest_path, "videos": entries}, indent=2))


if __name__ == "__main__":
    main()
