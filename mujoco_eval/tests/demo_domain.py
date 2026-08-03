"""Check that a demo file's frames match what the eval renders, and dump demo videos.

A dataset rendered with a different geom set trains a proxy that predicts well offline and acts in
the wrong place at rollout, with nothing wrong-looking in the loss curve.

Above ~2 mean|diff| means a mismatch; re-convert with setup/convert_mimicgen.py, which carries the
visual-only re-assert.

    python -m mujoco_eval.tests.demo_domain --task stack
"""
from __future__ import annotations

import argparse
import sys

import cv2
import h5py
import imageio
import numpy as np

from .. import paths
from ..env.mujoco_env import MuJoCoEnv

REFERENCE = 0.50      # stack demo_224 vs live, the known-good value
THRESHOLD = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="stack")
    ap.add_argument("--dataset", default="demo_224.hdf5")
    ap.add_argument("--fk_fit", default="fk_fit_stack_d0.json")
    ap.add_argument("--videos", type=int, default=3, help="demo videos to write (0 = none)")
    args = ap.parse_args()

    h5 = paths.task_data(args.task, args.dataset)
    out = paths.results_dir(args.task, "_demo_check")
    f = h5py.File(h5, "r")
    env = MuJoCoEnv(str(h5), str(paths.fk_fit(args.fk_fit)))

    g = f["data"]["demo_0"]
    demo = np.asarray(g["obs"]["table_cam"][0])
    env.reset_to(np.asarray(g["states"]["mujoco"][0]))
    live = env.rgb("agentview", hw=demo.shape[0])
    diff = np.abs(demo.astype(int) - live.astype(int)).mean()
    verdict = "OK" if diff < THRESHOLD else "DOMAIN MISMATCH"
    print(f"{verdict}: demo vs live mean|diff| = {diff:.2f} "
          f"(reference {REFERENCE:.2f}, threshold {THRESHOLD})")
    cv2.imwrite(str(out / "frame_demo.png"), cv2.cvtColor(demo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "frame_live.png"), cv2.cvtColor(live, cv2.COLOR_RGB2BGR))
    print(f"frames: {out}/frame_demo.png  {out}/frame_live.png")

    for name in list(f["data"].keys())[: args.videos]:
        table = np.asarray(f["data"][name]["obs"]["table_cam"])
        wrist = np.asarray(f["data"][name]["obs"]["wrist_cam"])
        path = out / f"{name}.mp4"
        with imageio.get_writer(str(path), fps=20) as w:
            for fr in np.concatenate([table, wrist], axis=2):     # table | wrist
                w.append_data(fr)
        print(f"wrote {path} ({len(table)} frames)")

    return 0 if diff < THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
