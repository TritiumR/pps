"""Inspect a MimicGen/robomimic hdf5: demo count, tree, obs keys, actions, env_args, masks.

    python setup/05_inspect_hdf5.py <path.hdf5> [...]
"""

import json
import sys

import h5py
import numpy as np


def print_tree(g, prefix="", max_children=40):
    names = list(g.keys())
    for name in names[:max_children]:
        item = g[name]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{name}  {item.shape} {item.dtype}")
        else:
            print(f"{prefix}{name}/")
            print_tree(item, prefix + "  ", max_children)
    if len(names) > max_children:
        print(f"{prefix}... ({len(names) - max_children} more)")


def inspect_file(path):
    print("=" * 78)
    print("FILE:", path)
    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        n = len(demos)
        print(f"num demos: {n}")

        # file-level attrs
        for k in data.attrs:
            v = data.attrs[k]
            if k == "env_args":
                print("env_args:")
                print(json.dumps(json.loads(v), indent=2))
            else:
                print(f"data.attrs[{k}] = {v}")

        # episode lengths
        lengths = np.array([data[d]["actions"].shape[0] for d in demos])
        print(f"episode lengths: min={lengths.min()} mean={lengths.mean():.1f} "
              f"max={lengths.max()} total={lengths.sum()}")

        # one demo in detail
        d0 = data[demos[0]]
        print(f"--- tree of data/{demos[0]} ---")
        print_tree(d0, "  ")
        for k in d0.attrs:
            v = d0.attrs[k]
            vs = str(v)
            print(f"  attrs[{k}] = {vs[:120]}{'...' if len(vs) > 120 else ''}")

        acts = d0["actions"][()]
        print(f"actions: shape={acts.shape} dtype={acts.dtype}")
        print(f"  per-dim min: {np.round(acts.min(axis=0), 3)}")
        print(f"  per-dim max: {np.round(acts.max(axis=0), 3)}")

        print("states present:", "states" in d0,
              "shape:", d0["states"].shape if "states" in d0 else None)
        if "obs" in d0:
            print("obs keys:", {k: str(d0["obs"][k].shape) for k in d0["obs"]})

        # datagen info / subtask signals — check first demo and count coverage
        if "datagen_info" in d0:
            print("--- datagen_info of demo_0 ---")
            print_tree(d0["datagen_info"], "  ")
            cov = sum("datagen_info" in data[d] for d in demos)
            print(f"datagen_info present in {cov}/{n} demos")
        else:
            print("datagen_info: ABSENT in demo_0")
        # any other subtask-ish keys
        extra = [k for k in d0 if k not in
                 ("actions", "states", "obs", "next_obs", "rewards", "dones",
                  "datagen_info")]
        if extra:
            print("other demo keys:", extra)

        if "mask" in f:
            print("mask groups:",
                  {k: f["mask"][k].shape for k in f["mask"]})
        else:
            print("mask groups: none")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect_file(p)
