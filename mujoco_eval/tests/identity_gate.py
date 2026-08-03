"""Prove the module reproduces a reference tree bit for bit.

Runs the same (task, seed, config) through both trees and compares every jsonl field except
wall-clock, plus the video md5. MBD is winner-take-all, so any changed planner input shows up as a
different trajectory -- which makes this a sharp test.

Exit status is non-zero if any case differs.

    python -m mujoco_eval.tests.identity_gate --legacy /path/to/old/tree
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

WALL_FIELDS = {"wall_s", "episode_wall_s", "replan_wall_median_s", "replan_wall_mean_s",
               "proxy_wall_s", "proxy_server_s", "proxy_embed_s"}

# (task, seed, config here, the same config's name in the legacy tree, extra args)
CASES = [
    ("stack", 42, "base", "mg_isaac_parity", []),
    ("stack", 7, "base", "mg_isaac_parity", []),
    ("can", 42, "base", "mg_isaac_parity", []),
    ("stack", 42, "rekep", "mg_rekep", ["--ground", "rekep"]),
]


def _records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _strip(rec):
    return {k: v for k, v in rec.items() if k not in WALL_FIELDS}


def compare(old_jsonl, new_jsonl, old_mp4, new_mp4):
    """(ok, [messages]) — record counts, per-field diffs, and the video md5."""
    msgs = []
    a, b = _records(old_jsonl), _records(new_jsonl)
    if len(a) != len(b):
        return False, [f"record count {len(a)} != {len(b)}"]
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = _strip(x), _strip(y)
        for k in sorted(set(sx) | set(sy)):
            if sx.get(k) != sy.get(k):
                msgs.append(f"record {i} ({x.get('kind')}) {k}: {sx.get(k)!r} != {sy.get(k)!r}")
    md5 = [hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest() for p in (old_mp4, new_mp4)]
    if md5[0] != md5[1]:
        msgs.append(f"video md5 {md5[0]} != {md5[1]}")
    return not msgs, msgs


def run_case(legacy, task, seed, config, legacy_config, extra, candidates, results, data):
    env = dict(os.environ, OMP_NUM_THREADS="4", MUJOCO_EVAL_DATA=data,
               MUJOCO_EVAL_RESULTS=results)
    common = ["--task", task, "--seed", str(seed), "--candidates", str(candidates), *extra]
    tag = f"_idgate_{task}_{seed}_{config}"
    subprocess.run(
        [sys.executable, "eval_mg.py", "--exp", f"{tag}_old",
         "--config", f"bench/configs/{legacy_config}.yaml", *common],
        cwd=legacy, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        [sys.executable, "-m", "mujoco_eval.eval", "--exp", f"{tag}_new",
         "--config", config, *common],
        cwd=str(pathlib.Path(__file__).resolve().parents[2]), env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    old = pathlib.Path(legacy) / "results" / task / f"{tag}_old"
    new = pathlib.Path(results) / task / f"{tag}_new"
    return compare(old / f"{seed}.jsonl", new / f"{seed}.jsonl",
                   old / f"{seed}.mp4", new / f"{seed}.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", required=True, help="the pre-refactor pps-mg checkout")
    ap.add_argument("--data", required=True, help="shared data root (both trees read it)")
    ap.add_argument("--results", required=True, help="writable results root for the new tree")
    ap.add_argument("--candidates", type=int, default=512)
    args = ap.parse_args()

    failed = 0
    for task, seed, config, legacy_config, extra in CASES:
        ok, msgs = run_case(args.legacy, task, seed, config, legacy_config, extra,
                            args.candidates, args.results, args.data)
        print(f"{'PASS' if ok else 'FAIL'}  {task} seed {seed} {config} {' '.join(extra)}")
        for m in msgs[:5]:
            print(f"      {m}")
        failed += not ok
    print(f"\n{len(CASES) - failed}/{len(CASES)} cases identical")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
