"""Persistent two-worker RoboLab evaluation launcher.

Each worker owns one Isaac application and evaluates a disjoint seed list sequentially.  This
keeps simulator state and success accounting independent while amortizing Isaac startup over the
worker's episodes.  Native ``num_envs>1`` is intentionally not used: the current ReKep bridge,
world adapter, action loop, and local proxy RPC are scalar/env-0 interfaces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .eval import _seed_values


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seeds", required=True, help="for example 5101-5120 or 1,4,8")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2),
                        help="one 48GB GPU safely supports at most two measured Spoon workers")
    parser.add_argument("--full_video_seeds", default=None,
                        help="representative seeds retaining full video; default is the first seed")
    parser.add_argument("--timeout", type=float, default=7200.0, help="seconds per worker")
    parser.add_argument("forward", nargs=argparse.REMAINDER,
                        help="arguments after -- forwarded to robolab_eval.eval")
    return parser.parse_args()


def _summary_for_seed(out_dir: Path, seed: int):
    traces = sorted(out_dir.glob(f"*/trace/{seed}_*.jsonl"))
    if not traces:
        return {"seed": seed, "error": "missing trace"}
    with traces[-1].open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    episode = next((row for row in reversed(rows) if row.get("kind") == "episode"), None)
    if episode is None:
        return {"seed": seed, "error": "missing episode row", "trace": str(traces[-1])}
    return {
        "seed": seed, "success": bool(episode["success"]),
        "steps": int(episode["env_steps"]), "episode_wall_s": float(episode["episode_wall_s"]),
        "stage_final": int(episode["stage_final"]), "trace": str(traces[-1]),
    }


def main():
    args = _args()
    seeds = _seed_values(args.seeds)
    full_video = ({seeds[0]} if args.full_video_seeds is None
                  else set(_seed_values(args.full_video_seeds)))
    unknown_video = full_video.difference(seeds)
    if unknown_video:
        raise SystemExit(f"full-video seeds are outside --seeds: {sorted(unknown_video)}")
    forbidden = {"--task", "--exp", "--seed", "--seed_list", "--artifact_mode",
                 "--full_video_seeds", "--num_envs"}
    forward = list(args.forward)
    if forward[:1] == ["--"]:
        forward = forward[1:]
    overlap = forbidden.intersection(forward)
    if overlap:
        raise SystemExit(f"launcher-owned options may not be forwarded: {sorted(overlap)}")

    # Importing paths is safe here; it does not start Isaac.
    from . import paths
    out_dir = paths.results_dir(args.task, args.exp)
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    chunks = [chunk for chunk in chunks if chunk]
    jobs = []
    started = time.perf_counter()
    for worker, chunk in enumerate(chunks):
        video_chunk = sorted(full_video.intersection(chunk))
        command = [
            sys.executable, "-m", "robolab_eval.eval", "--task", args.task, "--exp", args.exp,
            "--seed_list", ",".join(map(str, chunk)), "--artifact_mode", "summary",
        ]
        if video_chunk:
            command += ["--full_video_seeds", ",".join(map(str, video_chunk))]
        command += forward
        log_path = out_dir / f"worker{worker}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        jobs.append({"worker": worker, "seeds": chunk, "process": process,
                     "log": log, "log_path": str(log_path), "start": time.perf_counter()})
        print(f"[robolab-par] worker{worker} pid={process.pid} seeds={chunk} log={log_path}", flush=True)

    while any(job["process"].poll() is None for job in jobs):
        elapsed = time.perf_counter() - started
        done = sum(job["process"].poll() is not None for job in jobs)
        print(f"[robolab-par] workers {done}/{len(jobs)} complete, wall={elapsed:.0f}s", flush=True)
        if elapsed > args.timeout:
            for job in jobs:
                if job["process"].poll() is None:
                    job["process"].terminate()
            raise SystemExit(f"worker timeout after {elapsed:.0f}s")
        time.sleep(10)

    for job in jobs:
        job["log"].close()
        job["returncode"] = int(job["process"].returncode)
        job["wall_s"] = float(time.perf_counter() - job["start"])
    episode_rows = [_summary_for_seed(out_dir, seed) for seed in seeds]
    total_steps = sum(row.get("steps", 0) for row in episode_rows)
    wall_s = time.perf_counter() - started
    receipt = {
        "task": args.task, "exp": args.exp, "seeds": seeds, "workers": len(jobs),
        "representative_full_video_seeds": sorted(full_video), "wall_s": wall_s,
        "aggregate_control_steps_per_wall_s": total_steps / wall_s if wall_s else None,
        "episodes_per_hour": len(episode_rows) * 3600.0 / wall_s if wall_s else None,
        "workers_receipt": [{k: v for k, v in job.items()
                             if k not in ("process", "log", "start")} for job in jobs],
        "episodes": episode_rows,
    }
    receipt_path = out_dir / "parallel_summary.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[robolab-par] receipt: {receipt_path}", flush=True)
    print(f"[robolab-par] {len(episode_rows)} episodes in {wall_s:.1f}s; "
          f"{receipt['episodes_per_hour']:.2f} episodes/hour", flush=True)
    if any(job["returncode"] != 0 for job in jobs) or any("error" in row for row in episode_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
