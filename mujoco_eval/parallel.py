"""Run evaluation seeds across parallel worker subprocesses."""

import argparse
import json
import os
import subprocess
import sys
import time

from . import paths


def parse_seeds(spec):
    """Parse comma-separated seeds and inclusive ranges."""
    seeds = []

    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))

    return seeds


def episode_success(jsonl_path):
    """Return whether an episode log contains a successful result."""
    try:
        with open(jsonl_path, encoding="utf-8") as file:
            return any(
                json.loads(line).get("success")
                for line in file
                if line.strip()
            )
    except (OSError, json.JSONDecodeError):
        return False


def worker_env(workers, budget=None):
    """Build the environment shared by rollout workers.

    The thread budget starts from the IDLE core count, not the total: 8 workers x 4 threads is 32
    on a 32-core box, which saturates it before any other tenant runs and made replans 14x slower
    than the same work measured alone.
    """
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "3")
    cores = os.cpu_count() or 8
    if budget is None:
        budget = max(workers, int(cores - os.getloadavg()[0]))   # >= 1 thread per worker
    env["OMP_NUM_THREADS"] = str(max(1, min(4, budget // max(1, workers))))
    return env


def main():
    """Run all requested seeds with bounded parallelism."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument(
        "--seeds",
        required=True,
        help="e.g. 1-20 or 42,43,44",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="per-episode seconds",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="total thread budget across workers; default is the idle core count",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="args after -- go to the eval CLI",
    )
    args = parser.parse_args()

    passthrough = (
        args.rest[1:]
        if args.rest[:1] == ["--"]
        else args.rest
    )

    seeds = parse_seeds(args.seeds)
    env = worker_env(args.workers, args.threads)
    log_dir = paths.results_dir(args.task, args.exp)

    print(
        f"[par] {len(seeds)} seeds, {args.workers} workers, "
        f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']} -> {log_dir}",
        flush=True,
    )

    def launch(seed):
        command = [
            sys.executable,
            "-m",
            "mujoco_eval.eval",
            "--task",
            args.task,
            "--exp",
            args.exp,
            "--seed",
            str(seed),
            *passthrough,
        ]
        log = open(
            log_dir / f"{seed}.log",
            "w",
            encoding="utf-8",
        )
        return subprocess.Popen(
            command,
            cwd=str(paths.REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    running = {}
    queue = list(seeds)
    results = {}
    deadline = {}

    while queue or running:
        while queue and len(running) < args.workers:
            seed = queue.pop(0)
            running[seed] = launch(seed)
            deadline[seed] = time.time() + args.timeout
            print(
                f"[par] seed {seed} launched "
                f"({len(running)} running)",
                flush=True,
            )

        time.sleep(5)

        for seed, process in list(running.items()):
            return_code = process.poll()

            if (
                return_code is None
                and time.time() > deadline[seed]
            ):
                process.kill()
                return_code = -9

            if return_code is not None:
                del running[seed]
                success = episode_success(
                    log_dir / f"{seed}.jsonl"
                )
                results[seed] = (return_code, success)
                print(
                    f"[par] seed {seed} done "
                    f"rc={return_code} success={success}",
                    flush=True,
                )

    successful = sum(
        1
        for _, success in results.values()
        if success
    )
    clean_exits = sum(
        1
        for return_code, _ in results.values()
        if return_code == 0
    )
    timeouts = sum(
        1
        for return_code, _ in results.values()
        if return_code == -9
    )

    print(
        f"[par] TALLY {args.task}/{args.exp}: "
        f"{successful}/{len(seeds)} success "
        f"({clean_exits} clean exits, {timeouts} timeouts)",
        flush=True,
    )


if __name__ == "__main__":
    main()