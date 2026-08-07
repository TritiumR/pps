"""Select the best rollout per seed with a subgoal-based verifier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import paths

paths.ensure_repo_on_path()

from .grounding.gt import CAN_SEAT, EXTENTS, _CAN_XY, _ON_XY
from .parallel import parse_seeds


def place_goals(task):
    """Return placement goals for a supported task."""
    if task == "stack":
        seat = np.array(
            [0.0, 0.0, EXTENTS["cubeB"][2] + EXTENTS["cubeA"][2]]
        )
        return [("cubeA", "cubeB", seat, _ON_XY)]

    if task == "can":
        return [("can", None, CAN_SEAT, float(min(_CAN_XY)))]

    raise ValueError(f"no place goals defined for task {task!r}")


def subgoal_score(objects, stage_max, goals):
    """Score terminal beliefs by stage progress and placement residual."""
    score = float(stage_max)

    for payload, dest, offset, eps in goals:
        payload_pos = objects.get(payload)
        destination = offset if dest is None else objects.get(dest)

        if payload_pos is None or destination is None:
            return score

        goal = (
            np.asarray(destination, dtype=np.float64)
            if dest is None
            else np.asarray(destination, dtype=np.float64) + offset
        )
        residual = float(
            np.linalg.norm(
                np.asarray(payload_pos, dtype=np.float64) - goal
            )
        )

        if residual < eps:
            score += 1.0
        else:
            return score + max(
                0.0,
                1.0 - residual / (4.0 * eps),
            )

    return score


def read_episode(jsonl_path):
    """Read terminal beliefs, stage progress, and oracle success from JSONL."""
    stage_max = 0
    objects = None
    success = False
    complete = False

    try:
        with open(jsonl_path, encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                kind = record.get("kind")

                if kind == "stage":
                    stage_max = max(stage_max, int(record["to"]))
                elif kind == "replan":
                    stage_max = max(
                        stage_max,
                        int(record["stage_idx"]),
                    )
                    objects = record["objects"]
                elif kind == "episode":
                    stage_max = max(
                        stage_max,
                        int(record.get("stage_max", 0)),
                    )
                    objects = record["objects_final"]
                    success = bool(record["success"])
                    complete = True
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass

    return {
        "stage_max": stage_max,
        "objects": objects,
        "success": success,
        "complete": complete,
    }


def score_arm(task, base_dir, seeds, m):
    """Compare verifier and oracle selections across rollout candidates."""
    goals = place_goals(task)
    per_seed = {}

    agreement_count = 0
    mixed_count = 0
    mixed_agreement_count = 0
    picked_successes = 0
    oracle_successes = 0

    for seed in seeds:
        scores = []
        successes = []

        for sampler_seed in range(m):
            episode = read_episode(
                paths.seed_artifact(
                    base_dir / f"k{sampler_seed}", seed, "trace", "jsonl"
                )
                or base_dir / f"k{sampler_seed}" / f"{seed}.jsonl"
            )
            score = (
                round(
                    subgoal_score(
                        episode["objects"],
                        episode["stage_max"],
                        goals,
                    ),
                    4,
                )
                if episode["objects"] is not None
                else -1.0
            )
            scores.append(score)
            successes.append(episode["success"])

        pick = int(np.argmax(scores))
        oracle_pick = successes.index(True) if any(successes) else 0
        agrees = successes[pick] == any(successes)

        per_seed[seed] = {
            "scores": scores,
            "success": successes,
            "pick": pick,
            "pick_success": successes[pick],
            "oracle_pick": oracle_pick,
            "oracle_success": any(successes),
            "agree": agrees,
        }

        picked_successes += successes[pick]
        oracle_successes += any(successes)
        agreement_count += agrees

        if any(successes) and not all(successes):
            mixed_count += 1
            mixed_agreement_count += agrees

    count = len(seeds)
    summary = {
        "n_seeds": count,
        "m": m,
        "pick_success": picked_successes,
        "oracle_success": oracle_successes,
        "agreement": (
            round(agreement_count / count, 4)
            if count
            else None
        ),
        "mixed_seeds": mixed_count,
        "agreement_mixed": (
            round(mixed_agreement_count / mixed_count, 4)
            if mixed_count
            else None
        ),
    }
    return per_seed, summary


def run_pool(args, seeds, passthrough):
    """Run missing seed and sampler combinations with bounded parallelism."""
    source_dir = (paths.find_run(args.task, args.from_exp)
                  or paths.RESULTS / args.task / args.from_exp)

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "3")
    env["OMP_NUM_THREADS"] = str(args.omp)

    jobs = []
    for seed in seeds:
        for sampler_seed in range(args.m):
            episode_path = (
                paths.seed_artifact(
                    source_dir / f"k{sampler_seed}", seed, "trace", "jsonl"
                )
                or source_dir / f"k{sampler_seed}" / f"{seed}.jsonl"
            )
            if read_episode(episode_path)["complete"]:
                continue
            jobs.append((seed, sampler_seed))

    print(
        f"[select] {len(jobs)} rollouts to run "
        f"({len(seeds)} seeds x M={args.m}, resumed the rest)",
        flush=True,
    )

    def launch(seed, sampler_seed):
        experiment = f"{args.from_exp}/k{sampler_seed}"
        command = [
            sys.executable,
            "-m",
            "mujoco_eval.eval",
            "--task",
            args.task,
            "--exp",
            experiment,
            "--seed",
            str(seed),
            "--sampler_seed",
            str(sampler_seed),
            *passthrough,
        ]

        log_dir = source_dir / f"k{sampler_seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
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
    deadlines = {}
    completed = 0
    queue = list(jobs)

    while queue or running:
        while queue and len(running) < args.workers:
            seed, sampler_seed = queue.pop(0)
            key = (seed, sampler_seed)
            running[key] = launch(seed, sampler_seed)
            deadlines[key] = time.time() + args.timeout

        time.sleep(5)

        for key, process in list(running.items()):
            return_code = process.poll()

            if (
                return_code is None
                and time.time() > deadlines[key]
            ):
                process.kill()
                return_code = -9

            if return_code is not None:
                del running[key]
                completed += 1
                print(
                    f"[select] seed {key[0]} k{key[1]} "
                    f"done rc={return_code} "
                    f"({completed}/{len(jobs)})",
                    flush=True,
                )


def main():
    """Run rollouts if needed, score them, and write the selection report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--exp",
        required=True,
        help="selection report name (results/<task>/<exp>)",
    )
    parser.add_argument(
        "--from_exp",
        default=None,
        help=(
            "rollout pool to score (default: --exp); "
            "lets M=2 reuse the m4 pool"
        ),
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="e.g. 101-150",
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--omp",
        type=int,
        default=2,
        help="OMP_NUM_THREADS per rollout",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="per-episode seconds",
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="skip rollouts; score an existing pool",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="args after -- go to the eval CLI",
    )
    args = parser.parse_args()

    args.from_exp = args.from_exp or args.exp
    passthrough = (
        args.rest[1:]
        if args.rest[:1] == ["--"]
        else args.rest
    )
    seeds = parse_seeds(args.seeds)

    if not args.score_only:
        run_pool(args, seeds, passthrough)

    per_seed, summary = score_arm(
        args.task,
        (paths.find_run(args.task, args.from_exp)
         or paths.RESULTS / args.task / args.from_exp),
        seeds,
        args.m,
    )

    out_dir = paths.results_dir(args.task, args.exp)
    report = {
        "task": args.task,
        "exp": args.exp,
        "from_exp": args.from_exp,
        "seeds": args.seeds,
        "summary": summary,
        "per_seed": {
            str(seed): per_seed[seed]
            for seed in seeds
        },
    }

    with open(
        out_dir / "selection.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=1)

    print(
        f"[select] TALLY {args.task}/{args.exp} M={args.m}: "
        f"pick {summary['pick_success']}/{summary['n_seeds']} | "
        f"oracle {summary['oracle_success']}/{summary['n_seeds']} | "
        f"agree {summary['agreement']} "
        f"(mixed {summary['mixed_seeds']}: "
        f"{summary['agreement_mixed']})",
        flush=True,
    )
    print(
        f"[select] report: {out_dir / 'selection.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()