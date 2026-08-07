"""Aggregate a finished run into metrics, and optionally publish them to wandb.

Modelled on Cory's sim_infra `scripts/eval_pick_ball_mpc_steered_videos_wandb.py`: namespaced
scalars, a per-episode wandb.Table, and one wandb.Video per episode, logged once per run.

Deliberately a POST-HOC aggregator over a run directory rather than a hook in the rollout loop:

  * the filed traces are the source of truth, so a worker that died mid-sweep costs one episode
    rather than the whole summary;
  * it can backfill any run that already exists, including archived flat-layout ones;
  * it is re-runnable when wandb is unreachable, and the eval itself never depends on network.

`summarize` needs no wandb at all and always writes summary.json, so the metrics are available
whether or not the package is installed.

    python -m mujoco_eval.wandb_log --task can --exp base_10          # summary.json only
    python -m mujoco_eval.wandb_log --task can --exp base_10 --wandb  # + publish
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics

from . import paths


def _episode_and_replans(trace):
    """Split one trace into (episode summary, [replan records])."""
    episode, replans = None, []
    with open(trace, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "episode":
                episode = rec
            elif rec.get("kind") == "replan":
                replans.append(rec)
    return episode, replans


def _mean(values):
    values = [float(v) for v in values if v is not None and v == v]
    return float(statistics.fmean(values)) if values else None


def _per_level(records, key, field):
    """Mean of `field` at each denoise level across every replan carrying `key`."""
    buckets = collections.defaultdict(list)
    for rec in records:
        for level in rec.get(key) or ():
            value = level.get(field)
            if value is not None and value == value:
                buckets[int(level["it"])].append(float(value))
    return {it: float(statistics.fmean(vals)) for it, vals in sorted(buckets.items())}


def summarize(run_dir):
    """Metrics, per-episode rows, and video paths for one run directory."""
    rows, replans, videos = [], [], []
    for trace in paths.iter_traces(run_dir):
        episode, per_replan = _episode_and_replans(trace)
        if episode is None:
            continue
        replans.extend(per_replan)
        rows.append({
            "seed": episode.get("seed"),
            "success": bool(episode.get("success")),
            "stage_max": episode.get("stage_max"),
            "stage_final": episode.get("stage_final"),
            "replans": episode.get("replans"),
            "env_steps": episode.get("env_steps"),
            "episode_wall_s": episode.get("episode_wall_s"),
            "replan_wall_median_s": episode.get("replan_wall_median_s"),
        })
        video = paths.seed_artifact(run_dir, episode.get("seed"), "videos", "mp4")
        if video is not None:
            videos.append((episode.get("seed"), bool(episode.get("success")), str(video)))

    rows.sort(key=lambda r: (r["seed"] is None, r["seed"]))
    n = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    metrics = {
        "eval/n_episodes": n,
        "eval/n_success": n_success,
        "eval/success_rate": (n_success / n) if n else None,
        "eval/episode_wall_s_mean": _mean(r["episode_wall_s"] for r in rows),
        "eval/replan_wall_median_s_mean": _mean(r["replan_wall_median_s"] for r in rows),
        "cost/cost_min_mean": _mean(r.get("cost_min") for r in replans),
        "cost/cost_weighted_mean": _mean(r.get("cost_weighted") for r in replans),
        "cost/weight_ess_last_level_mean": _mean(r.get("weight_ess") for r in replans),
    }
    # Where episodes die, not just how many: the histogram is what separates "never grasped"
    # from "grasped then dropped".
    for stage, count in sorted(collections.Counter(
            r["stage_max"] for r in rows if not r["success"]).items(),
            key=lambda kv: (kv[0] is None, kv[0])):
        metrics[f"eval/fail_at_stage_{stage}"] = count

    for field in ("ess", "cost_std"):
        for it, value in _per_level(replans, "base_levels", field).items():
            metrics[f"denoise/{field}_level_{it:02d}"] = value
    for field in ("cos_ref_base", "ref_over_base", "ratio", "supp_d_mean", "cos_task_base"):
        for it, value in _per_level(replans, "steer_levels", field).items():
            metrics[f"steer/{field}_level_{it:02d}"] = value

    return {"metrics": {k: v for k, v in metrics.items() if v is not None},
            "episodes": rows, "videos": videos}


def _run_id(task, exp):
    """Stable wandb run id for one arm, so re-publishing updates rather than duplicates."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", f"{task}-{exp}")[:60]


def publish(task, exp, project="mujoco-eval", entity=None, max_videos=12, use_wandb=False):
    """Write summary.json into the run directory and optionally log it to wandb."""
    run_dir = paths.find_run(task, exp)
    if run_dir is None:
        raise SystemExit(f"no run {task}/{exp} found under {paths.RESULTS}")
    report = summarize(run_dir)
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    out = run_dir / "summary.json"
    out.write_text(json.dumps({"task": task, "exp": exp, "config": config,
                               "metrics": report["metrics"], "episodes": report["episodes"]},
                              indent=2), encoding="utf-8")
    print(f"[wandb-log] {task}/{exp}: {report['metrics'].get('eval/n_success')}/"
          f"{report['metrics'].get('eval/n_episodes')} success -> {out}", flush=True)
    if not use_wandb:
        return report

    import wandb

    # wandb.init mints a NEW run per call unless given an id, so publishing a run twice (a partial
    # mid-sweep, then the full sweep) left two entries for one arm. A deterministic id derived
    # from task/exp makes the second publish overwrite the first.
    run = wandb.init(project=project, entity=entity, name=f"{task}/{exp}",
                     id=_run_id(task, exp), resume="allow",
                     config={"task": task, "exp": exp, **config}, reinit=True)
    payload = dict(report["metrics"])
    if report["episodes"]:
        columns = list(report["episodes"][0])
        payload["episodes/table"] = wandb.Table(
            columns=columns, data=[[row[c] for c in columns] for row in report["episodes"]])
    # Key videos by SEED, not by outcome. wandb aligns panels across runs by key, so
    # videos/success_130 in one arm and videos/failure_130 in another are different panels and
    # cannot be compared. videos/seed_130 is the same panel in every arm, which is what makes
    # toggling two runs show the same seed side by side. Upload the first N seeds in sorted
    # order so every arm carries the same set; the outcome moves into the caption.
    for seed, success, path in sorted(report["videos"])[:max_videos]:
        payload[f"videos/seed_{seed}"] = wandb.Video(
            path, format="mp4", caption=f"seed {seed}: {'success' if success else 'failure'}")
    wandb.log(payload)
    run.finish()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--wandb", action="store_true", help="publish; otherwise summary.json only")
    p.add_argument("--wandb_project", default="mujoco-eval")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--max_videos", type=int, default=12,
                   help="first N seeds, sorted; same set in every arm so panels align")
    a = p.parse_args()
    publish(a.task, a.exp, project=a.wandb_project, entity=a.wandb_entity,
            max_videos=a.max_videos, use_wandb=a.wandb)


if __name__ == "__main__":
    main()
