#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments" / "weight_eval_sweep"
MANIFEST = RUN_ROOT / "manifest.tsv"
STATUS_DIR = RUN_ROOT / "status"
OUTPUT = RUN_ROOT / "progress.md"
RESULT_ROOT = ROOT / "results" / "Isaac-Weight-Droid-Visuomotor-v0"
SEED_RE = re.compile(r"^(\d+)_(success|fail)\.mp4$")


def read_manifest() -> list[dict[str, str]]:
    rows = []
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        rows.append(dict(zip(header, line.split("\t"), strict=True)))
    return rows


def read_status(experiment: str) -> dict[str, object]:
    path = STATUS_DIR / f"{experiment}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "invalid-status"}


def episode_outcomes(experiment: str) -> dict[int, str]:
    outcomes: dict[int, tuple[int, str]] = {}
    result_root = RESULT_ROOT / experiment
    if not result_root.exists():
        return {}
    for video in result_root.rglob("*.mp4"):
        match = SEED_RE.match(video.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if not 1 <= seed <= 20:
            continue
        mtime = video.stat().st_mtime_ns
        if seed not in outcomes or mtime > outcomes[seed][0]:
            outcomes[seed] = (mtime, match.group(2))
    return {seed: outcome for seed, (_, outcome) in outcomes.items()}


def rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return path


def main() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    rendered = []
    states = Counter()
    for row in rows:
        experiment = row["experiment"]
        status = read_status(experiment)
        outcomes = episode_outcomes(experiment)
        complete = len(outcomes)
        successes = sum(value == "success" for value in outcomes.values())
        failures = sum(value == "fail" for value in outcomes.values())
        state = str(status.get("state", "queued"))
        if complete >= 20:
            state = "complete"
        elif row["kind"] == "task_only" and complete:
            state = "running"
        states[state] += 1
        message = str(status.get("message", "-"))
        if message.startswith(str(ROOT)):
            message = rel(message)
        rendered.append(
            "| {experiment} | {kind} | {scale} | {job}/{slot} | {gpus}×{workers} | "
            "{state} | {complete}/20 | {successes} | {failures} | {message} |".format(
                **row,
                state=state,
                complete=complete,
                successes=successes,
                failures=failures,
                message=message.replace("|", "\\|"),
            )
        )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
    contents = f"""# Weight eval 实验进度

- 最后更新：{now}
- 种子：1–20（命令为 `--seed_start 1 --seed_end 21`）
- 汇总：{summary}
- ref-only 已按要求停止；停止前完成 13/20 seeds（0 success / 13 fail），不属于本轮待完成实验。
- 628885 的首次 scan 随已完成的 task-only tmux server 被 Slurm 清理；已于 03:36 EDT 使用独立 socket 恢复为两个并行队列。
- scale 0.3 后续在 18/20 时再次随外部 step 结束；已于 13:06 EDT 仅补跑 seeds 19–20 并完成。

| 实验 | 类型 | scale | job/slot | GPU×worker | 状态 | seeds | success | fail | 日志/备注 |
|---|---|---:|---|---:|---|---:|---:|---:|---|
{chr(10).join(rendered)}

调度约束：每个节点最多同时运行 2 个 eval。job 640102 的两个 eval 分别使用逻辑 GPU `0,1` 和 `2,3`，每个 eval 内部由 2 workers 均分 seeds。
"""
    temp_output = OUTPUT.with_name(f"{OUTPUT.name}.tmp.{os.getpid()}")
    temp_output.write_text(contents, encoding="utf-8")
    temp_output.replace(OUTPUT)


if __name__ == "__main__":
    main()
