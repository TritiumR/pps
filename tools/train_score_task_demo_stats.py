#!/usr/bin/env python3
"""Prepare and train image-only bidirectional task proxies with demo stats."""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI = ROOT / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "scripts"))

import train_proxy_score_pytorch as trainer  # noqa: E402
from openpi.training import config as training_config  # noqa: E402


TASKS = ("weight", "tea", "pot", "capsule")
DEFAULT_EXP_NAME = "task_eps_bidir_openpi_image_only_demo_stats"
DEFAULT_STATS_ROOT = pathlib.Path(
    "/home/yl4535/sharefs/pps/demo_stats/cn356"
)


def build_config(args: argparse.Namespace):
    config_name = f"score_task_{args.task}"
    repo_id = f"cn356/isaaclab_{args.task}"
    stats_root = pathlib.Path(
        os.environ.get("DEMO_STATS_ROOT", str(DEFAULT_STATS_ROOT))
    )
    stats_dir = stats_root / f"isaaclab_{args.task}"
    if not (stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Demo stats not found: {stats_dir}")

    config = training_config.get_config(config_name)
    data = dataclasses.replace(
        config.data,
        repo_id=repo_id,
        assets=dataclasses.replace(config.data.assets, asset_id=repo_id),
        norm_stats_dir=str(stats_dir),
        use_quantile_norm=True,
    )
    model = dataclasses.replace(
        config.model,
        prediction_type="epsilon",
        bidirectional_attention=True,
    )
    return dataclasses.replace(
        config,
        model=model,
        data=data,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=0,
        overwrite=args.overwrite,
        resume=args.resume,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.mode == "prepare":
        trainer.init_logging()
        trainer.prepare_task_cache(
            config,
            args.cache_path,
            num_workers=args.num_workers,
        )
        return

    os.environ[trainer.TASK_CACHE_ENV] = str(args.cache_path)
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
