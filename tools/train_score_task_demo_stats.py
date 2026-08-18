#!/usr/bin/env python3
"""Prepare and train image-only bidirectional task proxies with demo mean/std stats."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI = ROOT / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "scripts"))

import train_proxy_score_pytorch as trainer  # noqa: E402
from openpi.training import config as training_config  # noqa: E402


TASKS = ("weight", "tea", "pot", "capsule")
DEFAULT_EXP_NAME = "task_eps_bidir_openpi_image_only_demo_meanstd"
DEFAULT_STATS_ROOT = pathlib.Path(
    "/autodl-fs/data/yl4535/pps/demo_stats/cn356"
)


def _validate_mean_std_stats(stats_dir: pathlib.Path, task: str) -> None:
    """Require demo mean/std and cross-check standalone eval stats when present."""
    payload = json.loads((stats_dir / "norm_stats.json").read_text())
    norm_stats = payload.get("norm_stats", payload)
    actions = norm_stats.get("actions", {})
    missing = [field for field in ("mean", "std") if field not in actions]
    if missing:
        raise ValueError(f"{stats_dir}: actions are missing mean/std fields: {missing}")

    # eval's demo_delta decoder consumes the standalone file. It is redundant with
    # the action entry in the complete OpenPI bundle, so verify rather than copy it.
    action_stats_path = stats_dir.parent.parent / f"{task}_action_norm_stats.json"
    if not action_stats_path.is_file():
        return
    action_stats = json.loads(action_stats_path.read_text())
    for field in ("mean", "std"):
        embedded = np.asarray(actions[field], dtype=np.float64)
        standalone = np.asarray(action_stats[field], dtype=np.float64)
        if embedded.shape != standalone.shape or not np.allclose(
            embedded, standalone, rtol=0.0, atol=1e-7
        ):
            raise ValueError(
                f"{field} mismatch between {stats_dir / 'norm_stats.json'} and "
                f"{action_stats_path}"
            )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_init_checkpoint(init_from: pathlib.Path, stats_dir: pathlib.Path) -> pathlib.Path:
    """Require a strict model checkpoint whose normalization is byte-identical to task training."""
    checkpoint_dir = init_from if init_from.is_dir() else init_from.parent
    model_path = checkpoint_dir / "model.safetensors" if init_from.is_dir() else init_from
    if not model_path.is_file():
        raise FileNotFoundError(f"Initial model checkpoint not found: {model_path}")
    norm_files = sorted((checkpoint_dir / "assets").glob("**/norm_stats.json"))
    if len(norm_files) != 1:
        raise ValueError(
            f"Expected exactly one norm_stats.json under {checkpoint_dir / 'assets'}, "
            f"found {len(norm_files)}"
        )
    task_norm = stats_dir / "norm_stats.json"
    if _sha256(norm_files[0]) != _sha256(task_norm):
        raise ValueError(
            f"Init checkpoint normalization differs from task training: {norm_files[0]} vs {task_norm}"
        )
    return model_path


def build_config(args: argparse.Namespace):
    config_name = f"score_task_{args.task}"
    repo_id = f"cn356/isaaclab_{args.task}"
    stats_root = pathlib.Path(
        os.environ.get("DEMO_STATS_ROOT", str(DEFAULT_STATS_ROOT))
    )
    stats_dir = stats_root / f"isaaclab_{args.task}"
    if not (stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Demo stats not found: {stats_dir}")
    _validate_mean_std_stats(stats_dir, args.task)

    config = training_config.get_config(config_name)
    data = dataclasses.replace(
        config.data,
        repo_id=repo_id,
        assets=dataclasses.replace(config.data.assets, asset_id=repo_id),
        norm_stats_dir=str(stats_dir),
        use_quantile_norm=False,
    )
    model = dataclasses.replace(
        config.model,
        prediction_type="epsilon",
        bidirectional_attention=True,
    )
    init_path = None
    if args.init_from is not None:
        if args.resume:
            raise ValueError("--init-from initializes a new run and cannot be combined with --resume.")
        init_path = _validate_init_checkpoint(args.init_from, stats_dir)
    return dataclasses.replace(
        config,
        model=model,
        data=data,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=0,
        overwrite=args.overwrite,
        resume=args.resume,
        pytorch_weight_path=str(init_path) if init_path is not None else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--init-from",
        type=pathlib.Path,
        default=None,
        help="Ref checkpoint directory (or model.safetensors) used to initialize a new task run.",
    )
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
