#!/usr/bin/env python3
import argparse
import copy
import json
import os
import pathlib
import random

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare JAX and converted PyTorch policy outputs on identical random inputs."
    )
    parser.add_argument("--config-name", default="pi05_droid_jointpos")
    parser.add_argument(
        "--jax-checkpoint",
        type=pathlib.Path,
        default=repo_root / "checkpoints" / "pi05_droid_jointpos",
    )
    parser.add_argument(
        "--torch-checkpoint",
        type=pathlib.Path,
        default=repo_root / "checkpoints" / "pytorch" / "pi05_droid_jointpos",
    )
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument(
        "--pytorch-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--dump-json", type=pathlib.Path, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_random_obs_and_noise(
    seed: int,
    *,
    prompt: str,
    action_horizon: int,
    action_dim: int,
) -> tuple[dict[str, np.ndarray | str], np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = {
        "observation/exterior_image_1_left": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/joint_position": rng.random(7, dtype=np.float32),
        "observation/gripper_position": rng.random(1, dtype=np.float32),
        "prompt": prompt,
    }
    noise = rng.normal(size=(action_horizon, action_dim)).astype(np.float32)
    return obs, noise


def load_policy(
    config_name: str, checkpoint_dir: pathlib.Path, *, pytorch_device: str | None = None
):
    config = _config.get_config(config_name)
    return policy_config.create_trained_policy(
        config,
        checkpoint_dir.resolve(),
        pytorch_device=pytorch_device,
    )


def compare_case(
    jax_policy,
    torch_policy,
    obs: dict[str, np.ndarray | str],
    noise: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    with torch.no_grad():
        jax_actions = np.asarray(
            jax_policy.infer(copy.deepcopy(obs), noise=noise)["actions"]
        )
        torch_actions = np.asarray(
            torch_policy.infer(copy.deepcopy(obs), noise=noise)["actions"]
        )

    diff = torch_actions - jax_actions
    abs_diff = np.abs(diff)
    per_dim_max_abs_diff = abs_diff.max(axis=0)
    arm_abs_diff = abs_diff[:, :7] if abs_diff.shape[1] >= 7 else abs_diff
    gripper_abs_diff = abs_diff[:, 7] if abs_diff.shape[1] >= 8 else None
    return {
        "jax_shape": list(jax_actions.shape),
        "torch_shape": list(torch_actions.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "allclose": bool(np.allclose(jax_actions, torch_actions, atol=atol, rtol=rtol)),
        "arm_max_abs_diff": float(arm_abs_diff.max()),
        "arm_allclose": bool(
            np.allclose(
                jax_actions[:, : arm_abs_diff.shape[1]],
                torch_actions[:, : arm_abs_diff.shape[1]],
                atol=atol,
                rtol=rtol,
            )
        ),
        "gripper_max_abs_diff": (
            float(gripper_abs_diff.max()) if gripper_abs_diff is not None else None
        ),
        "gripper_allclose": (
            bool(
                np.allclose(
                    jax_actions[:, 7],
                    torch_actions[:, 7],
                    atol=atol,
                    rtol=rtol,
                )
            )
            if gripper_abs_diff is not None
            else None
        ),
        "per_dim_max_abs_diff": per_dim_max_abs_diff.tolist(),
        "jax_action_head": jax_actions[:2, :8].tolist(),
        "torch_action_head": torch_actions[:2, :8].tolist(),
        "diff_head": diff[:2, :8].tolist(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.base_seed)

    print(f"Loading JAX checkpoint from: {args.jax_checkpoint.resolve()}")
    jax_policy = load_policy(args.config_name, args.jax_checkpoint)

    print(f"Loading PyTorch checkpoint from: {args.torch_checkpoint.resolve()}")
    print(f"Using PyTorch device: {args.pytorch_device}")
    torch_policy = load_policy(
        args.config_name,
        args.torch_checkpoint,
        pytorch_device=args.pytorch_device,
    )

    train_config = _config.get_config(args.config_name)
    action_horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim

    results = []
    max_abs_values = []
    mean_abs_values = []
    rmses = []

    for index in range(args.num_cases):
        case_seed = args.base_seed + index
        seed_everything(case_seed)
        obs, noise = make_random_obs_and_noise(
            case_seed,
            prompt=args.prompt,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        case_result = compare_case(
            jax_policy,
            torch_policy,
            obs,
            noise,
            atol=args.atol,
            rtol=args.rtol,
        )
        case_result["seed"] = case_seed
        results.append(case_result)
        max_abs_values.append(case_result["max_abs_diff"])
        mean_abs_values.append(case_result["mean_abs_diff"])
        rmses.append(case_result["rmse"])

        print(
            "case={seed} shape={shape} max_abs_diff={max_abs:.6f} "
            "mean_abs_diff={mean_abs:.6f} rmse={rmse:.6f} allclose={allclose}".format(
                seed=case_seed,
                shape=tuple(case_result["jax_shape"]),
                max_abs=case_result["max_abs_diff"],
                mean_abs=case_result["mean_abs_diff"],
                rmse=case_result["rmse"],
                allclose=case_result["allclose"],
            )
        )

    summary = {
        "config_name": args.config_name,
        "jax_checkpoint": str(args.jax_checkpoint.resolve()),
        "torch_checkpoint": str(args.torch_checkpoint.resolve()),
        "pytorch_device": args.pytorch_device,
        "num_cases": args.num_cases,
        "base_seed": args.base_seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "global_max_abs_diff": float(max(max_abs_values)),
        "avg_mean_abs_diff": float(np.mean(mean_abs_values)),
        "avg_rmse": float(np.mean(rmses)),
        "all_cases_allclose": bool(all(item["allclose"] for item in results)),
        "cases": results,
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary JSON to: {args.dump_json}")


if __name__ == "__main__":
    main()
