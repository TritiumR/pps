import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME


def compute_ranges(ds, fields):
    # We keep running per-dimension stats for each field:
    # - min, max
    # - count, sum, sum_sq (for mean/std)
    # - all finite values (for exact 1st / 99th quantiles)
    stats = {
        f: {
            "min": None,      # torch.Tensor[num_dims] or None
            "max": None,      # torch.Tensor[num_dims] or None
            "count": None,    # torch.LongTensor[num_dims] or None
            "sum": None,      # torch.DoubleTensor[num_dims] or None
            "sum_sq": None,   # torch.DoubleTensor[num_dims] or None
            "values": None,   # list[list[torch.Tensor]] or None, size [num_dims][*]
        }
        for f in fields
    }

    for i in range(len(ds)):
        item = ds[i]
        for field in fields:
            if field not in item:
                continue

            t = torch.as_tensor(item[field], dtype=torch.float32)

            # Ensure we always interpret the last (or only) dimension as "feature" dimension.
            # Shapes:
            #   scalar ()      -> (1, 1)   (1 feature)
            #   vector (D,)    -> (1, D)   (D features)
            #   tensor (...,D) -> (-1, D)  (D features)
            if t.ndim == 0:
                dim = 1
                t_2d = t.reshape(1, 1)
            elif t.ndim == 1:
                dim = t.shape[0]
                t_2d = t.reshape(1, dim)
            else:
                dim = t.shape[-1]
                t_2d = t.reshape(-1, dim)

            finite_mask = torch.isfinite(t_2d)
            if not finite_mask.any():
                continue

            field_stats = stats[field]

            # Initialize running stats for this field the first time we see it.
            if field_stats["min"] is None:
                field_stats["min"] = torch.full((dim,), float("inf"), dtype=torch.float32)
                field_stats["max"] = torch.full(
                    (dim,), float("-inf"), dtype=torch.float32
                )
                field_stats["count"] = torch.zeros(dim, dtype=torch.long)
                field_stats["sum"] = torch.zeros(dim, dtype=torch.float64)
                field_stats["sum_sq"] = torch.zeros(dim, dtype=torch.float64)
                field_stats["values"] = [[] for _ in range(dim)]
            else:
                # Sanity check: ensure dimension is consistent for a given field
                prev_dim = field_stats["min"].shape[0]
                if dim != prev_dim:
                    raise ValueError(
                        f"Inconsistent last-dimension size for field '{field}': "
                        f"previous={prev_dim}, current={dim}"
                    )

            # Update per-dimension stats
            for d in range(dim):
                valid = t_2d[finite_mask[:, d], d]
                if valid.numel() == 0:
                    continue

                cur_min = valid.min()
                cur_max = valid.max()

                field_stats["min"][d] = torch.minimum(field_stats["min"][d], cur_min)
                field_stats["max"][d] = torch.maximum(field_stats["max"][d], cur_max)

                n = valid.numel()
                field_stats["count"][d] += n
                field_stats["sum"][d] += valid.double().sum()
                field_stats["sum_sq"][d] += (valid.double() ** 2).sum()
                field_stats["values"][d].append(valid.double())

        if i % 1000 == 0 and i > 0:
            print(f"Scanned {i} items...")

    # Finalize derived statistics (mean, std, 1st and 99th quantiles).
    for field, field_stats in stats.items():
        if field_stats["min"] is None:
            # Field never encountered
            continue

        count = field_stats["count"]
        dim = count.shape[0]

        count_float = count.to(torch.float64).clamp(min=1.0)
        sum_ = field_stats["sum"]
        sum_sq = field_stats["sum_sq"]

        mean = sum_ / count_float
        var = (sum_sq / count_float) - mean**2
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        q01 = torch.full((dim,), float("nan"), dtype=torch.float64)
        q99 = torch.full((dim,), float("nan"), dtype=torch.float64)

        for d in range(dim):
            if count[d] == 0:
                continue
            vals = torch.cat(field_stats["values"][d], dim=0)
            q01[d] = torch.quantile(vals, 0.01)
            q99[d] = torch.quantile(vals, 0.99)

        field_stats["mean"] = mean
        field_stats["std"] = std
        field_stats["q01"] = q01
        field_stats["q99"] = q99

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Show value ranges (min/max) for key numeric fields in a LeRobot dataset "
            "stored in parquet format."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Root directory of the LeRobot dataset (directory that contains 'data/').",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help=(
            "Optional HF repo id under HF_LEROBOT_HOME, e.g. 'your_hf_username/my_droid_dataset'. "
            "Used if --dataset_dir is not provided."
        ),
    )

    args = parser.parse_args()

    if args.dataset_dir is not None:
        root = Path(args.dataset_dir)
    elif args.repo_id is not None:
        root = HF_LEROBOT_HOME / args.repo_id
    else:
        # Default to your sample official dataset path
        root = Path(
            "/home/chuanruo/.cache/huggingface/lerobot/your_hf_username/my_droid_dataset"
        )

    data_dir = root / "data"
    print(f"Loading dataset from {data_dir}...")

    ds = load_dataset("parquet", data_dir=str(data_dir), split="train")

    fields = ["actions", "joint_position", "gripper_position"]
    print(f"Computing ranges for fields: {fields}")
    stats = compute_ranges(ds, fields)

    print("\n=== Ranges ===")
    for field in fields:
        field_stats = stats[field]
        if field_stats["min"] is None:
            print(f"- {field}: not found or no finite values")
        else:
            print(f"- {field}:")
            min_vals = field_stats["min"]
            max_vals = field_stats["max"]
            count_vals = field_stats["count"]
            mean_vals = field_stats["mean"]
            std_vals = field_stats["std"]
            q01_vals = field_stats["q01"]
            q99_vals = field_stats["q99"]

            num_dims = int(min_vals.shape[0])
            for d in range(num_dims):
                c = int(count_vals[d].item())
                if c == 0:
                    print(f"  dim {d}: no finite values")
                    continue

                print(
                    "  dim {d}: "
                    "min={min_val:.6f}, max={max_val:.6f}, "
                    "mean={mean_val:.6f}, std={std_val:.6f}, "
                    "q01={q01_val:.6f}, q99={q99_val:.6f}, "
                    "count={count}".format(
                        d=d,
                        min_val=float(min_vals[d].item()),
                        max_val=float(max_vals[d].item()),
                        mean_val=float(mean_vals[d].item()),
                        std_val=float(std_vals[d].item()),
                        q01_val=float(q01_vals[d].item()),
                        q99_val=float(q99_vals[d].item()),
                        count=c,
                    )
                )


if __name__ == "__main__":
    main()


