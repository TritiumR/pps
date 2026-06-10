'''
This script checks for NaNs, Infs and huge values in the dataset. 
Created and used because the pi0.5 droid lora finetuning failed for unknown reasons.
The dataset seems clean, so the issue must be somewhere else.
'''
import torch
import numpy as np
from datasets import load_dataset
from lerobot.common.constants import HF_LEROBOT_HOME
from pathlib import Path

# Replace with your dataset repo ID
repo_id = "tw559/flower_stem_vase_data_one_view"
root = HF_LEROBOT_HOME / repo_id
data_dir = root / "data"


def check_tensor(name: str, value, index: int, max_abs: float = 1e6):
    """Check a numeric field for NaNs, Infs and extreme magnitudes."""
    t = torch.as_tensor(value)
    problems = []

    if torch.isnan(t).any():
        problems.append("NaN")
    if torch.isinf(t).any():
        problems.append("Inf")
    if t.numel() > 0 and torch.isfinite(t).any():
        max_val = t[torch.isfinite(t)].abs().max().item()
        if max_val > max_abs:
            problems.append(f"abs(val)>{max_abs} (max={max_val})")

    if problems:
        print(f"[BAD] index={index}, field='{name}': {', '.join(problems)}")
        # Print a small slice to avoid flooding logs
        flat = t.flatten()
        preview = flat[: min(16, flat.numel())]
        print(f"       preview: {preview.tolist()}")
        return True
    return False


print(f"Loading dataset from {data_dir}...")
# Load the parquet files directly
ds = load_dataset("parquet", data_dir=str(data_dir), split="train")

fields_to_check = ["actions", "joint_position", "gripper_position"]

print(f"Checking {len(ds)} items for NaNs / Infs / huge values...")
num_bad = 0
for i in range(len(ds)):
    item = ds[i]

    for field in fields_to_check:
        if field not in item:
            continue
        if check_tensor(field, item[field], i):
            num_bad += 1

    if i % 1000 == 0:
        print(f"Checked {i} items... (bad so far: {num_bad})")

print(f"Check complete. Total bad items (any field in {fields_to_check}): {num_bad}")