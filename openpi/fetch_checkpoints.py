"""Download the PROXY steering checkpoints for the pps IsaacLab evaluations.

Checkpoints are not committed to the repository (they are large and gitignored).
This pulls them into ``openpi/checkpoints/``.

    python openpi/fetch_checkpoints.py

The destination matters: ``eval_steering.py`` derives the training-config name
from the checkpoint path by locating the ``checkpoints`` path segment and
reading the segment after it, so the files must land under a directory named
``checkpoints`` for the config lookup to succeed.
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "Tritiumac/PPS_checkpoints"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--local_dir", default=CHECKPOINT_DIR)
    parser.add_argument(
        "--task",
        nargs="*",
        choices=["pot", "tea", "weight"],
        default=None,
        help="Only fetch these tasks (default: all).",
    )
    args = parser.parse_args()

    if os.path.basename(os.path.normpath(args.local_dir)) != "checkpoints":
        parser.error(
            f"--local_dir must be named 'checkpoints' (got {args.local_dir!r}); "
            "eval_steering.py derives the config name from that path segment."
        )

    allow = None
    if args.task:
        allow = [f"proxy_isaaclab_droid_{t}_pi05_jointpos/*" for t in args.task]

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=allow,
    )
    print(f"Checkpoints downloaded to: {path}")


if __name__ == "__main__":
    main()
