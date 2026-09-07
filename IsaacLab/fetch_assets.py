"""Download the IsaacLab scene/object assets for the pps tasks.

Assets are not committed to the repository (they are large and gitignored).
This pulls them from the Hugging Face dataset into ``IsaacLab/assets/``, which
is where the task configs resolve their ``__file__``-relative asset paths.

    python IsaacLab/fetch_assets.py

The dataset bundles the project's own assets together with the six
``ArtVIP/Interactive_scene`` scenes the tasks load, so one download is enough.
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "Tritiumac/PPS_assets"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--local_dir", default=ASSETS_DIR)
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional glob(s) to fetch a subset, e.g. 'ArtVIP/Interactive_scene/kitchen/*'.",
    )
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns,
    )
    print(f"Assets downloaded to: {path}")


if __name__ == "__main__":
    main()
