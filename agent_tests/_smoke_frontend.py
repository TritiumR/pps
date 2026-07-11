"""Offline smoke test for the ReKep front-end port (no IsaacLab needed).

Builds a synthetic (rgb, points, masks) frame with two fake objects, runs the
keypoint proposer (DINOv2 + KMeans + overlay), and -- if --vlm is passed and
OPENAI_API_KEY is set -- runs GPT-4o constraint generation on the overlay image.
This validates the port mechanically (imports, DINOv2 load, clustering, projection,
prompt/parse) before the IsaacLab obs adapter can be exercised (which needs assets).

Run:  /isaac-sim/python.sh rekep/_smoke_frontend.py [--vlm]
"""

import argparse
import os
import sys

import numpy as np

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in agent_tests/)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from rekep.constraint_generation import ConstraintGenerator  # noqa: E402
from rekep.keypoint_proposal import KeypointProposer  # noqa: E402
from rekep.utils import load_default_config  # noqa: E402


def _synthetic_frame(height=224, width=224):
    """Two solid-color squares on a gray background, with matching masks + world points."""
    rng = np.random.default_rng(0)
    rgb = np.full((height, width, 3), 120, dtype=np.uint8)
    rgb += rng.integers(-8, 8, size=rgb.shape, dtype=np.int16).astype(np.uint8)  # mild texture
    masks = np.zeros((height, width), dtype=np.int32)
    # object 1 (reddish) and object 2 (bluish)
    rgb[40:90, 40:90] = (200, 60, 60)
    masks[40:90, 40:90] = 1
    rgb[130:180, 140:190] = (60, 80, 200)
    masks[130:180, 140:190] = 2

    # Dense world points on a tilted plane inside the (generous) config bounds.
    vv, uu = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = np.stack(
        [
            uu / width * 0.6,            # x in [0, 0.6]
            (vv / height - 0.5) * 0.6,   # y in [-0.3, 0.3]
            0.72 + uu / width * 0.05,    # z ~ 0.72..0.77
        ],
        axis=-1,
    ).astype(np.float32)
    return rgb, points, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", action="store_true", help="also call GPT-4o constraint generation")
    args = parser.parse_args()

    config = load_default_config()
    out_dir = os.path.join(_REPO_DIR, "results", "rekep", "_smoke")
    os.makedirs(out_dir, exist_ok=True)

    rgb, points, masks = _synthetic_frame()
    print(f"synthetic frame: rgb{rgb.shape} points{points.shape} masks ids={np.unique(masks).tolist()}")

    proposer = KeypointProposer(config["keypoint_proposer"])
    keypoints, projected = proposer.get_keypoints(rgb, points, masks)
    print(f"proposed {len(keypoints)} keypoints; sample coords:\n{np.round(keypoints[:5], 3)}")

    import cv2

    kp_path = os.path.join(out_dir, "keypoints.png")
    cv2.imwrite(kp_path, projected[..., ::-1])
    print(f"saved keypoint overlay -> {kp_path}")

    if args.vlm:
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
        gen = ConstraintGenerator(config["constraint_generator"])
        task_dir = gen.generate(
            projected,
            "pick up the red square and place it on the blue square",
            metadata={"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)},
            task_dir=os.path.join(out_dir, "vlm"),
        )
        print(f"constraint program written -> {task_dir}")
        print("files:", sorted(os.listdir(task_dir)))

    print("SMOKE_FRONTEND_OK")


if __name__ == "__main__":
    main()
