"""Run ReKep keypoint and constraint generation inside the container."""

import json
import pathlib
import sys

import numpy as np
import yaml

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_HERE = pathlib.Path(__file__).resolve().parent


def main():
    """Generate keypoints, overlays, and optional real constraints."""
    argv = sys.argv[1:]
    instruction = None
    constraints_only = "--constraints_only" in argv

    if constraints_only:
        argv.remove("--constraints_only")

    if "--real_constraints" in argv:
        i = argv.index("--real_constraints")
        instruction = argv[i + 1]
        argv = argv[:i]

    npz_path = pathlib.Path(argv[0]) if argv else _HERE / "frame.npz"
    frame = np.load(npz_path, allow_pickle=True)
    geometry = str(frame["geometry_txt"]) if "geometry_txt" in frame else ""

    if constraints_only:
        import cv2

        projected = cv2.cvtColor(
            cv2.imread(str(_HERE / "overlay_frozen.png")),
            cv2.COLOR_BGR2RGB,
        )
        print(
            "[propose-container] constraints_only: frozen overlay, proposal skipped",
            flush=True,
        )
    else:
        rgb = frame["rgb"]
        points = frame["points"].astype(np.float64)
        masks = frame["masks"]
        obj_pos = frame["obj_pos"]

        with open(
            f"{_REPO}/rekep/configs/default.yaml",
            encoding="utf-8",
        ) as fh:
            cfg = dict(yaml.safe_load(fh)["keypoint_proposer"])

        cfg["device"] = "cpu"
        margin = float(frame["margin"]) if "margin" in frame else 0.6
        cfg["bounds_min"] = (obj_pos.min(axis=0) - margin).tolist()
        cfg["bounds_max"] = (obj_pos.max(axis=0) + margin).tolist()

        from rekep.keypoint_proposal import KeypointProposer

        proposer = KeypointProposer(cfg)
        keypoints, projected = proposer.get_keypoints(rgb, points, masks)

        out = {
            "keypoints": np.asarray(
                keypoints,
                dtype=np.float64,
            ).tolist(),
            "bounds_min": cfg["bounds_min"],
            "bounds_max": cfg["bounds_max"],
            "npz": str(npz_path.name),
            "device": cfg["device"],
            "dino_model": cfg.get(
                "dino_model",
                "dinov2_vits14",
            ),
        }

        with open(
            _HERE / "keypoints.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(out, fh, indent=2)

        import cv2

        cv2.imwrite(
            str(_HERE / "overlay.png"),
            cv2.cvtColor(
                projected,
                cv2.COLOR_RGB2BGR,
            ),
        )
        print(
            f"[propose-container] {len(out['keypoints'])} keypoints -> "
            f"{_HERE / 'keypoints.json'}",
            flush=True,
        )

    if instruction is not None:
        with open(
            f"{_REPO}/rekep/configs/default.yaml",
            encoding="utf-8",
        ) as fh:
            gen_cfg = yaml.safe_load(fh)["constraint_generator"]

        from rekep.constraint_generation import ConstraintGenerator

        out_dir = _HERE / "rekep_real"

        if out_dir.is_dir():
            for file in out_dir.iterdir():
                if (
                    file.name.endswith("_constraints.txt")
                    or file.name == "metadata.json"
                ):
                    file.unlink()

        meta = ConstraintGenerator(gen_cfg).generate(
            projected,
            instruction,
            {},
            str(out_dir),
            geometry=geometry,
        )
        print(
            f"[propose-container] real constraints: "
            f"{json.dumps(meta)} -> {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()