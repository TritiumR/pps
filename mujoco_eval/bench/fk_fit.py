"""Fit the robot base transform and TCP offset to recorded end-effector poses."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch

from .. import paths

paths.ensure_repo_on_path()

from sim_free_mpc.fk import PandaFK  # noqa: E402

DATA = paths.DATA
OUT = paths.FK_FITS


def load_frames(hdf5, n_demos, stride):
    """Load sampled joint and end-effector poses from demonstrations."""
    q, eef_pos, eef_quat = [], [], []

    with h5py.File(hdf5, "r") as f:
        names = sorted(
            f["data"].keys(),
            key=lambda s: int(s.split("_")[1]),
        )[:n_demos]

        for name in names:
            demo = f[f"data/{name}"]
            q.append(
                np.asarray(
                    demo["obs/robot0_joint_pos"][::stride],
                    dtype=np.float64,
                )
            )
            eef_pos.append(
                np.asarray(
                    demo["obs/robot0_eef_pos"][::stride],
                    dtype=np.float64,
                )
            )
            eef_quat.append(
                np.asarray(
                    demo["obs/robot0_eef_quat"][::stride],
                    dtype=np.float64,
                )
            )

    return (
        np.concatenate(q),
        np.concatenate(eef_pos),
        np.concatenate(eef_quat),
    )


def quat_to_mat(q_wxyz):
    """Convert batched wxyz quaternions to rotation matrices."""
    w, x, y, z = np.moveaxis(q_wxyz, -1, 0)

    return np.stack(
        [
            np.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                -1,
            ),
            np.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ],
                -1,
            ),
            np.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ],
                -1,
            ),
        ],
        -2,
    )


def mat_to_quat(m):
    """Convert one rotation matrix to a normalized wxyz quaternion."""
    w = 0.5 * np.sqrt(
        max(1.0 + m[0, 0] + m[1, 1] + m[2, 2], 0.0)
    )
    x = np.copysign(
        0.5
        * np.sqrt(
            max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)
        ),
        m[2, 1] - m[1, 2],
    )
    y = np.copysign(
        0.5
        * np.sqrt(
            max(1.0 - m[0, 0] + m[1, 1] - m[2, 2], 0.0)
        ),
        m[0, 2] - m[2, 0],
    )
    z = np.copysign(
        0.5
        * np.sqrt(
            max(1.0 - m[0, 0] - m[1, 1] + m[2, 2], 0.0)
        ),
        m[1, 0] - m[0, 1],
    )

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def kabsch(X, Y):
    """Return the rigid transform minimizing row-wise squared error."""
    x_center = X.mean(0)
    y_center = Y.mean(0)
    covariance = (X - x_center).T @ (Y - y_center)

    U, _, Vt = np.linalg.svd(covariance)
    correction = np.diag(
        [
            1.0,
            1.0,
            np.sign(np.linalg.det(Vt.T @ U.T)),
        ]
    )
    rotation = Vt.T @ correction @ U.T

    return rotation, y_center - rotation @ x_center


def fit(p8, R8, Y, iters=200):
    """Fit the base transform and link-frame TCP offset."""
    tcp_offset = np.zeros(3)
    base_rotation = np.eye(3)
    previous_error = np.inf
    error = None

    for _ in range(iters):
        matrix = np.concatenate(
            [
                np.einsum("ij,njk->nik", base_rotation, R8),
                np.broadcast_to(
                    np.eye(3),
                    (len(Y), 3, 3),
                ),
            ],
            axis=2,
        ).reshape(-1, 6)
        target = (Y - p8 @ base_rotation.T).reshape(-1)

        solution, *_ = np.linalg.lstsq(
            matrix,
            target,
            rcond=None,
        )
        tcp_offset = solution[:3]
        base_position = solution[3:]

        transformed = p8 + np.einsum(
            "nij,j->ni",
            R8,
            tcp_offset,
        )
        base_rotation, base_position = kabsch(
            transformed,
            Y,
        )
        error = np.linalg.norm(
            Y
            - (
                transformed @ base_rotation.T
                + base_position
            ),
            axis=1,
        )

        if abs(previous_error - error.mean()) < 1e-15:
            break

        previous_error = error.mean()

    return (
        base_rotation,
        base_position,
        tcp_offset,
        error,
    )


def quat_mean(quaternions):
    """Return the principal-eigenvector mean of sign-aligned quaternions."""
    quaternions = quaternions * np.sign(
        quaternions[:, :1] + 1e-300
    )
    _, vectors = np.linalg.eigh(
        (
            quaternions[:, :, None]
            * quaternions[:, None, :]
        ).mean(0)
    )
    quaternion = vectors[:, -1]

    return quaternion * np.sign(quaternion[0] + 1e-300)


def geodesic_deg(rotation_a, rotation_b):
    """Return batched rotational distance in degrees."""
    trace = np.clip(
        (
            np.einsum(
                "nji,njk->nik",
                rotation_a,
                rotation_b,
            ).trace(axis1=1, axis2=2)
            - 1
        )
        / 2,
        -1,
        1,
    )
    return np.degrees(np.arccos(trace))


def main():
    """Fit FK calibration parameters and write them to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stack_d0")
    parser.add_argument("--n_demos", type=int, default=60)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    q, eef_pos, eef_quat_raw = load_frames(
        DATA / args.task / "demo.hdf5",
        args.n_demos,
        args.stride,
    )
    print(
        f"{args.task}: {len(q)} frames from "
        f"{args.n_demos} demos (stride {args.stride})"
    )

    fk = PandaFK(ee_offset=(0.0, 0.0, 0.0))
    output = fk.forward(torch.as_tensor(q))
    matrices = output.ee_matrix.numpy()
    p8 = matrices[:, :3, 3]
    R8 = matrices[:, :3, :3]

    base_rotation, base_position, tcp_offset, error = fit(
        p8,
        R8,
        eef_pos,
    )
    error_mm = error * 1000.0

    print(
        f"position fit: base pos {base_position.round(6)}  "
        f"base quat(wxyz) "
        f"{mat_to_quat(base_rotation).round(6)}"
    )
    print(
        f"tcp offset (link8 frame) "
        f"{tcp_offset.round(6)}"
    )
    print(
        f"residual mm: mean {error_mm.mean():.4f}  "
        f"p95 {np.percentile(error_mm, 95):.4f}  "
        f"max {error_mm.max():.4f}"
    )

    orientation = {}

    for convention in ("xyzw", "wxyz"):
        q_wxyz = (
            np.roll(eef_quat_raw, 1, axis=1)
            if convention == "xyzw"
            else eef_quat_raw
        )
        eef_rotation = quat_to_mat(q_wxyz)
        offset_rotations = np.einsum(
            "nji,jk,nkl->nil",
            R8,
            base_rotation.T,
            eef_rotation,
        )
        offset_quaternions = np.stack(
            [
                mat_to_quat(rotation)
                for rotation in offset_rotations
            ]
        )
        offset_quaternion = quat_mean(
            offset_quaternions
        )
        offset_rotation = quat_to_mat(
            offset_quaternion[None]
        )[0]
        geodesic = geodesic_deg(
            np.broadcast_to(
                offset_rotation,
                offset_rotations.shape,
            ),
            offset_rotations,
        )

        orientation[convention] = {
            "R_off_quat_wxyz": (
                offset_quaternion.round(6).tolist()
            ),
            "geodesic_deg_mean": float(
                geodesic.mean()
            ),
            "geodesic_deg_p95": float(
                np.percentile(geodesic, 95)
            ),
            "geodesic_deg_max": float(
                geodesic.max()
            ),
        }

        print(
            f"orientation ({convention} assumed): "
            f"R_off quat(wxyz) "
            f"{offset_quaternion.round(4)}  "
            f"geodesic deg mean "
            f"{geodesic.mean():.4f} "
            f"p95 "
            f"{np.percentile(geodesic, 95):.4f}"
        )

    best = min(
        orientation,
        key=lambda convention: orientation[convention][
            "geodesic_deg_mean"
        ],
    )
    yaw = (
        np.degrees(
            np.arctan2(
                orientation[best]["R_off_quat_wxyz"][3],
                orientation[best]["R_off_quat_wxyz"][0],
            )
        )
        * 2
    )

    print(
        f"stored convention: {best} "
        f"(constant offset); "
        f"R_off yaw about z ~ {yaw:.2f} deg"
    )

    gate = float(np.percentile(error_mm, 95)) < 5.0
    print(
        f"GATE p95 < 5mm: "
        f"{'PASS' if gate else 'FAIL'}"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "task": args.task,
        "n_frames": int(len(q)),
        "n_demos": args.n_demos,
        "stride": args.stride,
        "base_pos": base_position.round(8).tolist(),
        "base_quat_wxyz": (
            mat_to_quat(base_rotation).round(8).tolist()
        ),
        "tcp_offset_link8": tcp_offset.round(8).tolist(),
        "pos_residual_mm": {
            "mean": float(error_mm.mean()),
            "p95": float(
                np.percentile(error_mm, 95)
            ),
            "max": float(error_mm.max()),
        },
        "orientation": orientation,
        "stored_quat_convention": best,
        "gate_p95_lt_5mm": gate,
    }

    output_path = OUT / f"fk_fit_{args.task}.json"
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=1)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()