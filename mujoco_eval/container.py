"""Map host paths and commands into the Isaac Sim container."""

from __future__ import annotations

import os
import pathlib

from . import paths

NAME = os.environ.get("MUJOCO_EVAL_CONTAINER", "pps-jeremysiburian")
HOST_MOUNT = pathlib.Path(
    os.environ.get("MUJOCO_EVAL_HOST_MOUNT", str(paths.REPO.parent))
)
MOUNT = os.environ.get(
    "MUJOCO_EVAL_CONTAINER_MOUNT",
    f"/workspace/{paths.REPO.parent.name}",
)

PYTHON = "/isaac-sim/python.sh"


def to_container(host_path):
    """Convert a host path under the bind mount to its container path."""
    path = pathlib.Path(host_path).resolve()

    try:
        relative = path.relative_to(HOST_MOUNT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{path} is outside the mount {HOST_MOUNT}; "
            "the container cannot see it"
        ) from exc

    return f"{MOUNT}/{relative}"


def exchange_dir():
    """Return the shared directory used for keypoint proposal artifacts."""
    return paths.REPO / "results" / "mg_rekep_exchange"


def exec_cmd(argv, env=None):
    """Build a docker exec command with explicit environment variables."""
    command = ["docker", "exec"]

    for key, value in (env or {}).items():
        command.extend(["-e", f"{key}={value}"])

    return command + [NAME, *argv]