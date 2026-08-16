"""Resolve cost configurations by path or name."""

from __future__ import annotations

import pathlib

CONFIGS = pathlib.Path(__file__).resolve().parent / "configs"
TEST_CONFIGS = CONFIGS / "test_configs"


def resolve(name_or_path, repo_root=None) -> pathlib.Path:
    """Return the first matching cost configuration path."""
    given = pathlib.Path(name_or_path)
    candidates = [given]

    if repo_root is not None and not given.is_absolute():
        candidates.append(pathlib.Path(repo_root) / given)

    filename = given.name if given.suffix else f"{given.name}.yaml"
    candidates.extend(
        [
            CONFIGS / filename,
            TEST_CONFIGS / filename,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"no cost config {name_or_path!r}; "
        f"looked in {CONFIGS} and {TEST_CONFIGS}"
    )