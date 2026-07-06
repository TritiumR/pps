"""Load the single YAML config for the minimal_base pipeline.

Sections: ``run`` / ``grounding`` / ``engine`` / ``sampler`` / ``cost`` (see ``configs/base.yaml``).
``flat`` merges run+grounding+engine+sampler into one namespace for the task/driver; the ``cost`` section
maps directly onto ``minimal_base_cost.CostParams``.
"""
from __future__ import annotations

import types

import yaml


def load_config(path: str) -> dict:
    """Read the YAML config at ``path`` into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flat(config: dict) -> types.SimpleNamespace:
    """Merge run+grounding+engine+sampler sections into one namespace (the ``cost`` section is separate)."""
    merged = {}
    for section in ("run", "grounding", "engine", "sampler"):
        merged.update(config.get(section, {}))
    return types.SimpleNamespace(**merged)
