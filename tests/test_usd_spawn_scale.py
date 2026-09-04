import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/_usd_scale.py"
)
_SPEC = importlib.util.spec_from_file_location("usd_scale_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
compose_spawn_scale = _MODULE.compose_spawn_scale


def test_spawn_scale_is_not_compounded_when_clone_already_has_it():
    assert compose_spawn_scale((0.7, 0.7, 0.7), (0.7, 0.7, 0.7)) == pytest.approx(
        (0.7, 0.7, 0.7)
    )


def test_spawn_scale_preserves_distinct_authored_scale_as_multiplier():
    assert compose_spawn_scale((2.0, 3.0, 4.0), (0.5, 0.25, 2.0)) == pytest.approx(
        (1.0, 0.75, 8.0)
    )


def test_spawn_scale_rejects_non_xyz_values():
    with pytest.raises(ValueError, match="exactly three"):
        compose_spawn_scale((1.0, 1.0), (0.7, 0.7, 0.7))
