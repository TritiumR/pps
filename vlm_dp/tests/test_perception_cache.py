"""CPU regressions for pre-ReKep perception cache semantics."""

import sys
import types

import numpy as np

# Perception imports the simulator camera adapter, which is irrelevant to observe_frame().
helpers = types.ModuleType("rekep.isaaclab_helpers")
helpers.camera_to_rekep_inputs = None
helpers.workspace_bounds_from_scene = None
_helper_name = "rekep.isaaclab_helpers"
_previous_helper = sys.modules.get(_helper_name)
sys.modules[_helper_name] = helpers
try:
    from vlm_dp.perception import Perception
finally:
    if _previous_helper is None:
        del sys.modules[_helper_name]
    else:
        sys.modules[_helper_name] = _previous_helper


def test_read_cache_is_only_used_for_the_initial_frame(monkeypatch):
    """Recovery perception must observe the live scene, not replay the reset cache."""
    perception = Perception({"egg": "egg"})
    loaded = []
    segmented = []

    def load(path):
        loaded.append(path)
        perception.masks = {}

    def segment(_rgb):
        segmented.append(True)
        return {}

    monkeypatch.setattr(perception, "_load_cache", load)
    monkeypatch.setattr(perception, "_segment", segment)
    monkeypatch.setattr(perception, "_verify_identities", lambda: None)
    monkeypatch.setenv("VLMDP_PERCEPTION_CACHE_READ_PATH", "/tmp/reset-cache")
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    points = np.zeros((2, 2, 3), dtype=np.float32)

    perception.observe_frame(rgb, points)
    perception.observe_frame(rgb, points)

    assert loaded == ["/tmp/reset-cache"]
    assert segmented == [True]
