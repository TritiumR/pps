"""Offline priority-label context (vlm_dp.offline_context): stage inference and key contract."""

import numpy as np

from vlm_dp.offline_context import (
    _on_place,
    _weight_constraints,
    _weight_stage,
    check_representable,
    infer_stage,
    priority_context,
    weight_frame_context,
    WEIGHT_EXTENTS,
)


def _objects(pear, apple, scale=(0.66, 1.44, 0.20)):
    return {"pear": {"pos": np.array(pear)}, "apple": {"pos": np.array(apple)},
            "scale": {"pos": np.array(scale)}}


SCALE = (0.66, 1.44, 0.20)
FAR_EE = (0.50, 1.55, 0.45)


def test_stage_walkthrough():
    # 1: nothing placed, hand away and open -> grasp pear
    g, p, t, placed = infer_stage(_objects((0.22, 1.38, 0.30), (0.11, 1.41, 0.28)), FAR_EE, False)
    assert (g, p, t) == ("pear", None, None) and not placed
    # 2: pear in hand but still away from the scale -> LIFT (grasp_obj and payload, no destination).
    # The runtime ladder is grasp -> lift -> place; collapsing this frame into place gave every lift
    # frame the place stage's cost, so proxy labels came from a different cost than evaluation.
    g, p, t, placed = infer_stage(_objects((0.50, 1.53, 0.44), (0.11, 1.41, 0.28)),
                                  (0.50, 1.55, 0.45), True)
    assert (g, p, t) == ("pear", "pear", None) and not placed
    # 3: pear on the scale, hand free -> grasp apple
    g, p, t, placed = infer_stage(_objects((0.66, 1.45, 0.26), (0.11, 1.41, 0.28)), FAR_EE, False)
    assert (g, p, t) == ("apple", None, None) and placed == frozenset({"pear"})
    # 4: both placed -> terminal place stage, released
    g, p, t, placed = infer_stage(_objects((0.66, 1.45, 0.26), (0.64, 1.42, 0.26)), FAR_EE, False)
    assert (g, p, t) == (None, "apple", "scale") and placed == frozenset({"pear", "apple"})


def test_hover_at_seat_in_hand_is_still_the_place_stage():
    # pear directly above the scale, IN HAND: the on-place geometry fires but held must win --
    # these are the release-critical frames and they belong to place_pear, not grasp_apple
    ee = (0.66, 1.45, 0.30)
    g, p, t, placed = infer_stage(_objects((0.66, 1.45, 0.26), (0.11, 1.41, 0.28)), ee, True)
    assert (g, p, t) == (None, "pear", "scale")
    assert "pear" not in placed


def test_near_ee_open_hand_is_not_held():
    # hovering at the pear with an OPEN hand is still the grasp stage, not a carry
    g, p, t, _ = infer_stage(_objects((0.50, 1.53, 0.44), (0.11, 1.41, 0.28)),
                             (0.50, 1.55, 0.45), False)
    assert (g, p, t) == ("pear", None, None)


def test_context_contract():
    # Held AT the seat, so this is the place stage and the full key set applies (a lift stage
    # correctly carries no place_point).
    base = {"objects": _objects((0.66, 1.45, 0.26), (0.11, 1.41, 0.28)),
            "eef_pos": np.array([0.66, 1.45, 0.30], np.float32),
            "subtasks": {}, "task": "weight"}
    ctx = priority_context(base, gripper_closed=True)
    for key in ("objects", "grasp_obj", "payload", "place_target", "destination", "placed",
                "target", "place_point", "contact", "orient", "place_mode", "gripper_intent",
                "z_table", "plan_ref"):
        assert key in ctx, f"missing context key {key!r}"
    assert ctx["payload"] == "pear" and ctx["place_target"] == "scale"
    # seat = scale root + half_z (the release_gt top convention)
    assert abs(float(ctx["place_point"][2]) - (0.20 + WEIGHT_EXTENTS["scale"][2])) < 1e-6
    # objects carry extents for CompositeCost.__call__
    assert ctx["objects"]["pear"]["extents"] == WEIGHT_EXTENTS["pear"]
    # grasp-stage variant targets the object (pear away from the seat, hand open)
    away = {**base, "objects": _objects((0.50, 1.53, 0.44), (0.11, 1.41, 0.28)),
            "eef_pos": np.array([0.50, 1.55, 0.45], np.float32)}
    ctx2 = priority_context(away, gripper_closed=False)
    assert ctx2["grasp_obj"] == "pear" and abs(float(ctx2["target"][0]) - 0.50) < 1e-6
    # and the lift stage (held, away from the seat) carries no destination or seat point
    ctx3 = priority_context(away, gripper_closed=True)
    assert ctx3["payload"] == "pear" and ctx3["place_target"] is None
    assert "place_point" not in ctx3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")



def _weight_sig():
    return {
        "bounds": (10, 20, 30, 40, 50, 60, 70, 80),
        "initial": {
            "pear": np.array([0.2, 1.4, 0.23]),
            "apple": np.array([0.1, 1.3, 0.24]),
            "scale": np.array([0.66, 1.38, 0.20]),
        },
        "place": {
            "pear": np.array([0.66, 1.33, 0.29]),
            "apple": np.array([0.66, 1.33, 0.287]),
        },
        "held_offset": {"pear": np.zeros(3), "apple": np.zeros(3)},
        "eef": np.zeros((90, 3), dtype=np.float32),
        "z_table": 0.17,
    }


def test_weight_eight_stage_boundaries():
    sig = _weight_sig()
    assert [_weight_stage(sig, i) for i in (0, 10, 20, 30, 40, 50, 60, 70, 89)] == [0, 1, 2, 3, 4, 5, 6, 7, 7]


def test_weight_constraints_match_rendered_formulas():
    import torch

    sig = _weight_sig()
    kp = torch.tensor([[0.2, 1.4, 0.30], [0.1, 1.3, 0.24], [0.66, 1.38, 0.20]])[:, None, None, :]
    ee = torch.zeros(1, 1, 3)
    lift, lift_paths = _weight_constraints(sig, 1)
    assert torch.allclose(lift(ee, kp), torch.tensor([[0.08]]))
    assert torch.allclose(lift_paths[0](ee, kp), torch.zeros(1, 1))

    carry, carry_paths = _weight_constraints(sig, 2)
    hover = torch.tensor(sig["place"]["pear"] + [0.0, 0.0, 0.10])
    expected = torch.linalg.vector_norm(kp[0, 0, 0] - hover)
    assert torch.allclose(carry(ee, kp).reshape(()), expected.float(), atol=1e-6)
    assert torch.allclose(carry_paths[1](ee, kp), torch.tensor([[0.05]]), atol=1e-6)

    place, place_paths = _weight_constraints(sig, 3)
    expected = torch.linalg.vector_norm(kp[0, 0, 0] - torch.tensor(sig["place"]["pear"]))
    assert torch.allclose(place(ee, kp).reshape(()), expected.float(), atol=1e-6)
    assert len(place_paths) == 2


def test_weight_context_has_all_active_simple_auth_fields():
    sig = _weight_sig()
    raw = _objects((0.64, 1.34, 0.30), (0.1, 1.3, 0.24), (0.66, 1.38, 0.20))
    raw.update({"board": {"pos": np.array([0.2, 1.3, 0.2])}})
    base = {"objects": raw, "eef_pos": np.array([0.64, 1.34, 0.34]), "joint_pos": np.zeros(13)}
    ctx = weight_frame_context(base, sig, 35)
    assert ctx["stage_label"] == "place"
    assert ctx["payload"] == "pear" and ctx["place_target"] == "scale"
    for key in ("constraint", "path_fns", "keypoints", "held_idx", "held_offset",
                "payload_age_s", "eef_hist"):
        assert key in ctx


def test_scale_placement_uses_eval_y_offset_and_threshold():
    objects = _objects((0.66, 1.33, 0.28), (0.1, 1.3, 0.24), (0.66, 1.38, 0.20))
    assert _on_place(objects, "pear", "scale")
    objects["pear"]["pos"] = np.array([0.66, 1.451, 0.28])
    assert not _on_place(objects, "pear", "scale")


def test_current_simple_auth_is_representable():
    import pathlib
    import yaml

    cfg_path = pathlib.Path(__file__).parents[1] / "configs/test_configs/simple_auth.yaml"
    check_representable(yaml.safe_load(cfg_path.read_text()))
