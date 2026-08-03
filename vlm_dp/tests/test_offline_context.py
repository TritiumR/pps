"""Offline priority-label context (vlm_dp.offline_context): stage inference and key contract."""

import numpy as np

from vlm_dp.offline_context import infer_stage, priority_context, WEIGHT_EXTENTS


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
