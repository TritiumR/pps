"""The rekep_keypose term must respond to the keypose row and ONLY the keypose row.

That is its entire reason to exist: `rekep_subgoal` reduces over every row, so a one-row
perturbation moves it by ~1/H. Measured on the real steering loop, the resulting proposal cost
spread was 3.0% relative -- too flat for the Feynman-Kac softmax to discriminate.
"""

from __future__ import annotations

import torch

from vlm_dp.cost.terms import TERMS


class _Inputs:
    """Minimal CostInputs stand-in: the term reads ee_pos, context and geom only."""

    def __init__(self, ee_pos, constraint):
        self.ee_pos = ee_pos
        self.ee_quat = None
        self.real_actions = torch.zeros(*ee_pos.shape[:2], 8)
        self.extents = {}
        self.geom = torch.nn.Module()          # attribute bag; getattr defaults apply
        self.context = {
            "constraint": constraint,
            "keypoints": torch.zeros(1, 3),
            "held_idx": (),
            "held_offset": None,
        }


def _height_subgoal(ee, kp):
    """Toy sub-goal: distance of the end effector above the origin, per [K, H]."""
    del kp
    return ee[..., 2]


def test_keypose_term_ignores_non_terminal_rows():
    term = TERMS["rekep_keypose"]
    ee = torch.zeros(2, 8, 3)
    base = term(_Inputs(ee, _height_subgoal))
    assert torch.allclose(base, torch.zeros(2))

    moved_middle = ee.clone()
    moved_middle[:, 3, 2] = 5.0                     # perturb a middle row
    assert torch.allclose(term(_Inputs(moved_middle, _height_subgoal)), torch.zeros(2)), \
        "a middle-row change must not move the keypose term"

    moved_last = ee.clone()
    moved_last[:, -1, 2] = 5.0                      # perturb the keypose row
    assert torch.allclose(term(_Inputs(moved_last, _height_subgoal)),
                          torch.full((2,), 5.0))


def test_keypose_term_is_undiluted_versus_rekep_subgoal():
    """The point of the term: full response, not 1/H of it."""
    kp_term, sub_term = TERMS["rekep_keypose"], TERMS["rekep_subgoal"]
    horizon = 16
    ee = torch.zeros(1, horizon, 3)
    bumped = ee.clone()
    bumped[:, -1, 2] = 1.0

    kp_delta = float(kp_term(_Inputs(bumped, _height_subgoal))
                     - kp_term(_Inputs(ee, _height_subgoal)))
    sub_delta = float(sub_term(_Inputs(bumped, _height_subgoal))
                      - sub_term(_Inputs(ee, _height_subgoal)))
    assert abs(kp_delta - 1.0) < 1e-6
    # rekep_subgoal SUMS by default, so it also sees 1.0 here; the dilution appears once the term
    # is averaged (subgoal_mean) or, in practice, alongside the 50 terms that use mean(dim=1).
    assert kp_delta >= sub_delta


def test_inert_without_a_constraint():
    term = TERMS["rekep_keypose"]
    inputs = _Inputs(torch.zeros(3, 8, 3), None)
    inputs.context["constraint"] = None
    assert torch.allclose(term(inputs), torch.zeros(3))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all rekep_keypose tests passed")
