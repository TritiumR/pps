import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.score_steering import combine_scores  # noqa: E402


def test_full_score_steering_uses_task_minus_ref():
    base = torch.tensor([[[1.0, 2.0]]])
    task = torch.tensor([[[4.0, 8.0]]])
    ref = torch.tensor([[[2.0, 3.0]]])

    combined = combine_scores(
        base,
        task,
        mode="full",
        steer_scale=0.5,
        ref_score=ref,
    )

    assert torch.allclose(combined, base + 0.5 * (task - ref))


def test_task_score_steering_does_not_require_ref():
    base = torch.tensor([[[1.0, 2.0]]])
    task = torch.tensor([[[4.0, 8.0]]])

    combined = combine_scores(base, task, mode="task", steer_scale=0.25)

    assert torch.allclose(combined, base + 0.25 * task)


def test_full_score_steering_requires_ref():
    score = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="requires a reference"):
        combine_scores(score, score, mode="full", steer_scale=1.0)


def test_score_steering_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="base/task score shapes must match"):
        combine_scores(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 2),
            mode="task",
            steer_scale=1.0,
        )
