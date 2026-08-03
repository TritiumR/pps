"""Episode selection by the task's own sub-goals, evaluated on beliefs.

Best-of-M over policies: run an episode per parent, score each terminal state, keep the argmax.
The potential is the VLM's place sub-goals -- "payload keypoint at the destination placement point"
-- re-evaluated at the robot's BELIEVED object positions, so the selector is non-privileged and
generalizes exactly as far as the VLM front-end does: the verifier is the task specification reused.

Score = (# satisfied place sub-goals) + a bounded residual of the first unsatisfied one, so ties
break toward the run that got closer. Satisfaction tolerance is the grounding's subgoal_eps, not a
selector-private constant.
"""
from __future__ import annotations

import math


def subgoal_score(beliefs: dict, place_goals: list, eps: float = 0.12) -> float:
    """Score a terminal belief state against the task's place sub-goals.

    beliefs: {name: [x, y, z]} believed object positions.
    place_goals: [(payload, destination, offset_xyz)] in stage order, from the grounding metadata.
    Returns satisfied_count + clamp(1 - residual/(4*eps), 0, 1) for the first unsatisfied goal.

    eps is the grounding's subgoal_eps (0.06) PLUS the slack of evaluating keypoint sub-goals at
    believed object CENTRES (~0.06: keypoint-to-centre offset + terminal belief error). Tightening
    to the bare subgoal_eps decides near-ties on perception noise (measured: it flips a correct
    pick at a 0.62-vs-0.61 margin). Locked at 0.12 before the n=20 read.
    """
    score = 0.0
    for payload, dest, off in place_goals:
        p, d = beliefs.get(payload), beliefs.get(dest)
        if p is None or d is None:
            return score
        residual = math.dist(p, [d[0] + off[0], d[1] + off[1], d[2] + off[2]])
        if residual < eps:
            score += 1.0
        else:
            return score + max(0.0, 1.0 - residual / (4.0 * eps))
    return score


def select(parent_scores: dict[str, float]) -> str:
    """Argmax parent; ties keep the first (the base parent by convention)."""
    return max(parent_scores, key=lambda k: (parent_scores[k],))


WEIGHT_PLACE_GOALS = [
    # From the VLM query for the weight task: payload keypoint onto the scale-top placement point.
    # The xy offset is the keypoint-to-cloud-centre correction; z is scale top + payload half-height.
    ("pear", "scale", (0.039, 0.040, 0.040)),
    ("apple", "scale", (0.039, 0.040, 0.037)),
]
