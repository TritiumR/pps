"""Per-task RoboLab scene facts: gym id, tracked bodies, roles and object extents.

The RoboLab twin of mujoco_eval/grounding/gt.py's TASKS table, minus the stage ladders: stages
here come from the plan (gt_vlm_output/<task>/raw.txt), never from this file.

EXTENTS are half-widths in the (grip, keepout, half_height) convention every extents triple in
vlm_dp uses. They seed the synthetic point cloud `load_rekep_context` builds from live poses, and
they are OVERWRITTEN at context-build time by the measured AABB from WorldState.get_dimensions --
these are the fallback for a scene that has not been measured yet, not the authority.
"""

from __future__ import annotations

TASKS = {
    "banana_in_bowl": {
        "gym_id": "BananaInBowlTask",
        "prompt": "put the banana in the bowl",
        # Bodies tracked through the episode; every one must answer WorldState.get_pose.
        "movable": ("banana",),
        "fixtures": ("bowl", "table"),
        "grasp_objs": ("banana",),
        "place_obj": "bowl",
        # 15 Hz control; the env's own episode_length_s is 50 s = 750 steps.
        "max_steps": 750,
        "extents": {"banana": (0.018, 0.018, 0.020),
                    "bowl": (0.075, 0.075, 0.030),
                    "table": (0.40, 0.40, 0.02)},
    },
    "mustard_left_bin": {
        "gym_id": "MustardInLeftBinTask",
        "prompt": "put the mustard bottle in the left grey bin",
        "movable": ("mustard",),
        "fixtures": ("grey_bin_left", "grey_bin_right", "table"),
        "grasp_objs": ("mustard",),
        "place_obj": "grey_bin_left",
        # 15 Hz control; the env's own episode_length_s is 30 s = 450 steps.
        "max_steps": 450,
        "extents": {"mustard": (0.030, 0.030, 0.095),
                    "grey_bin_left": (0.10, 0.13, 0.05),
                    "grey_bin_right": (0.10, 0.13, 0.05),
                    "table": (0.40, 0.40, 0.02)},
    },
}


def spec(task):
    """Return one task's scene facts."""
    if task not in TASKS:
        raise SystemExit(f"[robolab-eval] unknown task {task!r} (have {sorted(TASKS)})")
    return TASKS[task]


def scene_objects(task):
    """Every body the grounding may build a SceneObject for: tracked bodies and fixtures."""
    s = spec(task)
    # `table` is the support plane, not an obstacle to plan around: it would otherwise be the
    # largest sphere obstacle in the scene and the clearance terms would push the arm off it.
    return [n for n in (*s["movable"], *s["fixtures"]) if n != "table"]
