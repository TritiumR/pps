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
    "spoon_insertion": {
        "gym_id": "InsertSpaghettiSpoonTask",
        "prompt": "Insert the spaghetti spoon into the utensil holder.",
        # The spatula is not manipulated, but it remains a named/tracked obstacle.  Omitting it
        # would give the Spoon controller a less complete scene than the Weight/Capsule perception
        # stack, where named distractors remain available to the clearance terms.
        # The holder looks like a fixture but is a dynamic rigid body in this scene: contact can
        # tip and translate it.  It must therefore be segmented and tracked just like the two
        # utensils; freezing its reset-time instance mask would make the insertion mouth stale.
        "movable": ("pink_spaghetti_spoon", "spatula", "utensil_holder"),
        "fixtures": ("table",),
        "grasp_objs": ("pink_spaghetti_spoon",),
        "place_obj": "utensil_holder",
        "max_steps": 1350,             # task episode_length_s=90 at the 15 Hz control rate
        "continuous_gripper": True,
        "vocab": {
            "pink_spaghetti_spoon": (
                "pink spaghetti spoon . pink pasta server . pink slotted serving spoon"
            ),
            "spatula": "grey spatula . metal spatula . turner",
            "utensil_holder": (
                "wooden utensil holder . wooden utensil crock . cylindrical utensil container"
            ),
        },
        # Standard fields consumed by the generic insertion decorator in grounding/rekep.py.  The
        # live mouth position comes from a tracked keypoint; these are semantic dimensions of the
        # requested insertion, not simulator poses.  They are rendered into the plan receipt.
        "insertion": {
            "depth": 0.075,
            "hover": 0.12,
            "capture": 0.10,
            "seat_radius": 0.025,
        },
        # Conservative fallbacks only.  The perception rung replaces these with measured clouds.
        "extents": {
            "pink_spaghetti_spoon": (0.014, 0.165, 0.014),
            "spatula": (0.018, 0.165, 0.018),
            "utensil_holder": (0.070, 0.070, 0.095),
            "table": (0.40, 0.40, 0.02),
        },
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
