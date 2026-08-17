from sim_free_mpc.trace_action_chunks import BatchedReplanTracker


def _lane(phase: str = "grasp_pear", *, active: bool = True):
    subtasks = {
        "grasp_pear": phase == "place_pear",
        "pear_on_scale": phase == "grasp_apple",
        "grasp_apple": phase == "place_apple",
    }
    return {"lane": 0, "valid": True, "active": active, "subtasks": subtasks}


def test_periodic_replans_follow_execution_horizon():
    tracker = BatchedReplanTracker(steps_per_inference=4)
    generated = [step for step in range(10) if tracker.observe_frame(step, [_lane()])]
    assert generated == [0, 4, 8]


def test_phase_transition_forces_and_restarts_replan_clock():
    tracker = BatchedReplanTracker(steps_per_inference=4)
    phases = ["grasp_pear", "grasp_pear", "place_pear"] + ["place_pear"] * 6
    generated = [
        step
        for step, phase in enumerate(phases)
        if tracker.observe_frame(step, [_lane(phase)])
    ]
    assert generated == [0, 2, 6]


def test_inactive_lane_does_not_force_replan():
    tracker = BatchedReplanTracker(steps_per_inference=4)
    assert tracker.observe_frame(0, [_lane("grasp_pear")])
    assert not tracker.observe_frame(1, [_lane("place_pear", active=False)])
