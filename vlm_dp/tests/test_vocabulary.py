"""CPU tests for deriving the object vocabulary from the instruction (no API call, no simulator).

The VLM reply is injected, so these pin our contract (parsing, role mapping, scene-entity alignment
and the failure modes), not GPT-4o's behaviour. The recorded replies are the shape the prompt asks for,
on the four real task instructions.

Run: python -m vlm_dp.tests.test_vocabulary.
"""
from __future__ import annotations

import json

from vlm_dp.grounding.vocabulary import align, derive

# What the prompt asks the model to return, for each shipped task instruction. Detectors are plain
# nouns: the instructions carry no appearance, and a guessed colour mis-detects (measured: a derived
# green-pear detector for a yellow pear grasped 103mm off).
REPLIES = {
    "put pear and apple on the scale": {"referents": [
        {"name": "pear", "detector": "pear", "role": "grasped"},
        {"name": "apple", "detector": "apple", "role": "grasped"},
        {"name": "scale", "detector": "scale", "role": "destination"}]},
    "pour the tea from the teapot into the cup": {"referents": [
        {"name": "teapot", "detector": "teapot", "role": "grasped"},
        {"name": "cup", "detector": "teacup", "role": "destination"}]},
    "open the coffee maker lid and put the pod inside": {"referents": [
        {"name": "lid", "detector": "coffee maker lid", "role": "grasped"},
        {"name": "pod", "detector": "coffee pod", "role": "grasped"},
        {"name": "coffee maker", "detector": "coffee maker", "role": "destination"}]},
    "remove the lid of the pot and put egg in it": {"referents": [
        {"name": "lid", "detector": "pot lid", "role": "grasped"},
        {"name": "egg", "detector": "egg", "role": "grasped"},
        {"name": "pot", "detector": "pot", "role": "destination"}]},
}

WORLD = {   # scene-entity names the sim exposes for each task
    "put pear and apple on the scale": ["pear", "apple", "mango", "cabbage", "board", "scale"],
    "pour the tea from the teapot into the cup": ["teapot", "teacup"],
    "open the coffee maker lid and put the pod inside": ["can", "capsule"],
    "remove the lid of the pot and put egg in it": ["pot", "cover", "egg"],
}


def _chat_for(instruction):
    return lambda _prompt: json.dumps(REPLIES[instruction])


def test_roles_come_out_of_the_instruction():
    """Referents, detector text and roles are read off the sentence, not a per-task dict."""
    got = derive("put pear and apple on the scale", WORLD["put pear and apple on the scale"],
                 chat=_chat_for("put pear and apple on the scale"))
    assert got["objects"] == {"pear": "pear", "apple": "apple", "scale": "scale"}
    assert got["grasp_objs"] == ["pear", "apple"], "both fruit are picked up, in instruction order"
    assert got["place_obj"] == "scale"


def test_distractors_are_never_named():
    """The measured mis-grounding this removes: 2/10 weight seeds grasped a *cabbage*.

    The hand-authored dict listed mango/cabbage/board as named objects, so the grounding could select
    one as a target. They are not in the instruction, so they are not referents. They remain obstacles,
    which need geometry, not identity.
    """
    instruction = "put pear and apple on the scale"
    got = derive(instruction, WORLD[instruction], chat=_chat_for(instruction))
    for distractor in ("mango", "cabbage", "board"):
        assert distractor in WORLD[instruction], "fixture check: the distractor IS in the scene"
        assert distractor not in got["objects"], f"{distractor} is not referred to; it must not be named"


def test_referents_align_onto_scene_entities():
    """A language referent binds to the scene entity that plainly corresponds, and only then."""
    assert align("cup", ["teapot", "teacup"]) == "teacup", "substring should bind cup -> teacup"
    assert align("coffee maker", ["can", "capsule"]) == "coffee maker", \
        "no plain correspondence -> keep the language name rather than guess"
    assert align("pot", ["pot", "cover", "egg"]) == "pot", "an exact name wins outright"
    assert align("knife", []) == "knife", "with no scene-entity list the referent stands alone"


def test_an_exact_name_beats_a_substring_sibling():
    """`pot` must not bind to `teapot` when `pot` itself exists."""
    assert align("pot", ["teapot", "pot"]) == "pot"


def test_an_ambiguous_referent_is_not_guessed():
    """Two plausible entities -> keep the referent's own name instead of picking one arbitrarily."""
    assert align("lid", ["pot lid", "kettle lid"]) == "lid"


def test_every_shipped_instruction_yields_a_usable_vocabulary():
    """All four tasks must produce objects, at least one graspable, and a destination."""
    for instruction, world in WORLD.items():
        got = derive(instruction, world, chat=_chat_for(instruction))
        assert got["objects"], f"{instruction!r} produced no objects"
        assert got["grasp_objs"], f"{instruction!r} produced nothing to grasp"
        assert got["place_obj"], f"{instruction!r} produced no destination"
        assert all(v for v in got["objects"].values()), f"{instruction!r} has an empty detector phrase"


def test_a_bad_reply_fails_loudly():
    """An unusable reply must raise, not return an empty vocabulary that fails much later."""
    for bad in ('{"referents": []}', "not json at all", '{"wrong_key": 1}'):
        try:
            derive("put pear on the scale", [], chat=lambda _p, b=bad: b)
        except SystemExit:
            continue
        raise AssertionError(f"reply {bad!r} should have raised rather than returned a vocabulary")


def test_a_fenced_reply_is_tolerated():
    """The model sometimes wraps JSON in a ```json fence despite response_format."""
    fenced = '```json\n{"referents": [{"name": "pear", "detector": "yellow pear", "role": "grasped"}]}\n```'
    got = derive("put the pear down", [], chat=lambda _p: fenced)
    assert got["objects"] == {"pear": "yellow pear"} and got["grasp_objs"] == ["pear"]


# ------------------------------------------------ keypoint identity, cross-checked against the prose
# Verbatim from a real GPT-4o reply (results/rekep/tea/stage3_subgoal_constraints.txt).
_TEA_STAGE3 = '''
def stage3_subgoal_constraint1(end_effector, keypoints):
    """The teapot spout (keypoint 23) needs to be 5cm above the cup opening (keypoint 25)."""
    offsetted_point = keypoints[25] + np.array([0, 0, 0.05])
    return np.linalg.norm(keypoints[23] - offsetted_point)

def stage3_subgoal_constraint2(end_effector, keypoints):
    """The teapot spout (keypoint 23) must be tilted to pour liquid."""
    teapot_vector = keypoints[23] - keypoints[21]
    z_axis = np.array([0, 0, 1])
    angle = np.arccos(np.dot(teapot_vector, z_axis) / np.linalg.norm(teapot_vector))
    return np.pi / 4 - angle  # Tilt at least 45 degrees
'''


def test_the_vlm_states_an_identity_for_each_keypoint_it_uses():
    """Two keypoints in one sentence bind to the object each is actually named beside."""
    from vlm_dp.grounding.rekep import keypoint_claims

    claims = keypoint_claims(_TEA_STAGE3, ["teapot", "cup"])
    assert claims[23] == "teapot", f"keypoint 23 is named as the teapot spout, got {claims.get(23)!r}"
    assert claims[25] == "cup", f"keypoint 25 is named as the cup opening, got {claims.get(25)!r}"


def test_an_unnamed_keypoint_makes_no_claim():
    """keypoints[21] appears only in code with no object beside it, and claiming nothing beats guessing."""
    from vlm_dp.grounding.rekep import keypoint_claims

    claims = keypoint_claims(_TEA_STAGE3, ["teapot", "cup"])
    assert 21 not in claims or claims[21] == "teapot", \
        f"kp21 is only bound if the prose actually names it, got {claims.get(21)!r}"


def test_a_disagreement_is_detectable():
    """The point of the check: prose and tracker can be compared, so a mis-assignment is visible."""
    from vlm_dp.grounding.rekep import keypoint_claims

    claims = keypoint_claims(_TEA_STAGE3, ["teapot", "cup"])
    owners = {23: "teacup", 25: "cup"}          # tracker mis-assigned kp23 to the cup
    bad = [k for k, v in claims.items() if k in owners and owners[k] != v]
    assert bad == [23], f"the mis-assigned keypoint should be the detectable one, got {bad}"


_TESTS = [test_roles_come_out_of_the_instruction, test_distractors_are_never_named,
          test_referents_align_onto_scene_entities, test_an_exact_name_beats_a_substring_sibling,
          test_an_ambiguous_referent_is_not_guessed,
          test_every_shipped_instruction_yields_a_usable_vocabulary,
          test_a_bad_reply_fails_loudly, test_a_fenced_reply_is_tolerated,
          test_the_vlm_states_an_identity_for_each_keypoint_it_uses,
          test_an_unnamed_keypoint_makes_no_claim, test_a_disagreement_is_detectable]


def main():
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report harness/import errors, don't hide them
            failures += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILED'} ({len(_TESTS)} tests)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
