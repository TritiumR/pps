"""Write rekep_context.json for a RoboLab task from measured scene geometry.

The RoboLab twin of mujoco_eval/grounding/make_context.py, and it writes the SAME schema, so
`mujoco_eval.grounding.rekep.load_rekep_context` reads it unchanged: each keypoint is stored as
{owner, offset_local, world_at_capture, local_extents} -- rigidly attached to a named body rather
than frozen in the world -- and is recomputed against live owner poses at episode start.

Two deliberate differences from the MuJoCo builder:

* OWNERSHIP IS DECLARED, NOT INFERRED. There, keypoints arrive as a flat array from a task's stage
  ladder and have to be attributed back to bodies by box-surface distance. Here each keypoint is
  CONSTRUCTED on a named body, so its owner is known exactly and no heuristic can mis-assign it.
  That matters directly: `held_idx` is "every keypoint owned by the grasped body", so an owner
  error would make a static place target ride the payload and satisfy the place rule for free.

* GEOMETRY IS MEASURED. Half-extents come from `WorldState.get_bbox`, not from a hand-written
  table; the table in tasks.py is the fallback for a scene that has not been measured yet.
  `get_bbox` and not `get_dimensions`: the latter reports 2 mm x 3 mm x 0.7 mm for a grey bin
  that is really 42 cm x 28 cm x 10.5 cm (it mishandles the prim's scale), which would have made
  the bins invisible to every clearance term and put the release point inside the table.

  `get_bbox` also gives the box CENTRE, which is not the body origin: a grey bin's origin sits on
  its base, 5.25 cm below its own centre. Both numbers are stored -- the centre offset is what
  keeps the synthetic cloud on the object and the mouth keypoint on the rim.

Keypoints per task, and why each one exists:

  banana_in_bowl   banana middle (the grasp point), banana end A, banana end B (the stage-4 rule
                   that both ends stay over the mouth needs them), bowl mouth centre (the static
                   anchor every transport target is an offset from).
  mustard_left_bin bottle middle (grasp), bottle top (the upright and mouth rules), left bin mouth
                   centre (anchor), right bin mouth centre (the keep-out that makes "left" a
                   constraint rather than a comment).
"""

from __future__ import annotations

import json

import numpy as np

from .. import paths
from ..tasks import scene_objects, spec

# Fraction of the half-length at which an "end" keypoint sits. A keypoint at the full half-length
# is on the bounding box corner, off the body; 0.75 keeps it on geometry the fingers could reach.
_END_FRAC = 0.75
_TOP_FRAC = 0.75


def _extent_triple(world_half):
    """Half-extents in vlm_dp's (grip, keepout, half_height) convention.

    grip is the body's NARROWEST half-extent over all three axes -- the tightest the jaws could
    ever need to open on it. Two consumers want exactly that number: `_contact_for` types a body
    as a pinch or a press by comparing it against the aperture, and the grasp terms use it as
    their feasibility radius.

    Narrowest over ALL axes, not over the two horizontal ones. A banana lying flat is 3.7 cm
    thick and 10.9 cm across in x -- but the 10.9 cm is the span of its CURVE, not the width of
    anything the fingers would close on. Taking the horizontal minimum reported a 109 mm body
    against an 85 mm aperture and typed the banana a press: the gripper would have pushed it
    rather than picked it up, and every grasp-mechanics term would have gone inert.

    This is still a whole-body approximation of what is really a keypoint-LOCAL question. The
    MimicGen path answers it from a real point cloud in a ball around the grasp keypoint
    (`local_grasp`); with only a bounding box available here, the narrowest extent is the closest
    honest stand-in, and it is exact for a body whose narrow axis is uniform along its length.

    keepout is the widest horizontal half-extent, which is the footprint the clearance and seat
    terms use; half_height is the vertical half-extent.
    """
    hx, hy, hz = (float(v) for v in world_half)
    return (min(hx, hy, hz), max(hx, hy), hz)


def _boxes(env, names):
    """Measure every body: world AABB centre, half-extents, and the centre in the body frame."""
    out = {}
    for name in names:
        centre, half = env.object_box(name)
        pos, rot = env.object_pose(name)
        out[name] = {"centre": centre, "half": half,
                     "centre_local": np.asarray(rot).T @ (centre - pos),
                     "rot": np.asarray(rot)}
        skew = float(np.abs(np.asarray(rot) - np.eye(3)).max())
        if skew > 0.05:
            # `get_bbox` is a WORLD axis-aligned box. For a body sitting square to the world it
            # is also the body's own box; for a rotated one it is an over-approximation, and the
            # cloud built from it is correspondingly fat. Say so rather than let it pass.
            print(f"[make-context] WARNING: {name} is rotated (max |R - I| = {skew:.3f}); its "
                  f"world AABB over-approximates the body box", flush=True)
    return out


def _long_axis(half):
    """Index of the body's longest axis."""
    return int(np.argmax(np.asarray(half, dtype=np.float64)))


def _keypoints_for(task, boxes):
    """Return [(label, owner, world_point)] for one task's declared keypoints."""
    def mouth(name):
        """Centre of a container's top face -- where a payload is released into it."""
        b = boxes[name]
        return b["centre"] + np.array([0.0, 0.0, b["half"][2]])

    if task == "banana_in_bowl":
        b = boxes["banana"]
        step = np.zeros(3)
        step[_long_axis(b["half"])] = _END_FRAC * b["half"][_long_axis(b["half"])]
        return [("banana_middle", "banana", b["centre"]),
                ("banana_end_a", "banana", b["centre"] + step),
                ("banana_end_b", "banana", b["centre"] - step),
                ("bowl_mouth", "bowl", mouth("bowl"))]

    if task == "mustard_left_bin":
        b = boxes["mustard"]
        return [("mustard_body", "mustard", b["centre"]),
                ("mustard_top", "mustard",
                 b["centre"] + np.array([0.0, 0.0, _TOP_FRAC * b["half"][2]])),
                ("left_bin_mouth", "grey_bin_left", mouth("grey_bin_left")),
                ("right_bin_mouth", "grey_bin_right", mouth("grey_bin_right"))]

    raise SystemExit(f"[make-context] no keypoint recipe for task {task!r}")


def build_context(task, env):
    """Return the rekep_context.json payload for one RoboLab task."""
    names = scene_objects(task)
    boxes = _boxes(env, names)
    measured = {n: _extent_triple(b["half"]) for n, b in boxes.items()}

    entries, labels = [], []
    for label, owner, point in _keypoints_for(task, boxes):
        pos, rot = env.object_pose(owner)
        entries.append({
            "owner": owner,
            "offset_local": (np.asarray(rot).T @ (np.asarray(point) - np.asarray(pos))).tolist(),
            "world_at_capture": np.asarray(point, dtype=np.float64).tolist(),
            # The geometry the fingers would close on AT this keypoint. Declared per body here:
            # every keypoint of a body lies on that body, and none of these scenes has a thin
            # sub-part (a drawer handle) whose local width differs from its owner's.
            "local_extents": [round(float(x), 6) for x in measured[owner]],
            "local_extents_from": owner,
        })
        labels.append(label)

    return {
        "mode": "measured",
        "mask_source": "worldstate_aabb",
        "names": names,
        "labels": labels,
        "keypoints": entries,
        "static_unowned": 0,
        # Measured half-extents, in the (grip, keepout, half_height) convention. The grounding
        # reads these instead of the tasks.py fallback table.
        "extents": {n: [round(float(x), 6) for x in e] for n, e in measured.items()},
        # What the synthetic point cloud is built from: half-extents, plus the box centre in the
        # BODY frame. The second is not always zero -- a grey bin's origin is on its base -- and
        # without it the cloud would sit half a bin below the bin.
        "box_half": {n: [round(float(x), 6) for x in b["half"]] for n, b in boxes.items()},
        "box_center_local": {n: [round(float(x), 6) for x in b["centre_local"]]
                             for n, b in boxes.items()},
        "source": "robolab_eval.grounding.make_context",
        "task": task,
        "gym_id": spec(task)["gym_id"],
    }


def write(task, env):
    """Build and store the context beside the task's other artifacts."""
    payload = build_context(task, env)
    out = paths.task_data(task, "rekep_context.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    owners = [e["owner"] for e in payload["keypoints"]]
    print(f"[make-context] {task}: {len(owners)} keypoints "
          f"{list(zip(payload['labels'], owners))} extents={payload['extents']} -> {out}",
          flush=True)
    return out
