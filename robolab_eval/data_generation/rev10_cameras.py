"""Front-facing table cameras, and the camera presets the rev10 runs select from.

Why this file exists
--------------------
The utensil insertions are recorded through ``over_shoulder_left_camera``, which
sits at (0.05, 0.57, 0.66) behind and to the left of the workspace and looks
across it at the crock. The arm reaches the crock from behind as well, so for the
whole of the reorientation and the insertion the forearm and wrist are between
that camera and the thing being demonstrated: at the moment the tool slides into
the mouth, the mouth, the tool and the fingers are all behind the wrist. A
demonstration whose money frames show a white forearm is not a demonstration of
an insertion.

RoboLab already ships the view that does not have this problem. The front camera
used for the ``banana_in_bowl`` recordings -- ``EgocentricMirroredCameraCfg``,
the ``viewport_cam`` of the jointpos registration -- stands at (1.5, 0, 1.0) in
*front* of the robot and looks back at it, 45 deg down, through a 47.5 deg lens.
The arm then approaches away from the camera rather than across it, and the
table, the container and the payload are all in open view. The classes below
reuse that pose and that lens; nothing about the angle is invented here.

Three variants are defined because two questions could not be answered from the
configs alone and had to be rendered:

* ``FrontRefCameraCfg`` is a byte-for-byte clone of the banana view, 864x480 with
  ``vertical_aperture`` 15.29. That aperture is inconsistent with the 864x480
  frame (20.955/15.29 = 1.37 against an image aspect of 1.80), so either the
  renderer ignores it and the vertical field follows the resolution, or it does
  not and the reference frames are anamorphic. Rendering this next to
  ``FrontTableCameraCfg`` answers it.
* ``FrontTableCameraCfg`` is the same pose and the same horizontal field at
  1280x720, with the vertical aperture made consistent with that frame.
* ``FrontWideCameraCfg`` is the same pose with a 60 deg lens.

Both questions were answered by rendering all three against
``over_shoulder_left_camera`` on one InsertSpatulaTask rollout
(``rev10_camprobe.log``, frames in ``rev10_frames/cmp``):

* The renderer ignores ``vertical_aperture``; the vertical field follows the
  resolution. ``FrontRefCameraCfg`` and ``FrontTableCameraCfg`` frame the scene
  identically, so 1280x720 reproduces the reference view at 720p.
* **The reference lens is too narrow for this task.** At the top of the carry
  (LIFT, tool at z = 0.372) the 47.5 deg field has the gripper entirely outside
  the frame -- only the spatula's blade hangs into the top-right corner -- and
  at REORIENT the wrist is cut off at the top edge. The brief for this view is
  that the tool, the gripper and the crock stay visible *throughout*, which the
  reference lens does not do. At 60 deg, from the same stand and along the same
  optical axis, the whole gripper and tool stay inside the frame at the carry's
  peak with margin to spare, and the insertion is still large enough to read.
  So ``WRIST_FRONT`` uses ``FrontWideCameraCfg``: the reference *angle* is
  reproduced exactly, and only the field of view is opened up, which is the
  smallest departure that meets the requirement.

Nothing in the RoboLab tree is modified. ``auto_register_droid_envs`` already
takes the observed-camera list as an argument, so a run selects its cameras by
passing one of the presets below.

Demonstration contract
----------------------
The training schema's ``obs/table_cam`` is fed by whichever scene camera the run
registered. For runs that use ``WRIST_FRONT`` that source is
``front_wide_camera``, not ``over_shoulder_left_camera``: same key, different
physical viewpoint. See ``rev10_run.py``'s module docstring.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.robots.droid import WristCameraCfg
from robolab.variations.camera import OverShoulderLeftCameraCfg

# Pose of RoboLab's EgocentricMirroredCameraCfg, i.e. the banana_in_bowl front
# view: 1.5 m in front of the robot, 1.0 m up, looking back and 45 deg down.
# The optical axis meets the table at (0.518, 0, 0.02), which is 32 mm from the
# utensil crock at (0.55, 0) -- the insertion happens on the lens axis.
FRONT_POS = (1.5, 0.0, 1.0)
FRONT_ROT = (0.653, 0.271, 0.271, 0.653)

# Lens of the same camera: 47.5 deg horizontal.
FRONT_FOCAL = 24.0
FRONT_FOCUS = 400.0
FRONT_H_APERTURE = 20.955


def _front_spawn(focal=FRONT_FOCAL, h_aperture=FRONT_H_APERTURE, aspect=1280 / 720):
    """Pinhole with the front lens, vertical aperture consistent with the frame."""
    return sim_utils.PinholeCameraCfg(
        focal_length=focal,
        focus_distance=FRONT_FOCUS,
        horizontal_aperture=h_aperture,
        vertical_aperture=h_aperture / aspect,
    )


@configclass
class FrontTableCameraCfg:
    """The banana_in_bowl front view at 720p. This is the recording camera."""

    front_table_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_table_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=_front_spawn(),
        offset=TiledCameraCfg.OffsetCfg(pos=FRONT_POS, rot=FRONT_ROT, convention="opengl"),
    )


@configclass
class FrontRefCameraCfg:
    """Exact clone of EgocentricMirroredCameraCfg, for the framing cross-check."""

    front_ref_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_ref_camera",
        height=480,
        width=864,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FRONT_FOCAL,
            focus_distance=FRONT_FOCUS,
            horizontal_aperture=FRONT_H_APERTURE,
            vertical_aperture=15.29,
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=FRONT_POS, rot=FRONT_ROT, convention="opengl"),
    )


@configclass
class FrontWideCameraCfg:
    """Same stand and axis, 60 deg lens. This is the recording camera.

    The extra 12.5 deg buys the top of the carry, which the reference lens loses.
    """

    front_wide_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_wide_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=_front_spawn(focal=18.0),
        offset=TiledCameraCfg.OffsetCfg(pos=FRONT_POS, rot=FRONT_ROT, convention="opengl"),
    )


# --------------------------------------------------------------------- presets
# The stock preset, unchanged: what every rev1-rev9 recording used.
WRIST_LEFT = [OverShoulderLeftCameraCfg, WristCameraCfg]

# What the utensil insertions record through from rev10 on: the banana_in_bowl
# stand and optical axis, opened from 47.5 to 60 deg so the carry stays framed.
WRIST_FRONT = [FrontWideCameraCfg, WristCameraCfg]

# Everything at once. Only for the camera probe: it renders four 720p-class
# streams per step, so it is far too slow to record demonstrations with.
PROBE = [OverShoulderLeftCameraCfg, FrontTableCameraCfg, FrontRefCameraCfg,
         FrontWideCameraCfg, WristCameraCfg]

PRESETS = {
    "over_shoulder": WRIST_LEFT,
    "front": WRIST_FRONT,
    "probe": PROBE,
}

# The scene camera each preset feeds obs/table_cam from.
PRESET_TABLE_CAM = {
    "over_shoulder": "over_shoulder_left_camera",
    "front": "front_wide_camera",
    "probe": "front_table_camera",
}
