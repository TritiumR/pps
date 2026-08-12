"""224 px RoboLab registration for single-spoon demonstration collection.

This is the utensil/front-camera counterpart of ``scratch/robolab_env224.py``.
It preserves the task's continuous gripper action rather than replacing it with
RoboLab's default binary gripper registration.
"""

from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.droid import (
    _WRIST_CAM,
    DroidCfg,
    DroidContinuousGripperActionCfg,
    ProprioceptionObservationCfg,
    contact_gripper,
)
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.lighting import SphereLightCfg

from robolab_eval.data_generation.rev10_cameras import FrontWideCameraCfg


IMG_SIZE = 224
ENV_POSTFIX = "_SpoonSingle50_224"
TABLE_CAM_KEY = "front_wide_camera"
WRIST_CAM_KEY = "wrist_cam"

# The demonstration recorder and success predicates consume these six contact
# streams.  The other generated pairwise sensors do not affect actions, labels,
# success, or the serialized dataset, but force an expensive history refresh at
# every Isaac scene update.
_SPOON_DEMO_CONTACT_SENSORS = frozenset(
    {
        "gripper__spatula",
        "gripper__pink_spaghetti_spoon",
        "gripper__utensil_holder",
        "pink_spaghetti_spoon__utensil_holder",
        "gripper__table",
        "spatula__utensil_holder",
    }
)

_FRONT_224 = FrontWideCameraCfg().front_wide_camera.replace(height=IMG_SIZE, width=IMG_SIZE)
_WRIST_224 = _WRIST_CAM.replace(height=IMG_SIZE, width=IMG_SIZE)


@configclass
class FrontWide224Cfg:
    front_wide_camera = _FRONT_224


@configclass
class Droid224Cfg(DroidCfg):
    wrist_cam = _WRIST_224


@configclass
class WristCamera224Cfg:
    """Observation holder; the physical camera is attached by ``Droid224Cfg``."""

    wrist_cam = _WRIST_224


def apply_spoon_contact_sensor_diet(scene_cfg, *, demonstration: bool = True) -> list[str]:
    """Disable contact sensors that have no Spoon pipeline consumer.

    Generation deliberately retains two more sensors than rollout: the
    gripper/table event trace and the spatula/holder negative-control predicate.
    Fail closed if this helper is accidentally requested for another contract.
    """
    if not demonstration:
        raise ValueError("this data-generation helper only defines the demonstration contract")
    disabled: list[str] = []
    for name, value in vars(scene_cfg).items():
        class_type = str(getattr(value, "class_type", ""))
        if "ContactSensor" not in class_type or name in _SPOON_DEMO_CONTACT_SENSORS:
            continue
        setattr(scene_cfg, name, None)
        disabled.append(name)
    return sorted(disabled)


def register_spoon_env(task: str = "InsertSpaghettiSpoonTask") -> str:
    image_obs_cfg = generate_image_obs_from_cameras([FrontWide224Cfg, WristCamera224Cfg])
    observations_cfg = generate_obs_cfg(
        {
            "image_obs": image_obs_cfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
        }
    )
    created = auto_discover_and_create_cfgs(
        tasks=task,
        env_prefix="",
        env_postfix=ENV_POSTFIX,
        observations_cfg=observations_cfg(),
        actions_cfg=DroidContinuousGripperActionCfg(),
        robot_cfg=Droid224Cfg,
        camera_cfg=[FrontWide224Cfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
    cfg_cls = next(iter(created.values()))
    return cfg_cls.__name__[: -len("EnvCfg")]
