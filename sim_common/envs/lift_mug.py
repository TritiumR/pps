"""Lift-Mug task (Franka + mug) -- registered as ``Isaac-Lift-Mug-Franka-v0``.

Subclasses the stock ``FrankaCubeLiftEnvCfg`` and swaps only the object to a mug, reusing the whole lift
framework without touching upstream IsaacLab. A mug's distinct parts (handle / rim / body) give several
ReKep keypoints and a real grasp-which-part affordance, unlike the symmetric cube's single keypoint.

Import AFTER AppLauncher has booted (it imports IsaacLab) to register the task.
"""
import gymnasium as gym

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

# Nucleus mug (properly textured); the repo-local mug USD references textures by an absolute path, so it
# renders untextured and DINOv2 finds no features.
_MUG_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"


@configclass
class FrankaMugLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # swap the cube for the mug; spawn slightly high so it settles onto the table regardless of origin
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.0, 0.10], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=_MUG_USD,
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


if "Isaac-Lift-Mug-Franka-v0" not in gym.registry:
    gym.register(
        id="Isaac-Lift-Mug-Franka-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:FrankaMugLiftEnvCfg"},
    )
