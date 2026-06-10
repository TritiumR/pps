# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_bin_stack_ik_rel_mimic_env_cfg import FrankaBinStackIKRelMimicEnvCfg
from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv
from .franka_stack_ik_abs_mimic_env_cfg import FrankaCubeStackIKAbsMimicEnvCfg
from .franka_stack_ik_rel_blueprint_mimic_env_cfg import (
    FrankaCubeStackIKRelBlueprintMimicEnvCfg,
)
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv
from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from .franka_stack_ik_rel_skillgen_env_cfg import FrankaCubeStackIKRelSkillgenEnvCfg
from .franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_custom_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCustomMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_ood_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorOODMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_custom_id_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCustomIDMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_custom_gap_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCustomGapMimicEnvCfg,
)
from .franka_stack_ik_rel_visuomotor_custom_gap_id_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCustomGapIDMimicEnvCfg,
)

from .droid_stack_ik_rel_mimic_env import DroidCubeStackIKRelMimicEnv
from .droid_stack_ik_rel_visuomotor_mimic_env_cfg import (
    DroidCubeStackIKRelVisuomotorMimicEnvCfg,
)
from .droid_stack_ik_rel_visuomotor_sim_mimic_env_cfg import (
    DroidCubeStackIKRelVisuomotorSimMimicEnvCfg,
)
from .droid_stack_ik_rel_visuomotor_sim_id_mimic_env_cfg import (
    DroidCubeStackIKRelVisuomotorSimIDMimicEnvCfg,
)
from .droid_stack_ik_rel_visuomotor_ood_mimic_env_cfg import (
    DroidCubeStackIKRelVisuomotorOODMimicEnvCfg,
)

from .droid_water_ik_rel_mimic_env import DroidWaterIKRelMimicEnv
from .droid_water_ik_rel_visuomotor_mimic_env_cfg import (
    DroidWaterIKRelVisuomotorMimicEnvCfg,
)

from .droid_water_joint_pos_visuomotor_mimic_env_cfg import (
    DroidWaterJointPosVisuomotorMimicEnvCfg,
)

from .droid_bread_ik_rel_mimic_env import DroidBreadIKRelMimicEnv
from .droid_bread_ik_rel_visuomotor_mimic_env_cfg import (
    DroidBreadIkRelVisuomotorMimicEnvCfg,
)
from .droid_can_ik_rel_mimic_env import DroidCanIKRelMimicEnv
from .droid_can_ik_rel_pointcloud_mimic_env_cfg import (
    DroidCanIKRelPointCloudMimicEnvCfg,
)
from .droid_can_ik_rel_visuomotor_mimic_env_cfg import (
    DroidCanIKRelVisuomotorMimicEnvCfg,
)
from .droid_capsule_ik_rel_mimic_env import DroidCapsuleIKRelMimicEnv
from .droid_capsule_ik_rel_visuomotor_mimic_env_cfg import (
    DroidCapsuleIKRelVisuomotorMimicEnvCfg,
)

from .droid_water_align_ik_rel_visuomotor_mimic_env_cfg import (
    DroidWaterAlignIKRelVisuomotorMimicEnvCfg,
)

from .droid_block_ik_rel_mimic_env import DroidBlockIKRelMimicEnv
from .droid_block_ik_rel_visuomotor_mimic_env_cfg import (
    DroidBlockIKRelVisuomotorMimicEnvCfg,
)
from .droid_drink_ik_rel_mimic_env import DroidDrinkIKRelMimicEnv
from .droid_drink_ik_rel_pointcloud_mimic_env_cfg import (
    DroidDrinkIKRelPointCloudMimicEnvCfg,
)
from .droid_drink_ik_rel_visuomotor_mimic_env_cfg import (
    DroidDrinkIKRelVisuomotorMimicEnvCfg,
)
from .droid_holder_ik_rel_mimic_env import DroidHolderIKRelMimicEnv
from .droid_holder_ik_rel_visuomotor_mimic_env_cfg import (
    DroidHolderIKRelVisuomotorMimicEnvCfg,
)

from .droid_cylinder_ik_rel_mimic_env import DroidCylinderIKRelMimicEnv
from .droid_cylinder_ik_rel_visuomotor_mimic_env_cfg import (
    DroidCylinderIKRelVisuomotorMimicEnvCfg,
)

from .droid_laptop_ik_rel_mimic_env import DroidLaptopIKRelMimicEnv
from .droid_laptop_ik_rel_pointcloud_mimic_env_cfg import (
    DroidLaptopIKRelPointCloudMimicEnvCfg,
)
from .droid_laptop_ik_rel_visuomotor_mimic_env_cfg import (
    DroidLaptopIKRelVisuomotorMimicEnvCfg,
)
from .droid_oven_ik_rel_mimic_env import DroidOvenIKRelMimicEnv
from .droid_oven_ik_rel_pointcloud_mimic_env_cfg import (
    DroidOvenIKRelPointCloudMimicEnvCfg,
)
from .droid_oven_ik_rel_visuomotor_mimic_env_cfg import (
    DroidOvenIKRelVisuomotorMimicEnvCfg,
)

from .droid_plate_ik_rel_mimic_env import DroidPlateIKRelMimicEnv
from .droid_plate_ik_rel_pointcloud_mimic_env_cfg import (
    DroidPlateIKRelPointCloudMimicEnvCfg,
)
from .droid_plate_ik_rel_visuomotor_mimic_env_cfg import (
    DroidPlateIKRelVisuomotorMimicEnvCfg,
)
from .droid_pen_ik_rel_mimic_env import DroidPenIKRelMimicEnv
from .droid_pen_ik_rel_pointcloud_mimic_env_cfg import (
    DroidPenIKRelPointCloudMimicEnvCfg,
)
from .droid_pen_ik_rel_visuomotor_mimic_env_cfg import (
    DroidPenIKRelVisuomotorMimicEnvCfg,
)
from .droid_phone_ik_rel_mimic_env import DroidPhoneIKRelMimicEnv
from .droid_phone_ik_rel_sound_mimic_env_cfg import (
    DroidPhoneIKRelSoundMimicEnvCfg,
)
from .droid_pot_ik_rel_mimic_env import DroidPotIKRelMimicEnv
from .droid_pot_ik_rel_pointcloud_mimic_env_cfg import (
    DroidPotIKRelPointCloudMimicEnvCfg,
)
from .droid_pot_ik_rel_visuomotor_mimic_env_cfg import (
    DroidPotIKRelVisuomotorMimicEnvCfg,
)
from .droid_scissor_ik_rel_mimic_env import DroidScissorIKRelMimicEnv
from .droid_scissor_ik_rel_visuomotor_mimic_env_cfg import (
    DroidScissorIKRelVisuomotorMimicEnvCfg,
)
from .droid_tea_ik_rel_mimic_env import DroidTeaIKRelMimicEnv
from .droid_hot_tea_ik_rel_mimic_env import DroidHotTeaIKRelMimicEnv
from .droid_hot_tea_ik_rel_thermal_mimic_env_cfg import (
    DroidHotTeaIKRelThermalMimicEnvCfg,
)
from .droid_tea_ik_rel_pointcloud_masked_mimic_env_cfg import (
    DroidTeaIKRelPointCloudMaskedMimicEnvCfg,
)
from .droid_tea_ik_rel_pointcloud_mimic_env_cfg import (
    DroidTeaIKRelPointCloudMimicEnvCfg,
)
from .droid_tea_ik_rel_visuomotor_mimic_env_cfg import (
    DroidTeaIKRelVisuomotorMimicEnvCfg,
)
from .droid_tea_wood_ik_rel_pointcloud_masked_mimic_env_cfg import (
    DroidTeaWoodIKRelPointCloudMaskedMimicEnvCfg,
)
from .droid_tea_wood_ik_rel_pointcloud_mimic_env_cfg import (
    DroidTeaWoodIKRelPointCloudMimicEnvCfg,
)
from .droid_tea_wood_ik_rel_visuomotor_mimic_env_cfg import (
    DroidTeaWoodIKRelVisuomotorMimicEnvCfg,
)
from .droid_weight_ik_rel_mimic_env import DroidWeightIKRelMimicEnv
from .droid_weight_ik_rel_pointcloud_mimic_env_cfg import (
    DroidWeightIKRelPointCloudMimicEnvCfg,
)
from .droid_weight_ik_rel_pointcloud_masked_mimic_env_cfg import (
    DroidWeightIKRelPointCloudMaskedMimicEnvCfg,
)
from .droid_weight_ik_rel_visuomotor_mimic_env_cfg import (
    DroidWeightIKRelVisuomotorMimicEnvCfg,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_mimic_env_cfg.FrankaCubeStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_abs_mimic_env_cfg.FrankaCubeStackIKAbsMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_custom_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCustomMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-ID-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_custom_id_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCustomIDMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Gap-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_custom_gap_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCustomGapMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Gap-ID-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_custom_gap_id_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCustomGapIDMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-OOD-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_ood_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorOODMimicEnvCfg,
    },
    disable_env_checker=True,
)


##
# SkillGen
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_skillgen_env_cfg.FrankaCubeStackIKRelSkillgenEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_bin_stack_ik_rel_mimic_env_cfg.FrankaBinStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Galbot Stack Cube with RmpFlow - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_rel_mimic_env:RmpFlowGalbotCubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_rel_mimic_env_cfg:RmpFlowGalbotLeftArmGripperCubeStackRelMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_rel_mimic_env:RmpFlowGalbotCubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_rel_mimic_env_cfg:RmpFlowGalbotRightArmSuctionCubeStackRelMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# Galbot Stack Cube with RmpFlow - Absolute Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-Abs-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_abs_mimic_env:RmpFlowGalbotCubeStackAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_abs_mimic_env_cfg:RmpFlowGalbotLeftArmGripperCubeStackAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-Abs-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_abs_mimic_env:RmpFlowGalbotCubeStackAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_abs_mimic_env_cfg:RmpFlowGalbotRightArmSuctionCubeStackAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# Agibot Left Arm: Place Upright Mug with RmpFlow - Relative Pose Control
##
gym.register(
    id="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agibot_place_upright_mug_mimic_env_cfg:RmpFlowAgibotPlaceUprightMugMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
##
# Agibot Right Arm: Place Toy2Box: RmpFlow - Relative Pose Control
##
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agibot_place_toy2box_mimic_env_cfg:RmpFlowAgibotPlaceToy2BoxMimicEnvCfg",
    },
    disable_env_checker=True,
)


##
# Droid Stack Cube with IK - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_stack_ik_rel_visuomotor_mimic_env_cfg.DroidCubeStackIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Visuomotor-Sim-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_stack_ik_rel_visuomotor_sim_mimic_env_cfg.DroidCubeStackIKRelVisuomotorSimMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Visuomotor-Sim-ID-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_stack_ik_rel_visuomotor_sim_id_mimic_env_cfg.DroidCubeStackIKRelVisuomotorSimIDMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Droid-IK-Rel-Visuomotor-OOD-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_stack_ik_rel_visuomotor_ood_mimic_env_cfg.DroidCubeStackIKRelVisuomotorOODMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Water
##

gym.register(
    id="Isaac-Water-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidWaterIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_water_ik_rel_visuomotor_mimic_env_cfg.DroidWaterIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Water-Droid-Joint-Pos-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidWaterIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_water_joint_pos_visuomotor_mimic_env_cfg.DroidWaterJointPosVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Water Align
##

gym.register(
    id="Isaac-Water-Align-Droid-Visuomotor-IK-Rel-v0",
    entry_point="isaaclab_mimic.envs:DroidWaterIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_water_align_ik_rel_visuomotor_mimic_env_cfg.DroidWaterAlignIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)
##
# Droid Bread
##

gym.register(
    id="Isaac-Bread-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidBreadIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_bread_ik_rel_visuomotor_mimic_env_cfg.DroidBreadIkRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)


##
# Droid Can
##

gym.register(
    id="Isaac-Can-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCanIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidCanIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Can-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCanIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidCanIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)


##
# Droid Capsule
##

gym.register(
    id="Isaac-Capsule-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCapsuleIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidCapsuleIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)


##
# Droid Block
##

gym.register(
    id="Isaac-Block-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidBlockIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_block_ik_rel_visuomotor_mimic_env_cfg.DroidBlockIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Drink
##

gym.register(
    id="Isaac-Drink-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidDrinkIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidDrinkIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Drink-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidDrinkIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidDrinkIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Holder
##

gym.register(
    id="Isaac-Holder-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidHolderIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidHolderIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Cylinder
##

gym.register(
    id="Isaac-Cylinder-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidCylinderIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_cylinder_ik_rel_visuomotor_mimic_env_cfg.DroidCylinderIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Laptop
##

gym.register(
    id="Isaac-Laptop-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidLaptopIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_laptop_ik_rel_visuomotor_mimic_env_cfg.DroidLaptopIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Laptop-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidLaptopIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidLaptopIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Droid Oven
##

gym.register(
    id="Isaac-Oven-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidOvenIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_oven_ik_rel_visuomotor_mimic_env_cfg.DroidOvenIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Oven-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidOvenIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidOvenIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)


##
# Droid Plate
##

gym.register(
    id="Isaac-Plate-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPlateIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_plate_ik_rel_visuomotor_mimic_env_cfg.DroidPlateIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Plate-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPlateIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPlateIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pen-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPenIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPenIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pen-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPenIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPenIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Phone-Droid-IK-Rel-Sound-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPhoneIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPhoneIKRelSoundMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pot-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPotIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPotIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pot-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidPotIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidPotIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Scissor-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidScissorIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": droid_scissor_ik_rel_visuomotor_mimic_env_cfg.DroidScissorIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tea-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tea-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tea-Droid-IK-Rel-PointCloud-Masked-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaIKRelPointCloudMaskedMimicEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Hot-Tea-Droid-IK-Rel-Thermal-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidHotTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidHotTeaIKRelThermalMimicEnvCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Tea-Wood-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaWoodIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tea-Wood-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaWoodIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tea-Wood-Droid-IK-Rel-PointCloud-Masked-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidTeaIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidTeaWoodIKRelPointCloudMaskedMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Weight-Droid-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidWeightIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidWeightIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Weight-Droid-IK-Rel-PointCloud-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidWeightIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidWeightIKRelPointCloudMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Weight-Droid-IK-Rel-PointCloud-Masked-Mimic-v0",
    entry_point="isaaclab_mimic.envs:DroidWeightIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": DroidWeightIKRelPointCloudMaskedMimicEnvCfg,
    },
    disable_env_checker=True,
)
