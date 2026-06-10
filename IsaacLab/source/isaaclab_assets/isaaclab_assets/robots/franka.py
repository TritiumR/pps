# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from pathlib import Path

ASSET_PATH = Path(__file__).parent / "../../../../asset"
##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

DROID_CFG = FRANKA_PANDA_CFG.copy()
DROID_CFG.spawn.usd_path = f"{ASSET_PATH}/droid/droid.usd"
DROID_CFG.spawn.articulation_props.fix_root_link = True
# disable self collisions
# DROID_CFG.spawn.articulation_props.enabled_self_collisions = False
DROID_CFG.spawn.rigid_props.disable_gravity = True
DROID_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
# FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
DROID_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=400.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=400.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        # "right_outer_knuckle_joint" is its mimic joint
        joint_names_expr=["finger_joint"],
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,  # Reduced from 10.0 to 2.0 for gentler movement
        stiffness=5,  # Reduced from 17 to 5.0 for less aggressive movement
        damping=0.1,  # Increased from 0.02 to 0.1 for smoother movement
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,  # Reduced from 10.0 to 2.0 for gentler movement
        stiffness=0.2,  # Keep low for gentle movement
        damping=0.001,  # Increased from 0.001 to 0.01 for smoother movement
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


"""Configuration of DROID_CFG robot."""
DROID_STIFF_CFG = DROID_CFG.copy()

DROID_STIFF_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=400.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=400.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        # "right_outer_knuckle_joint" is its mimic joint
        joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1,
        velocity_limit_sim=2.0,  # Reduced from 10.0 to 2.0 for gentler movement
        stiffness=5.0,  # Reduced from 17 to 5.0 for less aggressive movement
        damping=0.1,  # Increased from 0.02 to 0.1 for smoother movement
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=1,
        velocity_limit_sim=2.0,  # Reduced from 10.0 to 2.0 for gentler movement
        stiffness=0.2,  # Keep low for gentle movement
        damping=0.01,  # Increased from 0.001 to 0.01 for smoother movement
    ),
    # "gripper_knuckle": ImplicitActuatorCfg(
    #     joint_names_expr=["right_outer_finger_joint", "left_outer_finger_joint"],
    #     effort_limit_sim=180,
    #     velocity_limit_sim=130,
    #     stiffness=0.05,
    #     damping=0.0,
    # ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_inner_finger_knuckle_joint",
            "right_outer_finger_joint",
            "left_outer_finger_joint",
        ],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.05,
        damping=0.0,
    ),
}
# DROID_STIFF_CFG.actuators["panda_shoulder"].stiffness = 400.0
# DROID_STIFF_CFG.actuators["panda_shoulder"].damping = 20.0
# DROID_STIFF_CFG.actuators["panda_forearm"].stiffness = 400.0
# DROID_STIFF_CFG.actuators["panda_forearm"].damping = 20.0

DROID_STIFF_CFG.actuators["panda_shoulder"].stiffness = 400.0
DROID_STIFF_CFG.actuators["panda_shoulder"].damping = 80.0
DROID_STIFF_CFG.actuators["panda_forearm"].stiffness = 400.0
DROID_STIFF_CFG.actuators["panda_forearm"].damping = 80.0

# DROID_STIFF_CFG.actuators["gripper_drive"].stiffness = 1.0
# DROID_STIFF_CFG.actuators["gripper_drive"].damping = 0.1
# DROID_STIFF_CFG.actuators["gripper_finger"].stiffness = 0.05
# DROID_STIFF_CFG.actuators["gripper_finger"].damping = 0.05

DROID_STIFF_CFG.actuators["gripper_drive"].stiffness = 5.0
DROID_STIFF_CFG.actuators["gripper_drive"].damping = 1
DROID_STIFF_CFG.actuators["gripper_finger"].stiffness = 0.05
DROID_STIFF_CFG.actuators["gripper_finger"].damping = 0.05

# # for gripper velocity control
# DROID_STIFF_CFG.actuators["gripper_drive"].stiffness = 1.0
# DROID_STIFF_CFG.actuators["gripper_drive"].damping = 5000.0
# DROID_STIFF_CFG.actuators["gripper_finger"].stiffness = 1.0
# DROID_STIFF_CFG.actuators["gripper_finger"].damping = 5000.0
# DROID_STIFF_CFG.actuators["gripper_passive"].stiffness = 1.0
# DROID_STIFF_CFG.actuators["gripper_passive"].damping = 5000.0

DROID_ONLINE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/droid_arhan.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=None,
            damping=None,
            velocity_limit=1.0,
        ),
    },
)
