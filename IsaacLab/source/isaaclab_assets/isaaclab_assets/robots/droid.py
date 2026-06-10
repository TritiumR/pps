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

ASSET_PATH = Path(__file__).parent / "../../../../assets/droid"
##
# Configuration
##
DROID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/droid.usd",
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
            # effort_limit=200.0,
            velocity_limit=1.0,
        ),
    },
)
