import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pathlib import Path

ASSET_PATH = Path(__file__).parent / "../../../../assets/tianji" 

# Resolved by Isaac Lab at load-time from duplicate gripper payload names.
LEFT_GRIPPER_JOINT = "finger_joint"
RIGHT_GRIPPER_JOINT = "finger_joint_0"
LEFT_RIGHT_OUTER_KNUCKLE_JOINT = "right_outer_knuckle_joint"
RIGHT_RIGHT_OUTER_KNUCKLE_JOINT = "right_outer_knuckle_joint_0"
LEFT_LEFT_INNER_FINGER_JOINT = "left_inner_finger_joint"
RIGHT_LEFT_INNER_FINGER_JOINT = "left_inner_finger_joint_0"
LEFT_RIGHT_INNER_FINGER_JOINT = "right_inner_finger_joint"
RIGHT_RIGHT_INNER_FINGER_JOINT = "right_inner_finger_joint_0"
LEFT_LEFT_INNER_FINGER_KNUCKLE_JOINT = "left_inner_finger_knuckle_joint"
RIGHT_LEFT_INNER_FINGER_KNUCKLE_JOINT = "left_inner_finger_knuckle_joint_0"
LEFT_RIGHT_INNER_FINGER_KNUCKLE_JOINT = "right_inner_finger_knuckle_joint"
RIGHT_RIGHT_INNER_FINGER_KNUCKLE_JOINT = "right_inner_finger_knuckle_joint_0"

TIANJI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(
            f"{ASSET_PATH}/tianji_marvin_CCS_with_robotiq_2f85.usd"
        ),
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
            # Left arm
            "left_joint1": 0.0,
            "left_joint2": 0.0,
            "left_joint3": 0.0,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 0.0,
            "left_joint7": 0.0,
            # Right arm
            "right_joint1": 0.0,
            "right_joint2": 0.0,
            "right_joint3": 0.0,
            "right_joint4": 0.0,
            "right_joint5": 0.0,
            "right_joint6": 0.0,
            "right_joint7": 0.0,
            # Left gripper
            LEFT_GRIPPER_JOINT: 0.0,
            LEFT_RIGHT_OUTER_KNUCKLE_JOINT: 0.0,
            LEFT_LEFT_INNER_FINGER_JOINT: 0.0,
            LEFT_RIGHT_INNER_FINGER_JOINT: 0.0,
            LEFT_LEFT_INNER_FINGER_KNUCKLE_JOINT: 0.0,
            LEFT_RIGHT_INNER_FINGER_KNUCKLE_JOINT: 0.0,
            # Right gripper
            RIGHT_GRIPPER_JOINT: 0.0,
            RIGHT_RIGHT_OUTER_KNUCKLE_JOINT: 0.0,
            RIGHT_LEFT_INNER_FINGER_JOINT: 0.0,
            RIGHT_RIGHT_INNER_FINGER_JOINT: 0.0,
            RIGHT_LEFT_INNER_FINGER_KNUCKLE_JOINT: 0.0,
            RIGHT_RIGHT_INNER_FINGER_KNUCKLE_JOINT: 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        # --- Arm actuators ---
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-4]"],
            effort_limit=108.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "left_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[5-7]"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[1-4]"],
            effort_limit=108.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "right_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[5-7]"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        # --- Gripper actuators (Droid-equivalent per side) ---
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=[LEFT_GRIPPER_JOINT],
            stiffness=None,
            damping=None,
            velocity_limit=1.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=[RIGHT_GRIPPER_JOINT],
            stiffness=None,
            damping=None,
            velocity_limit=1.0,
        ),
    },
)