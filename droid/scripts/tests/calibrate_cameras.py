from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import calibrate_camera

# Make the robot env
env = RobotEnv()
env.camera_reader.enable_advanced_calibration()
controller = VRPolicy()
# camera_id = "19824535"
camera_id = "31079775"

print("Ready")
calibrate_camera(env, camera_id, controller)
