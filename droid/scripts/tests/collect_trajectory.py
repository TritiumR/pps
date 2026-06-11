import os
import argparse


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Collect trajectory data from the robot environment.")
    parser.add_argument("--save_filepath", type=str, default=None, help="Filepath to save the collected trajectory data.")
    return parser.parse_args()

args = parse_args()

if os.path.exists(args.save_filepath):
    print(f"WARNING: Overwriting existing file at {args.save_filepath}")
    # delete existing file
    if_delete = input("press y to delete existing file")
    if if_delete.lower() == 'y':
        os.remove(args.save_filepath)
    else:
        print("Exiting without overwriting the file.")
        exit(0)

import sys
sys.path.append('droid')

from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import collect_trajectory

# Make the robot env
env = RobotEnv()
controller = VRPolicy()

print("Ready")
collect_trajectory(env, controller=controller, self_save_filepath=args.save_filepath)
