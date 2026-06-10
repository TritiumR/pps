# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class DroidWeightIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Droid Weight IK Rel env.
    """

    should_terminate = dict[int, bool]()

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        ee_frame = self.scene["ee_frame"]
        robot = self.scene["robot"]

        eef_pos_base, eef_quat_base = PoseUtils.subtract_frame_transforms(
            robot.data.root_pos_w[env_ids],
            robot.data.root_quat_w[env_ids],
            ee_frame.data.target_pos_w[env_ids, 0, :],
            ee_frame.data.target_quat_w[env_ids, 0, :],
        )
        return PoseUtils.make_pose(
            eef_pos_base, PoseUtils.matrix_from_quat(eef_quat_base)
        )

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = {}

        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                obj_state["root_pose"][env_ids, :3],
                obj_state["root_pose"][env_ids, 3:7],
            )
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                pos_obj_base, PoseUtils.matrix_from_quat(quat_obj_base)
            )

        for art_name, art_state in articulation_states.items():
            if art_name == "robot":
                continue
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                art_state["root_pose"][env_ids, :3],
                art_state["root_pose"][env_ids, 3:7],
            )
            object_pose_matrix[art_name] = PoseUtils.make_pose(
                pos_obj_base, PoseUtils.matrix_from_quat(quat_obj_base)
            )

        return object_pose_matrix

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(
            delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
        ).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(
            delta_rotation_axis
        )[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        ).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp_pear": subtask_terms["grasp_pear"][env_ids],
            "pear_on_scale": subtask_terms["pear_on_scale"][env_ids],
            "grasp_apple": subtask_terms["grasp_apple"][env_ids],
        }
