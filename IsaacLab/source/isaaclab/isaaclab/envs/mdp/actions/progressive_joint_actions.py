# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg


class ProgressiveJointAction(ActionTerm):
    """Base class for binary joint actions.

    This action term maps a binary action to the *open* or *close* joint configurations. These configurations are
    specified through the :class:`BinaryJointActionCfg` object. If the input action is a float vector, the action
    is considered binary based on the sign of the action values.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term can mostly be used for gripper actions, where the gripper is either open or closed. This
    helps in devising a mimicking mechanism for the gripper, since in simulation it is often not possible to
    add such constraints to the gripper.
    """

    cfg: actions_cfg.ProgressiveJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(
        self, cfg: actions_cfg.ProgressiveJointActionCfg, env: ManagerBasedEnv
    ) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        self._total_progress = self.cfg.total_progress

        # assume gripper is open at the start
        self._grasping = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.bool
        )
        self._progress = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._previous_joint_pos = self._open_command.clone().repeat(self.num_envs, 1)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask following Droid convention
        close_command_mask = actions > 0.5

        # print(f"self._progress: {self._progress.shape}")
        # print(f"self._grasping: {self._grasping.shape}")

        # open the gripper with open command
        self._progress[~close_command_mask] = torch.clamp(
            self._progress[~close_command_mask] - 1,
            min=0,
        )
        self._grasping[~close_command_mask] = False

        # check if the gripper is grasping
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # print(f"current_joint_pos: {current_joint_pos.shape}")
        # print(f"self._previous_joint_pos: {self._previous_joint_pos.shape}")
        previous_target = (
            self._open_command
            + (self._close_command - self._open_command)
            * self._progress
            / self._total_progress
        )
        move_angle = current_joint_pos[:, 0:1] - self._previous_joint_pos[:, 0:1]
        to_target_angle = previous_target[:, 0:1] - current_joint_pos[:, 0:1]
        # print(f"progress: {self._progress}")
        # print(f"grasping: {self._grasping}")
        # print(f"move_angle: {move_angle}")
        non_move_mask = (
            move_angle.abs()
            < (self._close_command[0] - self._open_command[0]).abs()
            / self._total_progress
            / 2
        )
        non_open_mask = self._progress / self._total_progress > 0.2
        # print(f"non_move_mask: {non_move_mask.shape}")
        # print(f"close_command_mask: {close_command_mask.shape}")
        # print(f"non_open_mask: {non_open_mask.shape}")
        new_grasping_mask = (
            non_move_mask & close_command_mask & non_open_mask & ~self._grasping
        )
        self._grasping[new_grasping_mask] = True
        # relax the gripper when grasping state is newly reached
        self._progress[new_grasping_mask] = torch.clamp(
            self._progress[new_grasping_mask] - 1 / 2, min=0
        )

        # close the gripper with close command and non grasping state and close to the previous target
        close_to_previous_target_mask = (
            to_target_angle.abs()
            < (self._close_command[0] - self._open_command[0]).abs()
            / self._total_progress
            / 2
        )
        self._progress[
            close_command_mask & ~self._grasping & close_to_previous_target_mask
        ] = torch.clamp(
            self._progress[
                close_command_mask & ~self._grasping & close_to_previous_target_mask
            ]
            + 1,
            max=self._total_progress,
        )

        # update the previous joint positions
        self._previous_joint_pos = current_joint_pos.clone()

        # compute the command
        self._processed_actions = (
            self._open_command
            + (self._close_command - self._open_command)
            * self._progress
            / self._total_progress
        )

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class ProgressiveJointPositionAction(ProgressiveJointAction):
    """Progressive joint action that sets the progressive action into joint position targets."""

    cfg: actions_cfg.ProgressiveJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )
