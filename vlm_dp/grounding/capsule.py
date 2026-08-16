"""Ground the capsule task from privileged simulator state."""

from __future__ import annotations

import numpy as np

from vlm_dp.grounding import Grounding, SceneObject, Stage
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_MACHINE = "capsule"
_POD = "can"
_LID_BODY = "E_shell_8"
_LID_JOINT = "RevoluteJoint_capsule_coffee_maker_3_up"

_LID_OPEN = -0.5
_LID_OPEN_DISP = 0.10
_LID_LIP_LOCAL = np.array([-0.071, -0.234, -0.004])
_LID_EXTENTS = (0.010, 0.015, 0.010)

_BAY_LOCAL = np.array([0.0, 0.0, 0.27])
_BAY_XY = 0.09
_BAY_Z = (0.255, 0.325)

_LIFT_HEIGHT = 0.15
_LIFT_CONFIRM = 0.05

_RETREAT = np.array([0.0, 0.0, 0.12])
_RETREAT_EPS = 0.06

_LID_OPEN_LIFT = np.array([0.0, 0.0, 0.12])


def gt_keypoints(env):
    """Return fixed capsule keypoints and their ownership metadata."""
    from vlm_dp.world import GTWorld

    gt = GTWorld(env.env)
    root, rot = gt.object_pose(_MACHINE)
    pos, body_rot = gt.body_pose(_MACHINE, _LID_BODY)
    lip = pos + body_rot @ _LID_LIP_LOCAL

    keypoints = np.array(
        [
            lip,
            lip + _LID_OPEN_LIFT,
            gt.object_pose(_POD)[0],
            root + rot @ _BAY_LOCAL,
            root,
        ],
        dtype=np.float64,
    )
    metadata = {
        "owners": [_MACHINE, None, _POD, _MACHINE, _MACHINE],
        "virtual": {1},
        "grasp_extent": {0: float(_LID_EXTENTS[0])},
    }
    return keypoints, metadata


class CapsuleGrounding:
    """Open the coffee-maker lid, then place the pod in the bay."""

    def __init__(
        self,
        grasp_obj: str = _POD,
        place_obj: str = _MACHINE,
        grasp_objs=None,
        seat_shift: bool = True,
    ):
        self.pod = grasp_obj
        self.machine = place_obj

    def _calibrate_machine(self, env):
        """Return the static machine pose and closed-lid lip position."""
        from vlm_dp.world import GTWorld

        gt = GTWorld(env.env)
        root, rot = gt.object_pose(self.machine)
        pos, body_rot = gt.body_pose(self.machine, _LID_BODY)
        return root, rot, pos + body_rot @ _LID_LIP_LOCAL

    def calibration_points(self, env) -> dict:
        """Return the calibrated lid point used to seed tracking."""
        return {"lid": self._calibrate_machine(env)[2]}

    def ground(self, env, world) -> Grounding:
        extents = usd_extents(env, [self.pod])
        extents[_MACHINE] = (0.15, 0.15, 0.15)
        extents["lid"] = _LID_EXTENTS

        pod_pos = lambda: world.object_pose(self.pod)[0]
        root0, rot0, lip0 = self._calibrate_machine(env)
        machine_pose = lambda: (root0, rot0)
        sensed = not hasattr(world, "joint_angle")

        if sensed:
            lip_pos = lambda: world.object_pose("lid")[0]
            lid_open = lambda: bool(
                np.linalg.norm(lip_pos() - lip0) > _LID_OPEN_DISP
            )
        else:

            def lip_pos():
                pos, rot = world.body_pose(self.machine, _LID_BODY)
                return pos + rot @ _LID_LIP_LOCAL

            def lid_open():
                return (
                    world.joint_angle(self.machine, _LID_JOINT)
                    <= _LID_OPEN
                )

        def bay_seat():
            root, rot = machine_pose()
            return (
                root
                + rot @ _BAY_LOCAL
                - np.array(
                    [
                        0.0,
                        0.0,
                        extents.get(self.pod, _DEFAULT_EXTENT)[2],
                    ]
                )
            )

        def pod_in_bay():
            root, rot = machine_pose()
            local = rot.T @ (
                np.asarray(pod_pos(), dtype=np.float64) - root
            )
            return bool(
                np.linalg.norm(local[:2] - _BAY_LOCAL[:2]) < _BAY_XY
                and _BAY_Z[0] <= local[2] <= _BAY_Z[1]
            )

        def retreat_point():
            return lip_pos() + _RETREAT

        def retreated():
            return bool(
                np.linalg.norm(
                    np.asarray(env.tcp()) - retreat_point()
                )
                < _RETREAT_EPS
            )

        pod_z0 = float(pod_pos()[2])
        lift_target = (
            lambda z=pod_z0 + _LIFT_HEIGHT: np.array(
                [*np.asarray(pod_pos())[:2], z],
                dtype=np.float64,
            )
        )

        joint = (
            "sensed"
            if sensed
            else f"{world.joint_angle(self.machine, _LID_JOINT):.3f}"
        )
        print(
            f"[capsule] machine={np.round(root0, 3)} "
            f"lip={np.round(lip_pos(), 3)} "
            f"pod={np.round(pod_pos(), 3)} "
            f"joint={joint} "
            f"seat={np.round(bay_seat(), 3)} "
            f"pod_extents="
            f"{tuple(round(x, 4) for x in extents.get(self.pod, _DEFAULT_EXTENT))}",
            flush=True,
        )

        objects = [
            SceneObject(
                name=self.pod,
                pos=pod_pos,
                extents=extents.get(self.pod, _DEFAULT_EXTENT),
            ),
            SceneObject(
                name="lid",
                pos=lip_pos,
                extents=_LID_EXTENTS,
            ),
            SceneObject(
                name=self.machine,
                pos=lambda: machine_pose()[0],
                extents=extents[_MACHINE],
            ),
        ]

        stages = [
            Stage(
                name="open lid",
                gripper="close",
                grasp_obj="lid",
                payload=None,
                target=lip_pos,
                done=lid_open,
                advance_on_done=True,
            ),
            Stage(
                name="release lid",
                gripper="open",
                grasp_obj=None,
                payload=None,
                target=retreat_point,
                done=retreated,
                done_flag="open_coffee_lid",
            ),
            Stage(
                name=f"grasp {self.pod}",
                gripper="close",
                grasp_obj=self.pod,
                payload=None,
                target=pod_pos,
                done_flag="grasp_pod",
            ),
            Stage(
                name=f"lift {self.pod}",
                gripper="hold",
                grasp_obj=self.pod,
                payload=self.pod,
                target=lift_target,
                done=(
                    lambda z=pod_z0: float(pod_pos()[2])
                    > z + _LIFT_CONFIRM
                ),
            ),
            Stage(
                name=f"place {self.pod} in {self.machine}",
                gripper="place",
                grasp_obj=None,
                payload=self.pod,
                place_target=self.machine,
                target=lambda: bay_seat() + _RETREAT,
                done=pod_in_bay,
                place_point=bay_seat,
                carry_z=lambda: float(bay_seat()[2]) + 0.25,
            ),
        ]

        return Grounding(
            objects=objects,
            stages=stages,
            manipulated=frozenset(
                {self.pod, self.machine, "lid"}
            ),
        )