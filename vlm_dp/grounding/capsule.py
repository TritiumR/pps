"""Capsule-task grounding: open the coffee maker's articulated lid, then insert the pod.

Stage ladder: grasp lid lip, pull it open (joint-based advance), release, grasp pod, lift, place into
the bay. Bay seat and lid thresholds mirror the env's own success terms, which is fixture knowledge
like the scale-platform calibration.
"""
from __future__ import annotations

import numpy as np

from vlm_dp.grounding import Grounding, SceneObject, Stage
from vlm_dp.sim_helpers import DEFAULT_EXTENT as _DEFAULT_EXTENT, usd_extents

_MACHINE = "capsule"
_POD = "can"
_LID_BODY = "E_shell_8"
_LID_JOINT = "RevoluteJoint_capsule_coffee_maker_3_up"
_LID_OPEN = -0.5          # env's open_threshold in rad. Opening drives the joint negative.
_LID_OPEN_DISP = 0.10     # sensed twin of the joint test: the lip travels ~0.12m to the open angle
_LID_LIP_LOCAL = np.array([-0.071, -0.234, -0.004])   # lip 2cm inboard of the shell's free edge (probed)
_LID_EXTENTS = (0.010, 0.015, 0.010)           # thin lip: grasped across its thin dimension
_BAY_LOCAL = np.array([0.0, 0.0, 0.27])        # env's target_local_pos (machine frame)
_BAY_XY = 0.09            # tighter than the env's 0.10 success radius
_BAY_Z = (0.255, 0.325)   # inside the env's [0.25, 0.33] band
_LIFT_HEIGHT = 0.15
_LIFT_CONFIRM = 0.05
_RETREAT = np.array([0.0, 0.0, 0.12])          # clear the lid after release
_RETREAT_EPS = 0.06

_LID_OPEN_LIFT = np.array([0.0, 0.0, 0.12])    # GT diagnostic: lift the front lip ~12cm to rotate the lid open


def gt_keypoints(env):
    """GT diagnostic keypoints for capsule, occlusion-free.

    The perception image has the arm over the lid so no keypoint lands on it. Fixed order the capsule
    fake-VLM expects: [0] lid lip (grasp to open), [1] lid-open goal (a virtual point with no object
    owner), [2] pod, [3] bay (pod destination), [4] machine body. Reuses the same GT calibration as
    CapsuleGrounding.

    Returns (keypoints[5,3], meta) where meta carries the ground truth the tracker's center-distance
    association cannot recover (the machine is a large fixture whose centre is far from its lid):
      owners: per-keypoint object identity, None for a free-space goal kept static,
      virtual: indices with no owner,
      grasp_extent: per-keypoint grasp half-width (the thin lip). Unset falls back to the object extent.
    """
    from vlm_dp.world import GTWorld
    gt = GTWorld(env.env)
    root, rot = gt.object_pose(_MACHINE)
    pos, brot = gt.body_pose(_MACHINE, _LID_BODY)
    lip = pos + brot @ _LID_LIP_LOCAL
    kps = np.array([lip, lip + _LID_OPEN_LIFT, gt.object_pose(_POD)[0], root + rot @ _BAY_LOCAL, root],
                   dtype=np.float64)
    meta = {"owners": [_MACHINE, None, _POD, _MACHINE, _MACHINE],
            "virtual": {1},
            "grasp_extent": {0: float(_LID_EXTENTS[0])}}   # thin lid lip, a pinchable half-width
    return kps, meta


class CapsuleGrounding:
    """Open-lid + insert-pod task from GT poses (the privileged rung of this task)."""

    def __init__(self, grasp_obj: str = _POD, place_obj: str = _MACHINE, grasp_objs=None,
                 seat_shift: bool = True):
        self.pod = grasp_obj
        self.machine = place_obj

    def _calibrate_machine(self, env):
        """One-shot workcell calibration: the machine is a fixture and never moves, so its root
        pose and the closed-lid lip point are static facts, not perception."""
        from vlm_dp.world import GTWorld
        gt = GTWorld(env.env)
        root, rot = gt.object_pose(self.machine)
        pos, brot = gt.body_pose(self.machine, _LID_BODY)
        return root, rot, pos + brot @ _LID_LIP_LOCAL

    def calibration_points(self, env) -> dict:
        """Seed the sensed world with the calibrated lip: tracked (FK/CoTracker) thereafter."""
        return {"lid": self._calibrate_machine(env)[2]}

    def ground(self, env, world) -> Grounding:
        extents = usd_extents(env, [self.pod])
        extents[_MACHINE] = (0.15, 0.15, 0.15)   # articulation, approximate, excluded as destination anyway
        extents["lid"] = _LID_EXTENTS

        pod_pos = lambda: world.object_pose(self.pod)[0]
        root0, rot0, lip0 = self._calibrate_machine(env)
        machine_pose = lambda: (root0, rot0)
        sensed = not hasattr(world, "joint_angle")   # SensedWorld: no articulation oracle

        if sensed:
            lip_pos = lambda: world.object_pose("lid")[0]
            # Sensed twin of the joint test: the lip has travelled away from its closed pose.
            lid_open = lambda: bool(np.linalg.norm(lip_pos() - lip0) > _LID_OPEN_DISP)
        else:
            def lip_pos():
                pos, rot = world.body_pose(self.machine, _LID_BODY)
                return pos + rot @ _LID_LIP_LOCAL

            def lid_open():
                return world.joint_angle(self.machine, _LID_JOINT) <= _LID_OPEN

        def bay_seat():
            root, rot = machine_pose()
            return (root + rot @ _BAY_LOCAL
                    - np.array([0.0, 0.0, extents.get(self.pod, _DEFAULT_EXTENT)[2]]))

        def pod_in_bay():
            root, rot = machine_pose()
            local = rot.T @ (np.asarray(pod_pos(), dtype=np.float64) - root)
            return bool(np.linalg.norm(local[:2] - _BAY_LOCAL[:2]) < _BAY_XY
                        and _BAY_Z[0] <= local[2] <= _BAY_Z[1])

        def retreat_point():
            return lip_pos() + _RETREAT

        def retreated():
            return bool(np.linalg.norm(np.asarray(env.tcp()) - retreat_point()) < _RETREAT_EPS)

        pod_z0 = float(pod_pos()[2])
        lift_target = (lambda z=pod_z0 + _LIFT_HEIGHT:
                       np.array([*np.asarray(pod_pos())[:2], z], dtype=np.float64))

        # Grounding-time calibration echo (lip offset is tuned against these prints + a frame).
        joint = "sensed" if sensed else f"{world.joint_angle(self.machine, _LID_JOINT):.3f}"
        print(f"[capsule] machine={np.round(root0, 3)} lip={np.round(lip_pos(), 3)} "
              f"pod={np.round(pod_pos(), 3)} joint={joint} "
              f"seat={np.round(bay_seat(), 3)} "
              f"pod_extents={tuple(round(x, 4) for x in extents.get(self.pod, _DEFAULT_EXTENT))}", flush=True)

        objects = [SceneObject(name=self.pod, pos=pod_pos, extents=extents.get(self.pod, _DEFAULT_EXTENT)),
                   SceneObject(name="lid", pos=lip_pos, extents=_LID_EXTENTS),
                   SceneObject(name=self.machine, pos=(lambda: machine_pose()[0]),
                               extents=extents[_MACHINE])]
        stages = [
            # One stage: the grasp terms hook the lip and lever it open. The goal is the joint angle,
            # not a certified pinch, since a press-hook opens it without ever stalling in-band.
            Stage(name="open lid", gripper="close", grasp_obj="lid", payload=None, target=lip_pos,
                  done=lid_open, advance_on_done=True),
            Stage(name="release lid", gripper="open", grasp_obj=None, payload=None,
                  target=retreat_point, done=retreated, done_flag="open_coffee_lid"),
            Stage(name=f"grasp {self.pod}", gripper="close", grasp_obj=self.pod, payload=None,
                  target=pod_pos, done_flag="grasp_pod"),
            Stage(name=f"lift {self.pod}", gripper="hold", grasp_obj=self.pod, payload=self.pod,
                  target=lift_target,
                  done=(lambda z=pod_z0: float(pod_pos()[2]) > z + _LIFT_CONFIRM)),
            Stage(name=f"place {self.pod} in {self.machine}", gripper="place", grasp_obj=None,
                  payload=self.pod, place_target=self.machine, target=(lambda: bay_seat() + _RETREAT),
                  done=pod_in_bay, place_point=bay_seat,
                  carry_z=(lambda: float(bay_seat()[2]) + 0.25)),   # clear the machine rim during transit
        ]
        return Grounding(objects=objects, stages=stages,
                         manipulated=frozenset({self.pod, self.machine, "lid"}))
