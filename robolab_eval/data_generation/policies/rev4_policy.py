"""rev3 scripted policy plus a measured re-plan of the *place* target.

rev3 corrected the insertion by measuring the tool again once it hung upright.
The place branch never got the same treatment: it aims with
``R_obj_at_lift``, the object's orientation as read the instant the fingers
closed -- while the object is still lying on the table. For a compact payload
like the YCB drill that reading survives the lift. For a long one it does not: a
331 mm hammer pinched 42 mm from its head swings handle-down under its own
weight as soon as the table stops carrying it, and by the time it is up it hangs
some 200 mm below the fingertips instead of the 30 mm the table-flat reading
predicts.

Everything downstream is then wrong by that difference. The place pose is
computed to leave the object's lowest point a set clearance above the container
floor, but "lowest point" is taken in the stale orientation, so the whole
descent -- and the standoff above it, from which the arm crosses the table -- is
planned ~170 mm too low. Observed: the hammer's dangling handle tip rode at the
left bin's rim height throughout the traverse, struck the bin, and was knocked
out of the pinch.

The correction here is rev3's, applied to the other branch: hold still after the
lift until the payload stops swinging, measure the in-hand transform and the
object's orientation *again*, and re-plan the place pose and its standoff from
that. The standoff is additionally floored at whatever height the carried
object's lowest point needs to clear the container rim, and the arm rises to it
before it moves across rather than while it moves across.
"""

import numpy as np

from .rev2_geom import matrix_from_quat
from .rev3_policy import Rev3Policy


class Rev4Policy(Rev3Policy):
    """Scripted pick-and-place with a re-measured, rim-aware carry."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        c = dict(
            # Steps held still after the lift, before the payload is measured.
            # A long tool leaves the lift swinging from the pinch; measured
            # mid-arc it reads as pointing almost anywhere, and a place pose
            # planned from that reading is worse than the stale one.
            carry_settle_hold=90,
            # Clearance the carried object's lowest point keeps above the
            # container rim while the arm crosses to it.
            carry_clear=0.03,
            recompute_place=True,
            # finger_joint reading that means the jaws met with nothing between
            # them; the Robotiq's closed stop is pi/4.
            finger_closed_stop=0.77,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)
        self._replanned = False

    # ------------------------------------------------------------- geometry
    def _object_low_world(self):
        """World z of the object's lowest mesh point, from its live pose."""
        o_p, o_q = self._pose(self.obj_name)
        return float((self.obj_pts @ matrix_from_quat(o_q).T)[:, 2].min() + o_p[2])

    def _carried_tilt_deg(self):
        """How far the object has rotated in the pinch since the fingers closed."""
        _, o_q = self._pose(self.obj_name)
        R = matrix_from_quat(o_q) @ self.R_obj_at_close.T
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))

    # ------------------------------------------------------------- planning
    def _plan_tail(self):
        """rev3's tail, with a settle held after the lift in place mode."""
        _, o_q = self._pose(self.obj_name)
        self.R_obj_at_close = matrix_from_quat(o_q)
        super()._plan_tail()
        if self.mode != "place" or not self.cfg["recompute_place"]:
            return
        k = next(i for i, s in enumerate(self.stages) if s["name"] == "LIFT")
        lift = self.stages[k]
        self.stages.insert(k + 1, self._stage(
            "SETTLE", lift["pos"], lift["quat"], 1.0,
            hold=self.cfg["carry_settle_hold"],
            timeout=self.cfg["carry_settle_hold"] + 5))

    def _replan_place(self):
        """Re-measure the settled payload and re-plan the descent from it."""
        low_before = self._object_low_world()
        tilt = self._carried_tilt_deg()
        cf0 = self._container_frame()

        # A re-measurement is only worth acting on while the object is still in
        # the hand. If the grasp has failed the object is on the table or the
        # floor, and planning a place pose from *that* pose asks the arm to
        # reach as far below the container as the object has fallen -- i.e. to
        # drive itself upward by the same amount. Report and keep the planner's
        # own target rather than command it.
        held = float(self.robot.data.joint_pos[0, self.finger_id])
        # Two ways to have lost it, and both must be tested: the object may have
        # fallen away (lowest point far below the container floor) or it may
        # simply still be on the table, which reads at the same height as the
        # floor. The second shows up in the finger joint instead -- driven to
        # its closed stop, the jaws have nothing between them.
        empty = held > self.cfg["finger_closed_stop"]
        dropped = low_before < cf0["floor_z"] - 0.05 or empty
        self.report.update({
            "carry_tilt_since_close_deg": round(tilt, 2),
            "carry_object_low_world_before": round(low_before, 5),
            "carry_finger_joint_after_settle": round(held, 4),
        })
        if dropped:
            self.report["carry_replan_applied"] = False
            self._replanned = True
            self.log(f"[carry] payload's lowest point is at z={low_before:.4f} "
                     f"(container floor {cf0['floor_z']:.4f}) and the finger joint reads "
                     f"{held:.4f} rad ({'jaws closed on nothing' if empty else 'jaws hold something'})"
                     " -- the object is no longer in the hand; keeping the planner's "
                     "target rather than re-planning from a lost payload")
            return

        self._measure_in_hand()
        _, o_q = self._pose(self.obj_name)
        self.R_obj_at_lift = matrix_from_quat(o_q)
        p_des, q_des = self._place_target()
        cf = self._cf

        # Standoff: the planner's own height, but never less than the height at
        # which the carried object's lowest point clears the rim. At the place
        # pose that lowest point sits release_clear above the floor by
        # construction, so lifting the pose by dz lifts it to
        # floor + release_clear + dz.
        need = (cf["rim_z"] - cf["floor_z"] + self.cfg["carry_clear"]
                - self.cfg["release_clear"])
        dz = max(self.cfg["pretarget_dz"], need)
        pre = p_des.copy()
        pre[2] = p_des[2] + dz
        retreat = p_des.copy()
        retreat[2] = pre[2] + 0.06

        p_now, q_now = self._eef_pose()
        up = p_now.copy()
        up[2] = max(p_now[2], pre[2])

        self.report.update({
            "carry_replan_applied": True,
            "carry_pretarget_dz_m": round(float(dz), 5),
            "carry_rim_clearance_at_pretarget_m": round(
                float(cf["floor_z"] + self.cfg["release_clear"] + dz - cf["rim_z"]), 5),
            "object_in_gripper_pos_after_lift": self.report["object_in_gripper_pos"],
            "object_in_gripper_quat_after_lift": self.report["object_in_gripper_quat"],
        })
        self.log(f"[carry] payload has turned {tilt:.1f} deg in the pinch since close; "
                 f"its lowest point is at z={low_before:.4f} "
                 f"(rim {cf['rim_z']:.4f})")
        self.log(f"[carry] re-planned place from the measured pose: standoff dz={dz:.4f} "
                 f"(planner default {self.cfg['pretarget_dz']:.3f}), lowest point will "
                 f"clear the rim by "
                 f"{cf['floor_z'] + self.cfg['release_clear'] + dz - cf['rim_z']:.4f} m; "
                 f"rising to z={up[2]:.4f} before crossing")

        self._replanned = True
        self.stages = self.stages[:self.i] + [
            self._stage("CARRYRAISE", up, q_now, 1.0, speed=self.cfg["lift_speed"],
                        timeout=420),
            self._stage("PRETARGET", pre, q_des, 1.0, timeout=460),
            self._stage("PLACE", p_des, q_des, 1.0, speed=self.cfg["fine_speed"],
                        tol=self.cfg["pos_tol_fine"], timeout=360),
            self._stage("RELEASE", p_des, q_des, 0.0, hold=self.cfg["open_hold"],
                        timeout=self.cfg["open_hold"] + 5),
            self._stage("RETREAT", retreat, q_des, 0.0, speed=self.cfg["fine_speed"]),
        ]

    # -------------------------------------------------------------- hooking
    def _advance(self, st, p_now, q_now, settled):
        before = self.i
        super()._advance(st, p_now, q_now, settled)
        if self.i == before:
            return
        if (st["name"] == "SETTLE" and self.mode == "place"
                and self.cfg["recompute_place"] and not self._replanned):
            self._replan_place()
        elif st["name"] == "PLACE":
            self._check_grip("place")
            self.report["object_low_world_at_release"] = round(
                self._object_low_world(), 5)
            self.log(f"[place] object's lowest point at release: "
                     f"z={self.report['object_low_world_at_release']:.4f} "
                     f"(floor {self._cf['floor_z']:.4f})")
