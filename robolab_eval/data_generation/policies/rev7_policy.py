"""rev5 scripted policy driving a *continuous-aperture* gripper.

With the binary gripper the grasp command is always "fully closed", and that
single fact produced both hammer failure modes measured in rev6. At stock gains
the servo keeps winning against friction and creeps the jaws shut around a
smooth handle until it is squeezed out -- observed as a monotone
0.47 -> 0.57 -> 0.59 -> 0.785 rad drift that loses the tool after roughly
250-300 steps whatever the arm is doing. Raising the gains does not fix it, it
inverts it: at effort 60 and above the jaws slam through the handle and eject it
on contact (275 N at effort 200).

Both are symptoms of commanding a *position* the object is not allowed to
occupy. Given a continuous aperture the fix is to stop doing that: close until
the object is felt, then latch the command at the width where contact happened
(less a small squeeze) and hold it. The servo target becomes the handle itself,
so there is no closure drive left to creep and no error left to spike.

Release then has two forms, both now expressible: open fully, or open part way
to an aperture that no longer pinches but still guides, so a tool slides down
through the fingertips under gravity instead of being swept aside by the links.

The gripper channel keeps the existing convention: 1.0 fully closed, 0.0 open.
"""

import numpy as np

from .rev2_geom import align_rotation, matrix_from_quat, quat_from_matrix
from .rev2_policy import _np
from .rev5_policy import Rev5Policy


class Rev7Policy(Rev5Policy):
    """Scripted policy that grasps by holding a measured width."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        c = dict(
            # Close on the object instead of driving to the closed stop.
            hold_at_width=True,
            # Aperture command per control step while closing (1.0 = closed).
            close_rate=0.02,
            # Contact force that counts as "the object is between the jaws".
            contact_threshold_N=0.5,
            # Extra closure past the contact width, so the hold is a grip rather
            # than a touch. In command units, i.e. fractions of the finger's
            # travel.
            squeeze_margin=0.05,
            # Ceiling on the latched command, so a missed contact cannot end up
            # driving to the stop anyway.
            max_hold=0.92,
            # Lay the object down instead of setting it on end. A tool that
            # hangs from the pinch arrives at the bin vertical, and a 0.33 m
            # hammer released upright in a 0.13 m deep bin has three quarters of
            # its length above the rim: it topples over the wall and out. Placing
            # it with its long axis horizontal puts the whole tool inside the
            # bin's footprint, which is what the bin is shaped for.
            place_horizontal=False,
            # Leave the jaws at the slide aperture for the withdrawal instead of
            # opening them fully. Once the tool has slid down and seated there is
            # nothing left to release, and the only thing a full open can still
            # do is swing the finger links through whatever part of the tool is
            # beside them. Retreating at the slide width avoids that last sweep.
            retreat_at_slide_grip=False,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)
        self._closing_cmd = 0.0
        self._hold_cmd = None

    def _letgo_stages(self, p_des, q_des):
        """rev5's let-go, optionally retreating at the slide aperture."""
        stages = super()._letgo_stages(p_des, q_des)
        if self.cfg["slide_grip"] is None or not self.cfg["retreat_at_slide_grip"]:
            return stages
        for st in stages:
            if st["name"] in ("RELEASE", "WITHDRAW", "RETREAT"):
                st["grip"] = float(self.cfg["slide_grip"])
        self.report["retreat_grip"] = round(float(self.cfg["slide_grip"]), 4)
        self.log(f"[slide] withdrawing at the slide aperture "
                 f"{self.cfg['slide_grip']:.2f} rather than opening fully")
        return stages

    # -------------------------------------------------------------- placing
    def _place_target(self):
        """rev2's place pose, optionally with the object laid flat in the bin."""
        if not self.cfg["place_horizontal"]:
            return super()._place_target()

        # Turn the measured carried orientation so the object's own long axis
        # becomes horizontal, keeping the rotation to the smallest one that does
        # it so the wrist is asked for as little as possible.
        axis_w = self.R_obj_at_lift @ self.limb["axis"]
        n = float(np.linalg.norm(axis_w))
        axis_w = axis_w / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        flat = np.array([axis_w[0], axis_w[1], 0.0])
        if float(np.linalg.norm(flat)) < 1e-6:
            flat = np.array([1.0, 0.0, 0.0])
        flat = flat / np.linalg.norm(flat)
        tilt = float(np.degrees(np.arccos(np.clip(abs(axis_w @ flat), -1.0, 1.0))))
        self.R_obj_at_lift = align_rotation(axis_w, flat) @ self.R_obj_at_lift
        self.report["place_horizontal_tilt_removed_deg"] = round(tilt, 2)
        self.log(f"[flat] carried tool sits {tilt:.1f} deg off horizontal; "
                 "laying it flat for the placement so it lands inside the bin "
                 "footprint rather than standing above the rim")
        return super()._place_target()

    # ------------------------------------------------------------ grip state
    def _contact_N(self):
        try:
            f = _np(self.world.get_contact_force(self.obj_name, "gripper", env_id=0))
            return float(np.linalg.norm(f))
        except Exception:  # noqa: BLE001
            return 0.0

    def _grip_command(self):
        """Aperture to command while a stage asks for a closed gripper."""
        if self._hold_cmd is not None:
            return self._hold_cmd

        force = self._contact_N()
        if force >= self.cfg["contact_threshold_N"] and self._closing_cmd > 0.0:
            self._hold_cmd = float(min(self._closing_cmd + self.cfg["squeeze_margin"],
                                       self.cfg["max_hold"]))
            fj = float(self.robot.data.joint_pos[0, self.finger_id])
            self.report.update({
                "hold_contact_force_N": round(force, 3),
                "hold_command_latched": round(self._hold_cmd, 4),
                "hold_finger_joint_at_contact": round(fj, 4),
                "hold_width_rad_at_contact": round(float(self._closing_cmd * np.pi / 4), 5),
            })
            self.log(f"[width] contact at {force:.2f} N with the aperture command at "
                     f"{self._closing_cmd:.3f} (finger_joint {fj:.4f} rad); latching the "
                     f"hold at {self._hold_cmd:.3f} and stopping the closure drive")
            return self._hold_cmd

        self._closing_cmd = float(min(1.0, self._closing_cmd + self.cfg["close_rate"]))
        return self._closing_cmd

    def act(self):
        """rev5's action, with the gripper channel replaced by a held width."""
        action = super().act()
        if not self.cfg["hold_at_width"]:
            return action

        commanded = float(action[0, -1])
        if commanded <= 0.0:
            # An open command releases the hold, so a later re-grasp re-measures.
            self._closing_cmd = 0.0
            self._hold_cmd = None
            return action
        if commanded < 1.0:
            # A deliberate partial aperture (guided slide) passes through, and
            # ends the hold so the slide is not overridden.
            self._hold_cmd = None
            self._closing_cmd = 0.0
            return action
        action[0, -1] = self._grip_command()
        return action

    def _check_grip(self, when):
        super()._check_grip(when)
        self.report[f"hold_command_at_{when}"] = (
            None if self._hold_cmd is None else round(self._hold_cmd, 4))
