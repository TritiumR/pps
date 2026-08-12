"""rev8 scripted policy reshaped into a *demonstration* generator.

rev2-rev8 were written to answer "does this task succeed at all". They succeed,
but they move like a script: the setpoint chases each waypoint at a constant
speed and stops dead on arrival, stages end after a counted number of steps, and
every episode is the same trajectory. All three are visible in a recorded
demonstration and all three are things behaviour cloning will faithfully copy.

Three changes, in the order they matter.

**Motion.** The setpoint is no longer teleported at constant speed. It is driven
by an acceleration-limited profile whose speed is capped both by a maximum and
by ``sqrt(2*a*d)`` -- the speed it can still bleed off before the target -- so
it accelerates out of a waypoint and decelerates into the next one instead of
starting and stopping instantaneously. Because the velocity itself is rate
limited, a target that changes mid-flight bends the path rather than breaking
it; transit waypoints are therefore *blended*, handed over as soon as the arm is
within ``blend_radius``, and the corner is rounded by the accel limit rather
than squared off by a full stop. Orientation is driven the same way, which also
removes the wrist snap that used to set the payload swinging.

**Transitions.** Every stage now ends on a condition about the world -- the hand
is within the grasp region, the grasp is established, clearance is reached, the
tool is aligned, the relation is achieved, the release is verified -- and the
step counts that used to end them survive only as timeouts. Nothing waits out a
fixed dwell, so there are no idle frames at waypoints.

**Grasp by commanded width, with no contact event at all.** rev7 latched its
held aperture on a *force* threshold, and rev9's first draft tried to latch it
on the finger stalling behind its command. Both were measured to be unusable on
this Robotiq: the drive joint keeps rotating through the object to its
mechanical stop, so the stall only fires at the stop (0.784 rad of a 0.785 rad
stop) with 29.8 N already in the pads, and the "grasp" is whatever squeeze the
servo happens to have reached by then.

The fix is to stop waiting for an event. The finger is a position servo and the
object's width is a geometric property, so the grasp simply *commands the
aperture that fits the object*: the fingers are told to close to the measured
width of the object across the direction they close along, less a small squeeze.
``rev9_calibrate.py`` measures the map from the action channel to the distance
between the pads once, in free air, and writes it out with its provenance;
``rev9_gripper_cal.json`` is that table. Nothing about the grasp then depends on
force, on contact, or on any quantity a camera cannot see -- the only input is
the object's own geometry. Force is still read, but only as a cross-check that
gets reported.

On top of that, each episode draws a *variant*: where along the handle to grasp,
from which direction and height to approach it, how high to carry, how much to
bow the transport path, where inside the target to release, and which way and
how far to retreat. The ranges are chosen so every draw is a good solution --
the intent is to vary the solution, not the quality.
"""

import json
import os

import numpy as np
import torch

from .rev2_geom import matrix_from_quat, quat_from_matrix
from .rev2_policy import _np, _qmul, slerp_step
from .rev8_policy import Rev8Policy

# Stages the arm may hand over early, as soon as it is within blend_radius.
# These are places the hand is passing through, not places it must arrive at.
# RETREAT is deliberately *not* one of them: it is the last thing the demo does,
# and handing it over early would end the episode with the arm still moving, so
# the recorded command would step from "flying" to "hold" in one frame.
VIA_STAGES = {"PREGRASP", "LIFT", "TRANSIT", "CARRYRAISE", "PRETARGET", "REALIGN",
              "SIDESTEP", "WITHDRAW"}

DEFAULT_GRIPPER_CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "rev9_gripper_cal.json")


class GripperCal:
    """The measured map between the aperture command and the jaw separation.

    Produced by ``rev9_calibrate.py``; see that script for how it is measured.
    The table is inverted by interpolation rather than through the fit, so the
    command the policy issues is the one that was actually observed to give that
    separation.
    """

    def __init__(self, path=DEFAULT_GRIPPER_CAL):
        with open(path) as fh:
            cal = json.load(fh)
        self.path = path
        self.provenance = cal["provenance"]
        cmd = np.array([s["command"] for s in cal["samples"]], dtype=float)
        sep = np.array([s["separation_m"] for s in cal["samples"]], dtype=float)
        order = np.argsort(sep)  # np.interp needs an increasing abscissa
        self.sep, self.cmd = sep[order], cmd[order]
        self.open_m = float(sep.max())
        self.closed_m = float(sep.min())

    def command_for(self, separation_m):
        """Aperture command that puts the pads ``separation_m`` apart."""
        return float(np.interp(separation_m, self.sep, self.cmd))

    def separation_for(self, command):
        """Jaw separation the given aperture command produces in free air."""
        return float(np.interp(command, self.cmd[::-1], self.sep[::-1]))


class Rev9Policy(Rev8Policy):
    """rev8's task logic with demonstration-quality motion and timing."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None,
                 log=print):
        c = dict(
            # ---- motion shaping (per control step; dt = 1 step) ----
            max_accel=0.0012,          # m/step^2, transit
            fine_max_accel=0.0007,     # m/step^2, descend / insert / place
            max_ang_accel=0.005,       # rad/step^2
            max_ang_speed=0.05,        # rad/step
            blend_radius=0.025,        # hand over a via waypoint this far out
            # ---- closed-loop tolerances ----
            reach_tol_via=0.012,
            reach_tol=0.006,
            reach_tol_fine=0.003,
            ang_tol_stage=0.06,
            still_speed=0.0004,        # m/step that counts as stopped
            still_ang=0.004,           # rad/step that counts as stopped
            # ---- grasp by commanded width ----
            # The aperture is commanded to the object's own measured width less
            # a squeeze. The squeeze is what makes it a grip rather than a
            # touch: the finger is a position servo, so asking it for a
            # separation narrower than the object is what develops the normal
            # force.
            #
            # A fixed 3 mm is not enough, and the reason is geometric rather
            # than about force. The object turns a little in the pads as it
            # takes up its load, and a turned object presents a *different*
            # width to the jaws; on a 28 mm oval hammer handle that change is
            # several millimetres, so a 3 mm squeeze is used up by the settling
            # and the grip goes slack -- measured as 11.4 N at closing, 4.4 N
            # once the load was taken and 0 N by the lift, with the hammer left
            # on the table. The squeeze therefore scales with the width, which
            # is what bounds how much the presented width can change: a fifth of
            # it reproduces the apertures that carried every object in
            # rev7/rev8, with a floor for the near-round handles.
            squeeze_margin_m=0.003,
            squeeze_frac=0.20,
            gripper_cal=DEFAULT_GRIPPER_CAL,
            # Widest object the jaws will be asked to take; beyond this the
            # measurement is reported and the grasp is refused rather than
            # silently commanding a fully open hand.
            max_grasp_width_m=0.078,
            # The grasp is complete when the whole aperture command has been
            # issued and the finger has stopped moving against the object --
            # both readable from gripper_pos and the policy's own command.
            # rev2 sized ``open_hold`` as a dwell -- 25 steps of holding the
            # jaws open. It is now a time-out on a condition that needs longer
            # than that to be true at all: the aperture is rate limited, so
            # opening from a held width takes ~16 steps by itself, and the test
            # that the object has come to rest is measured over a 10-step
            # window on top. At 25 the stage could only ever end on its cap.
            open_hold=70,
            grasp_still_rad=1e-4,
            grasp_still_steps=4,
            # Consecutive steps of a stopped arm that end a fine-positioning
            # stage the differential IK cannot converge any further.
            fine_stall_steps=10,
            # Retained from rev9's first draft, no longer used to decide
            # anything; see ``_aperture_stalled``.
            stall_rate_frac=0.45,
            stall_confirm_steps=4,
            stall_arm_rad=0.20,
            # The aperture is an action channel like any other, so a stage that
            # changes it (1.0 -> slide width -> open) must not step it. Rate
            # limiting it keeps the gripper column of an action chunk as smooth
            # as the arm columns.
            grip_rate=0.05,            # max change in grip command per step
            # ---- payload settling, measured not counted ----
            # The payload has settled when it has stopped moving *in the hand*
            # and stopped turning *in the room*.
            inhand_still_m=0.00025,    # m/step of drift in the gripper frame
            inhand_still_rad=0.0025,   # rad/step of drift in the gripper frame
            world_turn_still_rad=0.0012,  # rad/step, over a window (see below)
            world_turn_window=10,
            rest_still_m=0.0002,       # m/step of object travel, over the window
            swing_still_steps=8,       # consecutive quiet steps that confirm it
            # Retained, reported, no longer consulted: the payload's motion in
            # the room has an arm-servo floor it never gets below.
            swing_still_radps=0.08,
            swing_still_mps=0.008,
            swing_still_deg=0.15,
            # A long tool hanging from the pinch is a slow pendulum; rev4's
            # 90-step allowance was sized for a dwell, not for a condition, and
            # a hammer is still turning 0.2 deg a step when it expires.
            carry_settle_hold=150,
            # ---- variation ----
            variant_seed=None,
            vary=True,
            variant_scale=1.0,         # 1.0 normal, <1 conservative (hammer)
            # Values pinned over the draw, so a demonstration can be asked for a
            # *named* solution -- grasp the far end, carry low, leave to the left
            # -- rather than for one more sample from the same cloud. A
            # validation set wants modes, not perturbations.
            variant_force=None,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)

        self.vel = np.zeros(3)
        self.ang_vel = 0.0
        self._stall_count = 0
        self._prev_finger = None
        self._last_grip = None
        self._last_action = None
        self._grip_still = 0
        self._settle_still = 0
        self._fine_stalled = 0
        self._settle_seen = [float("inf")] * 3
        self._quat_hist = []
        self._rest_hist = []
        self._still_low = 0
        self._prev_inhand = None
        self._prev_eef = None
        self._eef_speed = float("inf")
        self._prev_obj_quat = None
        self._obj_spin_deg = float("inf")
        self._cond_met_at = {}
        self.cal = GripperCal(self.cfg["gripper_cal"])
        self.width_cmd = None
        self.variant = self._draw_variant()

    # ------------------------------------------------------------- variation
    def _draw_variant(self):
        """Sample the episode's geometric choices and fold them into the cfg.

        Every quantity here changes *which* good solution is demonstrated. None
        of them changes how well it is executed.

        Which of them are live depends on the branch. ``target_off_m`` moves
        only a *placement* (see ``_place_target``), and ``retreat_dz`` only an
        *insertion*, because the place branch re-plans its own tail after the
        carry and sizes the retreat from the rim clearance it just computed.
        Both are reported either way, so what the episode was asked for and what
        it could act on stay distinguishable.
        """
        if not self.cfg["vary"]:
            self.report["variant"] = None
            return {}
        rng = np.random.default_rng(self.cfg["variant_seed"])
        s = float(self.cfg["variant_scale"])

        def u(lo, hi):
            mid = 0.5 * (lo + hi)
            return float(mid + s * (rng.uniform(lo, hi) - mid))

        v = {
            # where along the handle the fingers close
            "grasp_shift_m": u(-0.022, 0.022),
            # how the hand is turned about the approach axis when it closes
            "grasp_yaw_deg": u(-7.0, 7.0),
            # how high the pre-grasp stands, and how far it is offset laterally,
            # so the descent is a gentle diagonal rather than a plumb line
            "approach_h": u(0.115, 0.165),
            "pregrasp_off_m": [u(-0.028, 0.028), u(-0.028, 0.028)],
            # how high the tool is carried
            "lift_h": u(0.30, 0.375),
            # sideways bow of the transport path, so it is not a straight line
            "transit_bow_m": u(-0.065, 0.065),
            # where inside the target the object is set down / lined up
            "target_off_m": [u(-0.030, 0.030), u(-0.030, 0.030)],
            # how high above the target the tool is turned upright
            "over_dz": u(0.165, 0.215),
            # how the hand leaves
            "retreat_dz": u(0.10, 0.16),
            "lateral_offset": u(0.070, 0.100),
            # Which way the hand leaves. Only honoured when it is *forced*:
            # left to itself the planner picks the side with free workspace,
            # which is a better default than a coin toss.
            "lateral_sign": int(rng.choice([-1, 1])),
        }
        forced = dict(self.cfg["variant_force"] or {})
        v.update(forced)
        # Hand the ones the inherited planners already read straight to them.
        self.cfg["approach_h"] = v["approach_h"]
        self.cfg["lift_h"] = v["lift_h"]
        self.cfg["reorient_dz"] = v["over_dz"]
        self.cfg["retreat_dz"] = v["retreat_dz"]
        self.cfg["lateral_offset"] = v["lateral_offset"]
        self.cfg["lateral_offset_max"] = max(self.cfg["lateral_offset_max"],
                                             v["lateral_offset"])
        if "lateral_sign" in forced:
            self.cfg["lateral_sign"] = int(forced["lateral_sign"])
        if forced:
            self.report["variant_forced"] = {k: forced[k] for k in sorted(forced)}
        self.report["variant"] = {k: (np.round(x, 5).tolist()
                                      if isinstance(x, list) else
                                      (round(x, 5) if isinstance(x, float) else x))
                                  for k, x in v.items()}
        self.log("[variant] " + "  ".join(
            f"{k}={np.round(x, 4).tolist() if isinstance(x, list) else x}"
            for k, x in v.items()))
        return v

    def _plan_grasp(self):
        """Inherited grasp, moved by the variant, then sized to the object."""
        super()._plan_grasp()
        self._apply_variant_grasp()
        self._plan_width_grasp()

    def _apply_variant_grasp(self):
        """Move the grasp along the handle and turn it, as the variant asks."""
        if not self.variant:
            return
        shift = self.variant["grasp_shift_m"]
        lf = self.limb
        # Slide the grasp along the limb's own axis, staying inside the limb.
        axis_w = matrix_from_quat(self._pose(self.obj_name)[1]) @ lf["axis"]
        t_now = float(lf.get("grasp_t", 0.0))
        lo, hi = min(lf["limb_t"]), max(lf["limb_t"])
        room = 0.20 * (hi - lo)
        shift = float(np.clip(shift, lo + room - t_now, hi - room - t_now))
        self.grasp_pos = self.grasp_pos + shift * axis_w
        self.grasp_point_w = self.grasp_point_w + shift * axis_w

        yaw = np.radians(self.variant["grasp_yaw_deg"])
        rz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.grasp_quat = _qmul(rz, self.grasp_quat)

        self.report["variant_grasp_shift_applied_m"] = round(shift, 5)
        self.report["grasp_point_world"] = np.round(self.grasp_point_w, 5).tolist()
        self.log(f"[variant] grasp moved {shift:+.4f} m along the handle and turned "
                 f"{self.variant['grasp_yaw_deg']:+.1f} deg")

    # -------------------------------------------------------- grasp by width
    def _pad_footprint(self):
        """Where the finger pads are, in the eef frame: what they can touch.

        The width to command is not the object's width somewhere near the grasp;
        it is the width of the material that ends up *between the pads*. The
        pads are 27 mm deep and reach 57 mm back from the fingertips, so on a
        curved or tapered object -- a banana, a hammer handle flaring into its
        head -- most of a band along the limb never comes near them, and
        measuring the band over-reads the width by a centimetre.
        """
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
        base_prim = self._find_prim(robot_prim, "base_link")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        pts = np.concatenate([
            self._mesh_points(self._find_prim(robot_prim, name), base_prim, cache)
            for name in ("left_inner_finger", "right_inner_finger")])

        # base_link and eef share an origin, so the eef axes expressed in
        # base_link are all that is needed to read the pads in the eef frame.
        R_down = matrix_from_quat(self.q_down)
        close_local = R_down.T @ np.array([np.cos(self.finger_axis_angle),
                                           np.sin(self.finger_axis_angle), 0.0])
        approach_local = np.array([0.0, 0.0, 1.0])
        escape_local = np.cross(approach_local, close_local)
        R_base_eef = matrix_from_quat(self.eef_offset.detach().cpu().numpy().astype(float))
        pts_e = pts @ R_base_eef

        a = pts_e @ approach_local
        e = pts_e @ escape_local
        fp = {"approach_lo": float(a.min()), "approach_hi": float(a.max()),
              "half_escape": float(np.abs(e).max()),
              "close_local": close_local, "escape_local": escape_local}
        self.report["pad_footprint_eef"] = {
            "approach_span_m": [round(fp["approach_lo"], 5), round(fp["approach_hi"], 5)],
            "half_depth_across_escape_axis_m": round(fp["half_escape"], 5),
        }
        self.log(f"[pads] the pads occupy {fp['approach_lo']:.4f}..{fp['approach_hi']:.4f} m "
                 f"along the approach axis and reach "
                 f"{fp['half_escape'] * 1000:.1f} mm either side of the jaw centre line")
        return fp

    def _measure_grasp_width(self):
        """How wide the object is where the pads will close on it.

        Object points are taken into the eef frame at the grasp pose, cut down
        to those lying inside the pads' own footprint, and measured across the
        closing axis. That is the distance the jaws have to span, as opposed to
        ``grasp_cross_section``, which is an extent along the object's principal
        axes near the grasp and answers a different question.
        """
        fp = self._pad_footprint()
        o_p, o_q = self._pose(self.obj_name)
        R_grasp = matrix_from_quat(self.grasp_quat)
        pts_w = o_p + self.obj_pts @ matrix_from_quat(o_q).T
        pts_e = (pts_w - self.grasp_pos) @ R_grasp

        a = pts_e[:, 2]
        e = pts_e @ fp["escape_local"]
        inside = ((a >= fp["approach_lo"]) & (a <= fp["approach_hi"])
                  & (np.abs(e) <= fp["half_escape"]))
        if int(inside.sum()) < 8:
            return None, int(inside.sum())
        c = pts_e[inside] @ fp["close_local"]
        return float(c.max() - c.min()), int(inside.sum())

    def _plan_width_grasp(self):
        """Turn the measured width into the aperture the fingers will be told."""
        width, t_g = self._measure_grasp_width()
        if width is None:
            self.log(f"[width] only {t_g} object points fall inside the pads' "
                     "footprint, too few to measure a width; falling back to "
                     "commanding the closed stop")
            self.width_cmd = 1.0
            self.report["width_grasp"] = {"measured": False,
                                          "points_inside_pad_footprint": t_g}
            return

        squeeze = max(float(self.cfg["squeeze_margin_m"]),
                      float(self.cfg["squeeze_frac"]) * width)
        want = width - squeeze
        cmd = self.cal.command_for(want)
        cmd = float(np.clip(cmd, 0.0, self.cfg["max_hold"]))
        self.width_cmd = cmd
        too_wide = width > self.cfg["max_grasp_width_m"]
        self.report["width_grasp"] = {
            "object_width_across_jaws_m": round(width, 5),
            "squeeze_m": round(squeeze, 5),
            "squeeze_frac_of_width": round(squeeze / max(width, 1e-9), 4),
            "commanded_separation_m": round(want, 5),
            "aperture_command": round(cmd, 4),
            "aperture_command_rad": round(cmd * np.pi / 4, 5),
            "free_air_separation_at_command_m": round(self.cal.separation_for(cmd), 5),
            "pad_footprint_points_on_object": t_g,
            "wider_than_jaws": bool(too_wide),
            "calibration": self.cal.path,
            "calibration_provenance": self.cal.provenance,
        }
        self.log(f"[width] the object is {width * 1000:.2f} mm across the closing axis "
                 f"at the grasp band; commanding the jaws to "
                 f"{want * 1000:.2f} mm (squeeze {squeeze * 1000:.1f} mm), "
                 f"which the calibration gives as aperture {cmd:.4f} "
                 f"({cmd * np.pi / 4:.4f} rad)")
        if too_wide:
            self.log(f"[width] WARNING {width * 1000:.1f} mm is wider than the "
                     f"{self.cfg['max_grasp_width_m'] * 1000:.0f} mm this grasp is "
                     "sized for; the jaws will close on it as far as they can")

    def reset(self):
        """Inherited reset, with the pre-grasp offset off the plumb line."""
        super().reset()
        self.vel = np.zeros(3)
        self.ang_vel = 0.0
        self._stall_count = 0
        self._prev_finger = None
        self._last_grip = None
        self._last_action = None
        self._grip_still = 0
        self._settle_still = 0
        self._fine_stalled = 0
        self._settle_seen = [float("inf")] * 3
        self._quat_hist = []
        self._rest_hist = []
        self._still_low = 0
        self._prev_inhand = None
        self._prev_eef = None
        self._eef_speed = float("inf")
        self._hold_cmd = None
        self._prev_obj_quat = None
        if not self.variant:
            return
        off = self.variant["pregrasp_off_m"]
        for st in self.stages:
            if st["name"] == "PREGRASP":
                st["pos"] = st["pos"] + np.array([off[0], off[1], 0.0])

    def _place_target(self):
        """Inherited place pose, set down off the container's exact centre.

        Only the place branch takes this. The insert branch aims at the crock's
        mouth centre and stays there: the mouth is 109 x 78 mm and the whole
        insertion is already spending that margin on getting the tool to stand
        up in it, so moving the aim point is variation bought with the one
        quantity the task cannot spare.
        """
        p_des, q_des = super()._place_target()
        off = (self.variant or {}).get("target_off_m")
        if not off:
            return p_des, q_des
        p_des = p_des + np.array([off[0], off[1], 0.0])
        self.report["place_target_offset_m"] = [round(off[0], 5), round(off[1], 5)]
        self.log(f"[variant] setting the object down {np.round(off, 4).tolist()} off the "
                 "container's centre")
        return p_des, q_des

    def _letgo_stages(self, p_des, q_des):
        """Inherited let-go, rising by the variant's own retreat height."""
        stages = super()._letgo_stages(p_des, q_des)
        dz = self.cfg.get("retreat_dz")
        if dz is None:
            return stages
        for st in stages:
            if st["name"] == "RETREAT":
                # rev3 builds the rise as withdraw + 0.12; re-base it on the
                # height this episode asked for.
                st["pos"] = st["pos"] + np.array([0.0, 0.0, float(dz) - 0.12])
        return stages

    def _plan_tail(self):
        """Inherited tail, with a bowed transport path instead of a straight one."""
        super()._plan_tail()
        if not self.variant:
            return
        k = next((i for i, s in enumerate(self.stages) if s["name"] == "TRANSIT"), None)
        if k is None:
            return
        start = self.stages[k - 1]["pos"]
        end = self.stages[k]["pos"]
        span = end - start
        if float(np.linalg.norm(span[:2])) < 0.05:
            return
        # A via point half way along, pushed sideways in the horizontal plane.
        side = np.array([-span[1], span[0], 0.0])
        side = side / max(float(np.linalg.norm(side)), 1e-9)
        mid = 0.5 * (start + end) + side * self.variant["transit_bow_m"]
        mid[2] = max(start[2], end[2])
        self.stages.insert(k, self._stage("TRANSIT", mid, self.stages[k]["quat"],
                                          self.stages[k]["grip"], timeout=420))
        self.report["transit_via_world"] = np.round(mid, 5).tolist()
        self.log(f"[variant] transport bowed {self.variant['transit_bow_m']:+.3f} m "
                 f"through {np.round(mid, 3).tolist()}")

    # -------------------------------------------------- observable grasp state
    def _aperture_stalled(self):
        """Kept for reference; **not** what decides the grasp any more.

        The idea was that the finger's rate of progress would collapse when it
        met the object. Measured on this Robotiq it does not: the drive keeps
        turning through the object to its mechanical stop, so the only stall is
        the stop itself, reached at 0.784 rad of a 0.785 rad limit with 29.8 N
        already in the pads. rev9 commands the width instead; see
        ``_plan_width_grasp``.
        """
        fj = float(self.robot.data.joint_pos[0, self.finger_id])
        prev, self._prev_finger = self._prev_finger, fj
        cmd_rad = float(self._closing_cmd) * np.pi / 4.0
        if prev is None or cmd_rad < self.cfg["stall_arm_rad"]:
            return False
        step_cmd = float(self.cfg["close_rate"]) * np.pi / 4.0
        stalled = (fj - prev) < self.cfg["stall_rate_frac"] * step_cmd
        self._stall_count = self._stall_count + 1 if stalled else 0
        return self._stall_count >= self.cfg["stall_confirm_steps"]

    def _grip_command(self):
        """The aperture that fits the object, decided before the arm ever moved.

        There is nothing to latch and nothing to wait for: the command was
        computed from the object's measured width at planning time, and the rate
        limiter in ``act`` is what ramps the jaws onto it.
        """
        if self._hold_cmd is None:
            self._hold_cmd = float(self.width_cmd)
            self.report.update({
                "hold_command_latched": round(self._hold_cmd, 4),
                "hold_trigger": "commanded_width",
                # reported for cross-check only; nothing depends on it
                "hold_contact_force_N_observed": round(self._contact_N(), 3),
            })
            self.log(f"[width] closing to the commanded aperture {self._hold_cmd:.4f} "
                     f"({self._hold_cmd * np.pi / 4:.4f} rad), which the calibration "
                     f"puts {self.cal.separation_for(self._hold_cmd) * 1000:.2f} mm "
                     "between the pads")
        return self._hold_cmd

    # ------------------------------------------------------------- conditions
    def _obj_spin(self):
        """Degrees of object rotation per step -- how hard the payload is swinging."""
        _, q = self._pose(self.obj_name)
        if self._prev_obj_quat is not None:
            d = abs(float(np.dot(q, self._prev_obj_quat)))
            self._obj_spin_deg = float(np.degrees(2.0 * np.arccos(min(1.0, d))))
        self._prev_obj_quat = q
        return self._obj_spin_deg

    def _inhand_drift(self):
        """How fast the payload is still moving *relative to the hand*.

        This is the quantity the settle stage exists to stabilise: the next
        thing done after it is to re-measure the object-in-gripper transform and
        re-plan from it, so what has to have stopped changing is that transform,
        not the object's motion in the room. Asking the latter cannot work here
        -- the arm's own servo never fully stops, and it carries the payload
        with it, so the payload's world speed has a floor of 10-26 mm/s that the
        stage was left waiting out to its cap. In the hand the same payload is
        quiet within a few tens of steps.
        """
        e_p, e_q = self._eef_pose()
        o_p, o_q = self._pose(self.obj_name)
        R_e = matrix_from_quat(e_q)
        p_rel = R_e.T @ (o_p - e_p)
        q_rel = quat_from_matrix(R_e.T @ matrix_from_quat(o_q))
        prev = self._prev_inhand
        self._prev_inhand = (p_rel, q_rel)
        if prev is None:
            return float("inf"), float("inf")
        dp = float(np.linalg.norm(p_rel - prev[0]))
        dang = 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(q_rel, prev[1]))))))
        return dp, dang

    def _world_turn_rate(self):
        """How fast the payload is turning in the room, over a window.

        Differencing consecutive poses cannot answer this: the per-step rotation
        never reads below about 0.15 deg however still the object is, because
        that is the noise in the readback rather than motion. Over a window the
        noise cancels and a real drift does not, so the same data answers the
        question at a tenth of the scale.
        """
        _, q = self._pose(self.obj_name)
        self._quat_hist.append(q)
        n = int(self.cfg["world_turn_window"])
        if len(self._quat_hist) <= n:
            return float("inf")
        del self._quat_hist[:-(n + 1)]
        d = abs(float(np.dot(self._quat_hist[-1], self._quat_hist[0])))
        return 2.0 * float(np.arccos(min(1.0, d))) / n

    def _obj_motion(self):
        """The payload's own linear and angular speed, in the room.

        Differencing the pose between steps was the obvious way to ask whether
        the payload had stopped swinging, and it does not work below about a
        quarter of a degree a step: the quaternion read back has enough noise at
        that scale that the measure never settles, and the stage sat out its
        entire allowance -- 145 idle frames on the spatula, a fifth of the demo
        -- waiting for a number that could not get there. The body's velocity is
        the same question asked of a quantity that is actually resolved.
        """
        body = self.env.scene[self.obj_name]
        v = float(np.linalg.norm(_np(body.data.root_lin_vel_w[0])))
        w = float(np.linalg.norm(_np(body.data.root_ang_vel_w[0])))
        return v, w

    def _stage_done(self, st, p_now, q_now):
        """Whether this stage's *task condition* is satisfied, right now."""
        name = st["name"]
        pos_err = float(np.linalg.norm(st["pos"] - p_now))
        ang_err = 2.0 * np.arccos(min(1.0, abs(float(np.dot(q_now, st["quat"])))))
        speed = float(np.linalg.norm(self.vel))

        if name in VIA_STAGES:
            # Passing through: hand over as soon as the corner is in reach.
            return pos_err < max(self.cfg["blend_radius"], self.cfg["reach_tol_via"])

        if name == "DESCEND":
            # Within the grasp region, and no longer travelling.
            return (pos_err < self.cfg["reach_tol_fine"]
                    and ang_err < self.cfg["ang_tol_stage"]
                    and speed < self.cfg["still_speed"])

        if name == "CLOSE":
            # Grasp established: the whole aperture command has been issued --
            # the rate limiter has finished ramping onto it -- and the finger has
            # stopped moving, because the object is now what is stopping it.
            issued = (self._last_grip is not None and self.width_cmd is not None
                      and self._last_grip >= self.width_cmd - 1e-6)
            return issued and self._grip_still >= self.cfg["grasp_still_steps"]

        if name == "TAKELOAD":
            # The load has been taken up: at height, with the aperture holding.
            return pos_err < self.cfg["reach_tol"] and self._hold_cmd is not None

        if name == "REORIENT":
            # Aligned, and the wrist has actually stopped turning.
            return (ang_err < self.cfg["ang_tol_stage"]
                    and abs(self.ang_vel) < self.cfg["still_ang"])

        if name == "SETTLE":
            # Measured, not counted: the payload has stopped swinging, and the
            # hand it hangs from has stopped moving. One quiet step is not
            # enough -- a pendulum is momentarily still at each end of its arc,
            # and the whole point of this stage is that the *next* thing done is
            # to re-measure where the payload is. Reading it at the top of a
            # swing plans the placement from a pose the tool is about to leave,
            # which is how a hammer ended up asked for a wrist the arm could not
            # reach and spent 320 steps failing to converge on it.
            # Two things have to have stopped, and they are different things.
            # The payload can only move relative to the hand by turning in the
            # pinch, and that is what the *insert* branch re-measures. But the
            # *place* branch re-measures the payload's orientation in the room,
            # and that keeps changing for as long as the arm does -- LIFT is a
            # via stage, so this stage begins with the hand still flying at it.
            # Waiting only on the in-hand drift let the hammer be re-measured
            # ten steps in, mid-approach, and planned a placement the arm then
            # spent its whole allowance failing to reach.
            dp, dang = self._inhand_drift()
            world = self._world_turn_rate()
            self._settle_seen = [min(self._settle_seen[0], dp),
                                 min(self._settle_seen[1], dang),
                                 min(self._settle_seen[2], world)]
            quiet = (dp < self.cfg["inhand_still_m"]
                     and dang < self.cfg["inhand_still_rad"]
                     and world < self.cfg["world_turn_still_rad"])
            self._settle_still = (self._settle_still + 1) if quiet else 0
            return self._settle_still >= self.cfg["swing_still_steps"]

        if name in ("INSERT", "PLACE"):
            # The relation is achieved -- or the arm has stopped being able to
            # improve it. Both are world conditions; only the second is new.
            #
            # Asking solely for the first assumes the wrist can be tracked to
            # the stated tolerance everywhere, and reaching into the far bin it
            # cannot: the hammer's placement settles at a 7.8 deg residual the
            # differential IK will not remove however long it is given, so the
            # stage sat out its 360-step cap with the arm motionless against it
            # -- 115 consecutive idle frames, and the placement itself had been
            # correct for most of them. "The setpoint has arrived and the arm
            # has stopped moving" ends it on the same evidence a person would
            # use, and the residual is reported rather than waited out.
            arrived = (pos_err < self.cfg["reach_tol_fine"]
                       and ang_err < self.cfg["ang_tol_stage"]
                       and speed < self.cfg["still_speed"])
            stalled = (speed < self.cfg["still_speed"]
                       and self._eef_speed < self.cfg["still_speed"])
            self._fine_stalled = (self._fine_stalled + 1) if stalled else 0
            if arrived:
                self.report[f"{name.lower()}_exit"] = "converged"
                return True
            if self._fine_stalled >= self.cfg["fine_stall_steps"]:
                self.report[f"{name.lower()}_exit"] = "stalled"
                self.report[f"{name.lower()}_residual"] = [
                    round(pos_err, 5), round(float(np.degrees(ang_err)), 3)]
                return True
            return False

        if name == "SLIDE":
            # The tool has to have *started* down through the fingertips before
            # "it has stopped" means anything: at the moment the aperture opens
            # it is still held, and its lowest point is momentarily as constant
            # as it will be when it has finished seating.
            low = self._object_low_world()
            entry = self._cond_met_at.setdefault(("slide_low", self.i), low)
            descended = (entry - low) > 0.02
            return descended and self._obj_settled()

        if name == "RELEASE":
            # Released *and let go of*: the jaws have reached the commanded
            # aperture and the object has come to rest. Leaving before it has
            # stopped is what dragged the spatula out of the crock -- the opening
            # finger links carry the tool sideways, and a hand that sidesteps
            # during that carries it further.
            fj = float(self.robot.data.joint_pos[0, self.finger_id])
            want = float(self._last_grip if self._last_grip is not None else st["grip"])
            return (abs(fj - float(st["grip"]) * np.pi / 4.0) < 0.03
                    and abs(want - float(st["grip"])) < 1e-6
                    and self._obj_settled())

        if name == "RETREAT":
            # The last thing the demo does, so it ends stopped rather than
            # mid-flight: arrived, and no longer travelling.
            return (pos_err < self.cfg["reach_tol"]
                    and float(np.linalg.norm(self.vel)) < self.cfg["still_speed"])

        return pos_err < self.cfg["reach_tol"] and ang_err < self.cfg["ang_tol_stage"]

    def _track_rest(self):
        """Update, every step, how long the object has been at rest.

        Kept here rather than inside the stage that asks, because a window that
        only starts filling when the question is first asked answers "not yet"
        for its own length however long the object has actually been still. At
        the release that cost a 16-frame pause in every demonstration -- the
        hand stopped, the jaws open, the object already settled, and nothing
        happening while the evidence was gathered.
        """
        p, q = self._pose(self.obj_name)
        self._rest_hist.append((p, q))
        n = int(self.cfg["world_turn_window"])
        if len(self._rest_hist) <= n:
            self._still_low = 0
            return
        del self._rest_hist[:-(n + 1)]
        dp = float(np.linalg.norm(self._rest_hist[-1][0] - self._rest_hist[0][0])) / n
        d = abs(float(np.dot(self._rest_hist[-1][1], self._rest_hist[0][1])))
        dang = 2.0 * float(np.arccos(min(1.0, d))) / n
        quiet = dp < self.cfg["rest_still_m"] and dang < self.cfg["world_turn_still_rad"]
        self._still_low = (self._still_low + 1) if quiet else 0

    def _obj_settled(self, need=6):
        """True once the object has come to rest, over a window.

        The first version compared the object's lowest point between successive
        steps. That cannot decide it: the readback's own noise is of the same
        size as the threshold, so a tool that had finished seating minutes ago
        still failed the test and the stage ran to its cap -- 265 steps of
        guided slide, 127 of them with nothing moving at all. Measured as a net
        displacement across a window the noise cancels and only real motion is
        left, which is the same correction the settle stage needed.
        """
        return self._still_low >= need

    # ------------------------------------------------------------- stepping
    def act(self):
        """rev2's action with an acceleration-limited setpoint and live exits."""
        if self.i >= len(self.stages):
            self.stage_name = "DONE"
            return self._hold_action()

        st = self.stages[self.i]
        self.stage_name = st["name"]
        fine = st["speed"] <= self.cfg["fine_speed"] + 1e-9
        v_max = float(st["speed"])
        a_max = self.cfg["fine_max_accel"] if fine else self.cfg["max_accel"]

        # ---- translation: accelerate out, decelerate in, never jump ----
        delta = st["pos"] - self.setpoint_pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-9:
            # the fastest we may still be going and stop exactly on the target
            v_arrival = float(np.sqrt(max(0.0, 2.0 * a_max * dist)))
            v_des = delta / dist * min(v_max, v_arrival)
        else:
            v_des = np.zeros(3)
        dv = v_des - self.vel
        n = float(np.linalg.norm(dv))
        if n > a_max:
            dv *= a_max / n
        self.vel = self.vel + dv
        self.setpoint_pos = self.setpoint_pos + self.vel

        # ---- orientation: the same profile, so the payload is never snapped ----
        d = min(1.0, abs(float(np.dot(self.setpoint_quat, st["quat"]))))
        ang = 2.0 * np.arccos(d)
        w_arrival = float(np.sqrt(max(0.0, 2.0 * self.cfg["max_ang_accel"] * ang)))
        w_des = min(self.cfg["max_ang_speed"], w_arrival)
        dw = float(np.clip(w_des - self.ang_vel, -self.cfg["max_ang_accel"],
                           self.cfg["max_ang_accel"]))
        self.ang_vel = max(0.0, self.ang_vel + dw)
        if ang > 1e-6:
            self.setpoint_quat = slerp_step(self.setpoint_quat, st["quat"], self.ang_vel)

        p_now, q_now = self._eef_pose()
        self._track_rest()
        if self._prev_eef is not None:
            self._eef_speed = float(np.linalg.norm(p_now - self._prev_eef))
        self._prev_eef = p_now
        # The integral term only ever ran once the setpoint had stopped, which
        # made it a burst of correction at the end of a move. Feed it whenever
        # the hand is close and slow, so it trims continuously instead.
        if dist < 0.02 and float(np.linalg.norm(self.vel)) < 2.0 * self.cfg["still_speed"]:
            self.err_int = np.clip(self.err_int + self.cfg["ki"] * (self.setpoint_pos - p_now),
                                   -self.cfg["int_clip"], self.cfg["int_clip"])

        cmd_pos = self.setpoint_pos + self.err_int
        q_base_cmd = _qmul(self.setpoint_quat,
                           self.eef_offset_inv.detach().cpu().numpy().astype(float))
        cmd = torch.tensor(np.concatenate([cmd_pos, q_base_cmd]), dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        self.ik.set_command(cmd)
        p_base, q_base = self._base_link_root()
        q_des = self.ik.compute(p_base, q_base, self._jacobian_root(),
                                self.robot.data.joint_pos[:, self.arm_ids])
        if bool(torch.isnan(q_des).any()):
            q_des = (self._last_q.clone() if getattr(self, "_last_q", None) is not None
                     else self.robot.data.joint_pos[:, self.arm_ids].clone())
        lo = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 1]
        q_des = torch.clamp(q_des, lo, hi)
        self._last_q = q_des.clone()

        self._advance(st, p_now, q_now, True)
        grip = torch.full((1, 1), st["grip"], device=self.device)
        action = torch.cat([q_des, grip], dim=1)
        if self.cfg["hold_at_width"] and float(action[0, -1]) >= 1.0:
            # A stage asking for a closed hand gets the aperture that fits the
            # object. A partial aperture (the guided slide) and a full open are
            # deliberate widths in their own right and pass through unchanged.
            action[0, -1] = self._grip_command()
        # Rate limit the aperture channel itself, so a stage that changes the
        # grip ramps into it instead of stepping.
        want = float(action[0, -1])
        if self._last_grip is None:
            self._last_grip = want
        else:
            r = self.cfg["grip_rate"]
            self._last_grip = float(np.clip(want, self._last_grip - r,
                                            self._last_grip + r))
        action[0, -1] = self._last_grip
        self._track_finger()
        self._last_action = action.detach().clone()
        return action

    def _track_finger(self):
        """How long the finger has been standing still, in steps."""
        fj = float(self.robot.data.joint_pos[0, self.finger_id])
        prev, self._prev_finger = self._prev_finger, fj
        if prev is not None and abs(fj - prev) < self.cfg["grasp_still_rad"]:
            self._grip_still += 1
        else:
            self._grip_still = 0

    def _hold_action(self):
        """Hold the last *commanded* action, not the measured joint positions.

        The two differ by the servo's tracking error, so ending the episode on
        the measured configuration steps the recorded command by that error in a
        single frame -- a jump the learner sees at the very last boundary and has
        no way to explain. Replaying the last command keeps the action stream
        continuous through the end of the demonstration.
        """
        if self._last_action is not None:
            return self._last_action.clone()
        return super()._hold_action()

    def _advance(self, st, p_now, q_now, settled):
        """Advance on the stage's own condition; the step count is only a cap."""
        self.step_in_stage += 1
        done = self._stage_done(st, p_now, q_now)
        timed_out = self.step_in_stage >= st["timeout"]
        if not (done or timed_out):
            return
        pos_err = float(np.linalg.norm(st["pos"] - p_now))
        ang_err = 2.0 * np.arccos(min(1.0, abs(float(np.dot(q_now, st["quat"])))))
        if timed_out and not done:
            self.log(f"[timeout] {st['name']} hit its {st['timeout']}-step cap with "
                     f"pos_err={pos_err * 1000:.1f} mm ang_err={np.degrees(ang_err):.1f} deg")
        fj = float(self.robot.data.joint_pos[0, self.finger_id])
        self.log(f"[stage] {st['name']:9s} {'cond' if done else 'CAP '} in "
                 f"{self.step_in_stage:3d} steps  pos_err={pos_err * 1000:6.2f} mm  "
                 f"ang_err={np.degrees(ang_err):6.2f} deg  finger_joint={fj:.4f} rad")
        self.report.setdefault("stage_steps", {})[st["name"]] = self.step_in_stage
        self.report.setdefault("stage_exit", {})[st["name"]] = "cond" if done else "cap"

        self.i += 1
        self.step_in_stage = 0
        if self.i == len(self.stages) and not self.planned_tail:
            self._plan_tail()
        # Only now: rev3's _realign rebuilds ``stages[:self.i]``, so it must see
        # the index already advanced past the stage that just finished.
        self._on_stage_done(st)

    def _on_stage_done(self, st):
        """The measurement / replanning hooks rev3-rev8 hang off each stage."""
        name = st["name"]
        if name == "CLOSE":
            self._z_at_close = self._pose(self.obj_name)[0][2]
            self._check_grip("close")
        elif name == "TAKELOAD":
            self._check_grip("takeload")
        elif name == "LIFT":
            self._check_grip("lift")
        if name == "SETTLE":
            v, w, tw = self._settle_seen
            self.report["settle_quietest"] = {
                "inhand_drift_m_per_step": round(v, 6),
                "inhand_turn_rad_per_step": round(w, 6),
                "world_turn_rad_per_step": round(tw, 6)}
            self.log(f"[settle] payload quietest: {v * 1000:.3f} mm and "
                     f"{np.degrees(w):.3f} deg a step in the hand, "
                     f"{np.degrees(tw):.3f} deg a step in the room (asked for "
                     f"{self.cfg['inhand_still_m'] * 1000:.2f} mm, "
                     f"{np.degrees(self.cfg['inhand_still_rad']):.2f} deg, "
                     f"{np.degrees(self.cfg['world_turn_still_rad']):.3f} deg)")
        if name == "SETTLE" and self.mode == "insert" and self.cfg["realign"] \
                and not self._realigned:
            self._realign()
        elif name == "SETTLE" and self.mode == "place" and self.cfg["recompute_place"] \
                and not self._replanned:
            # rev4's carry re-measurement. rev9 replaced ``_advance`` wholesale
            # and this hook was left behind with it, so the place branch went
            # back to aiming with the orientation read while the object was
            # still lying on the table -- which for a 0.33 m hammer that swings
            # handle-down as soon as it is lifted is wrong by most of its
            # length. Measured consequence: a place pose the arm could not
            # reach, PLACE running its full 320-step cap 92 mm and 27 deg short,
            # and a quarter of the episode spent idle against it.
            self._replan_place()
        elif name == "INSERT" and self.mode == "insert":
            self._verify_before_release()
        elif name == "PLACE":
            self._check_grip("place")
            self.report["object_low_world_at_release"] = round(
                self._object_low_world(), 5)
        elif name == "SLIDE":
            o_p, _ = self._pose(self.obj_name)
            low = self._object_low_world()
            self.report["slide_object_low_world_after"] = round(low, 5)
            self.report["slide_object_pos_after"] = np.round(o_p, 5).tolist()
        elif name in ("SIDESTEP", "WITHDRAW", "RETREAT"):
            o_p, _ = self._pose(self.obj_name)
            low = self._object_low_world()
            self.report[f"object_pos_after_{name.lower()}"] = np.round(o_p, 5).tolist()
            self.report[f"object_low_world_after_{name.lower()}"] = round(low, 5)
