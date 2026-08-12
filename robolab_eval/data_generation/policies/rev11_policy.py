"""rev9's demonstration generator with the two grasp-siting defects fixed.

rev9 sizes the grasp from one number -- the object's extent across the closing
axis, inside the pads' footprint, at the yaw the planner happened to choose --
and moves the grasp along the handle by whatever the variant asks for. Both are
fine on a straight handle and neither survives the round the demonstrations are
being scaled for.

**The width is read at one yaw, and on a curved object that yaw matters.** The
jaws are asked for the measured width less a fixed fraction of it, so an
over-read width is an over-squeeze, not a slacker grip. On the banana the three
recorded demonstrations differ only in the yaw the hand closes at -- -6, 0 and
+6 degrees -- and read 45.5, 44.7 and 40.7 mm; the -6 one then carried 71 N
through the take-up and 68 N through the lift against the +6 one's 31 N and the
0 one's 2.4 N. Nothing about the banana changed. What changed is which chord of
its arc the pads were asked to span. ``width_from="min_over_rotation"`` reads
the width the way the object will actually present it -- the narrowest section
the pads can close on, minimised over rotation about the approach axis -- and
``grasp_align_to_min_width`` closes there rather than measuring one place and
squeezing another. The whole profile is reported either way, so the choice is
visible in the demonstration's own metadata.

**A variant may move the grasp off the object.** ``grasp_shift_m`` slides the
grasp along the limb, clipped only to stay inside the limb's own extent. On the
spaghetti spoon a +18 mm shift left 180 object points inside the pads' footprint
against the 1400 an unshifted grasp gets: the pads are then holding the very end
of the handle, the head's lever arm is at its longest, and the tool pivoted 79
degrees in the pinch during the wrist turn. The re-aim that exists to catch that
correctly refused to act on it (0.104 m of lateral correction against a 0.06
bound) -- and the policy inserted anyway, 148 mm off the mouth, and the utensil
finished lying across the rim. ``pad_contact_guard`` walks such a shift back
until the pads have a real grip to measure, and reports how far it walked.

Neither change touches what the tasks *are*: the stage machine, the tolerances,
the acceleration limits and every task setting are rev9's.
"""

import numpy as np

from .rev2_geom import matrix_from_quat
from .rev2_policy import _qmul
from .rev9_policy import Rev9Policy


class Rev11Policy(Rev9Policy):
    """rev9, with the grasp width and the grasp station both measured properly."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None,
                 log=print):
        c = dict(
            # "nominal" reproduces rev9 exactly: the extent across the closing
            # axis at the planner's own yaw. "min_over_rotation" takes the
            # smallest extent over rotation of the jaws about the approach axis.
            width_from="nominal",
            # Close where that minimum is, instead of measuring one place and
            # squeezing another. Only meaningful with min_over_rotation.
            grasp_align_to_min_width=False,
            # Yaws the sweep visits, in degrees, over a half turn: the jaws are
            # symmetric, so a yaw and its opposite present the same chord.
            width_sweep_deg=np.arange(-90.0, 90.0, 5.0).tolist(),
            # Walk a variant's grasp shift back until the pads have this
            # fraction of the object points an unshifted grasp would have, so a
            # shift cannot site the grasp where there is nothing to hold.
            pad_contact_guard=True,
            pad_points_min_frac=0.5,
            pad_points_min=200,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)

    # -------------------------------------------------------- width over yaw
    def _yawed_axes(self, fp, deg):
        """The pads' closing and escape axes, turned about the approach axis."""
        t = np.radians(float(deg))
        close = np.cos(t) * fp["close_local"] + np.sin(t) * fp["escape_local"]
        escape = -np.sin(t) * fp["close_local"] + np.cos(t) * fp["escape_local"]
        return close, escape

    def _width_at(self, pts_e, fp, deg):
        """(width across the jaws, points between the pads) at one jaw yaw."""
        close, escape = self._yawed_axes(fp, deg)
        a = pts_e[:, 2]
        inside = ((a >= fp["approach_lo"]) & (a <= fp["approach_hi"])
                  & (np.abs(pts_e @ escape) <= fp["half_escape"]))
        n = int(inside.sum())
        if n < 8:
            return None, n
        c = pts_e[inside] @ close
        return float(c.max() - c.min()), n

    def _width_profile(self):
        """How wide the object is across the jaws, as a function of jaw yaw.

        Object points in the grasp frame, the pads' footprint applied at each
        yaw. Yaw 0 is the planner's own choice, so the profile is read against
        the number rev9 would have used.
        """
        fp = self._pad_footprint()
        o_p, o_q = self._pose(self.obj_name)
        pts_w = o_p + self.obj_pts @ matrix_from_quat(o_q).T
        pts_e = (pts_w - self.grasp_pos) @ matrix_from_quat(self.grasp_quat)
        prof = []
        for deg in self.cfg["width_sweep_deg"]:
            w, n = self._width_at(pts_e, fp, deg)
            if w is not None:
                prof.append((float(deg), w, n))
        return prof

    def _plan_width_grasp(self):
        """rev9's aperture, sized from the width the object actually presents."""
        prof = self._width_profile()
        if not prof:
            super()._plan_width_grasp()
            return

        nominal = next((w for d, w, _ in prof if abs(d) < 1e-9), None)
        best = min(prof, key=lambda r: r[1])
        self.report["width_profile"] = {
            "yaw_deg": [round(d, 1) for d, _, _ in prof],
            "width_m": [round(w, 5) for _, w, _ in prof],
            "points_between_pads": [n for _, _, n in prof],
            "width_at_planner_yaw_m": (None if nominal is None else round(nominal, 5)),
            "min_width_m": round(best[1], 5),
            "min_width_at_yaw_deg": round(best[0], 1),
            "spread_m": round(max(w for _, w, _ in prof) - best[1], 5),
        }
        self.log(f"[width] across the jaws the object measures "
                 f"{best[1] * 1000:.2f} mm at its narrowest (jaw yaw "
                 f"{best[0]:+.0f} deg) and "
                 f"{max(w for _, w, _ in prof) * 1000:.2f} mm at its widest"
                 + ("" if nominal is None else
                    f"; the planner's own yaw reads {nominal * 1000:.2f} mm"))

        if self.cfg["width_from"] == "min_over_rotation" and \
                self.cfg["grasp_align_to_min_width"] and abs(best[0]) > 1e-9:
            yaw = np.radians(best[0])
            rz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            # The sweep is expressed in the grasp frame, so the correction is
            # applied there: post-multiplying turns the jaws about their own
            # approach axis, which is what the sweep varied.
            self.grasp_quat = _qmul(self.grasp_quat, rz)
            self.report["grasp_aligned_to_min_width_deg"] = round(best[0], 1)
            self.log(f"[width] turning the hand {best[0]:+.0f} deg about the "
                     "approach axis so the jaws close on that narrowest section "
                     "rather than measuring it and squeezing somewhere else")

        super()._plan_width_grasp()
        wg = self.report.get("width_grasp")
        if not wg or not wg.get("object_width_across_jaws_m"):
            return
        if self.cfg["width_from"] != "min_over_rotation":
            wg["width_source"] = "planner_yaw"
            return

        # Re-size the aperture from the minimum. The squeeze is a fraction of
        # the width, so which width it is a fraction of is the whole question.
        # The profile is re-read rather than re-used: if the hand was turned
        # just above, the jaws are now somewhere else on the object, and the
        # width at the *new* planner yaw is what says whether the turn went the
        # way it was meant to.
        after = self._width_profile()
        width = min(after, key=lambda r: r[1])[1]
        at_yaw = next((w for d, w, _ in after if abs(d) < 1e-9), None)
        if at_yaw is not None:
            self.report["width_profile"]["width_at_planner_yaw_after_align_m"] = round(
                at_yaw, 5)
        squeeze = max(float(self.cfg["squeeze_margin_m"]),
                      float(self.cfg["squeeze_frac"]) * width)
        want = width - squeeze
        cmd = float(np.clip(self.cal.command_for(want), 0.0, self.cfg["max_hold"]))
        self.width_cmd = cmd
        wg.update({
            "width_source": "min_over_rotation",
            "object_width_across_jaws_m": round(width, 5),
            "squeeze_m": round(squeeze, 5),
            "squeeze_frac_of_width": round(squeeze / max(width, 1e-9), 4),
            "commanded_separation_m": round(want, 5),
            "aperture_command": round(cmd, 4),
            "aperture_command_rad": round(cmd * np.pi / 4, 5),
            "free_air_separation_at_command_m": round(self.cal.separation_for(cmd), 5),
        })
        self.log(f"[width] sizing the grasp from the narrowest section: "
                 f"{width * 1000:.2f} mm less a {squeeze * 1000:.1f} mm squeeze "
                 f"-> aperture {cmd:.4f}")

    # ------------------------------------------------ grasp station on the object
    def _pad_points(self):
        """How many object points fall between the pads at the current grasp."""
        return self._measure_grasp_width()[1]

    def _apply_variant_grasp(self):
        """rev9's variant grasp, walked back until the pads have a hold."""
        if not self.variant or not self.cfg["pad_contact_guard"]:
            super()._apply_variant_grasp()
            return

        base_pos = self.grasp_pos.copy()
        base_point = self.grasp_point_w.copy()
        base_quat = self.grasp_quat.copy()
        unshifted = self._pad_points()
        floor = max(int(self.cfg["pad_points_min"]),
                    int(self.cfg["pad_points_min_frac"] * unshifted))

        asked = float(self.variant["grasp_shift_m"])
        scale, applied, got = 1.0, asked, None
        for _ in range(5):
            self.grasp_pos, self.grasp_point_w = base_pos.copy(), base_point.copy()
            self.grasp_quat = base_quat.copy()
            self.variant["grasp_shift_m"] = asked * scale
            super()._apply_variant_grasp()
            applied = float(self.report.get("variant_grasp_shift_applied_m", 0.0))
            got = self._pad_points()
            if got >= floor:
                break
            scale *= 0.5
        self.variant["grasp_shift_m"] = asked
        self.report["pad_contact_guard"] = {
            "points_unshifted": unshifted,
            "points_required": floor,
            "points_after_shift": got,
            "shift_requested_m": round(asked, 5),
            "shift_applied_m": round(applied, 5),
            "walked_back": bool(abs(applied - asked) > 1e-6 and scale < 1.0),
        }
        if scale < 1.0:
            self.log(f"[pads] a {asked * 1000:+.1f} mm shift left {got} object "
                     f"points between the pads against the {unshifted} an "
                     f"unshifted grasp gets; walked it back to "
                     f"{applied * 1000:+.1f} mm, where the pads hold {got}")
