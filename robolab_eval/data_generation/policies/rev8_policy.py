"""rev7 scripted policy with a *lateral* escape from a seated insertion.

rev7 gets the spatula seated: the guided slide drops it 110 mm through the
fingertips onto the crock floor and the retention predicate fires. What is left
is getting the hand off it. rev3's let-go backs the jaws straight out along the
gripper's own approach axis, which at an insertion is 0.16 m of pure rise, and a
rise is the one direction a parallel gripper cannot leave a standing tool by:
the blade is now beside the jaws, and lifting drags the finger links up through
it.

The jaws are only closed in one direction. Along the closing axis a finger
blocks the way out at half the aperture, so there is no escape there whatever
the tool is; along the perpendicular horizontal axis -- the open front of the
jaws -- nothing stands between the tool and clear air except the width of the
finger pads. Moving the hand along that axis first slides the tool out from
between the fingers sideways, and only then is a rise harmless.

So the tail becomes RELEASE -> SIDESTEP -> WITHDRAW -> RETREAT, with the
withdraw and retreat rebased onto the sidestep. The distance is sized from the
tool's own measured extent along that axis, not guessed.
"""

import numpy as np

from .rev2_geom import matrix_from_quat
from .rev7_policy import Rev7Policy


class Rev8Policy(Rev7Policy):
    """rev7, with the post-insertion retreat taken sideways before it rises."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        c = dict(
            # Step the hand sideways out of the tool before rising.
            lateral_retreat=False,
            # Nominal sidestep distance. Raised if the tool's own measured extent
            # along the escape axis needs more, capped at ``lateral_offset_max``.
            lateral_offset=0.12,
            lateral_offset_max=0.14,
            # Half-width of a finger pad across the escape axis, i.e. how much of
            # the hand still has to pass the tool after its edge is cleared.
            finger_half_width=0.015,
            lateral_margin=0.020,
            # +1 / -1 forces the side; None picks the one with free workspace.
            lateral_sign=None,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)

    # ------------------------------------------------------------- gripper axes
    def _jaw_axes(self, q_eef):
        """(closing axis, escape axis) in world for an eef orientation.

        The escape axis is the horizontal direction square to both the closing
        axis and the approach axis: the open front of the jaws.
        """
        R = matrix_from_quat(q_eef)
        R_down = matrix_from_quat(self.q_down)
        close_local = R_down.T @ np.array([np.cos(self.finger_axis_angle),
                                           np.sin(self.finger_axis_angle), 0.0])
        close_w = R @ close_local
        approach_w = R @ np.array([0.0, 0.0, 1.0])
        escape = np.cross(approach_w, close_w)
        escape[2] = 0.0
        n = float(np.linalg.norm(escape))
        if n < 1e-6:
            # The escape axis is vertical in this pose, so there is no lateral
            # way out; fall back to horizontalising the closing axis's normal.
            escape = np.array([-close_w[1], close_w[0], 0.0])
            n = float(np.linalg.norm(escape))
        return close_w, escape / n

    # ---------------------------------------------------------- tool geometry
    def _head_points_world(self, q_eef):
        """The tool's head (the end opposite the handle tip), in world."""
        lf = self.limb
        R_obj = matrix_from_quat(q_eef) @ self.R_e_obj
        t = (self.obj_pts - lf["centroid"]) @ lf["axis"]
        s = 1.0 if float(np.dot(lf["free_dir"], lf["axis"])) >= 0 else -1.0
        u = t * s
        cut = u.min() + 0.35 * (u.max() - u.min())
        head = self.obj_pts[u <= cut]
        return (head - lf["grasp_local"]) @ R_obj.T

    def _lateral_offset(self, q_eef, escape, close):
        """Sidestep distance, from how far the head reaches along each axis."""
        head = self._head_points_world(q_eef)
        reach_e = float(np.abs(head @ escape).max())
        reach_c = float(np.abs(head @ close).max())
        need = reach_e + self.cfg["finger_half_width"] + self.cfg["lateral_margin"]
        off = float(min(max(self.cfg["lateral_offset"], need),
                        self.cfg["lateral_offset_max"]))
        self.report.update({
            "lateral_head_reach_along_escape_m": round(reach_e, 5),
            "lateral_head_reach_along_close_m": round(reach_c, 5),
            "lateral_offset_needed_m": round(need, 5),
            "lateral_offset_used_m": round(off, 5),
        })
        self.log(f"[lateral] head reaches {reach_e:.4f} m along the escape axis and "
                 f"{reach_c:.4f} m along the closing axis; clearing it needs "
                 f"{need:.4f} m, stepping {off:.4f} m")
        return off

    # ------------------------------------------------------------- side choice
    def _other_bodies_xy(self):
        """Horizontal positions of everything that is not the tool or the target."""
        out = {}
        for name in self.env.scene.rigid_objects.keys():
            if name in (self.obj_name, self.container_name, "table"):
                continue
            out[name] = self._pose(name)[0][:2]
        return out

    def _pick_side(self, p_des, escape, off):
        """Which way along the escape axis, if the caller has not said."""
        forced = self.cfg["lateral_sign"]
        if forced is not None:
            self.report["lateral_sign"] = int(np.sign(forced))
            self.log(f"[lateral] side forced to {int(np.sign(forced)):+d}")
            return float(np.sign(forced))

        others = self._other_bodies_xy()
        r_now = float(np.linalg.norm(p_des[:2]))
        best, scores = None, {}
        for sgn in (1.0, -1.0):
            xy = (p_des + sgn * off * escape)[:2]
            clear = min((float(np.linalg.norm(xy - o)) for o in others.values()),
                        default=1.0)
            # Prefer not to ask the arm for more reach than it already has.
            score = clear - 0.5 * max(0.0, float(np.linalg.norm(xy)) - r_now)
            scores[int(sgn)] = {"clearance_m": round(clear, 4),
                                "reach_delta_m": round(float(np.linalg.norm(xy)) - r_now, 4),
                                "score": round(score, 4)}
            if best is None or score > best[0]:
                best = (score, sgn)
        self.report["lateral_side_scores"] = scores
        self.report["lateral_sign"] = int(best[1])
        self.log(f"[lateral] side scores {scores}; going {int(best[1]):+d} along "
                 f"{np.round(escape, 3).tolist()}")
        return best[1]

    # ---------------------------------------------------------------- planning
    def _letgo_stages(self, p_des, q_des):
        """rev7's let-go, taken sideways out of the tool before it rises."""
        stages = super()._letgo_stages(p_des, q_des)
        if not self.cfg["lateral_retreat"]:
            return stages

        close, escape = self._jaw_axes(q_des)
        off = self._lateral_offset(q_des, escape, close)
        step = self._pick_side(p_des, escape, off) * off * escape
        self.report["lateral_step_world_m"] = np.round(step, 5).tolist()

        out = []
        for st in stages:
            if st["name"] in ("WITHDRAW", "RETREAT"):
                st = dict(st, pos=st["pos"] + step)
            out.append(st)
        j = next(i for i, s in enumerate(out) if s["name"] == "RELEASE")
        out.insert(j + 1, self._stage("SIDESTEP", p_des + step, q_des, out[j]["grip"],
                                      speed=self.cfg["fine_speed"], timeout=320))
        self.log(f"[lateral] sidestepping {np.round(step, 4).tolist()} out of the "
                 "tool before the withdraw, which now rises from there")
        return out

    # ---------------------------------------------------------------- hooking
    def _advance(self, st, p_now, q_now, settled):
        before = self.i
        super()._advance(st, p_now, q_now, settled)
        if self.i == before or st["name"] not in ("SIDESTEP", "WITHDRAW", "RETREAT"):
            return
        o_p, _ = self._pose(self.obj_name)
        low = self._object_low_world()
        self.report[f"object_pos_after_{st['name'].lower()}"] = np.round(o_p, 5).tolist()
        self.report[f"object_low_world_after_{st['name'].lower()}"] = round(low, 5)
        self.log(f"[retreat] after {st['name']}: tool at {np.round(o_p, 4).tolist()}, "
                 f"lowest point z={low:.4f}")
