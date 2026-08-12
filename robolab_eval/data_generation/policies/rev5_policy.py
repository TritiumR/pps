"""rev3 scripted policy with the insertion released at a *partial* depth.

rev3 drives the handle tip as far down the holder as the fingers allow, so the
gripper is still rigidly holding the utensil while the utensil is already deep
inside the crock. Two things follow from that, both seen in the rev4 runs: the
held tool bottoms out on the cavity floor and the arm keeps pushing (INSERT ran
its full 420-step timeout), and the jaws end up at or below the rim, where
backing them out rakes the holder -- one run toppled it outright.

None of that depth has to be commanded. These crocks capture a utensil as soon
as its end is inside the mouth and below the rim: once the jaws let go, the tool
slides handle-first to the floor on its own. So the arm's job is only to get the
tip *captured*, then open and stay out of the way.

``partial_depth`` states that directly -- how far below the rim the handle tip
should sit at release -- instead of expressing depth as a by-product of how
close the fingers may come to the rim. It is a property of the holder, not of
the tool, so the same number transfers between utensils whose grasps sit at
very different heights above their tips. Two floors still apply: the tip may not
be driven into the cavity floor, and the fingertips must stay ``min_rim_clear``
above the rim, so the gripper never enters the mouth at all.
"""

import numpy as np

from .rev2_geom import matrix_from_quat
from .rev4_policy import Rev4Policy


class Rev5Policy(Rev4Policy):
    """Scripted pick-and-place/insert: rev4's re-measured carry, plus grasp
    siting and partial-depth release.

    Inherits Rev4Policy so the place branch keeps its post-lift re-measurement;
    the insert-only knobs below are inert in place mode and vice versa.
    """

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        c = dict(
            # Depth below the rim for the handle tip at release. None -> rev3's
            # "as deep as the fingers allow" behaviour.
            partial_depth=None,
            # Where to hold the tool, stated as a distance from the handle's free
            # end instead of as a bias along the limb.
            #
            # The release failure that survives partial insertion is the
            # gripper's own geometry: the Robotiq's finger links reach
            # ``finger_drop`` (0.150 m) back from the fingertips, and whatever
            # part of the tool lies inside that span is swept by the links as
            # they open. Holding a spatula mid-handle puts the blade's root
            # 0.104 m above the jaws -- squarely inside the linkage -- and
            # opening the jaws flings the tool even with the arm standing still.
            # The blade only clears if the grasp is within
            # (blade_root_to_tip - finger_drop) of the free end, which is a
            # distance from the *tip*, not a fraction of the limb.
            grasp_from_free_end=None,
            # Guided-slide release. Opening the jaws fully while the tool is
            # still held sweeps the finger links through whatever part of the
            # tool lies within finger_drop of the fingertips -- for a spatula
            # held mid-handle that is the blade, and it is flung even with the
            # arm standing still.
            #
            # A partial aperture avoids it. Opened just past the handle's own
            # width the tool is no longer pinched but is still loosely captured,
            # so it slides down through the fingertips under gravity and seats
            # in the holder. That descent carries the blade *below* the jaws,
            # and only then is opening fully safe. Expressed as the policy's
            # normalised grip command (1.0 = closed, 0.0 = open).
            slide_grip=None,
            slide_hold=150,
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)

    def _letgo_stages(self, p_des, q_des):
        """rev3's let-go, optionally preceded by a guided slide."""
        stages = super()._letgo_stages(p_des, q_des)
        if self.cfg["slide_grip"] is None:
            return stages
        self.report["slide_grip"] = round(float(self.cfg["slide_grip"]), 4)
        self.report["slide_hold_steps"] = int(self.cfg["slide_hold"])
        self.log(f"[slide] releasing in two steps: hold station at grip="
                 f"{self.cfg['slide_grip']:.2f} for {self.cfg['slide_hold']} steps so the "
                 "tool slides down through the fingertips and seats, then open fully")
        slide = self._stage("SLIDE", p_des, q_des, self.cfg["slide_grip"],
                            hold=self.cfg["slide_hold"],
                            timeout=self.cfg["slide_hold"] + 5)
        return [slide] + stages

    def _advance(self, st, p_now, q_now, settled):
        before = self.i
        super()._advance(st, p_now, q_now, settled)
        if self.i != before and st["name"] == "SLIDE":
            o_p, _ = self._pose(self.obj_name)
            low = self._object_low_world()
            self.report["slide_object_low_world_after"] = round(low, 5)
            self.report["slide_object_pos_after"] = np.round(o_p, 5).tolist()
            self.log(f"[slide] after the slide the tool's lowest point is at "
                     f"z={low:.4f} (was aimed {self.cfg['partial_depth']:.3f} below the "
                     "rim while held); opening fully now")

    def _plan_grasp(self):
        """rev3's grasp, optionally re-sited a stated distance from the free end."""
        super()._plan_grasp()
        d = self.cfg["grasp_from_free_end"]
        if d is None:
            return
        lf = self.limb
        if lf["free_end_local"] is None:
            self.log("[hold] this limb has no free end; keeping the planner's grasp")
            return

        axis = lf["axis"]
        t = (self.obj_pts - lf["centroid"]) @ axis
        t_free = float((lf["free_end_local"] - lf["centroid"]) @ axis)
        # sign of "toward the free end" along the axis
        s = 1.0 if float(np.dot(lf["free_dir"], axis)) >= 0 else -1.0
        t_want = float(np.clip(t_free - d * s, min(lf["limb_t"]), max(lf["limb_t"])))

        # Average a band about that station so the grasp lands on the limb's
        # centreline rather than beside it, exactly as limb_frame does.
        half = max(0.010, 0.06 * lf["limb_len"])
        band = np.abs(t - t_want) <= half
        if int(band.sum()) < 8:
            self.log(f"[hold] band about {d:.3f} m from the free end is too sparse; "
                     "keeping the planner's grasp")
            return
        grasp_local = self.obj_pts[band].mean(axis=0)

        obj_p, obj_q = self._pose(self.obj_name)
        grasp_w = obj_p + matrix_from_quat(obj_q) @ grasp_local
        self.grasp_pos = grasp_w + np.array(
            [0.0, 0.0, self.finger_drop - self.cfg["grasp_lower"]])
        self.grasp_point_w = grasp_w
        lf["grasp_local"] = grasp_local
        lf["grasp_t"] = t_want

        reach = float(np.linalg.norm(lf["free_end_local"] - grasp_local))
        self.report.update({
            "grasp_from_free_end_requested_m": round(float(d), 5),
            "grasp_from_free_end_achieved_m": round(reach, 5),
            "grasp_point_object_frame": np.round(grasp_local, 5).tolist(),
            "grasp_point_world": np.round(grasp_w, 5).tolist(),
            "grasp_to_free_end_m": round(reach, 5),
        })
        self.log(f"[hold] re-sited the grasp to {reach:.4f} m from the free end "
                 f"(requested {d:.4f}): obj={np.round(grasp_local, 4).tolist()} "
                 f"world={np.round(grasp_w, 4).tolist()}")

    def _tip_z(self, cf, R_obj_des, R_eef_des, free_end):
        """Height for the handle tip: captured by the mouth, jaws clear of it."""
        if self.cfg["partial_depth"] is None:
            return super()._tip_z(cf, R_obj_des, R_eef_des, free_end)

        off_hand = (R_eef_des @ self.p_e_obj)[2]
        off_free = (R_obj_des @ free_end)[2]
        off_tip = (R_eef_des @ np.array([0.0, 0.0, self.finger_drop]))[2]
        gap = off_tip - off_hand - off_free

        want = cf["rim_z"] - self.cfg["partial_depth"]
        floor_limit = cf["floor_z"] + self.cfg["insert_floor_clear"]
        jaw_limit = cf["rim_z"] + self.cfg["min_rim_clear"] - gap
        tip_z = max(want, floor_limit, jaw_limit)

        binding = ("requested partial depth" if tip_z <= want + 1e-9 else
                   ("cavity floor" if tip_z <= floor_limit + 1e-9 else "jaw clearance"))
        fingertip_z = tip_z + gap
        self.report.update({
            "partial_depth_requested_m": round(float(self.cfg["partial_depth"]), 5),
            "partial_depth_binding_limit": binding,
            "partial_tip_depth_below_rim_m": round(float(cf["rim_z"] - tip_z), 5),
            "partial_fingertip_above_rim_m": round(float(fingertip_z - cf["rim_z"]), 5),
        })
        self.log(f"[partial] gap(fingertip above tip)={gap:.4f} "
                 f"want={want:.4f} floor_limit={floor_limit:.4f} "
                 f"jaw_limit={jaw_limit:.4f} -> tip_z={tip_z:.4f} ({binding}); "
                 f"tip {cf['rim_z'] - tip_z:.4f} below the rim, fingertips "
                 f"{fingertip_z - cf['rim_z']:+.4f} above it")
        return tip_z
