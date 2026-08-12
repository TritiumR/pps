"""rev2 scripted policy plus a measured mid-flight correction of the insertion.

Two changes, both of the same kind -- measure again instead of trusting an
earlier measurement:

- **Re-measure after the wrist turns.** ``rev2`` measures the object-in-gripper
  transform once, right after the fingers close, and uses it to aim the handle
  tip at the holder. But turning a utensil from flat to upright swings its whole
  weight about the pinch and it rotates *inside* the fingers on the way; by the
  time it hangs vertically the transform measured at CLOSE no longer says where
  the tip is. rev2 aimed with the stale transform and set the spatula down
  beside the holder, where it toppled. Here the transform is measured a second
  time once the tool is upright and clear of the rim, and the remaining stages
  are re-planned from it.

- **Depth from the cavity floor.** rev2 fixed the fingertips a set distance
  *above the rim*, which left the tip only 7 cm into an 18.5 cm holder. The tip
  is now aimed at a clearance above the cavity *floor*, and the fingertips are
  allowed a (small, configurable) descent past the rim to get there.

Both corrections are checked, not assumed: the handle tip's world position is
recomputed from the object's live pose before the fingers open and tested
against the mouth rectangle, and the result is reported.
"""

import numpy as np

from .rev2_geom import align_rotation, cavity, matrix_from_quat, quat_from_matrix
from .rev2_policy import ScriptedPolicy, _np, _qmul


class Rev3Policy(ScriptedPolicy):
    """Scripted pick-and-insert with a re-measured, verified insertion."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        c = dict(
            # Tip clearance above the cavity floor. The tip is aimed here unless
            # the fingertips would have to go deeper past the rim than allowed.
            insert_floor_clear=0.035,
            # How far the fingertips may descend past the rim (negative = below).
            # The mouth is much wider than the closed gripper, so a small dip is
            # free and buys the tip the same distance again.
            min_rim_clear=-0.015,
            # Height above the insert pose at which the corrected pose is taken up.
            realign_dz=0.13,
            realign=True,
            # Steps held still after the wrist turns, before the tool is
            # re-measured. Turning the tool upright leaves it swinging from the
            # pinch like a pendulum; measured mid-arc it reads as pointing
            # almost anywhere, and the correction computed from that reading is
            # worse than no correction at all. The measurement is only
            # meaningful once the swing has damped out.
            settle_hold=140,
            # Bounds on a correction worth applying. A tool that has pivoted far
            # inside the pinch needs a wrist pose the arm cannot reach, and
            # driving at it is worse than descending with the original plan; past
            # these the correction is reported and abandoned, not attempted.
            # Only the *lateral* shift is bounded: the vertical part is just the
            # standing height above the insert pose and is always ~realign_dz.
            realign_max_xy=0.06,
            realign_max_tilt_deg=30.0,
            # Distance the jaws are backed off the tool, along the gripper's own
            # approach axis, before the arm rises. Rising straight up from an
            # insertion lifts the tool with it: the open jaws sit *below* its
            # head and catch it on the way out.
            withdraw=0.16,
            # Close the fingers across the narrowest *horizontal* direction of
            # the material at the grasp band, instead of assuming that is square
            # to the limb. On the drill the two differ and it matters: square to
            # the limb the fingers pinch the 46 mm grip and a 1.5 kg tool levers
            # straight back out of them, whereas along the limb they close on the
            # motor body above the grip and the battery below it, which block the
            # tool in both directions -- how a hand holds a drill.
            grasp_thin_axis=False,
            grasp_band_half=None,  # None -> max(0.02, 0.08 * object span)
        )
        if cfg:
            c.update(cfg)
        super().__init__(env, obj_name, container_name, mode=mode, cfg=c, log=log)
        self._realigned = False
        self._cav_cache = {}
        self._z_at_close = None

    # ------------------------------------------------------------- geometry
    def _container_frame(self):
        """rev2's container frame, plus the mouth rectangle in world."""
        cf = super()._container_frame()
        if self.container_name not in self._cav_cache:
            self._cav_cache[self.container_name] = cavity(
                self._body_points(self.container_name))
        cav = self._cav_cache[self.container_name]
        if cav is not None:
            _, cq = self._pose(self.container_name)
            R = matrix_from_quat(cq)
            cf["mouth_axes_w"] = [R @ np.array([1.0, 0.0, 0.0]),
                                  R @ np.array([0.0, 1.0, 0.0])]
            cf["mouth_half"] = 0.5 * np.asarray(cav["mouth_size"], dtype=float)
        else:
            cf["mouth_axes_w"] = None
            cf["mouth_half"] = None
        self._cf = cf
        return cf

    def _tip_z(self, cf, R_obj_des, R_eef_des, free_end):
        """Height for the handle tip, driven by the cavity floor.

        ``gap`` is how far the fingertips ride above the tip in this pose; it is
        what limits the depth, since the fingers must not be driven into the rim.
        """
        off_hand = (R_eef_des @ self.p_e_obj)[2]
        off_free = (R_obj_des @ free_end)[2]
        off_tip = (R_eef_des @ np.array([0.0, 0.0, self.finger_drop]))[2]
        gap = off_tip - off_hand - off_free

        want = cf["floor_z"] + self.cfg["insert_floor_clear"]
        allowed = cf["rim_z"] + self.cfg["min_rim_clear"] - gap
        tip_z = max(want, allowed)
        tip_z = min(tip_z, cf["rim_z"] - self.cfg["min_insert"])
        tip_z = max(tip_z, cf["floor_z"] + 0.012)
        self.log(f"[depth] gap(fingertip above tip)={gap:.4f} "
                 f"want={want:.4f} floor-limited={allowed:.4f} -> tip_z={tip_z:.4f} "
                 f"(depth below rim {cf['rim_z'] - tip_z:.4f}, "
                 f"{tip_z - cf['floor_z']:.4f} above floor)")
        return tip_z

    def _insert_pose(self, cf, q_ref):
        """Eef pose putting the handle tip down the cavity, nearest ``q_ref``."""
        free_dir = self.limb["free_dir"]
        free_end = self.limb["free_end_local"]
        down = -cf["axis_w"]

        forced = self.cfg["insert_spin_deg"]
        spins = ([np.radians(forced)] if forced is not None
                 else np.linspace(-np.pi, np.pi, 145))
        best = None
        for spin in spins:
            R_obj_des = align_rotation(free_dir, down, spin_axis=down, spin=spin)
            R_eef_des = R_obj_des @ self.R_e_obj.T
            q = quat_from_matrix(R_eef_des)
            travel = 2.0 * np.arccos(min(1.0, abs(float(np.dot(q, q_ref)))))
            if best is None or travel < best[0]:
                best = (travel, spin, R_obj_des, R_eef_des)
        travel, spin, R_obj_des, R_eef_des = best
        self.log(f"[spin] chose {spin:+.3f} rad -> wrist travel {np.degrees(travel):.1f} deg")

        tip_z = self._tip_z(cf, R_obj_des, R_eef_des, free_end)
        tip_target = np.array([cf["centre_w"][0], cf["centre_w"][1], tip_z])
        p_eef_des = tip_target - R_eef_des @ self.p_e_obj - R_obj_des @ free_end
        return p_eef_des, R_eef_des, R_obj_des, tip_target, spin

    def _insert_target(self):
        """Same contract as rev2, with the floor-driven depth."""
        cf = self._container_frame()
        p_eef_des, R_eef_des, R_obj_des, tip_target, spin = self._insert_pose(
            cf, self.grasp_quat)
        q_eef_des = quat_from_matrix(R_eef_des)
        fingertip_z = p_eef_des[2] + (R_eef_des @ np.array([0, 0, self.finger_drop]))[2]

        self.report.update({
            "container_rim_z": round(cf["rim_z"], 5),
            "container_floor_z": round(cf["floor_z"], 5),
            "container_mouth_centre_world": np.round(cf["centre_w"], 5).tolist(),
            "container_insertion_axis_world": np.round(cf["axis_w"], 5).tolist(),
            "container_mouth_size": (None if cf["mouth_size"] is None
                                     else np.round(cf["mouth_size"], 4).tolist()),
            "desired_object_quat_world": np.round(quat_from_matrix(R_obj_des), 5).tolist(),
            "insert_spin_rad": round(float(spin), 4),
            "tip_target_world": np.round(tip_target, 5).tolist(),
            "tip_depth_below_rim_m": round(float(cf["rim_z"] - tip_target[2]), 5),
            "tip_above_floor_m": round(float(tip_target[2] - cf["floor_z"]), 5),
            "fingertip_z_at_insert": round(float(fingertip_z), 5),
            "fingertip_clearance_above_rim_m": round(float(fingertip_z - cf["rim_z"]), 5),
        })
        self._log_reorient(q_eef_des)
        self.log(f"[target] insert: tip->{np.round(tip_target, 4).tolist()} "
                 f"eef->{np.round(p_eef_des, 4).tolist()}")
        return p_eef_des, q_eef_des

    # --------------------------------------------------------------- grasp
    def _plan_grasp(self):
        """rev2's grasp point, optionally re-aimed across the thinnest direction."""
        super()._plan_grasp()
        if not self.cfg["grasp_thin_axis"]:
            return

        lf = self.limb
        t = (self.obj_pts - lf["centroid"]) @ lf["axis"]
        span = lf["obj_t_range"][1] - lf["obj_t_range"][0]
        half = self.cfg["grasp_band_half"] or max(0.02, 0.08 * span)
        band = np.abs(t - lf["grasp_t"]) <= half
        if int(band.sum()) < 8:
            self.log("[thin] grasp band too sparse; keeping the limb-square yaw")
            return

        _, obj_q = self._pose(self.obj_name)
        xy = (self.obj_pts[band] @ matrix_from_quat(obj_q).T)[:, :2]
        xy = xy - xy.mean(axis=0)
        _, _, v = np.linalg.svd(xy, full_matrices=False)
        thin, wide = v[1], v[0]
        width = float((xy @ thin).max() - (xy @ thin).min())
        length = float((xy @ wide).max() - (xy @ wide).min())

        yaw = float(np.arctan2(thin[1], thin[0])) - self.finger_axis_angle
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if self.cfg["grasp_yaw_flip"]:
            yaw += np.pi
        self.grasp_yaw = yaw
        rz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.grasp_quat = _qmul(rz, self.q_down)

        self.report["grasp_thin_axis_width_m"] = round(width, 4)
        self.report["grasp_thin_axis_length_m"] = round(length, 4)
        self.report["grasp_quat_eef"] = np.round(self.grasp_quat, 5).tolist()
        self.report["grasp_yaw_rad"] = round(yaw, 5)
        self.log(f"[thin] band footprint {width:.4f} across x {length:.4f} along; "
                 f"closing across the {width:.4f} m direction, yaw={yaw:+.3f}")

    # ------------------------------------------------------------- planning
    def _letgo_stages(self, p_des, q_des):
        """Open, back the jaws off the standing tool, then rise clear of it."""
        back = matrix_from_quat(q_des) @ np.array([0.0, 0.0, self.cfg["withdraw"]])
        withdraw = p_des - back
        rise = withdraw + np.array([0.0, 0.0, 0.12])
        return [
            self._stage("RELEASE", p_des, q_des, 0.0, hold=self.cfg["open_hold"],
                        timeout=self.cfg["open_hold"] + 5),
            self._stage("WITHDRAW", withdraw, q_des, 0.0, speed=self.cfg["fine_speed"],
                        timeout=320),
            self._stage("RETREAT", rise, q_des, 0.0, speed=self.cfg["fine_speed"]),
        ]

    def _plan_tail(self):
        """rev2's tail, with a settle after the wrist turn and a clear withdrawal."""
        super()._plan_tail()
        if self.mode != "insert" or not self.cfg["realign"]:
            return
        k = next(i for i, s in enumerate(self.stages) if s["name"] == "REORIENT")
        turn = self.stages[k]
        self.stages.insert(k + 1, self._stage(
            "SETTLE", turn["pos"], turn["quat"], 1.0, hold=self.cfg["settle_hold"],
            timeout=self.cfg["settle_hold"] + 5))
        j = next(i for i, s in enumerate(self.stages) if s["name"] == "RELEASE")
        ins = self.stages[j - 1]
        self.stages = self.stages[:j] + self._letgo_stages(ins["pos"], ins["quat"])

    # ----------------------------------------------------------- correction
    def _tool_tilt_deg(self):
        """Angle between the tool's long axis and the container's interior axis."""
        _, o_q = self._pose(self.obj_name)
        axis_w = matrix_from_quat(o_q) @ self.limb["free_dir"]
        down = -self._cf["axis_w"]
        return float(np.degrees(np.arccos(np.clip(axis_w @ down, -1.0, 1.0))))

    def _free_end_world(self):
        """Handle tip in world, from the object's live pose (no hand transform)."""
        o_p, o_q = self._pose(self.obj_name)
        return o_p + matrix_from_quat(o_q) @ self.limb["free_end_local"]

    def _tip_in_mouth(self, tip_w):
        """Tip offset from the mouth centre, along the mouth's own axes."""
        cf = self._cf
        if cf.get("mouth_axes_w") is None:
            return None, None
        d = tip_w - np.array([cf["centre_w"][0], cf["centre_w"][1], tip_w[2]])
        off = np.array([float(d @ cf["mouth_axes_w"][0]), float(d @ cf["mouth_axes_w"][1])])
        return off, bool(np.all(np.abs(off) < cf["mouth_half"]))

    def _realign(self):
        """Re-measure the upright tool and re-plan the descent from it."""
        p_now, q_now = self._eef_pose()
        tip_before = self._free_end_world()
        off_before, in_before = self._tip_in_mouth(tip_before)

        self._measure_in_hand()
        cf = self._container_frame()
        tilt = self._tool_tilt_deg()
        self.report["realign_tool_tilt_from_axis_deg"] = round(tilt, 2)
        self.log(f"[realign] settled tool axis is {tilt:.1f} deg off the holder axis "
                 f"({'handle down' if tilt < 90 else 'HANDLE UP -- wrong way round'})")
        p_des, R_eef_des, R_obj_des, tip_target, spin = self._insert_pose(cf, q_now)
        q_des = quat_from_matrix(R_eef_des)

        shift = p_des - p_now
        self.log(f"[realign] tip was {np.round(tip_before, 4).tolist()} "
                 f"(mouth offset {None if off_before is None else np.round(off_before, 4).tolist()}, "
                 f"inside={in_before}); correcting eef by "
                 f"{np.round(shift, 4).tolist()} (|xy|={np.linalg.norm(shift[:2]):.4f} m)")

        self.report.update({
            "realign_tip_world_before": np.round(tip_before, 5).tolist(),
            "realign_tip_mouth_offset_before": (None if off_before is None
                                                else np.round(off_before, 5).tolist()),
            "realign_tip_inside_mouth_before": in_before,
            "realign_eef_correction_m": np.round(shift, 5).tolist(),
            "realign_eef_correction_xy_m": round(float(np.linalg.norm(shift[:2])), 5),
            "object_in_gripper_pos_after_reorient": self.report["object_in_gripper_pos"],
            "object_in_gripper_quat_after_reorient": self.report["object_in_gripper_quat"],
            "tip_target_world": np.round(tip_target, 5).tolist(),
            "tip_depth_below_rim_m": round(float(cf["rim_z"] - tip_target[2]), 5),
            "tip_above_floor_m": round(float(tip_target[2] - cf["floor_z"]), 5),
            "desired_object_quat_world": np.round(quat_from_matrix(R_obj_des), 5).tolist(),
            "insert_spin_rad": round(float(spin), 4),
        })

        self._realigned = True
        lateral = float(np.linalg.norm(shift[:2]))
        if lateral > self.cfg["realign_max_xy"] or tilt > self.cfg["realign_max_tilt_deg"]:
            self.report["realign_applied"] = False
            self.log(f"[realign] lateral {lateral:.3f} m / tilt {tilt:.1f} deg is outside "
                     f"the {self.cfg['realign_max_xy']:.3f} m / "
                     f"{self.cfg['realign_max_tilt_deg']:.0f} deg bound -- the tool has "
                     "pivoted too far in the pinch to be re-aimed; keeping the "
                     "original descent")
            return
        self.report["realign_applied"] = True

        over = p_des.copy()
        over[2] = p_des[2] + self.cfg["realign_dz"]
        self.stages = self.stages[:self.i] + [
            self._stage("REALIGN", over, q_des, 1.0, speed=self.cfg["fine_speed"],
                        timeout=420),
            self._stage("INSERT", p_des, q_des, 1.0, speed=self.cfg["fine_speed"],
                        tol=self.cfg["pos_tol_fine"], timeout=420),
        ] + self._letgo_stages(p_des, q_des)

    def _verify_before_release(self):
        """Where the handle tip actually is, just before the fingers open."""
        tip = self._free_end_world()
        off, inside = self._tip_in_mouth(tip)
        cf = self._cf
        self.report.update({
            "verify_tip_world_at_release": np.round(tip, 5).tolist(),
            "verify_tip_mouth_offset": None if off is None else np.round(off, 5).tolist(),
            "verify_tip_inside_mouth": inside,
            "verify_tip_below_rim_m": round(float(cf["rim_z"] - tip[2]), 5),
            "verify_tip_above_floor_m": round(float(tip[2] - cf["floor_z"]), 5),
        })
        self.log(f"[verify] tip at release {np.round(tip, 4).tolist()}: "
                 f"mouth offset {None if off is None else np.round(off, 4).tolist()} "
                 f"(half {None if cf['mouth_half'] is None else np.round(cf['mouth_half'], 4).tolist()}) "
                 f"inside={inside}  below_rim={cf['rim_z'] - tip[2]:.4f}  "
                 f"above_floor={tip[2] - cf['floor_z']:.4f}")

    # ------------------------------------------------------------ grip check
    def _check_grip(self, when):
        """Contact force between object and gripper, and whether it is carried."""
        try:
            f = _np(self.world.get_contact_force(self.obj_name, "gripper", env_id=0))
            mag = float(np.linalg.norm(f))
        except Exception as exc:  # noqa: BLE001
            self.log(f"[grip] contact force unavailable: {exc}")
            return
        fj = float(self.robot.data.joint_pos[0, self.finger_id])
        o_p, _ = self._pose(self.obj_name)
        carried = o_p[2] - self._z_at_close if self._z_at_close is not None else float("nan")
        self.report[f"grip_force_N_at_{when}"] = round(mag, 3)
        self.report[f"finger_joint_at_{when}"] = round(fj, 4)
        self.report[f"object_rise_since_close_m_at_{when}"] = round(carried, 5)
        self.log(f"[grip] {when}: |contact force|={mag:.2f} N  finger_joint={fj:.4f} rad  "
                 f"object risen {carried * 1000:+.1f} mm since close")

    # -------------------------------------------------------------- hooking
    def _advance(self, st, p_now, q_now, settled):
        before = self.i
        super()._advance(st, p_now, q_now, settled)
        if self.i == before:
            return
        if st["name"] == "CLOSE":
            self._z_at_close = self._pose(self.obj_name)[0][2]
            self._check_grip("close")
        elif st["name"] == "TAKELOAD":
            self._check_grip("takeload")
        elif st["name"] == "LIFT":
            self._check_grip("lift")
        if (st["name"] == "SETTLE" and self.mode == "insert"
                and self.cfg["realign"] and not self._realigned):
            self._realign()
        elif st["name"] == "INSERT" and self.mode == "insert":
            self._verify_before_release()
