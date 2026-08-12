"""Deterministic scripted manipulation policy over object-relative transforms.

State machine: PREGRASP -> DESCEND -> CLOSE -> LIFT -> REORIENT -> PRETARGET ->
INSERT/PLACE -> RELEASE -> RETREAT -> DONE.

Nothing is expressed in world coordinates. The grasp is a pose in the object's
body frame (found from the mesh by ``rev2_geom.limb_frame``), the target is a
pose in the container's body frame (its cavity, from ``rev2_geom.cavity``), and
the two are tied together by the object-in-gripper transform *measured* after
the fingers close -- so the policy reasons about where the tool actually ended
up in the hand rather than assuming the grasp was perfect.

Frame conventions follow p1b_expert (proven on this robot): the IK tracks body
``base_link``, ``eef = base_link (X) EEF_OFFSET_ROT`` with zero offset
translation, so positions pass through and only orientation is un-offset. The
fingertips sit ``finger_drop`` from base_link along the eef +z (approach) axis.
"""

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from robolab.core.utils import usd_utils
from robolab.core.world.world_state import get_world
from robolab.robots.droid import EEF_OFFSET_ROT

from .rev2_geom import align_rotation, cavity, limb_frame, matrix_from_quat, quat_from_matrix

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _np(x):
    """Tensor / list / array -> float numpy array (WorldState returns all three)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=float)


def slerp_step(q_from, q_to, max_ang):
    """Rotate q_from toward q_to by at most ``max_ang`` radians."""
    q_to = np.asarray(q_to, dtype=float)
    d = float(np.dot(q_from, q_to))
    if d < 0.0:
        q_to, d = -q_to, -d
    d = min(max(d, -1.0), 1.0)
    theta = np.arccos(d)
    if 2.0 * theta <= max_ang or theta < 1e-6:
        return q_to
    frac = max_ang / (2.0 * theta)
    s = np.sin(theta)
    q = (np.sin((1 - frac) * theta) * q_from + np.sin(frac * theta) * q_to) / s
    return q / np.linalg.norm(q)


class ScriptedPolicy:
    """Single-env scripted pick-and-(insert|place)."""

    def __init__(self, env, obj_name, container_name, mode="place", cfg=None, log=print):
        assert mode in ("place", "insert")
        self.env = env
        self.device = env.device
        self.mode = mode
        self.obj_name = obj_name
        self.container_name = container_name
        self.world = get_world(env)
        self.robot = env.scene["robot"]
        self.log = log

        c = dict(
            approach_h=0.14,       # pregrasp height above the grasp point
            grasp_lower=0.004,     # extra descent past the nominal grasp point
            lift_h=0.34,           # fingertip height after lifting
            pretarget_dz=0.13,     # height above the insert/place pose to line up
            release_clear=0.04,    # object clearance above the container floor (place)
            rim_clear=0.030,       # fingertip clearance above the rim (insert)
            min_insert=0.030,      # tip must end at least this far below the rim
            pos_tol=0.008,
            pos_tol_fine=0.003,    # grasp/insert must converge, not just get close:
                                   # an 8 mm shortfall on DESCEND grips the very top
                                   # of the object and the grasp shakes loose
            ang_tol=0.05,
            ki=0.35,
            int_clip=0.05,
            move_speed=0.010,      # m per control step, transit
            fine_speed=0.005,      # m per control step, descend/insert
            ang_speed=0.06,        # rad per control step
            close_hold=40,         # steps held closed before lifting
            open_hold=25,
            lift_speed=0.003,      # m per control step; slower than transit so a
                                   # freshly closed grasp is not shaken loose
            # "com" keeps the lever arm short for a freely carried object;
            # "middle" leaves the far half of the handle free for insertion.
            grasp_bias=None,       # None -> per-mode default
            reorient_dz=0.19,      # base_link height above the insert pose at which
                                   # the tool is rotated upright, clear of the rim
            insert_spin_deg=None,  # force the spin about the container axis
            # A parallel gripper grasps the same line at yaw and yaw+pi. The two
            # are identical for picking but not for what follows: they send the
            # wrist opposite ways during the reorientation, and one of them runs
            # panda_joint6 into its hard +3.75 limit.
            grasp_yaw_flip=False,
            takeload_h=0.025,      # first, tiny lift
            takeload_hold=55,      # held there so the payload rotates into a
                                   # hanging equilibrium while the table still
                                   # carries most of it, instead of being torn
                                   # out of the pinch by the full moment at once
        )
        if cfg:
            c.update(cfg)
        if c["grasp_bias"] is None:
            c["grasp_bias"] = "middle" if mode == "insert" else "com"
        self.cfg = c

        names = self.robot.data.joint_names
        self.arm_ids = [names.index(n) for n in ARM_JOINT_NAMES]
        self.finger_id = names.index("finger_joint")
        body_names = self.robot.data.body_names
        self.base_id = body_names.index("base_link")
        self.jacobi_body_idx = self.base_id - 1 if self.robot.is_fixed_base else self.base_id
        self.jacobi_joint_ids = (self.arm_ids if self.robot.is_fixed_base
                                 else [i + 6 for i in self.arm_ids])
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=self.device)
        self.eef_offset = torch.tensor(EEF_OFFSET_ROT, device=self.device, dtype=torch.float32)
        self.eef_offset_inv = math_utils.quat_inv(self.eef_offset.unsqueeze(0)).squeeze(0)
        self.calibrated = False
        self.stage_name = "INIT"
        self.report = {}

    # ------------------------------------------------------------ kinematics
    def _base_link_root(self):
        rp, rq = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        pw = self.robot.data.body_pos_w[:, self.base_id, :]
        qw = self.robot.data.body_quat_w[:, self.base_id, :]
        return math_utils.subtract_frame_transforms(rp, rq, pw, qw)

    def _jacobian_root(self):
        jac = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids]
        rot = math_utils.matrix_from_quat(math_utils.quat_inv(self.robot.data.root_quat_w))
        jac = jac.clone()
        jac[:, :3, :] = torch.bmm(rot, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(rot, jac[:, 3:, :])
        return jac

    def _eef_pose(self):
        """(pos[3], quat[4]) of the eef frame in the robot root frame, numpy."""
        p, q_base = self._base_link_root()
        q_eef = math_utils.quat_mul(q_base, self.eef_offset.unsqueeze(0))
        return (p[0].detach().cpu().numpy().astype(float),
                q_eef[0].detach().cpu().numpy().astype(float))

    # ---------------------------------------------------------- calibration
    @staticmethod
    def _mesh_points(prim, ref_prim, cache):
        inv_ref = cache.GetLocalToWorldTransform(ref_prim).GetInverse()
        pts = []
        for p in Usd.PrimRange(prim):
            if not p.IsA(UsdGeom.Mesh):
                continue
            raw = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not raw:
                continue
            to_ref = cache.GetLocalToWorldTransform(p) * inv_ref
            for v in raw:
                q = to_ref.Transform(Gf.Vec3d(v[0], v[1], v[2]))
                pts.append((q[0], q[1], q[2]))
        return np.asarray(pts, dtype=float)

    @staticmethod
    def _find_prim(root, name):
        for p in Usd.PrimRange(root):
            if p.GetName() == name:
                return p
        return None

    def _calibrate(self):
        """Fingertip drop and finger-axis yaw, measured from gripper geometry.

        ``robot.data.body_pos_w`` is useless here: in the flattened Robotiq USD
        every gripper link frame is co-located with base_link.
        """
        import omni.usd

        _, q_base = self._base_link_root()
        self.q_down = math_utils.quat_mul(
            q_base, self.eef_offset.unsqueeze(0))[0].detach().cpu().numpy().astype(float)

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
        base_prim = self._find_prim(robot_prim, "base_link")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        approach = math_utils.quat_apply(
            self.eef_offset.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device)).squeeze(0)
        grip = self._mesh_points(base_prim.GetParent(), base_prim, cache)
        self.finger_drop = float((grip @ approach.detach().cpu().numpy()).max())

        lc = self._mesh_points(self._find_prim(robot_prim, "left_inner_finger"),
                               base_prim, cache).mean(axis=0)
        rc = self._mesh_points(self._find_prim(robot_prim, "right_inner_finger"),
                               base_prim, cache).mean(axis=0)
        sep = math_utils.quat_apply(
            q_base[:1], torch.tensor((lc - rc), dtype=torch.float32,
                                     device=self.device).unsqueeze(0)).squeeze(0)
        sep = sep.detach().cpu().numpy()
        self.finger_axis_angle = float(np.arctan2(sep[1], sep[0]))
        self.calibrated = True
        self.log(f"[cal] finger_drop={self.finger_drop:.4f} m  "
                 f"finger_axis={self.finger_axis_angle:.4f} rad  "
                 f"q_down={np.round(self.q_down, 4).tolist()}")

    def _body_points(self, name):
        """Mesh vertices of a scene object in its own body frame, scene-scaled."""
        prim = self.world._get_prim(name, env_id=0)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        inv_root = cache.GetLocalToWorldTransform(prim).GetInverse()
        scale = usd_utils.get_scale(prim)
        pts = []
        for p in Usd.PrimRange(prim):
            if not p.IsA(UsdGeom.Mesh):
                continue
            raw = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not raw:
                continue
            to_body = cache.GetLocalToWorldTransform(p) * inv_root
            for v in raw:
                q = to_body.Transform(Gf.Vec3d(v[0], v[1], v[2]))
                pts.append((q[0] * scale[0], q[1] * scale[1], q[2] * scale[2]))
        return np.asarray(pts, dtype=float)

    # ------------------------------------------------------------- geometry
    def _pose(self, name):
        p, q = self.world.get_pose(name, env_id=0)
        return _np(p), _np(q)

    def _plan_grasp(self):
        """Grasp pose from the object's own limb geometry (body frame)."""
        pts = self._body_points(self.obj_name)
        lf = limb_frame(pts, bias=self.cfg["grasp_bias"])
        self.limb = lf
        self.obj_pts = pts

        try:
            masses = self.env.scene[self.obj_name].root_physx_view.get_masses()
            self.report["object_mass_kg"] = round(float(_np(masses).sum()), 5)
            self.log(f"[mass] {self.obj_name} = {self.report['object_mass_kg']} kg")
        except Exception as exc:  # noqa: BLE001
            self.log(f"[mass] unavailable: {exc}")

        obj_p, obj_q = self._pose(self.obj_name)
        R_obj = matrix_from_quat(obj_q)

        grasp_w = obj_p + R_obj @ lf["grasp_local"]
        # close the fingers across the limb: finger axis perpendicular to the
        # limb's horizontal direction
        limb_w = R_obj @ lf["axis"]
        limb_yaw = float(np.arctan2(limb_w[1], limb_w[0]))
        want = limb_yaw + np.pi / 2
        yaw = want - self.finger_axis_angle
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if self.cfg["grasp_yaw_flip"]:
            yaw += np.pi
        self.grasp_yaw = yaw
        rz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.grasp_quat = _qmul(rz, self.q_down)

        # base_link command so the fingertips land on the grasp point
        self.grasp_pos = grasp_w + np.array([0.0, 0.0, self.finger_drop - self.cfg["grasp_lower"]])
        self.grasp_point_w = grasp_w

        self.report["grasp_point_object_frame"] = np.round(lf["grasp_local"], 5).tolist()
        self.report["grasp_limb_bins"] = lf["limb_bins"]
        self.report["grasp_cross_section_m"] = [round(lf["grasp_width_u"], 4),
                                                round(lf["grasp_width_w"], 4)]
        self.report["grasp_bias"] = self.cfg["grasp_bias"]
        self.report["grasp_to_centroid_xy_m"] = round(lf["grasp_to_centroid_xy"], 5)
        self.report["grasp_point_world"] = np.round(grasp_w, 5).tolist()
        self.report["grasp_quat_eef"] = np.round(self.grasp_quat, 5).tolist()
        self.report["grasp_yaw_rad"] = round(yaw, 5)
        self.report["limb_axis_object_frame"] = np.round(lf["axis"], 5).tolist()
        if lf["free_end_local"] is not None:
            self.report["free_end_object_frame"] = np.round(lf["free_end_local"], 5).tolist()
            self.report["grasp_to_free_end_m"] = round(
                float(np.linalg.norm(lf["free_end_local"] - lf["grasp_local"])), 5)

        self.log(f"[plan] {self.obj_name}: grasp(obj)="
                 f"{np.round(lf['grasp_local'], 4).tolist()} limb_bins={lf['limb_bins']} "
                 f"cross_section={lf['grasp_width_u']:.4f}x{lf['grasp_width_w']:.4f} "
                 f"bias={self.cfg['grasp_bias']} "
                 f"lever={lf['grasp_to_centroid_xy']:.4f} "
                 f"grasp(world)={np.round(grasp_w, 4).tolist()} yaw={yaw:+.3f}")

    def _container_frame(self):
        """Container cavity in world: rim/floor z, mouth centre."""
        pts = self._body_points(self.container_name)
        cav = cavity(pts)
        cp, cq = self._pose(self.container_name)
        R = matrix_from_quat(cq)
        if cav is None:
            corners, centroid = self.world.get_bbox(self.container_name, env_id=0)
            corners, centre_w = _np(corners), _np(centroid)
            self.log(f"[warn] no mesh cavity for {self.container_name}; "
                     "falling back to its bounding box")
            return {"rim_z": float(corners[:, 2].max()),
                    "floor_z": float(corners[:, 2].min()),
                    "centre_w": np.array([centre_w[0], centre_w[1], 0.0]),
                    "axis_w": R @ np.array([0.0, 0.0, 1.0]),
                    "mouth_size": None}
        mouth_local = np.array([cav["mouth_centre"][0], cav["mouth_centre"][1],
                                0.5 * (cav["rim_z"] + cav["floor_z"])])
        centre_w = cp + R @ mouth_local
        return {
            "rim_z": float(cp[2] + (R @ np.array([0, 0, cav["rim_z"]]))[2]),
            "floor_z": float(cp[2] + (R @ np.array([0, 0, cav["floor_z"]]))[2]),
            "centre_w": centre_w,
            "axis_w": R @ np.array([0.0, 0.0, 1.0]),
            "mouth_size": cav["mouth_size"],
        }

    def _measure_in_hand(self):
        """Object pose in the eef frame, measured after the fingers close."""
        e_p, e_q = self._eef_pose()
        o_p, o_q = self._pose(self.obj_name)
        R_e = matrix_from_quat(e_q)
        self.p_e_obj = R_e.T @ (o_p - e_p)
        self.R_e_obj = R_e.T @ matrix_from_quat(o_q)
        q_e_obj = quat_from_matrix(self.R_e_obj)
        self.report["object_in_gripper_pos"] = np.round(self.p_e_obj, 5).tolist()
        self.report["object_in_gripper_quat"] = np.round(q_e_obj, 5).tolist()
        self.log(f"[hand] {self.obj_name} in gripper frame: "
                 f"p={np.round(self.p_e_obj, 4).tolist()} q={np.round(q_e_obj, 4).tolist()}")

    # -------------------------------------------------------------- targets
    def _insert_target(self):
        """Utensil: handle free end down the container's interior axis."""
        cf = self._container_frame()
        free_dir = self.limb["free_dir"]
        free_end = self.limb["free_end_local"]
        down = -cf["axis_w"]

        # Only the tool's long axis is constrained (it must point down the
        # container's interior axis); the spin about that axis is free. Pick the
        # spin keeping the wrist closest to the orientation it already holds
        # from the grasp: the arm is demonstrably in a good configuration there,
        # and minimising travel avoids the joint limits that made a "point the
        # gripper away from the robot" choice untrackable (29 deg residual).
        forced = self.cfg["insert_spin_deg"]
        spins = ([np.radians(forced)] if forced is not None
                 else np.linspace(-np.pi, np.pi, 145))
        best = None
        for spin in spins:
            R_obj_des = align_rotation(free_dir, down, spin_axis=down, spin=spin)
            R_eef_des = R_obj_des @ self.R_e_obj.T
            q = quat_from_matrix(R_eef_des)
            travel = 2.0 * np.arccos(min(1.0, abs(float(np.dot(q, self.grasp_quat)))))
            if best is None or travel < best[0]:
                best = (travel, spin, R_obj_des, R_eef_des)
        travel, spin, R_obj_des, R_eef_des = best
        self.log(f"[spin] chose {spin:+.3f} rad -> wrist travel "
                 f"{np.degrees(travel):.1f} deg")

        # tip depth chosen so the fingers stay clear of the rim
        off_hand = (R_eef_des @ self.p_e_obj)[2]
        off_free = (R_obj_des @ free_end)[2]
        off_tip = (R_eef_des @ np.array([0.0, 0.0, self.finger_drop]))[2]
        tip_z = cf["rim_z"] + self.cfg["rim_clear"] + off_hand + off_free - off_tip
        tip_z = min(tip_z, cf["rim_z"] - self.cfg["min_insert"])
        tip_z = max(tip_z, cf["floor_z"] + 0.012)

        tip_target = np.array([cf["centre_w"][0], cf["centre_w"][1], tip_z])
        p_eef_des = tip_target - R_eef_des @ self.p_e_obj - R_obj_des @ free_end

        q_obj_des = quat_from_matrix(R_obj_des)
        q_eef_des = quat_from_matrix(R_eef_des)
        fingertip_z = p_eef_des[2] + (R_eef_des @ np.array([0, 0, self.finger_drop]))[2]

        self.report["container_rim_z"] = round(cf["rim_z"], 5)
        self.report["container_floor_z"] = round(cf["floor_z"], 5)
        self.report["container_mouth_centre_world"] = np.round(cf["centre_w"], 5).tolist()
        self.report["container_insertion_axis_world"] = np.round(cf["axis_w"], 5).tolist()
        self.report["container_mouth_size"] = (None if cf["mouth_size"] is None
                                               else np.round(cf["mouth_size"], 4).tolist())
        self.report["desired_object_quat_world"] = np.round(q_obj_des, 5).tolist()
        self.report["insert_spin_rad"] = round(float(spin), 4)
        self.report["tip_target_world"] = np.round(tip_target, 5).tolist()
        self.report["tip_depth_below_rim_m"] = round(float(cf["rim_z"] - tip_z), 5)
        self.report["fingertip_z_at_insert"] = round(float(fingertip_z), 5)
        self.report["fingertip_clearance_above_rim_m"] = round(
            float(fingertip_z - cf["rim_z"]), 5)
        self._log_reorient(q_eef_des)
        self.log(f"[target] insert: tip->{np.round(tip_target, 4).tolist()} "
                 f"depth_below_rim={cf['rim_z'] - tip_z:.4f} "
                 f"fingertip_clear={fingertip_z - cf['rim_z']:+.4f} "
                 f"eef->{np.round(p_eef_des, 4).tolist()}")
        return p_eef_des, q_eef_des

    def _place_target(self):
        """Drill: keep the grasped orientation, lower into the container."""
        cf = self._container_frame()
        R_obj_des = self.R_obj_at_lift
        R_eef_des = R_obj_des @ self.R_e_obj.T
        # lowest point of the object in this orientation, relative to its origin
        low = float((self.obj_pts @ R_obj_des.T)[:, 2].min())
        obj_z = cf["floor_z"] + self.cfg["release_clear"] - low
        obj_target = np.array([cf["centre_w"][0], cf["centre_w"][1], obj_z])
        p_eef_des = obj_target - R_eef_des @ self.p_e_obj

        self.report["container_rim_z"] = round(cf["rim_z"], 5)
        self.report["container_floor_z"] = round(cf["floor_z"], 5)
        self.report["container_mouth_centre_world"] = np.round(cf["centre_w"], 5).tolist()
        self.report["container_mouth_size"] = (None if cf["mouth_size"] is None
                                               else np.round(cf["mouth_size"], 4).tolist())
        self.report["object_lowest_offset_m"] = round(low, 5)
        self.report["object_target_world"] = np.round(obj_target, 5).tolist()
        self.report["object_bottom_above_floor_m"] = round(self.cfg["release_clear"], 5)
        q_eef_des = quat_from_matrix(R_eef_des)
        self.report["desired_object_quat_world"] = np.round(
            quat_from_matrix(R_obj_des), 5).tolist()
        self._log_reorient(q_eef_des)
        self.log(f"[target] place: obj->{np.round(obj_target, 4).tolist()} "
                 f"(bottom {self.cfg['release_clear']:.3f} above floor "
                 f"{cf['floor_z']:.4f}) eef->{np.round(p_eef_des, 4).tolist()}")
        return p_eef_des, q_eef_des

    def _log_reorient(self, q_eef_des):
        """Wrist rotation commanded between grasp and target configurations."""
        dq = _qmul(q_eef_des, _qinv(self.grasp_quat))
        dq = dq / np.linalg.norm(dq)
        ang = 2.0 * np.arccos(min(1.0, abs(float(dq[0]))))
        axis = dq[1:] / max(np.linalg.norm(dq[1:]), 1e-9)
        if dq[0] < 0:
            axis = -axis
        self.report["reorient_quat_grasp_to_target"] = np.round(dq, 5).tolist()
        self.report["reorient_angle_deg"] = round(float(np.degrees(ang)), 3)
        self.report["reorient_axis_world"] = np.round(axis, 4).tolist()
        self.log(f"[reorient] {np.degrees(ang):.2f} deg about "
                 f"{np.round(axis, 3).tolist()} (grasp -> target)")

    # --------------------------------------------------------------- stages
    def _stage(self, name, pos, quat, grip, speed=None, hold=0, timeout=260, tol=None):
        return {"name": name, "pos": np.asarray(pos, dtype=float),
                "quat": np.asarray(quat, dtype=float), "grip": float(grip),
                "speed": speed or self.cfg["move_speed"], "hold": hold,
                "timeout": timeout, "tol": tol or self.cfg["pos_tol"]}

    def reset(self):
        if not self.calibrated:
            self._calibrate()
        p, q = self._eef_pose()
        self._plan_grasp()

        pre = self.grasp_pos + np.array([0.0, 0.0, self.cfg["approach_h"]])
        self.stages = [
            self._stage("PREGRASP", pre, self.grasp_quat, 0.0),
            self._stage("DESCEND", self.grasp_pos, self.grasp_quat, 0.0,
                        speed=self.cfg["fine_speed"], tol=self.cfg["pos_tol_fine"],
                        timeout=320),
            self._stage("CLOSE", self.grasp_pos, self.grasp_quat, 1.0,
                        hold=self.cfg["close_hold"], timeout=self.cfg["close_hold"] + 5),
        ]
        self.planned_tail = False
        self.i = 0
        self.step_in_stage = 0
        self.setpoint_pos = p.copy()
        self.setpoint_quat = q.copy()
        self.err_int = np.zeros(3)
        self.ik.reset()
        self.stage_name = self.stages[0]["name"]

    def _plan_tail(self):
        """Everything after the grasp, using the measured in-hand transform."""
        self._measure_in_hand()
        _, o_q = self._pose(self.obj_name)
        self.R_obj_at_lift = matrix_from_quat(o_q)

        lift = self.grasp_pos.copy()
        lift[2] = self.cfg["lift_h"] + self.finger_drop

        if self.mode == "insert":
            p_des, q_des = self._insert_target()
        else:
            p_des, q_des = self._place_target()

        pre = p_des.copy()
        pre[2] = p_des[2] + self.cfg["pretarget_dz"]
        retreat = p_des.copy()
        retreat[2] = p_des[2] + self.cfg["pretarget_dz"] + 0.06

        takeload = self.grasp_pos.copy()
        takeload[2] = self.grasp_pos[2] + self.cfg["takeload_h"]
        self.stages.append(
            self._stage("TAKELOAD", takeload, self.grasp_quat, 1.0,
                        speed=self.cfg["lift_speed"], hold=self.cfg["takeload_hold"],
                        timeout=self.cfg["takeload_hold"] + 5))
        self.stages.append(
            self._stage("LIFT", lift, self.grasp_quat, 1.0, speed=self.cfg["lift_speed"],
                        timeout=320))

        if self.mode == "insert":
            # Carry the tool to the holder in the *grasp* orientation, which the
            # arm tracks comfortably, and only rotate it upright once it is
            # already above the holder. Swinging across the table while holding
            # the awkward wrist pose left a 29 deg orientation residual.
            over = p_des.copy()
            over[2] = p_des[2] + self.cfg["reorient_dz"]
            self.stages += [
                self._stage("TRANSIT", over, self.grasp_quat, 1.0, timeout=420),
                self._stage("REORIENT", over, q_des, 1.0, speed=self.cfg["fine_speed"],
                            timeout=420),
                self._stage("INSERT", p_des, q_des, 1.0, speed=self.cfg["fine_speed"],
                            tol=self.cfg["pos_tol_fine"], timeout=360),
            ]
        else:
            self.stages += [
                self._stage("REORIENT", lift, q_des, 1.0, speed=self.cfg["fine_speed"],
                            timeout=320),
                self._stage("PRETARGET", pre, q_des, 1.0, timeout=420),
                self._stage("PLACE", p_des, q_des, 1.0, speed=self.cfg["fine_speed"],
                            tol=self.cfg["pos_tol_fine"], timeout=320),
            ]

        self.stages += [
            self._stage("RELEASE", p_des, q_des, 0.0,
                        hold=self.cfg["open_hold"], timeout=self.cfg["open_hold"] + 5),
            self._stage("RETREAT", retreat, q_des, 0.0, speed=self.cfg["fine_speed"]),
        ]
        self.planned_tail = True

    # ------------------------------------------------------------- stepping
    def act(self):
        if self.i >= len(self.stages):
            self.stage_name = "DONE"
            return self._hold_action()

        st = self.stages[self.i]
        self.stage_name = st["name"]

        delta = st["pos"] - self.setpoint_pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-9:
            self.setpoint_pos = self.setpoint_pos + delta / dist * min(dist, st["speed"])
        self.setpoint_quat = slerp_step(self.setpoint_quat, st["quat"], self.cfg["ang_speed"])

        p_now, q_now = self._eef_pose()
        settled = float(np.linalg.norm(st["pos"] - self.setpoint_pos)) < 1e-3
        if settled:
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
        # A NaN solve means the target is unreachable from here. Zero-filling it
        # commands the home configuration and the arm slams across the table
        # (seen as a 579 mm RETREAT error); hold the last good command instead.
        if bool(torch.isnan(q_des).any()):
            q_des = (self._last_q.clone() if getattr(self, "_last_q", None) is not None
                     else self.robot.data.joint_pos[:, self.arm_ids].clone())
        lo = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self.arm_ids, 1]
        q_des = torch.clamp(q_des, lo, hi)
        self._last_q = q_des.clone()

        self._advance(st, p_now, q_now, settled)
        grip = torch.full((1, 1), st["grip"], device=self.device)
        return torch.cat([q_des, grip], dim=1)

    def _advance(self, st, p_now, q_now, settled):
        self.step_in_stage += 1
        pos_err = float(np.linalg.norm(st["pos"] - p_now))
        ang_err = 2.0 * np.arccos(min(1.0, abs(float(np.dot(q_now, st["quat"])))))
        reached = settled and pos_err < st["tol"] and ang_err < self.cfg["ang_tol"]
        done = (self.step_in_stage >= st["hold"]) if st["hold"] else reached
        done = done or self.step_in_stage >= st["timeout"]
        if not done:
            return
        if self.step_in_stage >= st["timeout"] and not st["hold"]:
            q = _np(self.robot.data.joint_pos[0, self.arm_ids])
            lo = _np(self.robot.data.soft_joint_pos_limits[0, self.arm_ids, 0])
            hi = _np(self.robot.data.soft_joint_pos_limits[0, self.arm_ids, 1])
            near = [f"j{i+1}={q[i]:+.3f}[{lo[i]:+.2f},{hi[i]:+.2f}]"
                    f"{'  <-AT LIMIT' if min(q[i]-lo[i], hi[i]-q[i]) < 0.05 else ''}"
                    for i in range(7)]
            self.log(f"[timeout] {st['name']}: " + "  ".join(near))
        fj = float(self.robot.data.joint_pos[0, self.finger_id])
        self.log(f"[stage] {st['name']:9s} done in {self.step_in_stage:3d} steps  "
                 f"pos_err={pos_err * 1000:6.2f} mm  ang_err={np.degrees(ang_err):6.2f} deg  "
                 f"finger_joint={fj:.4f} rad")
        self.i += 1
        self.step_in_stage = 0
        if self.i == len(self.stages) and not self.planned_tail:
            self._plan_tail()

    def _hold_action(self):
        names = self.robot.data.joint_names
        ids = [names.index(n) for n in ARM_JOINT_NAMES] + [names.index("finger_joint")]
        return self.robot.data.joint_pos[:, ids].clone()

    @property
    def finished(self):
        return self.planned_tail and self.i >= len(self.stages)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qinv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / float(np.dot(q, q))
