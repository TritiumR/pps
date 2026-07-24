"""Shared base for the single-env IsaacLab wrappers (``DroidEnv`` / ``LiftEnv``).

Holds what they share: terminations-off, the arm joint indices + limits + control dt, and the
``q0`` / ``tcp`` / ``rgb`` / ``object_pose`` / ``keypoints_world`` accessors. A subclass builds its scene
cfg, runs ``gym.make`` + reset + settle, sets ``fk`` / ``cam`` / ``_grasp_offset`` / ``_obj_name``, then
calls ``_compute_arm_ids`` + ``_read_limits_dt``; only ``apply_arm`` (the action mapping) differs.

Construct only after AppLauncher has booted.
"""
import numpy as np

from sim_common.geometry import quat_wxyz_to_R


class IsaacLabEnv:
    """Common single-env base: terminations-off + joint limits/dt + shared GT accessors."""

    _grasp_offset = (0.0, 0.0, 0.0)   # subclass: TCP offset in the hand frame (tcp() default)
    _obj_name = None                  # subclass: default object scene key for object_pose()

    @staticmethod
    def _disable_terminations(cfg):
        """Turn every termination term off so an episode never resets mid-rollout."""
        if hasattr(cfg, "terminations"):
            for term in list(vars(cfg.terminations).keys()):
                try:
                    setattr(cfg.terminations, term, None)
                except Exception:
                    pass

    def _compute_arm_ids(self):
        """Panda arm joint indices (``self.arm_ids``) -- call before a settle loop that reads them."""
        jn = list(self.robot.data.joint_names)
        self.arm_ids = [jn.index(f"panda_joint{i}") for i in range(1, 8)]

    def _read_limits_dt(self, cfg):
        """Arm joint position limits (``q_lo``/``q_hi``) + control ``dt``, read after reset+settle."""
        lim = getattr(self.robot.data, "joint_pos_limits", None)
        if lim is None:
            lim = self.robot.data.soft_joint_pos_limits
        lim = lim[0, self.arm_ids]
        self.q_lo, self.q_hi = lim[:, 0].contiguous(), lim[:, 1].contiguous()
        try:
            self.dt = float(self.env.step_dt)
        except Exception:
            self.dt = float(self.env.sim.get_physics_dt()) * getattr(cfg, "decimation", 1)

    def q0(self):
        """Current arm joint positions [7] (detached)."""
        return self.robot.data.joint_pos[0, self.arm_ids].detach()

    def tcp(self, offset=None):
        """Grasp-point world position [3] via FK at the current config (``self._grasp_offset`` default)."""
        pos, _ = self.fk.grasp_point(self.q0().unsqueeze(0),
                                     self._grasp_offset if offset is None else offset)
        return pos[0].detach().cpu().numpy()

    def rgb(self):
        """Recording-camera RGB [H,W,3] uint8."""
        return self.cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def object_pose(self, name=None):
        """GT world pose (pos[3], R[3,3]) of a scene object (defaults to ``self._obj_name``)."""
        st = self.env.scene[name or self._obj_name].data.root_state_w[0, :7].detach().cpu().numpy()
        return st[:3], quat_wxyz_to_R(st[3:7])

    def keypoints_world(self, offsets, name=None):
        """Privileged GT keypoints: object pose applied to local ``offsets`` -> world [N,3]."""
        pos, R = self.object_pose(name)
        return pos[None] + np.asarray(offsets, dtype=np.float64) @ R.T

    def apply_arm(self, q_arm, grip_open):
        """Execute one arm config + gripper command (task-specific action mapping)."""
        raise NotImplementedError
