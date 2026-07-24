"""ReKep subgoal solver (IsaacLab port).

Optimizes a 6-DoF end-effector pose that minimizes the stage's sub-goal + path constraint
violations, plus a consistency term (stay near the current pose) and, on grasp stages, a
top-down grasp-orientation term. Upstream's SDF-collision and IK-feasibility terms are
omitted -- IsaacLab's IK-Rel action handles reachability at execution time. Quaternions are
scipy (x,y,z,w) here, converted at the IsaacLab boundary (w,x,y,z).
"""

import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.transform import Rotation as R

# cost-term weights
_W_CONSISTENCY = 1.0
_W_GRASP = 10.0
_W_CONSTRAINT = 200.0


def pose7_to_homo(pose7):
    """[x,y,z, qx,qy,qz,qw] -> 4x4 homogeneous transform."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(pose7[3:]).as_matrix()
    mat[:3, 3] = pose7[:3]
    return mat


def homo_to_pose7(mat):
    """4x4 -> [x,y,z, qx,qy,qz,qw]."""
    return np.concatenate([mat[:3, 3], R.from_matrix(mat[:3, :3]).as_quat()])


def normalize_vars(vars_, og_bounds):
    out = np.empty_like(vars_)
    for i, (lo, hi) in enumerate(og_bounds):
        out[i] = (vars_[i] - lo) / (hi - lo) * 2 - 1
    return out


def unnormalize_vars(norm_vars, og_bounds):
    out = np.empty_like(norm_vars)
    for i, (lo, hi) in enumerate(og_bounds):
        out[i] = (norm_vars[i] + 1) / 2 * (hi - lo) + lo
    return out


def consistency(pose_a_homo, pose_b_homo, rot_weight=1.5):
    """Position distance + weighted rotation angle between two homogeneous poses."""
    pos_dist = np.linalg.norm(pose_a_homo[:3, 3] - pose_b_homo[:3, 3])
    rel = pose_a_homo[:3, :3].T @ pose_b_homo[:3, :3]
    rot_dist = float(np.linalg.norm(R.from_matrix(rel).as_rotvec()))
    return pos_dist + rot_weight * rot_dist


def transform_keypoints(transform, keypoints, movable_mask):
    """Apply a 4x4 transform to the movable keypoints only (others unchanged)."""
    out = keypoints.copy()
    if movable_mask.sum() > 0:
        out[movable_mask] = keypoints[movable_mask] @ transform[:3, :3].T + transform[:3, 3]
    return out


def _objective(opt_vars, og_bounds, keypoints_centered, movable_mask, goal_constraints,
               path_constraints, init_pose_homo, is_grasp_stage):
    """Total cost of a candidate EE pose: consistency + grasp-orientation + constraint violations."""
    opt_pose = unnormalize_vars(opt_vars, og_bounds)
    opt_pose_homo = np.eye(4)
    opt_pose_homo[:3, :3] = R.from_euler("xyz", opt_pose[3:]).as_matrix()
    opt_pose_homo[:3, 3] = opt_pose[:3]

    cost = _W_CONSISTENCY * consistency(opt_pose_homo, init_pose_homo, rot_weight=1.5)

    if is_grasp_stage:
        # prefer a top-down gripper: reward the EE z-axis (the approach axis) pointing world-down
        grasp_cost = -np.dot(opt_pose_homo[:3, 2], np.array([0, 0, -1])) + 1.0
        cost += _W_GRASP * grasp_cost

    for constraints in (goal_constraints, path_constraints):
        if constraints:
            tk = transform_keypoints(opt_pose_homo, keypoints_centered, movable_mask)
            for constraint in constraints:
                cost += _W_CONSTRAINT * np.clip(float(constraint(tk[0], tk[1:])), 0, np.inf)
    return cost


class SubgoalSolver:
    """Optimize the next EE subgoal pose for a stage's constraints."""

    def __init__(self, config):
        self.config = config
        self.last_sol = None

    def solve(self, ee_pose7, keypoints, movable_mask, goal_constraints, path_constraints,
              is_grasp_stage, from_scratch=False):
        """Return (subgoal_pose7, debug). keypoints[0] is the EE, keypoints[1:] the scene keypoints."""
        ee_homo = pose7_to_homo(ee_pose7)
        ee_euler = np.concatenate([ee_pose7[:3], R.from_quat(ee_pose7[3:]).as_euler("xyz")])

        pos_lo, pos_hi = np.array(self.config["bounds_min"]), np.array(self.config["bounds_max"])
        og_bounds = [(pos_lo[i], pos_hi[i]) for i in range(3)] + [(-np.pi, np.pi)] * 3
        norm_bounds = [(-1, 1)] * 6

        # center keypoints in the current EE frame (so opt_pose moves the grasped ones rigidly)
        centering = np.linalg.inv(ee_homo)
        keypoints_centered = transform_keypoints(centering, keypoints, movable_mask)

        if not from_scratch and self.last_sol is not None:
            init_sol = self.last_sol
        else:
            init_sol = normalize_vars(ee_euler, og_bounds)
            from_scratch = True

        args = (og_bounds, keypoints_centered, movable_mask, goal_constraints,
                path_constraints, ee_homo, is_grasp_stage)
        if from_scratch:
            res = dual_annealing(
                _objective, bounds=norm_bounds, args=args,
                maxfun=self.config.get("sampling_maxfun", 2000), x0=init_sol, no_local_search=False,
                minimizer_kwargs={"method": "SLSQP", "options": self.config.get("minimizer_options", {"maxiter": 200})},
            )
        else:
            res = minimize(_objective, x0=init_sol, args=args, bounds=norm_bounds,
                           method="SLSQP", options=self.config.get("minimizer_options", {"maxiter": 200}))

        sol_euler = unnormalize_vars(res.x, og_bounds)
        sol_pose7 = np.concatenate([sol_euler[:3], R.from_euler("xyz", sol_euler[3:]).as_quat()])
        if res.success or "maximum" in str(res.message).lower() or "iteration" in str(res.message).lower():
            self.last_sol = res.x
        return sol_pose7, {"cost": float(res.fun), "message": str(res.message)}
