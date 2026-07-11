A.8 Implementation Details of Sub-Goal Solver
The sub-goal problems are implemented and solved using SciPy [125]. The decision variable is a
single end-effector pose (position and Euler angles) in R
6
for single-arm robots and two end-effector
poses in R
12 for bimanual robot. The bounds for the position terms are the pre-defined workspace
bounds, and the bounds for the rotation terms are that the half hemisphere where the end-effector
faces down (due to the joint limits of the Franka arm, it is often likely to reach joint limit when an
end-effector pose faces up). The decision variables are normalized to [−1, 1] based on the bounds.
For the first solving iteration, the initial guess is chosen to be the current end-effector pose. We
use sampling-based global optimization Dual Annealing [126] in the first iteration to quickly search
the full space, which is followed by a gradient-based local optimizer SLSQP [127] that refines the
solution. The full procedure takes around 1 second for this iteration. In subsequent iterations, we
use the solution from previous stage and only use local optimizer as it can quickly adjust to small
changes. The optimization is cut off with a fixed time budget represented as number of objective
function calls to keep the system running at a high frequency.
We discuss the cost terms in the objective function below.
Constraint Violation: We implement constraints as cost terms in the optimization problem, where
the returned costs by the ReKep functions are multiplied with large weights.
Scene Collision Avoidance: We use nvblox [149] with the PyTorch wrapper [58] to compute the
ESDF of the scene in a separate node that runs at 20 Hz. The ESDF calculation aggregates the
25
depth maps from all available cameras and excludes robot arms using cuRobo and any grasped rigid
objects (tracked via a masked tracker model Cutie [136]). A collision voxel grid is then calculated
using the ESDF and used by other modules in the system. In the sub-goal solver module, we first
downsample the gripper points and the grasped object points to have a maximum of 30 points using
farthest point sampling. Then we calculate the collision cost using the ESDF voxel grid with linear
interpolation with a threshold of 15cm.
Reachability: Since our decision variables are end-effector poses, which may not be always reachable by the robot arms, especially in confined spaces, we need to add a cost term that encourages
finding solutions with valid joint configurations. Therefore, we solve an IK problem in each iteration
of the sub-goal solver using PyBullet [133] and use its residual as a proxy for reachability. We find
that this takes around 40% of the time of the full objective function. Alternatively, one may solve
the problem in joint space, which would ensure the solution is within the joint limits by enforcing
the bounds. We find that this is inefficient with our Python-based implementation as we need to calculate forward kinematics for a magnitude of more times in the path solver, because the constraints
are evaluated in the task space. To address this while ensuring efficiency, future works can consider
using hardware-accelerated implementations to solve the problems in joint space [58].
Pose Regularization: We also add a small cost that encourages the sub-goal to be close to the
current end-effector pose.
Consistency: Since the solver iteratively solves the problem at a high frequency and the noise from
the perception pipeline may propagate to the solver, we find it useful to include a consistency cost
that encourages the solution to be close to the previous solution.
(Dual-Arm only) Self-Collision Avoidance: To avoid two arms collide with each other, we compute
the pairwise distance between the two point sets, each including the gripper points and grasped
object points.
A.9 Implementation Details of Path Solver
The path problems are implemented and solved using SciPy [125]. The number of decision variables
is calculated based on the distance between the current end-effector pose and the target end-effector
pose. Specifically, we define a fixed step size (20cm and 45 degree) and linearly approximate the
desired number of “intermediate poses”, which are used as decision variables. As in the sub-goal
problem, they are similarly represented using position and Euler angles with the same bounds. For
the first solving iteration, the initial guess is chosen to be linear interpolation between the start and
the target. We similarly use sampling-based global optimization followed by a gradient-based local
optimizer in the first iteration and only use local optimizer in subsequent iterations. After we obtain
the solution, represented as a number of intermediate poses, we fit a spline using the current pose,
the intermediate poses, and the target pose, which are then densely sampled to be executed by the
robot.
In the objective function, we first unnormalize the decision variables and use piecewise linear interpolation to obtain a dense sequence of discrete poses to represent the path (referred to as “dense
samples” below). A spline interpolation would be aligned with how we postprocess and execute the
solution, but we find linear interpolation to be computationally more efficient. Below we discuss the
individual cost terms in the objective function.
Constraint Violation: Similar to that in the sub-goal problem, we check violation of the ReKep
constraints for each dense sample along the path and penalize with large weights.
Scene Collision Avoidance: The calculation is similar to the sub-goal problem, except that it is
calculated for each dense sample. We ignore the collision calculation with a 5cm radius near the
start and the target poses, as this tends to stabilize the solution when solved at a high frequency due
to various real-world noises. We additionally add a table clearance cost that penalizes the path from
penetrating the table (or the bottom of the workspace for the wheeled single-arm robot).
26
Path Length: We approximate the path length using the dense samples by taking the sum of their
differences. Shorter paths are encouraged.
Reachability: We solve an IK problem for each intermediate pose inside the objective function as
in the sub-goal problem. See the sub-goal solver section for more details.
Consistency: As in the sub-goal problem, we encourage the solution to be close to the previous
one. Specifically, we store the dense samples from the previous iteration. To calculate the solution
consistency, we use the pairwise distance between the two sequences (treated as two sets) as an
efficient proxy. Alternatively, Hausdorff distance can be used.
(Dual-Arm only) Self-Collision Avoidance: We similarly compute self-collision avoidance for the
dual-arm platform as in the sub-goal problem. We also use pairwise distance between the two
sequences to efficiently calculate this cost.
A.1