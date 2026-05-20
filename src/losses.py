import torch
import torch.nn.functional as F
from utils import get_world_spheres_torch, query_sdf_differentiable

# ----------------------------
# Cost functions
# ----------------------------

def compute_joint_limits_cost(q, q_min, q_max, weight=1.0):
    """
    Computes the robot's joint limits cost.
    q: Tensor with the configurations.
    q_min/q_max: Tensors with the robot's joint limits.
    """
    # If q is within the limits, the clamp result is 0 and
    # there is no penalization
    lower_violation = torch.clamp(q_min - q, min=0)
    upper_violation = torch.clamp(q - q_max, min=0)

    return weight * torch.sum(lower_violation ** 2 + upper_violation ** 2)

def compute_smoothness_cost(q, dt=0.1, weight=1.0):
    """
    Computes the smoothness cost.
    q: Tensor with the configurations
    dt: Time step
    """
    # Penalize velocity
    vel = (q[..., 1:, :] - q[..., :-1, :]) / dt

    # Penalize acceleration
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt

    return weight * (torch.mean(vel ** 2) + torch.mean(acc ** 2))

def compute_collision_cost(distances, eps=0.1, weight=1.0):
    """
    Computes the collision cost.
    distances: Tensor with the SDF distances.
    eps: Security radio.
    """
    # Case 1: The point is inside the obstacle (negative distance)
    cost_inside  = -distances + 0.5 * eps

    # Case 2: The point is in the danger zone
    cost_danger  = 0.5 * (distances - eps) ** 2 / eps

     # Case 3: The point is in a safe position
    cost_safe    = torch.zeros_like(distances)

    # Combine the cases
    cost = torch.where(distances < 0, cost_inside,
           torch.where(distances <= eps, cost_danger, cost_safe))
    return weight * cost.sum()

def compute_trajectory_joint_limits_cost(q_traj, q_min, q_max, weight=1.0):
    """
    Batched joint limits cost over a full trajectory.
    q_traj:    [B, T, dof]
    q_min/q_max: Tensors with the robot's joint limits.
    """
    # Initialize the cost
    total_cost = 0

    # Compute the cost for each step
    for t in range(q_traj.shape[1]):
        total_cost += compute_joint_limits_cost(q_traj[:, t], q_min, q_max, weight=weight / q_traj.shape[1])
    return total_cost

def compute_trajectory_collision_cost(q_traj, sdf_batch, robot_info,
                                      grid_length=2.5, eps=0.1, weight=1.0):
    """
    Batched collision cost over a full trajectory.
    q_traj:    [B, T, dof]
    sdf_batch: [B, 1, H, W]
    """
    # Flatten the time for the kinematics
    B, T, dof = q_traj.shape
    q_flat = q_traj.reshape(B * T, dof)
    
    # Compute all the spheres once
    spheres = get_world_spheres_torch(q_flat, robot_info)      # [B*T, N_spheres, 2]
    
    # Regroup the spheres by batch
    N_spheres = spheres.shape[1]
    sphere_points = spheres.reshape(B, T * N_spheres, 2)

    # Repeat sdf_batch along time dimension for sampling
    sdf_rep = sdf_batch.repeat_interleave(1, dim=0)            # [B, 1, H, W] (already correct)
    distances = query_sdf_differentiable(sdf_rep, sphere_points, grid_length)  # [B, T*N_spheres]

    return compute_collision_cost(distances.reshape(-1), eps=eps, weight=weight / (B * T))