import torch
import torch.nn.functional as F

from utils import get_world_spheres_torch, query_sdf_batched


# ----------------------------
# Cost functions
# ----------------------------

def compute_joint_limits_cost(q, q_min, q_max, weight=1.0):
    lower_violation = torch.clamp(q_min - q, min=0)
    upper_violation = torch.clamp(q - q_max, min=0)
    return weight * torch.sum(lower_violation ** 2 + upper_violation ** 2)

def compute_trajectory_joint_limits_cost(q_traj, q_min, q_max, weight=1.0):
    total_cost = 0
    for t in range(q_traj.shape[1]):
        total_cost += compute_joint_limits_cost(q_traj[:, t], q_min, q_max, weight=weight / q_traj.shape[1])
    return total_cost


def compute_collision_cost(distances, eps=0.1, weight=1.0):
    cost_inside  = -distances + 0.5 * eps
    cost_danger  = 0.5 * (distances - eps) ** 2 / eps
    cost_safe    = torch.zeros_like(distances)
    cost = torch.where(distances < 0, cost_inside,
           torch.where(distances <= eps, cost_danger, cost_safe))
    return weight * cost.sum()


def compute_trajectory_collision_cost(q_traj, sdf_batch, robot_info,
                                      grid_length=2.5, eps=0.1, weight=1.0):
    """
    Batched collision cost over a full trajectory.
    q_traj:    [B, T, dof]
    sdf_batch: [B, 1, H, W]
    """
    B, T, dof = q_traj.shape
    q_flat = q_traj.reshape(B * T, dof)
    spheres = get_world_spheres_torch(q_flat, robot_info)      # [B*T, N_spheres, 2]
    N_spheres = spheres.shape[1]
    sphere_points = spheres.reshape(B, T * N_spheres, 2)

    # Repeat sdf_batch along time dimension for sampling
    sdf_rep = sdf_batch.repeat_interleave(1, dim=0)            # [B, 1, H, W] (already correct)
    distances = query_sdf_batched(sdf_rep, sphere_points, grid_length)  # [B, T*N_spheres]

    return compute_collision_cost(distances.reshape(-1), eps=eps, weight=weight / (B * T))


def compute_smoothness_cost(q, dt=0.1, weight=1.0):
    """q: [T, dof] or [B, T, dof]"""
    vel = (q[..., 1:, :] - q[..., :-1, :]) / dt
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    return weight * (torch.mean(vel ** 2) + torch.mean(acc ** 2))