import torch
import torch.nn.functional as F


def forward_kinematics_torch(q, link_lengths):
    q_cum = torch.cumsum(q, dim=-1)
    if not isinstance(link_lengths, torch.Tensor):
        link_lengths = torch.tensor(link_lengths, device=q.device, dtype=q.dtype)
    dx = torch.cos(q_cum) * link_lengths
    dy = torch.sin(q_cum) * link_lengths
    x = torch.cat([torch.zeros(q.shape[0], 1, device=q.device), torch.cumsum(dx, dim=-1)], dim=-1)
    y = torch.cat([torch.zeros(q.shape[0], 1, device=q.device), torch.cumsum(dy, dim=-1)], dim=-1)
    return torch.stack((x, y), dim=-1)


def get_world_spheres_torch(q, robot_info):
    q_cum = torch.cumsum(q, dim=-1)
    joint_positions = forward_kinematics_torch(q, robot_info.linklengths)
    sphere_offsets = torch.from_numpy(robot_info.spheres.xy).float().to(q.device)
    frame_indices = robot_info.spheres.frame_idx
    all_spheres = []
    for i, f_idx in enumerate(frame_indices):
        origin = joint_positions[:, f_idx, :]
        angle = q_cum[:, f_idx]
        local_dist = sphere_offsets[i, 0]
        s_x = origin[:, 0] + local_dist * torch.cos(angle)
        s_y = origin[:, 1] + local_dist * torch.sin(angle)
        all_spheres.append(torch.stack([s_x, s_y], dim=-1))
    return torch.stack(all_spheres, dim=1)  # [B, N_spheres, 2]


def query_sdf_differentiable(sdf_tensor, world_points, grid_length=2.5):
    """Single-environment SDF query. sdf_tensor: [H, W], world_points: [N, 2]."""
    grid_input = sdf_tensor.unsqueeze(0).unsqueeze(0)
    points_norm = world_points / (grid_length / 2.0)
    grid_coords = points_norm.unsqueeze(0).unsqueeze(2)
    sampled = F.grid_sample(grid_input, grid_coords, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return sampled.reshape(-1)


def query_sdf_batched(sdf_batch, world_points, grid_length=2.5):
    """Batched SDF query. sdf_batch: [B, 1, H, W], world_points: [B, N, 2]."""
    points_norm = (world_points / (grid_length / 2.0)).unsqueeze(2)  # [B, N, 1, 2]
    sampled = F.grid_sample(sdf_batch, points_norm, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return sampled.squeeze(1).squeeze(-1)  # [B, N]
