import torch
import torch.nn.functional as F


def forward_kinematics_torch(q, link_lengths):
    """
    PyTorch version for the kinematics.
    q: Tensor with the configurations
    link_lenghts: Tensor or list with the link lengths
    """
    # Cumulative sum of angles
    q_cum = torch.cumsum(q, dim=-1)

    if not isinstance(link_lengths, torch.Tensor):
        link_lengths = torch.tensor(link_lengths, device=q.device, dtype=q.dtype)
    
    dx = torch.cos(q_cum) * link_lengths
    dy = torch.sin(q_cum) * link_lengths

    # Calculate each joint position and concatenate the origin
    x = torch.cat([torch.zeros(q.shape[0], 1, device=q.device), torch.cumsum(dx, dim=-1)], dim=-1)
    y = torch.cat([torch.zeros(q.shape[0], 1, device=q.device), torch.cumsum(dy, dim=-1)], dim=-1)
    
    return torch.stack((x, y), dim=-1)


def get_world_spheres_torch(q, robot_info):
    """
    Simplified version for obtaining the spheres without needing to build every frame.
    q: Tensor with the configurations
    robot_info: Contains the link lenghts and the spheres local coordinates
    """
    # Cumulate sum of angles
    q_cum = torch.cumsum(q, dim=-1)

    # Obtain each link origin
    joint_positions = forward_kinematics_torch(q, robot_info.linklengths)
    
    # Position each sphere_points
    sphere_offsets = torch.from_numpy(robot_info.spheres.xy).float().to(q.device)
    frame_indices = robot_info.spheres.frame_idx
    all_spheres = []
    for i, f_idx in enumerate(frame_indices):
        # Link origin
        origin = joint_positions[:, f_idx, :]

        # Global angle
        angle = q_cum[:, f_idx]

        # Rotate the local offset to the global frame
        local_dist = sphere_offsets[i, 0]
        s_x = origin[:, 0] + local_dist * torch.cos(angle)
        s_y = origin[:, 1] + local_dist * torch.sin(angle)

        all_spheres.append(torch.stack([s_x, s_y], dim=-1))

    return torch.stack(all_spheres, dim=1)  # [B, N_spheres, 2]


def query_sdf_differentiable(sdf_batch, world_points, grid_length=2.5):
    """
    Environment differentiable encoding for a batch.
    sdf_batch: Batch of tensors with the distances
    world_points: Batch of tensors with positions in meters
    grid_length: Total size
    """
    # Check dimensions
    if sdf_batch.ndim == 2:
        sdf_batch = sdf_batch.unsqueeze(0).unsqueeze(0)
    elif sdf_batch.ndim == 3:
        sdf_batch = sdf_batch.unsqueeze(0)
    B = sdf_batch.size(0)
    if world_points.ndim == 2:
        world_points = world_points.unsqueeze(0)
    if world_points.size(0) != B:
        world_points = world_points.expand(B, -1, -1)
    
    # Normalize and prepare format [B, N, 1, 2]
    points_norm = (world_points / (grid_length / 2.0)).unsqueeze(2)

    # Sample with bilinear interpolation
    sampled = F.grid_sample(sdf_batch, points_norm, mode='bilinear',
                            padding_mode='border', align_corners=True)
    
    return sampled.squeeze(1).squeeze(-1) # [B, N]

