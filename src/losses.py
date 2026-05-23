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
    lower_violation = torch.clamp(q_min - q, min=0)
    upper_violation = torch.clamp(q - q_max, min=0)
    return weight * torch.sum(lower_violation ** 2 + upper_violation ** 2)

def compute_smoothness_cost(q, dt=0.1, weight=1.0):
    """
    Computes the smoothness cost.
    q: Tensor with the configurations
    dt: Time step
    """
    vel = (q[..., 1:, :] - q[..., :-1, :]) / dt
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    return weight * (torch.mean(vel ** 2) + torch.mean(acc ** 2))

def compute_collision_cost(distances, eps=0.1, weight=1.0):
    """
    Computes the collision cost.
    distances: Tensor with the SDF distances.
    eps: Security radio.
    """
    cost_inside  = -distances + 0.5 * eps
    cost_danger  = 0.5 * (distances - eps) ** 2 / eps
    cost = torch.where(distances < 0, cost_inside,
           torch.where(distances <= eps, cost_danger,
           torch.zeros_like(distances)))

    # Exponential repulsion: provides nonzero gradient even in the safe zone so
    # that smoothness/velocity terms cannot silently push the trajectory into
    # obstacles without opposition. Decays with distance, so it stays small far
    # from obstacles and doesn't dominate the other cost terms.
    # gradient w.r.t. d: -0.5 * exp(-d/eps) — always negative → always pushes
    # trajectory away from obstacles regardless of which zone it is in.
    repulsion = (eps * 0.5) * torch.exp(-distances.clamp(min=0) / eps)
    return weight * (cost + repulsion).sum()

def compute_trajectory_joint_limits_cost(q_traj, q_min, q_max, weight=1.0,
                                         return_per_step=False):
    """
    Batched joint limits cost over a full trajectory.
    q_traj:    [B, T, dof]
    q_min/q_max: Tensors with the robot's joint limits.
    return_per_step: if True, also returns per-timestep costs [B, T]
    """
    lower_violation = torch.clamp(q_min - q_traj, min=0)
    upper_violation = torch.clamp(q_traj - q_max, min=0)
    per_step = (lower_violation ** 2 + upper_violation ** 2).sum(dim=-1)  # [B, T]
    total = weight * per_step.sum() / q_traj.shape[1]
    if return_per_step:
        return total, per_step
    return total

def compute_exploration_cost(waypoints, q_start, q_goal, collision_cost,
                             threshold=0.5, weight=1.0):
    """
    Breaks the 'barely outside obstacle' local minimum by penalizing small deviations
    from the straight-line baseline whenever the trajectory is in collision.

    The idea: if the arm crashes AND barely deviates from the straight line, it is
    stuck in the wrong local minimum — it needs to deviate MORE to route around.
    Multiplying by collision_cost ensures this only activates during actual collisions
    and fades away once the trajectory is clear.

    collision_cost: detached collision loss scalar — acts as a gate
    threshold:      minimum mean offset norm (radians) required while in collision
    waypoints:      [B, C, dof]
    """
    C      = waypoints.shape[1]
    t_vals = torch.linspace(0, 1, C, device=waypoints.device)
    baseline   = q_start.unsqueeze(1) + t_vals.view(1, -1, 1) * (q_goal - q_start).unsqueeze(1)
    offsets    = (waypoints - baseline)[:, 1:-1]          # [B, C-2, dof] — interior only
    offset_mag = offsets.norm(dim=-1).mean(dim=-1)        # [B]
    return weight * collision_cost * torch.clamp(threshold - offset_mag, min=0).mean()


def compute_waypoint_spacing_cost(waypoints, weight=1.0):
    """
    Penalizes unequal spacing between consecutive waypoints (in joint space).
    Discourages the arm from rushing through obstacles by making large inter-waypoint
    gaps expensive — a fast pass through a collision zone creates a high-distance
    outlier that raises the variance.
    waypoints: [B, C, dof]
    """
    diffs     = waypoints[:, 1:] - waypoints[:, :-1]   # [B, C-1, dof]
    distances = torch.norm(diffs, dim=-1)               # [B, C-1]
    mean_dist = distances.mean(dim=-1, keepdim=True)    # [B, 1]
    variance  = ((distances - mean_dist) ** 2).mean()
    return weight * variance


def _compute_sphere_costs(q_traj, sdf_batch, robot_info,
                          grid_length=2.5, eps=0.1,
                          ccd=True, joint_weight_decay=0.5):
    """
    Shared inner function: returns per-sphere costs [B, T, N_spheres].

    eps: controls the danger zone width AND the repulsion range. Increase to
         create a wider gradient field that encourages routing around obstacles.
    ccd: expands each sphere's danger zone by half its travel distance to prevent
         tunneling through thin obstacles at high speed.
    joint_weight_decay: spheres near the base are weighted more than tip spheres.
         Set to 0 to disable.
    """
    B, T, dof = q_traj.shape

    # --- Step 1: Forward kinematics → world-space sphere positions ---
    # The robot arm is approximated by a set of spheres placed along its links.
    # get_world_spheres_torch runs FK for every (batch, timestep) configuration
    # and returns the 2D center position of each sphere in world coordinates.
    q_flat  = q_traj.reshape(B * T, dof)                     # flatten batch+time → [B*T, dof]
    spheres = get_world_spheres_torch(q_flat, robot_info)     # [B*T, N_spheres, 2]
    N_spheres = spheres.shape[1]

    # Restore the time dimension so we can compute CCD travel distances later.
    sphere_pos    = spheres.reshape(B, T, N_spheres, 2)       # [B, T, N_spheres, 2]

    # --- Step 2: SDF lookup — distance from each sphere center to the nearest obstacle ---
    # query_sdf_differentiable does bilinear interpolation on the SDF grid,
    # so gradients flow back through sphere positions into the joint angles.
    # Positive distance = sphere is outside all obstacles.
    # Negative distance = sphere center is inside an obstacle.
    sphere_points = sphere_pos.reshape(B, T * N_spheres, 2)
    distances     = query_sdf_differentiable(sdf_batch, sphere_points, grid_length)
    distances     = distances.reshape(B, T, N_spheres)        # [B, T, N_spheres]

    # --- Step 3: CCD — inflate the danger zone for fast-moving spheres ---
    # A sphere moving quickly can jump over a thin obstacle between two timesteps
    # (tunneling). To prevent this, we expand eps by half the sphere's travel
    # distance: a sphere that moves 0.1m in one step gets an extra 0.05m margin.
    if ccd:
        travel  = torch.norm(sphere_pos[:, 1:] - sphere_pos[:, :-1], dim=-1)  # [B, T-1, N_spheres]
        travel  = torch.cat([travel, travel[:, -1:]], dim=1)  # repeat last step → [B, T, N_spheres]
        eff_eps = eps + travel / 2                            # effective danger zone per sphere
    else:
        eff_eps = eps

    # --- Step 4: Per-sphere cost — three zones ---
    # Zone 1 (d < 0):          sphere is INSIDE obstacle → linear cost, grows with penetration depth
    # Zone 2 (0 ≤ d ≤ eff_eps): sphere is in DANGER ZONE → quadratic cost, zero at d=eff_eps
    # Zone 3 (d > eff_eps):    sphere is SAFE → zero structured cost
    # Repulsion (all zones):   exponential term that is always > 0, decaying with distance.
    #   This provides gradient everywhere in the workspace, so the arm is always
    #   gently pushed away from obstacles — even before entering the danger zone.
    #   Larger eps → repulsion reaches farther → arm routes around instead of squeezing through.
    cost_inside = -distances + 0.5 * eff_eps
    cost_danger = 0.5 * (distances - eff_eps) ** 2 / eff_eps
    repulsion   = (eps * 0.5) * torch.exp(-distances.clamp(min=0) / eps)
    per_sphere  = torch.where(distances < 0, cost_inside,
                  torch.where(distances <= eff_eps, cost_danger,
                  torch.zeros_like(distances))) + repulsion   # [B, T, N_spheres]

    # --- Step 5: Joint-position weighting ---
    # Base spheres cost more than tip spheres. This creates an incentive to shift
    # any collision from the forearm towards the fingers: to do so, the arm must
    # bend — and that bending motion is geometrically equivalent to routing around
    # the obstacle. Tip collision (cheap) → arm bends further → no collision (free).
    # relative_pos = 0 at base, 1 at tip → weight = 1 at base, (1-decay) at tip.
    if joint_weight_decay > 0:
        link_lengths = robot_info.linklengths
        total_length = sum(link_lengths)
        cum_lengths  = [0.0] + [sum(link_lengths[:i+1]) for i in range(len(link_lengths))]
        sphere_dist  = [
            cum_lengths[f] + float(robot_info.spheres.xy[i, 0])
            for i, f in enumerate(robot_info.spheres.frame_idx)
        ]
        relative_pos   = torch.tensor(
            [d / total_length for d in sphere_dist],
            dtype=torch.float32, device=q_traj.device,
        )
        sphere_weights = 1.0 - joint_weight_decay * relative_pos
        per_sphere     = per_sphere * sphere_weights.view(1, 1, -1)

    return per_sphere  # [B, T, N_spheres]


def compute_trajectory_collision_cost(q_traj, sdf_batch, robot_info,
                                      grid_length=2.5, eps=0.1, weight=1.0,
                                      ccd=True, joint_weight_decay=0.5,
                                      return_per_step=False):
    """Sum-based collision cost: penalizes total collision across the trajectory."""
    per_sphere = _compute_sphere_costs(q_traj, sdf_batch, robot_info, grid_length, eps, ccd, joint_weight_decay)
    _, T, _    = per_sphere.shape
    per_step   = per_sphere.amax(dim=-1)          # [B, T]
    total      = weight * per_step.sum() / T
    if return_per_step:
        return total, per_step
    return total


def compute_trajectory_max_collision_cost(q_traj, sdf_batch, robot_info,
                                          grid_length=2.5, eps=0.1, weight=1.0,
                                          ccd=True, joint_weight_decay=0.5,
                                          return_per_step=False):
    """Max-based collision cost: penalizes the single worst collision across the trajectory."""
    per_sphere = _compute_sphere_costs(q_traj, sdf_batch, robot_info, grid_length, eps, ccd, joint_weight_decay)
    per_step   = per_sphere.amax(dim=-1)          # [B, T]
    total      = weight * per_step.amax(dim=-1).mean()
    if return_per_step:
        return total, per_step
    return total
