"""
evaluation.py — trajectory quality metrics, shared by every planner.

A single source of truth for judging any [B, T, dof] trajectory (model output,
CHOMP output, ...) on the same objective, so a learned model and a classical
optimizer are compared on identical numbers. Cost-weight defaults match the
"Bigboy" training objective (train.ipynb); geometric feasibility uses the robot
sphere *surface* clearance, exactly like data.is_collision_free, and is therefore
weight-independent — the cleanest comparison signal.

Requires SimpleArm on sys.path before importing this module:
    sys.path.insert(0, "/path/to/SimpleArm/src")
"""

import torch

from simplearm.robot import RobotInfo

from utils import get_world_spheres_torch, query_sdf_differentiable, forward_kinematics_torch
from losses import (
    compute_trajectory_collision_cost,
    compute_trajectory_max_collision_cost,
    compute_trajectory_joint_limits_cost,
    compute_smoothness_cost,
)


def _as_batched_traj(traj: torch.Tensor) -> torch.Tensor:
    return traj.unsqueeze(0) if traj.ndim == 2 else traj


def _as_batched_sdf(sdf: torch.Tensor, B: int) -> torch.Tensor:
    if sdf.ndim == 2:        # [H, W] -> [1, 1, H, W]
        sdf = sdf.unsqueeze(0).unsqueeze(0)
    elif sdf.ndim == 3:      # [1, H, W] -> [1, 1, H, W]
        sdf = sdf.unsqueeze(0)
    if sdf.shape[0] == 1 and B > 1:
        sdf = sdf.expand(B, -1, -1, -1)
    return sdf


@torch.no_grad()
def surface_clearance(traj, sdf, robot, grid_length=2.5, sphere_rad=None):
    """
    Signed distance of every robot-sphere SURFACE to the nearest obstacle,
    shape [B, T*N]. Negative = penetration. sphere_rad defaults to robot.sphere_rad.
    """
    traj = _as_batched_traj(traj.float())
    B, T, dof = traj.shape
    sdf = _as_batched_sdf(sdf.float(), B)
    if sphere_rad is None:
        sphere_rad = float(getattr(robot, "sphere_rad", 0.0))

    spheres = get_world_spheres_torch(traj.reshape(B * T, dof), robot)  # [B*T, N, 2]
    N       = spheres.shape[1]
    dists   = query_sdf_differentiable(sdf, spheres.reshape(B, T * N, 2), grid_length)
    return dists - sphere_rad


@torch.no_grad()
def evaluate_trajectory(
    traj, sdf, robot, grid_length, q_min, q_max, *,
    eps=0.8, w_coll=50.0, w_joints=0.1, w_smooth=1.0, collision_agg="max",
    dt=0.1, ccd=True, joint_weight_decay=0.5, sphere_rad=None,
) -> dict:
    """
    Quality metrics for a [B, T, dof] trajectory. Inputs may be unbatched
    ([T, dof] / [H, W]). For B > 1 the result aggregates over the batch
    (feasibility = all samples; counts = totals; lengths/costs = means).

    Cost terms (weighted, default = Bigboy training objective):
      total, coll, coll_sum, coll_max, joints, smooth

    Geometric feasibility (weight-independent, sphere-surface based):
      collision_free  — True iff every sphere surface stays outside obstacles
      n_collision_pts — number of (timestep, sphere) samples in collision
      min_clearance   — minimum surface clearance (neg = deepest penetration)
      mean_clearance  — mean surface clearance over the whole trajectory

    Trajectory shape:
      joint_path_length — sum of |Δq| in joint space (radians, per sample mean)
      ee_path_length    — end-effector Cartesian path length (per sample mean)
    """
    traj   = _as_batched_traj(traj.float())
    device = traj.device
    B, T, dof = traj.shape
    sdf = _as_batched_sdf(sdf.float(), B)

    q_min = torch.as_tensor(q_min, dtype=torch.float32, device=device)
    q_max = torch.as_tensor(q_max, dtype=torch.float32, device=device)

    def _coll(agg):
        fn = compute_trajectory_collision_cost if agg == "sum" else compute_trajectory_max_collision_cost
        return fn(traj, sdf, robot, grid_length=grid_length, eps=eps, weight=w_coll,
                  ccd=ccd, joint_weight_decay=joint_weight_decay).item()

    coll_sum = _coll("sum")
    coll_max = _coll("max")
    coll     = coll_sum if collision_agg == "sum" else coll_max
    joints   = compute_trajectory_joint_limits_cost(traj, q_min, q_max, weight=w_joints).item()
    smooth   = compute_smoothness_cost(traj, dt=dt, weight=w_smooth).item()

    clearance = surface_clearance(traj, sdf, robot, grid_length, sphere_rad)  # [B, T*N]

    # Joint-space path length (per sample, then averaged over the batch).
    joint_len = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1).sum(dim=-1).mean().item()

    # End-effector Cartesian path length.
    link = robot.linklengths if isinstance(robot.linklengths, torch.Tensor) \
        else torch.as_tensor(list(robot.linklengths), dtype=torch.float32, device=device)
    ee = forward_kinematics_torch(traj.reshape(B * T, dof), link)[:, -1, :].reshape(B, T, 2)
    ee_len = (ee[:, 1:] - ee[:, :-1]).norm(dim=-1).sum(dim=-1).mean().item()

    return {
        "total":           coll + joints + smooth,
        "coll":            coll,
        "coll_sum":        coll_sum,
        "coll_max":        coll_max,
        "joints":          joints,
        "smooth":          smooth,
        "collision_free":  bool((clearance >= 0).all().item()),
        "n_collision_pts": int((clearance < 0).sum().item()),
        "min_clearance":   float(clearance.min().item()),
        "mean_clearance":  float(clearance.mean().item()),
        "joint_path_length": joint_len,
        "ee_path_length":    ee_len,
    }


class TrajectoryEvaluator:
    """
    Convenience wrapper holding the robot / grid / limits / eval weights, so a
    trajectory can be scored without a CHOMP optimizer. Build it from a dataset's
    metadata via `from_metadata`, then call `evaluate(traj, sdf)`.
    """

    def __init__(self, robot, grid_length, q_min, q_max, *,
                 eps=0.8, w_coll=50.0, w_joints=0.1, w_smooth=1.0,
                 collision_agg="max", dt=0.1, ccd=True, joint_weight_decay=0.5,
                 device="cpu"):
        self.robot       = robot
        self.grid_length = grid_length
        self.q_min       = torch.as_tensor(q_min, dtype=torch.float32, device=device)
        self.q_max       = torch.as_tensor(q_max, dtype=torch.float32, device=device)
        self.cfg = dict(eps=eps, w_coll=w_coll, w_joints=w_joints, w_smooth=w_smooth,
                        collision_agg=collision_agg, dt=dt, ccd=ccd,
                        joint_weight_decay=joint_weight_decay)
        self.device = device

    @classmethod
    def from_metadata(cls, meta: dict, device="cpu", **overrides):
        """Mirror training.train's robot / joint-limit setup (incl. wide base joint)."""
        robot = RobotInfo.from_linklengths(meta["linklengths"], sphere_rad=meta["sphere_rad"])
        robot.sphere_rad = meta["sphere_rad"]
        q_min = torch.tensor(meta["q_min"], dtype=torch.float32, device=device)
        q_max = torch.tensor(meta["q_max"], dtype=torch.float32, device=device)
        q_min[0] = -2 * torch.pi
        q_max[0] =  2 * torch.pi
        return cls(robot, meta["grid_length"], q_min, q_max, device=device, **overrides)

    def evaluate(self, traj, sdf) -> dict:
        return evaluate_trajectory(
            traj.to(self.device), sdf.to(self.device), self.robot, self.grid_length,
            self.q_min, self.q_max, **self.cfg,
        )