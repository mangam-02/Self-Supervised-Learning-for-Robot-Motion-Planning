"""
data.py — Environment generation, configuration sampling, CHOMP, dataset assembly.

Requires SimpleArm on sys.path before importing this module:
    sys.path.insert(0, "/path/to/SimpleArm/src")
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from simplearm.geom import SquareGrid, SE2, Obstacles
from simplearm.robot import RobotInfo

from utils import get_world_spheres_torch, query_sdf_differentiable
from losses import (
    compute_smoothness_cost,
    compute_trajectory_collision_cost,
    compute_joint_limits_cost,
)
from models import build_bspline_interpolation_matrix


# ── Environment ───────────────────────────────────────────────────────────────

def build_sdf_tensor(
    obstacles: Obstacles,
    grid_length: float = 2.5,
    n_vox: int = 128,
) -> torch.Tensor:
    """Convert an Obstacles object to a [1, H, W] SDF tensor."""
    grid = SquareGrid(
        data=np.zeros((n_vox, n_vox)),
        length=grid_length,
        origin=SE2.identity(),
    )
    x = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    y = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    X, Y = np.meshgrid(x, y)
    for i in range(len(obstacles.r)):
        dist = np.sqrt((X - obstacles.x[i]) ** 2 + (Y - obstacles.y[i]) ** 2)
        grid.data[dist <= obstacles.r[i]] = 1.0
    sdf = grid.derive_sdf_from_voxels().data
    return torch.from_numpy(sdf).float().unsqueeze(0)


def sample_circular_obstacles(
    n_obstacles: int = 3,
    r_min: float = 0.06,
    r_max: float = 0.18,
    workspace_radius: float = 1.0,
    min_separation: float = 0.05,
    rng: np.random.Generator = None,
    max_tries: int = 300,
) -> Obstacles:
    """
    Sample n_obstacles circular obstacles within the robot workspace.

    Obstacles are placed via polar coordinates (r in [0.15, 0.85*workspace_radius])
    so they always lie within the robot's reachable area. A minimum surface-to-surface
    separation is enforced to keep paths feasible.
    """
    if rng is None:
        rng = np.random.default_rng()

    positions, radii = [], []
    for _ in range(n_obstacles):
        for _ in range(max_tries):
            r     = rng.uniform(0.15, workspace_radius * 0.85)
            theta = rng.uniform(0, 2 * np.pi)
            x, y  = r * np.cos(theta), r * np.sin(theta)
            rad   = rng.uniform(r_min, r_max)
            ok = all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) >= rad + pr + min_separation
                for (px, py), pr in zip(positions, radii)
            )
            if ok:
                positions.append((x, y))
                radii.append(rad)
                break

    if not positions:
        positions, radii = [(0.5, 0.0)], [0.1]

    xy = np.array(positions)
    return Obstacles(x=xy[:, 0], y=xy[:, 1], r=np.array(radii))


def visualize_environment(
    sdf_tensor: torch.Tensor,
    grid_length: float = 2.5,
    ax=None,
):
    """Plot a single SDF environment (obstacles filled, boundary contoured)."""
    sdf = sdf_tensor.squeeze(0).numpy()
    xs  = np.linspace(-grid_length / 2, grid_length / 2, sdf.shape[1])
    ys  = np.linspace(-grid_length / 2, grid_length / 2, sdf.shape[0])
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("#f0f0f0")
    ax.contourf(xs, ys, sdf, levels=[-1e6, 0], colors=["#c0392b"], alpha=0.85)
    ax.contour(xs, ys, sdf, levels=[0], colors=["#7b241c"], linewidths=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-grid_length / 2, grid_length / 2)
    ax.set_ylim(-grid_length / 2, grid_length / 2)
    ax.grid(True, color="white", linewidth=0.8)
    if own_fig:
        plt.tight_layout()
        plt.show()


# ── Configuration sampling ────────────────────────────────────────────────────

def is_collision_free(
    q: np.ndarray,
    sdf_tensor: torch.Tensor,
    robot: RobotInfo,
    grid_length: float = 2.5,
    clearance: float = 0.0,
) -> bool:
    """Return True when all robot spheres at joint configuration q have SDF > clearance."""
    q_t     = torch.from_numpy(q).float().unsqueeze(0)
    sdf_hw  = sdf_tensor.squeeze(0)
    spheres = get_world_spheres_torch(q_t, robot)
    dists   = query_sdf_differentiable(sdf_hw, spheres.reshape(-1, 2), grid_length)
    return bool((dists > clearance).all())


def straight_line_blocked(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    sdf_tensor: torch.Tensor,
    robot: RobotInfo,
    grid_length: float = 2.5,
    n_check: int = 9,
) -> bool:
    """
    Return True if any interior point on the joint-space straight line is in collision.

    Only non-trivial pairs (straight line blocked) provide a useful training signal —
    a trivial pair would teach the network to output straight-line interpolations.
    """
    sdf_hw = sdf_tensor.squeeze(0)
    ts = np.linspace(0, 1, n_check + 2)[1:-1]
    for t in ts:
        q_mid = (1 - t) * q_start + t * q_goal
        if not is_collision_free(q_mid, sdf_hw.unsqueeze(0), robot, grid_length):
            return True
    return False


def sample_valid_pair(
    sdf_tensor: torch.Tensor,
    robot: RobotInfo,
    q_min: np.ndarray,
    q_max: np.ndarray,
    grid_length: float = 2.5,
    clearance: float = 0.0,
    require_nontrivial: bool = True,
    n_check_midpoints: int = 9,
    max_tries: int = 300,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample a (q_start, q_goal) pair satisfying:
      1. Both configs are collision-free (with optional clearance margin).
      2. (if require_nontrivial) The straight joint-space path is blocked.

    Returns None if no valid pair is found within max_tries attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries):
        q_start = rng.uniform(q_min, q_max)
        q_goal  = rng.uniform(q_min, q_max)

        if not is_collision_free(q_start, sdf_tensor, robot, grid_length, clearance):
            continue
        if not is_collision_free(q_goal, sdf_tensor, robot, grid_length, clearance):
            continue
        if require_nontrivial and not straight_line_blocked(
            q_start, q_goal, sdf_tensor, robot, grid_length, n_check_midpoints
        ):
            continue

        return q_start, q_goal

    return None


# ── CHOMP ─────────────────────────────────────────────────────────────────────

def compute_reference_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    sdf_tensor: torch.Tensor,
    robot: RobotInfo,
    T: int = 50,
    C: int = 10,
    n_iter: int = 500,
    lr: float = 0.03,
    w_smooth: float = 0.001,
    w_coll: float = 5.0,
    w_vel: float = 400.0,
    w_jl: float = 50.0,
    eps: float = 0.15,
    v_max: float = 0.10,
    n_restarts: int = 3,
    grid_length: float = 2.5,
    success_threshold: float = 0.02,
    q_min: np.ndarray = None,
    q_max: np.ndarray = None,
) -> tuple[np.ndarray, bool]:
    """
    B-spline CHOMP with sinusoidal restarts.

    Optimizes C-2 interior waypoints with fixed start and goal endpoints.
    Sinusoidal perturbations bias restarts toward globally different solutions.
    A velocity-limit penalty prevents tunneling (arm crossing obstacles between frames).

    Args:
      w_coll:             Collision weight (normalized by B*T inside compute_trajectory_collision_cost).
      w_vel:              Velocity limit weight — prevents tunneling.
      v_max:              Max joint change [rad] per timestep.
      w_jl:               Joint limit penalty weight (skipped when q_min/q_max are None).
      success_threshold:  Max fraction of colliding sphere-timestep pairs to count as success.

    Returns:
      (q_traj [T, dof], success)
    """
    q_s = torch.tensor(q_start, dtype=torch.float32)
    q_g = torch.tensor(q_goal,  dtype=torch.float32)
    dof = len(q_start)

    sdf_batch    = sdf_tensor.unsqueeze(0)  # [1, 1, H, W]
    M            = build_bspline_interpolation_matrix(T, C, degree=3)
    t_inner      = torch.linspace(0, 1, C)[1:-1]
    straight     = q_s * (1 - t_inner[:, None]) + q_g * t_inner[:, None]
    sin_envelope = torch.sin(torch.pi * t_inner)

    q_min_t = torch.tensor(q_min, dtype=torch.float32) if q_min is not None else None
    q_max_t = torch.tensor(q_max, dtype=torch.float32) if q_max is not None else None

    best_traj, best_coll_rate = None, float("inf")

    for restart in range(n_restarts):
        if restart == 0:
            init = straight.clone()
        else:
            direction    = torch.randn(dof)
            direction    = direction / (direction.norm() + 1e-8)
            amplitude    = 0.6 * restart
            perturbation = amplitude * sin_envelope[:, None] * direction[None, :]
            init         = straight + perturbation

        q_inner = init.detach().clone().requires_grad_(True)
        opt     = torch.optim.Adam([q_inner], lr=lr)

        for _ in range(n_iter):
            opt.zero_grad()
            waypoints = torch.cat([q_s.unsqueeze(0), q_inner, q_g.unsqueeze(0)], dim=0)
            q_traj    = M @ waypoints  # [T, dof]

            loss_smooth = compute_smoothness_cost(q_traj.unsqueeze(0), dt=1.0, weight=w_smooth)
            loss_coll   = compute_trajectory_collision_cost(
                q_traj.unsqueeze(0), sdf_batch, robot,
                grid_length=grid_length, eps=eps, weight=w_coll,
            )
            vel      = q_traj[1:] - q_traj[:-1]
            loss_vel = w_vel * torch.relu(vel.abs() - v_max).pow(2).sum()
            loss_jl  = (
                compute_joint_limits_cost(q_traj, q_min_t, q_max_t, weight=w_jl)
                if q_min_t is not None else 0.0
            )

            (loss_smooth + loss_coll + loss_vel + loss_jl).backward()
            opt.step()

        with torch.no_grad():
            wp      = torch.cat([q_s.unsqueeze(0), q_inner, q_g.unsqueeze(0)], dim=0)
            q_final = M @ wp  # [T, dof]

        sdf_hw    = sdf_tensor.squeeze(0)
        spheres   = get_world_spheres_torch(q_final, robot)
        dists     = query_sdf_differentiable(sdf_hw, spheres.reshape(-1, 2), grid_length)
        coll_rate = (dists < 0).float().mean().item()

        if coll_rate < best_coll_rate:
            best_coll_rate = coll_rate
            best_traj      = q_final.numpy()

    return best_traj, best_coll_rate <= success_threshold


# ── Dataset assembly ──────────────────────────────────────────────────────────

def generate_dataset(
    N_envs: int = 200,
    pairs_per_env: int = 5,
    robot: RobotInfo = None,
    sphere_rad: float = 0.08,
    q_min: np.ndarray = None,
    q_max: np.ndarray = None,
    n_obstacles_range: tuple[int, int] = (1, 4),
    r_range: tuple[float, float] = (0.06, 0.18),
    workspace_radius: float = 1.0,
    min_separation: float = 0.05,
    clearance: float = None,
    require_nontrivial: bool = True,
    n_check_midpoints: int = 9,
    max_pair_tries: int = 300,
    T: int = 50,
    C: int = 10,
    chomp_n_iter: int = 500,
    chomp_lr: float = 0.03,
    chomp_w_smooth: float = 0.001,
    chomp_w_coll: float = 15.0,
    chomp_w_vel: float = 400.0,
    chomp_w_jl: float = 50.0,
    chomp_eps: float = 0.15,
    chomp_v_max: float = 0.10,
    chomp_n_restarts: int = 3,
    success_threshold: float = 0.02,
    grid_length: float = 2.5,
    n_vox: int = 128,
    seed: int = 42,
    save_path: str = "data/training_dataset.pt",
) -> dict:
    """
    Full data generation pipeline for the WarmStartPlanner.

    For each of N_envs environments:
      1. Sample circular obstacles (workspace-aware, min separation enforced).
      2. Build an SDF tensor [1, H, W].
      3. Sample pairs_per_env valid (q_start, q_goal) pairs:
           - Both endpoints collision-free with clearance >= sphere_rad.
           - Straight joint-space line is blocked (non-trivial filter).
      4. Run CHOMP to produce a reference trajectory.
      5. Accept only trajectories with collision rate < success_threshold.

    Dataset fields:
      sdf:         [N, 1, H, W]     SDF grids
      q_start:     [N, dof]         Start joint configurations
      q_goal:      [N, dof]         Goal joint configurations
      q_traj:      [N, T, dof]      CHOMP reference trajectories (labels)
      obstacles:   [N, max_obs, 3]  Obstacle (x, y, r), zero-padded
      n_obstacles: [N]              Actual obstacle count per sample
      metadata:    dict             Robot, grid, and joint-limit parameters
    """
    if clearance is None:
        clearance = sphere_rad  # sphere surface must be outside obstacles

    rng = np.random.default_rng(seed)

    all_sdfs, all_q_starts, all_q_goals, all_trajs = [], [], [], []
    all_obs_x, all_obs_y, all_obs_r, all_n_obs    = [], [], [], []
    n_attempted, n_no_pair, n_chomp_failed         = 0, 0, 0

    for _ in tqdm(range(N_envs), desc="Generating dataset"):
        n_obs     = int(rng.integers(n_obstacles_range[0], n_obstacles_range[1] + 1))
        obstacles = sample_circular_obstacles(
            n_obstacles=n_obs, r_min=r_range[0], r_max=r_range[1],
            workspace_radius=workspace_radius, min_separation=min_separation, rng=rng,
        )
        sdf_tensor = build_sdf_tensor(obstacles, grid_length, n_vox)

        for _ in range(pairs_per_env):
            n_attempted += 1

            pair = sample_valid_pair(
                sdf_tensor, robot, q_min, q_max,
                grid_length=grid_length, clearance=clearance,
                require_nontrivial=require_nontrivial,
                n_check_midpoints=n_check_midpoints,
                max_tries=max_pair_tries, rng=rng,
            )
            if pair is None:
                n_no_pair += 1
                continue

            q_start, q_goal = pair
            q_traj, success = compute_reference_trajectory(
                q_start, q_goal, sdf_tensor, robot,
                T=T, C=C,
                n_iter=chomp_n_iter, lr=chomp_lr,
                w_smooth=chomp_w_smooth, w_coll=chomp_w_coll,
                w_vel=chomp_w_vel, w_jl=chomp_w_jl,
                eps=chomp_eps, v_max=chomp_v_max,
                n_restarts=chomp_n_restarts,
                grid_length=grid_length,
                success_threshold=success_threshold,
                q_min=q_min, q_max=q_max,
            )
            if not success:
                n_chomp_failed += 1
                continue

            all_sdfs.append(sdf_tensor)
            all_q_starts.append(torch.from_numpy(q_start).float())
            all_q_goals.append(torch.from_numpy(q_goal).float())
            all_trajs.append(torch.from_numpy(q_traj).float())
            all_obs_x.append(obstacles.x.copy())
            all_obs_y.append(obstacles.y.copy())
            all_obs_r.append(obstacles.r.copy())
            all_n_obs.append(len(obstacles.r))

    N_total = len(all_sdfs)
    print(f"\n{'─' * 48}")
    print(f"  Attempts:           {n_attempted}")
    print(f"  No valid pair:      {n_no_pair}")
    print(f"  CHOMP failed:       {n_chomp_failed}")
    print(f"  Successful samples: {N_total}  ({100 * N_total / max(n_attempted, 1):.1f}%)")
    print(f"{'─' * 48}")

    if N_total == 0:
        print("WARNING: No data points generated. Check parameters.")
        return {}

    max_n_obs  = max(all_n_obs)
    obs_padded = torch.zeros(N_total, max_n_obs, 3)
    for i, (x, y, r) in enumerate(zip(all_obs_x, all_obs_y, all_obs_r)):
        n = len(x)
        obs_padded[i, :n, 0] = torch.from_numpy(x).float()
        obs_padded[i, :n, 1] = torch.from_numpy(y).float()
        obs_padded[i, :n, 2] = torch.from_numpy(r).float()

    dataset = {
        "sdf":         torch.stack(all_sdfs),
        "q_start":     torch.stack(all_q_starts),
        "q_goal":      torch.stack(all_q_goals),
        "q_traj":      torch.stack(all_trajs),
        "obstacles":   obs_padded,
        "n_obstacles": torch.tensor(all_n_obs, dtype=torch.long),
        "metadata": {
            "N":           N_total,
            "T":           T,
            "C":           C,
            "grid_length": grid_length,
            "n_vox":       n_vox,
            "dof":         robot.n_dof,
            "linklengths": list(robot.linklengths),
            "sphere_rad":  sphere_rad,
            "q_min":       q_min.tolist() if q_min is not None else None,
            "q_max":       q_max.tolist() if q_max is not None else None,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(dataset, save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"  Saved: {save_path}  ({size_mb:.1f} MB)")
    return dataset
