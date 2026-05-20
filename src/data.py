"""
data.py — Environment generation, configuration sampling, dataset assembly.

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

from utils import get_world_spheres_torch, query_sdf_differentiable, forward_kinematics_torch


# ── Environment ───────────────────────────────────────────────────────────────

def build_sdf_tensor(
    obstacles: Obstacles,
    grid_length: float = 2.5,
    n_vox: int = 128,
) -> torch.Tensor:
    """
    Converts an Obstacles object to a [1, H, W] SDF tensor.
    """
    grid = SquareGrid(
        data=np.zeros((n_vox, n_vox)),
        length=grid_length,
        origin=SE2.identity(),
    )

    # Create circles in the grid
    x = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    y = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    X, Y = np.meshgrid(x, y)
    for i in range(len(obstacles.r)):
        dist = np.sqrt((X - obstacles.x[i]) ** 2 + (Y - obstacles.y[i]) ** 2)
        grid.data[dist <= obstacles.r[i]] = 1.0

    sdf = grid.derive_sdf_from_voxels().data

    # Convert the SDF to a tensor
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
    Samples n_obstacles circular obstacles within the robot workspace.
    Obstacles are placed via polar coordinates (r in [0.15, 0.85*workspace_radius])
    so they always lie within the robot's reachable area. A minimum surface-to-surface
    separation is enforced to keep paths feasible.
    """
    if rng is None:
        rng = np.random.default_rng()

    positions, radii = [], []
    for _ in range(n_obstacles):
        for _ in range(max_tries):
            # Polar coordinates to guarantee that the obstacles are created within
            # the workspace
            r     = rng.uniform(0.15, workspace_radius * 0.85)
            theta = rng.uniform(0, 2 * np.pi)
            x, y  = r * np.cos(theta), r * np.sin(theta)
            rad   = rng.uniform(r_min, r_max)

            # Validate that the obstacles do not collide between them
            ok = all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) >= rad + pr + min_separation
                for (px, py), pr in zip(positions, radii)
            )
            if ok:
                positions.append((x, y))
                radii.append(rad)
                break

    # Backup plan if there were not valid obstacles
    if not positions:
        positions, radii = [(0.5, 0.0)], [0.1]

    # Return the valid positions
    xy = np.array(positions)
    return Obstacles(x=xy[:, 0], y=xy[:, 1], r=np.array(radii))


def visualize_environment(
    sdf_tensor: torch.Tensor,
    grid_length: float = 2.5,
    ax=None,
    robot=None,
    q=None,
):
    """
    Plots a single SDF environment (obstacles filled, boundary contoured).
    If robot and q are provided, the arm configuration is drawn on top.
    """
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
    if robot is not None and q is not None:
        q_t    = torch.from_numpy(np.asarray(q, dtype=np.float32)).unsqueeze(0)
        joints = forward_kinematics_torch(q_t, robot.linklengths)[0].numpy()  # [dof+1, 2]
        ax.plot(joints[:, 0], joints[:, 1], "o-", color="#2980b9",
                linewidth=2.5, markersize=5, zorder=5)
        ax.plot(joints[0, 0],  joints[0, 1],  "s", color="#27ae60", markersize=8,  zorder=6)
        ax.plot(joints[-1, 0], joints[-1, 1], "*", color="#f39c12", markersize=12, zorder=6)
    if own_fig:
        plt.tight_layout()
        plt.show()


def browse_dataset(
    dataset: dict,
    grid_length: float = 2.5,
    start_idx: int | None = None,
) -> None:
    """
    Interactive dataset browser with Prev / Next buttons to navigate samples.
    Starts at a random index when start_idx is None.
    Requires ipywidgets (pip install ipywidgets) and a Jupyter environment.
    """
    import io
    import ipywidgets as widgets
    from IPython.display import display
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    N = dataset["metadata"]["N"]
    robot = RobotInfo.from_linklengths(
        dataset["metadata"]["linklengths"],
        sphere_rad=dataset["metadata"]["sphere_rad"],
    )

    # Mutable index stored in a list so the callbacks can update it
    idx = [int(torch.randint(0, N, (1,)).item()) if start_idx is None else int(start_idx)]

    btn_prev   = widgets.Button(description="◄ Prev", layout=widgets.Layout(width="100px"))
    btn_next   = widgets.Button(description="Next ►", layout=widgets.Layout(width="100px"))
    info_label = widgets.HTML()
    img_widget = widgets.Image(format="png")

    def render():
        i          = idx[0]
        q_start_np = dataset["q_start"][i].numpy()
        q_goal_np  = dataset["q_goal"][i].numpy()
        n_obs      = dataset["n_obstacles"][i].item()
        sdf_sample = dataset["sdf"][i]

        # Overwrite label value in-place — no Output widget, no accumulation
        info_label.value = (
            f"<b>Sample {i} / {N - 1}</b> &nbsp;|&nbsp; obstacles: {n_obs}<br>"
            f"&nbsp;&nbsp;q_start: {np.round(q_start_np, 3)}<br>"
            f"&nbsp;&nbsp;q_goal:&nbsp; {np.round(q_goal_np, 3)}"
        )

        # Render figure to PNG bytes via Agg — bypasses all pyplot display hooks
        fig = Figure(figsize=(10, 5))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        for ax, q, title in zip(axes, [q_start_np, q_goal_np], ["q_start", "q_goal"]):
            visualize_environment(sdf_sample, grid_length, ax=ax, robot=robot, q=q)
            ax.set_title(title)
        fig.suptitle(f"Sample {i} / {N - 1}", fontsize=13)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        img_widget.value = buf.getvalue()

    def on_prev(_):
        idx[0] = (idx[0] - 1) % N
        render()

    def on_next(_):
        idx[0] = (idx[0] + 1) % N
        render()

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)

    render()
    display(widgets.VBox([
        widgets.HBox([btn_prev, btn_next]),
        info_label,
        img_widget,
    ]))


# ── Configuration sampling ────────────────────────────────────────────────────

def is_collision_free(
    q: np.ndarray,
    sdf_tensor: torch.Tensor,
    robot: RobotInfo,
    grid_length: float = 2.5,
    clearance: float = 0.0,
) -> bool:
    """
    Returns True when all robot spheres at joint configuration q have SDF > clearance.
    """
    q_t     = torch.from_numpy(q).float().unsqueeze(0)
    spheres = get_world_spheres_torch(q_t, robot)
    dists   = query_sdf_differentiable(sdf_tensor, spheres, grid_length)
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
    Returns True if any interior point on the joint-space straight line is in collision.
    Only non-trivial pairs (straight line blocked) provide a useful training signal —
    a trivial pair would teach the network to output straight-line interpolations.
    """
    ts = np.linspace(0, 1, n_check + 2)[1:-1]
    for t in ts:
        q_mid = (1 - t) * q_start + t * q_goal
        if not is_collision_free(q_mid, sdf_tensor, robot, grid_length):
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
    Samples a (q_start, q_goal) pair satisfying:
      1. Both configs are collision-free (with optional clearance margin).
      2. (if require_nontrivial) The straight joint-space path is blocked.
    Returns None if no valid pair is found within max_tries attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries):
        # Random start and goal configurations
        q_start = rng.uniform(q_min, q_max)
        q_goal  = rng.uniform(q_min, q_max)

        # Verify that they are collision free
        if not is_collision_free(q_start, sdf_tensor, robot, grid_length, clearance):
            continue
        if not is_collision_free(q_goal, sdf_tensor, robot, grid_length, clearance):
            continue

        # Verify that they are non trivial
        if require_nontrivial and not straight_line_blocked(
            q_start, q_goal, sdf_tensor, robot, grid_length, n_check_midpoints
        ):
            continue

        return q_start, q_goal

    return None


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
    grid_length: float = 2.5,
    n_vox: int = 128,
    seed: int = 42,
    save_path: str = "data/training_dataset.pt",
    split: tuple[float, float, float] | None = None,
) -> dict:
    """
    Full data generation pipeline for the WarmStartPlanner. For each of N_envs
    environments:
      1. Sample circular obstacles (workspace-aware, min separation enforced).
      2. Build an SDF tensor [1, H, W].
      3. Sample pairs_per_env valid (q_start, q_goal) pairs:
           - Both endpoints collision-free with clearance >= sphere_rad.
           - Straight joint-space line is blocked (non-trivial filter).

    Dataset fields:
      sdf:         [N, 1, H, W]     SDF grids
      q_start:     [N, dof]         Start joint configurations
      q_goal:      [N, dof]         Goal joint configurations
      obstacles:   [N, max_obs, 3]  Obstacle (x, y, r), zero-padded
      n_obstacles: [N]              Actual obstacle count per sample
      metadata:    dict             Robot, grid, and joint-limit parameters
    """
    if clearance is None:
        clearance = sphere_rad  # sphere surface must be outside obstacles

    rng = np.random.default_rng(seed) # Random seed fixed

    # Define empty containers
    all_sdfs, all_q_starts, all_q_goals = [], [], []
    all_obs_x, all_obs_y, all_obs_r, all_n_obs = [], [], [], []
    n_attempted, n_no_pair = 0, 0

    #Environments loop
    for _ in tqdm(range(N_envs), desc="Generating dataset"):
        # Build an environment with obstacles
        n_obs     = int(rng.integers(n_obstacles_range[0], n_obstacles_range[1] + 1))
        obstacles = sample_circular_obstacles(
            n_obstacles=n_obs, r_min=r_range[0], r_max=r_range[1],
            workspace_radius=workspace_radius, min_separation=min_separation, rng=rng,
        )
        sdf_tensor = build_sdf_tensor(obstacles, grid_length, n_vox)

        for _ in range(pairs_per_env):
            n_attempted += 1

            # Find a pair (start, goal) that is collision-free but its straight
            # path is blocked
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

            # Store successfull samples
            q_start, q_goal = pair
            all_sdfs.append(sdf_tensor)
            all_q_starts.append(torch.from_numpy(q_start).float())
            all_q_goals.append(torch.from_numpy(q_goal).float())
            all_obs_x.append(obstacles.x.copy())
            all_obs_y.append(obstacles.y.copy())
            all_obs_r.append(obstacles.r.copy())
            all_n_obs.append(len(obstacles.r))

    # Print out the quality diagnosis
    N_total = len(all_sdfs)
    print(f"\n{'─' * 48}")
    print(f"  Attempts:           {n_attempted}")
    print(f"  No valid pair:      {n_no_pair}")
    print(f"  Successful samples: {N_total}  ({100 * N_total / max(n_attempted, 1):.1f}%)")
    print(f"{'─' * 48}")
    if N_total == 0:
        print("WARNING: No data points generated. Check parameters.")
        return {}

    # Obstacle padding as not all of the environments have the same amount
    # of obstacles
    max_n_obs  = max(all_n_obs)
    obs_padded = torch.zeros(N_total, max_n_obs, 3)
    for i, (x, y, r) in enumerate(zip(all_obs_x, all_obs_y, all_obs_r)):
        n = len(x)
        obs_padded[i, :n, 0] = torch.from_numpy(x).float()
        obs_padded[i, :n, 1] = torch.from_numpy(y).float()
        obs_padded[i, :n, 2] = torch.from_numpy(r).float()

    # Stack and save
    metadata = {
        "N":           N_total,
        "grid_length": grid_length,
        "n_vox":       n_vox,
        "dof":         robot.n_dof,
        "linklengths": list(robot.linklengths),
        "sphere_rad":  sphere_rad,
        "q_min":       q_min.tolist() if q_min is not None else None,
        "q_max":       q_max.tolist() if q_max is not None else None,
    }
    dataset = {
        "sdf":         torch.stack(all_sdfs),
        "q_start":     torch.stack(all_q_starts),
        "q_goal":      torch.stack(all_q_goals),
        "obstacles":   obs_padded,
        "n_obstacles": torch.tensor(all_n_obs, dtype=torch.long),
        "metadata":    metadata,
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if split is None:
        torch.save(dataset, save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  Saved: {save_path}  ({size_mb:.1f} MB)")
    else:
        assert abs(sum(split) - 1.0) < 1e-6, "split fractions must sum to 1.0"
        train_frac, val_frac, _ = split
        idx = torch.randperm(N_total, generator=torch.Generator().manual_seed(seed))
        n_train = int(N_total * train_frac)
        n_val   = int(N_total * val_frac)
        splits = {
            "train": idx[:n_train],
            "val":   idx[n_train:n_train + n_val],
            "test":  idx[n_train + n_val:],
        }
        base, ext = os.path.splitext(save_path)
        for name, indices in splits.items():
            subset = {
                "sdf":         dataset["sdf"][indices],
                "q_start":     dataset["q_start"][indices],
                "q_goal":      dataset["q_goal"][indices],
                "obstacles":   dataset["obstacles"][indices],
                "n_obstacles": dataset["n_obstacles"][indices],
                "metadata":    {**metadata, "N": len(indices)},
            }
            path = f"{base}_{name}{ext}"
            torch.save(subset, path)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  Saved: {path}  ({len(indices)} samples, {size_mb:.1f} MB)")

    return dataset


def generate_sdf_dataset(
    N: int = 12000,
    save_path: str | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generates N random SDF environments for environment autoencoder pre-training.
    Returns a [N, 1, H, W] tensor. Saves to save_path if given (loads from cache if exists).
    """
    if save_path is not None and os.path.exists(save_path):
        dataset = torch.load(save_path, weights_only=True)
        print(f"SDF dataset loaded: {save_path}  shape={tuple(dataset.shape)}")
        return dataset

    rng  = np.random.default_rng(seed)
    sdfs = []
    for _ in tqdm(range(N), desc="Generating SDF dataset"):
        n_obs = int(rng.integers(1, 5))
        obs   = sample_circular_obstacles(n_obstacles=n_obs, rng=rng)
        sdfs.append(build_sdf_tensor(obs))
    dataset = torch.stack(sdfs)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"SDF dataset saved: {save_path}  shape={tuple(dataset.shape)}")

    return dataset


def print_dataset_stats(dataset: dict | str, n_collision_check: int = 100):
    """
    Print dataset statistics and a start-config collision sanity check.
    dataset: either a loaded dataset dict or a path to a .pt file.
    """
    if isinstance(dataset, str):
        dataset = torch.load(dataset, weights_only=False)

    meta        = dataset["metadata"]
    N           = meta["N"]
    grid_length = meta["grid_length"]
    robot       = RobotInfo.from_linklengths(meta["linklengths"], sphere_rad=meta["sphere_rad"])

    print(f"Dataset: {N} samples")
    for key in ("sdf", "q_start", "q_goal", "obstacles", "n_obstacles"):
        print(f"  {key:<12} {tuple(dataset[key].shape)}")
    print(f"\nMetadata: {meta}")

    n_check = min(N, n_collision_check)
    coll_rates = []
    for i in range(n_check):
        sdf_i   = dataset["sdf"][i, 0]
        spheres = get_world_spheres_torch(dataset["q_start"][i:i+1], robot)
        dist    = query_sdf_differentiable(sdf_i, spheres.reshape(-1, 2), grid_length)
        coll_rates.append((dist < 0).float().mean().item())

    print(f"\nStart-config collision rate (first {n_check} samples): "
          f"mean={np.mean(coll_rates):.3f}  (should be ~0.0)")
