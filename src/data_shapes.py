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
from simplearm.perlin import perlin_noise_2d

from utils import get_world_spheres_torch, query_sdf_differentiable, forward_kinematics_torch


# ── Environment ───────────────────────────────────────────────────────────────

def build_sdf_tensor(
    grid_vox: SquareGrid,
) -> torch.Tensor:
    """
    Converts a SquareGrid containing arbitrary binary voxel data
    into a continuous [1, H, W] SDF tensor.
    """
    # Computes the exact signed distance field from the voxel data
    grid_sdf = grid_vox.derive_sdf_from_voxels()
    
    # Convert the array data to a PyTorch tensor
    return torch.from_numpy(grid_sdf.data).float().unsqueeze(0)

def generate_blob_mask(x_grid, y_grid, cx, cy, r_base, rng):
    """
    Genera formas orgánicas agresivas garantizando que NINGÚN punto del obstáculo
    exceda la distancia 'r_base' desde el centro (cx, cy).
    """
    # 1. Distancia y ángulo de cada píxel respecto al centro
    dx = x_grid - cx
    dy = y_grid - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 2. Perfil base (1.0 = círculo perfecto)
    profile = np.ones_like(angle)
    
    # 3. Aplastamiento direccional (Crea objetos largos)
    # Solo aplastamos hacia adentro (valores menores a 1), nunca expandimos
    squish_angle = rng.uniform(0, 2 * np.pi)
    squish_factor = rng.uniform(0.2, 0.8) 
    
    # Modulamos el perfil con una onda basada en la dirección del aplastamiento
    profile *= (squish_factor + (1 - squish_factor) * np.abs(np.cos(angle - squish_angle)))
    
    # 4. Ruido de armónicos agresivo para asimetría y formas irregulares
    n_harmonics = rng.integers(2, 5)
    for _ in range(n_harmonics):
        freq = rng.integers(2, 5) # Frecuencias bajas para deformar la topología
        amp = rng.uniform(0, 0.5) / n_harmonics 
        phase = rng.uniform(0, 2 * np.pi)
        profile += amp * np.sin(freq * angle + phase)
        
    # 5. Evitar valores negativos o cero si el ruido fue muy fuerte
    profile = np.clip(profile, 0.05, None)
    
    # 6. EL PASO CLAVE: Normalización estricta
    # Dividimos todo el perfil por su valor máximo. 
    # Ahora, el punto más lejano del centro vale exactamente 1.0.
    profile = profile / np.max(profile)
    
    # 7. Escalar al radio real permitido
    r_actual = r_base * profile
    
    # La máscara evalúa qué píxeles están dentro de nuestro nuevo radio seguro
    return dist <= r_actual

def sample_circular_obstacles(
    n_obstacles: int = 3,
    r_min: float = 0.06,
    r_max: float = 0.18,
    workspace_radius: float = 1.0,
    dist_from_origin: tuple[float, float] | None = None,
    min_separation: float = 0.05,
    min_base_clearance: float = 0.35,
    robot_sphere_rad: float = 0.08,  # <-- NUEVO: Para conocer el volumen base del robot
    rng: np.random.Generator = None,
    max_tries: int = 300,
) -> Obstacles:
    """
    Samples n_obstacles circular obstacles within the robot workspace.
    Obstacles are placed via polar coordinates. The distance parameters 
    now explicitly evaluate the distance from the closest boundary of the 
    obstacle to the physical spheres of the robot base.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # d_min y d_max ahora dictan la distancia desde el origen a la FRONTERA del obstáculo
    d_min, d_max = dist_from_origin if dist_from_origin is not None else (0.15, workspace_radius * 0.85)

    positions, radii = [], []
    for _ in range(n_obstacles):
        for _ in range(max_tries):
            # 1. Obtenemos el radio ANTES de ubicar el centro
            rad = rng.uniform(r_min, r_max)
            
            # 2. Calculamos los límites para el centro basándonos en la frontera
            # Queremos que la distancia del origen al borde del obstáculo sea al menos d_min.
            # Además, el borde no debe invadir las esferas físicas del robot (min_base_clearance + robot_sphere_rad).
            r_min_center = max(d_min + rad, min_base_clearance + robot_sphere_rad + rad)
            
            # El centro no debe estar tan lejos que el obstáculo se salga del workspace
            r_max_center = min(d_max + rad, workspace_radius - rad)
            
            if r_min_center >= r_max_center:
                continue  # Intentar con un nuevo radio si la geometría es imposible
                
            # 3. Muestreamos el centro usando los límites correctos
            r = rng.uniform(r_min_center, r_max_center)
            theta = rng.uniform(0, 2 * np.pi)
            x, y = r * np.cos(theta), r * np.sin(theta)

            # Validar que los obstáculos no colisionen entre sí
            ok = all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) >= rad + pr + min_separation
                for (px, py), pr in zip(positions, radii)
            )
            if ok:
                positions.append((x, y))
                radii.append(rad)
                break

    # Backup plan
    if not positions:
        positions, radii = [(0.6, 0.0)], [0.1]

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

    if isinstance(dataset, str):
        dataset = torch.load(dataset, weights_only=False)

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
        sdf_sample = dataset["sdf"][i]
        
        # Revisar si el dataset es de vóxeles o de obstáculos circulares
        if "n_obstacles" in dataset:
            n_obs = dataset["n_obstacles"][i].item()
            obs_info = f"obstacles: {n_obs}"
        else:
            obs_info = "Voxel Map (Perlin Noise)"

        # Overwrite label value in-place — no Output widget, no accumulation
        info_label.value = (
            f"<b>Sample {i} / {N - 1}</b> &nbsp;|&nbsp; {obs_info}<br>"
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
    diff = q_goal - q_start
    short_diff = np.arctan2(np.sin(diff), np.cos(diff))
    for t in ts:
        q_mid = q_start + t * short_diff
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
    dist_from_origin: tuple[float, float] | None = None, # <-- Añadido
    min_separation: float = 0.05,                        # <-- Añadido
    min_base_clearance: float = 0.35,                    # <-- Añadido
    clearance: float = None,
    require_nontrivial: bool = True,
    n_check_midpoints: int = 9,
    max_pair_tries: int = 500,
    grid_length: float = 2.5,
    n_vox: int = 128,
    seed: int = 42,
    save_path: str = "data/training_dataset.pt",
    split: tuple[float, float, float] | None = None,
    fixed_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """
    Full data generation pipeline using randomized organic blob geometry.
    """
    if clearance is None:
        clearance = sphere_rad

    rng = np.random.default_rng(seed)

    all_sdfs, all_voxels, all_q_starts, all_q_goals = [], [], [], []
    all_obs_x, all_obs_y, all_obs_r, all_n_obs = [], [], [], [] # <-- Recuperado para el dataset
    n_attempted, n_no_pair = 0, 0

    # Pre-calcular las coordenadas X e Y de la grilla para generar los blobs rápido
    lin = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    x_grid, y_grid = np.meshgrid(lin, lin)
    
    # RENOMBRADO a grid_dist para no sobreescribir el parámetro dist_from_origin
    grid_dist = np.sqrt(x_grid**2 + y_grid**2)

    for _ in tqdm(range(N_envs), desc="Generating dataset (Organic Blobs)"):
        # 1. Decidir cuántos obstáculos y obtener sus centros
        n_obs = int(rng.integers(n_obstacles_range[0], n_obstacles_range[1] + 1))
        
        obs = sample_circular_obstacles(
            n_obstacles=n_obs,
            r_min=r_range[0],
            r_max=r_range[1],
            workspace_radius=workspace_radius,
            dist_from_origin=dist_from_origin,       # <-- Pasado a la función
            min_separation=min_separation,           # <-- Pasado a la función
            min_base_clearance=min_base_clearance,   # <-- Pasado a la función
            robot_sphere_rad=sphere_rad,
            rng=rng
        )

        # 2. Construir la matriz de vóxeles y dibujar las formas orgánicas
        voxel_data = np.zeros((n_vox, n_vox), dtype=bool)
        
        for i in range(len(obs.x)):
            blob_mask = generate_blob_mask(
                x_grid, y_grid, 
                cx=obs.x[i], cy=obs.y[i], r_base=obs.r[i], 
                rng=rng
            )
            voxel_data = np.logical_or(voxel_data, blob_mask)
        
        # 3. Aplicar máscara de workspace usando grid_dist
        voxel_data[grid_dist > workspace_radius] = False

        grid_vox = SquareGrid.from_zero_centered(
            limits=(-grid_length / 2, grid_length / 2), 
            data=voxel_data
        )

        # 4. Derivar tensores
        sdf_tensor = build_sdf_tensor(grid_vox)
        vox_tensor = torch.from_numpy(grid_vox.data.astype(np.float32)).unsqueeze(0)

        n_pairs = 1 if fixed_pair is not None else pairs_per_env
        for _ in range(n_pairs):
            n_attempted += 1

            if fixed_pair is not None:
                q_s, q_g = fixed_pair
                valid = (
                    is_collision_free(q_s, sdf_tensor, robot, grid_length, clearance)
                    and is_collision_free(q_g, sdf_tensor, robot, grid_length, clearance)
                    and (not require_nontrivial or straight_line_blocked(
                        q_s, q_g, sdf_tensor, robot, grid_length, n_check_midpoints))
                )
                pair = fixed_pair if valid else None
            else:
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
            all_sdfs.append(sdf_tensor)
            all_voxels.append(vox_tensor)
            all_q_starts.append(torch.from_numpy(q_start).float())
            all_q_goals.append(torch.from_numpy(q_goal).float())
            
            # Guardamos la info original de los obstáculos por si necesitas visualizarla
            all_obs_x.append(obs.x.copy())
            all_obs_y.append(obs.y.copy())
            all_obs_r.append(obs.r.copy())
            all_n_obs.append(len(obs.r))

    N_total = len(all_sdfs)
    print(f"\n{'─' * 48}")
    print(f"  Attempts:           {n_attempted}")
    print(f"  No valid pair:      {n_no_pair}")
    print(f"  Successful samples: {N_total}  ({100 * N_total / max(n_attempted, 1):.1f}%)")
    print(f"{'─' * 48}")
    if N_total == 0:
        print("WARNING: No data points generated. Check parameters.")
        return {}

    # <-- Padding de los obstáculos recuperado del código antiguo
    max_n_obs  = max(all_n_obs) if all_n_obs else 0
    obs_padded = torch.zeros(N_total, max_n_obs, 3)
    for i, (x, y, r) in enumerate(zip(all_obs_x, all_obs_y, all_obs_r)):
        n = len(x)
        obs_padded[i, :n, 0] = torch.from_numpy(x).float()
        obs_padded[i, :n, 1] = torch.from_numpy(y).float()
        obs_padded[i, :n, 2] = torch.from_numpy(r).float()

    metadata = {
        "N":                N_total,
        "grid_length":      grid_length,
        "n_vox":            n_vox,
        "dof":              robot.n_dof,
        "linklengths":      list(robot.linklengths),
        "sphere_rad":       sphere_rad,
        "workspace_radius": workspace_radius,
        "q_min":            q_min.tolist() if q_min is not None else None,
        "q_max":            q_max.tolist() if q_max is not None else None,
    }
    
    dataset = {
        "sdf":         torch.stack(all_sdfs),
        "voxels":      torch.stack(all_voxels), 
        "q_start":     torch.stack(all_q_starts),
        "q_goal":      torch.stack(all_q_goals),
        "obstacles":   obs_padded,                          # <-- Añadido al dicc
        "n_obstacles": torch.tensor(all_n_obs, dtype=torch.long), # <-- Añadido al dicc
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
                "voxels":      dataset["voxels"][indices],
                "q_start":     dataset["q_start"][indices],
                "q_goal":      dataset["q_goal"][indices],
                "obstacles":   dataset["obstacles"][indices],       # <-- Añadido al split
                "n_obstacles": dataset["n_obstacles"][indices],     # <-- Añadido al split
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
    min_base_clearance: float = 0.25,
    grid_length: float = 2.5,
    n_vox: int = 128,
    workspace_radius: float = 1.0,
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
    
    # 1. Pre-calcular las coordenadas X e Y de la grilla 
    lin = np.linspace(-grid_length / 2, grid_length / 2, n_vox)
    x_grid, y_grid = np.meshgrid(lin, lin)
    dist_from_origin = np.sqrt(x_grid**2 + y_grid**2)

    for _ in tqdm(range(N), desc="Generating SDF dataset"):
        n_obs = int(rng.integers(1, 5))
        obs = sample_circular_obstacles(
            n_obstacles=n_obs, 
            r_min=0.06, 
            r_max=0.18, 
            min_base_clearance=min_base_clearance, 
            rng=rng
        )
        
        # 2. Construir la matriz de vóxeles y dibujar las formas orgánicas
        voxel_data = np.zeros((n_vox, n_vox), dtype=bool)
        for i in range(len(obs.x)):
            blob_mask = generate_blob_mask(
                x_grid, y_grid, 
                cx=obs.x[i], cy=obs.y[i], r_base=obs.r[i], 
                rng=rng
            )
            voxel_data = np.logical_or(voxel_data, blob_mask)
            
        # 3. Aplicar máscara de workspace para liberar espacio
        voxel_data[dist_from_origin > workspace_radius] = False
        
        # 4. Crear el objeto SquareGrid que necesita build_sdf_tensor
        grid_vox = SquareGrid.from_zero_centered(
            limits=(-grid_length / 2, grid_length / 2), 
            data=voxel_data
        )

        # 5. Generar y guardar el tensor
        sdfs.append(build_sdf_tensor(grid_vox))
        
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
    
    # Adaptación para aceptar tanto formato antiguo (obstacles) como nuevo (voxels)
    keys_to_print = ["sdf", "q_start", "q_goal"]
    if "voxels" in dataset:
        keys_to_print.append("voxels")
    if "obstacles" in dataset:
        keys_to_print.extend(["obstacles", "n_obstacles"])

    for key in keys_to_print:
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
