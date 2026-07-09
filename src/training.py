import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from simplearm.robot import RobotInfo

from models import WarmStartPlanner, StateAutoEncoder, EnvAutoEncoder
from losses import (
    compute_trajectory_collision_cost,
    compute_trajectory_max_collision_cost,
    compute_trajectory_joint_limits_cost,
    compute_smoothness_cost,
    compute_waypoint_spacing_cost,
    compute_exploration_cost,
)


def _load(dataset: dict | str) -> dict:
    """
    Load the dataset to use
    """
    if isinstance(dataset, str):
        return torch.load(dataset, weights_only=False)
    return dataset


def train(
    train_dataset: dict | str,
    val_dataset: dict | str | None = None,
    # Model
    T: int = 50,
    C: int = 10,
    # Pre-trained encoders
    env_encoder_path: str | None = None,
    state_encoder_path: str | None = None,
    freeze_encoders: bool = False,
    # Training
    n_epochs: int = 5000,
    batch_size: int = 64,
    lr: float = 1e-3,
    # Loss weights
    w_coll: float = 1.0,
    w_joints: float = 0.1,
    w_smooth: float = 0.01,
    w_spacing: float = 0.1,
    w_explore: float = 0.1,
    explore_threshold: float = 0.5,
    # Loss parameters
    collision_eps: float = 0.1,
    dt: float = 0.1,
    # Misc
    log_every: int = 100,
    device: str | None = None,
    save_path: str | None = None,
    save_env_encoder_path: str | None = None,
    save_state_encoder_path: str | None = None,
) -> tuple[WarmStartPlanner, dict]:
    """
    Train a WarmStartPlanner on a pre-generated dataset.

    Returns the trained model and a history dict with keys
    'train' (and 'val' if val_dataset is provided).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_ds = _load(train_dataset)
    val_ds   = _load(val_dataset) if val_dataset is not None else None
    meta        = train_ds["metadata"]
    grid_length = meta["grid_length"]
    robot       = RobotInfo.from_linklengths(meta["linklengths"], sphere_rad=meta["sphere_rad"])
    robot.sphere_rad = meta["sphere_rad"]
    q_min       = torch.tensor(meta["q_min"], dtype=torch.float32, device=device)
    q_max       = torch.tensor(meta["q_max"], dtype=torch.float32, device=device)
    q_min[0]    = -2 * torch.pi
    q_max[0]    =  2 * torch.pi
    dof         = meta["dof"]
    sdf     = train_ds["sdf"].to(device)
    q_start = train_ds["q_start"].to(device)
    q_goal  = train_ds["q_goal"].to(device)
    N       = sdf.shape[0]
    if val_ds is not None:
        val_sdf     = val_ds["sdf"].to(device)
        val_q_start = val_ds["q_start"].to(device)
        val_q_goal  = val_ds["q_goal"].to(device)
        N_val       = val_sdf.shape[0]

    # Model
    model = WarmStartPlanner(dof=dof, T=T, C=C, linklengths=meta["linklengths"]).to(device)

    # Load autoencoders
    if env_encoder_path is not None:
        model.env_encoder.load_state_dict(
            torch.load(env_encoder_path, map_location=device, weights_only=True)
        )
        print(f"Loaded env encoder:   {env_encoder_path}")
        if freeze_encoders:
            for p in model.env_encoder.parameters():
                p.requires_grad_(False)

    if state_encoder_path is not None:
        model.state_encoder.load_state_dict(
            torch.load(state_encoder_path, map_location=device, weights_only=True)
        )
        print(f"Loaded state encoder: {state_encoder_path}")
        if freeze_encoders:
            for p in model.state_encoder.parameters():
                p.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    # Decreasing learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Compute losses
    def compute_losses(waypoints, sdf_batch, q_s, q_g):
        traj      = model.trajectory(waypoints) # [B, T, dof] — full B-spline
        l_coll    = compute_trajectory_max_collision_cost(traj, sdf_batch, robot, grid_length=grid_length, eps=collision_eps, weight=w_coll)
        l_joint   = compute_trajectory_joint_limits_cost(traj, q_min, q_max, weight=w_joints)
        l_smooth  = compute_smoothness_cost(traj, dt=dt, weight=w_smooth)
        l_space   = compute_waypoint_spacing_cost(waypoints, weight=w_spacing)
        l_explore = compute_exploration_cost(waypoints, q_s, q_g, l_coll.detach(),
                                             threshold=explore_threshold, weight=w_explore)
        return l_coll + l_joint + l_smooth + l_space + l_explore, l_coll, l_joint, l_smooth, l_space, l_explore

    # Loss scale diagnostics
    model.eval()
    with torch.no_grad():
        _wp   = model(q_start[:min(batch_size, N)], q_goal[:min(batch_size, N)], sdf[:min(batch_size, N)])
        _traj = model.trajectory(_wp)
        _raw  = {
            "coll":    compute_trajectory_collision_cost(_traj, sdf[:min(batch_size, N)], robot, grid_length=grid_length, eps=collision_eps, weight=1.0),
            "joints":  compute_trajectory_joint_limits_cost(_traj, q_min, q_max, weight=1.0),
            "smooth":  compute_smoothness_cost(_traj, dt=dt, weight=1.0),
            "spacing": compute_waypoint_spacing_cost(_wp, weight=1.0),
            "explore": compute_exploration_cost(_wp, q_start[:min(batch_size, N)], q_goal[:min(batch_size, N)], torch.tensor(1.0), threshold=explore_threshold, weight=1.0),
        }
        _w = {"coll": w_coll, "joints": w_joints, "smooth": w_smooth, "spacing": w_spacing, "explore": w_explore}
    print(f"\n{'Loss':<10} {'Raw':>10}  {'Weight':>10}  {'Weighted':>10}")
    print("─" * 46)
    for name, val in _raw.items():
        weighted = _w[name] * val.item()
        print(f"{name:<10} {val.item():>10.4f}  {_w[name]:>10.4f}  {weighted:>10.4f}")
    print()

    # Training and validation loop
    history = {"train": [], "val": []}

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()

        if N <= batch_size:
            # Small dataset: use all samples every epoch
            idx = torch.arange(N, device=device)
        else:
            idx = torch.randperm(N, device=device)[:batch_size]
        waypoints = model(q_start[idx], q_goal[idx], sdf[idx])
        loss, l_coll, l_joint, l_smooth, l_space, l_explore = compute_losses(waypoints, sdf[idx], q_start[idx], q_goal[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["train"].append(loss.item())

        scheduler.step()

        if val_ds is not None and (epoch + 1) % log_every == 0:
            model.eval()
            val_loss_total = 0.0

            with torch.no_grad():
                num_batches = (N_val + batch_size - 1) // batch_size
                for b in range(num_batches):
                    b_start = b * batch_size
                    b_end = min((b + 1) * batch_size, N_val)
                    
                    val_wp_batch = model(val_q_start[b_start:b_end], val_q_goal[b_start:b_end], val_sdf[b_start:b_end])
                    v_loss, *_ = compute_losses(val_wp_batch, val_sdf[b_start:b_end], val_q_start[b_start:b_end], val_q_goal[b_start:b_end])
                    val_loss_total += v_loss.item() * (b_end - b_start)
                    
                val_loss_avg = val_loss_total / N_val
            history["val"].append(val_loss_avg)
            model.train()

        if (epoch + 1) % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch+1:5d} | lr={current_lr:.2e} | train={loss.item():.4f}  "
                   f"coll={l_coll.item():.4f}  joints={l_joint.item():.4f}  "
                   f"smooth={l_smooth.item():.4f}  spacing={l_space.item():.4f}  "
                   f"explore={l_explore.item():.4f}")
            if val_ds is not None and len(history["val"]) > 0:
                msg += f"  val={history['val'][-1]:.4f}"
            tqdm.write(msg)

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(history["train"], label="train", linewidth=0.8)
    if history["val"]:
        x_val = np.linspace(0, len(history["train"]), len(history["val"]))
        ax.plot(x_val, history["val"], label="val", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WarmStartPlanner Training")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")
    if save_env_encoder_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_env_encoder_path)), exist_ok=True)
        torch.save(model.env_encoder.state_dict(), save_env_encoder_path)
        print(f"Env encoder saved: {save_env_encoder_path}")
    if save_state_encoder_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_state_encoder_path)), exist_ok=True)
        torch.save(model.state_encoder.state_dict(), save_state_encoder_path)
        print(f"State encoder saved: {save_state_encoder_path}")

    return model, history


def train_state_autoencoder(
    file_name: str,
    linklengths: list[float],
    dof: int = 3,
    latent_dim: int = 12,
    N: int = 120_000,
    batch_size: int = 256,
    epochs: int = 5000,
    lr: float = 1e-3,
    log_every: int = 50,
    device: str | None = None,
    save_path: str | None = None,
) -> tuple[StateAutoEncoder, dict]:
    """
    Train a StateAutoEncoder on randomly sampled joint configurations.
    Input/output: [q_start, q_goal, fk_start, fk_goal] — no environment needed.
    save_path: full state_dict path; encoder is saved alongside as state_encoder.pt.
    Returns the trained model and a history dict with 'train' and 'val' keys.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    q_start_data = (torch.rand(N, dof) * 4 * np.pi - 2 * np.pi).float()
    q_goal_data  = (torch.rand(N, dof) * 4 * np.pi - 2 * np.pi).float()

    perm      = torch.randperm(N)
    n_train   = int(N * 0.75)
    n_val     = int(N * 0.125)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    model = StateAutoEncoder(dof=dof, latent_dim=latent_dim, linklengths=linklengths).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def _eval(indices):
        model.eval()
        with torch.no_grad():
            rec, features, _ = model(q_start_data[indices].to(device), q_goal_data[indices].to(device))
            return F.mse_loss(rec, features).item()

    history = {"train": [], "val": []}

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training State Autoencoder"):
        model.train()
        idx = train_idx[torch.randint(0, len(train_idx), (batch_size,))]
        rec, features, _ = model(q_start_data[idx].to(device), q_goal_data[idx].to(device))
        loss = F.mse_loss(rec, features)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history["train"].append(loss.item())
        if epoch % log_every == 0:
            history["val"].append(_eval(val_idx))

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(history["train"], label="train", linewidth=0.8)
    if history["val"]:
        ax.plot(np.linspace(0, len(history["train"]), len(history["val"])), history["val"], label="val", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("State Autoencoder Training")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Test loss: {_eval(test_idx):.6f}")

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        enc_path = os.path.join(os.path.dirname(save_path), file_name)
        torch.save(model.encoder.state_dict(), enc_path)
        print(f"Saved: {save_path}  (encoder -> {enc_path})")

    return model, history


def train_env_autoencoder(
    dataset_path: str,
    file_name: str,
    latent_dim: int = 64,
    batch_size: int = 64,
    epochs: int = 5000,
    lr: float = 1e-3,
    log_every: int = 50,
    device: str | None = None,
    save_path: str | None = None,
) -> tuple[EnvAutoEncoder, dict]:
    """
    Train an EnvAutoEncoder on SDF images.
    dataset_path: path to a dataset file containing a dict with key "sdf" ([N, 1, H, W]).
    save_path: full state_dict path; encoder is saved alongside as env_encoder.pt.
    Returns the trained model and a history dict with 'train' and 'val' keys.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _data       = _load(dataset_path)
    sdf_dataset = _data["sdf"] if isinstance(_data, dict) else _data
    N           = len(sdf_dataset)
    perm      = torch.randperm(N)
    n_train   = int(N * 0.75)
    n_val     = int(N * 0.125)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    model = EnvAutoEncoder(latent_dim=latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    def weighted_sdf_loss(pred, target):
        """
        Stronger penalization in areas near to the obstacles.
        """
        weight = torch.where(target < 0.15, 5.0, 1.0)
        return (weight * (pred - target) ** 2).mean()
    
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def _eval(indices):
        model.eval()
        with torch.no_grad():
            sdf = sdf_dataset[indices].to(device)
            sdf_rec, _ = model(sdf)
            return weighted_sdf_loss(sdf_rec, sdf).item()

    history = {"train": [], "val": []}

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Env Autoencoder"):
        model.train()
        idx = train_idx[torch.randint(0, len(train_idx), (batch_size,))]
        sdf = sdf_dataset[idx].to(device)
        sdf_rec, _ = model(sdf)
        loss = weighted_sdf_loss(sdf_rec, sdf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        history["train"].append(loss.item())
        if epoch % log_every == 0:
            history["val"].append(_eval(val_idx))

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(history["train"], label="train", linewidth=0.8)
    if history["val"]:
        ax.plot(np.linspace(0, len(history["train"]), len(history["val"])), history["val"], label="val", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Environment Autoencoder Training")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Test loss: {_eval(test_idx):.6f}")

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        enc_path = os.path.join(os.path.dirname(save_path), file_name)
        torch.save(model.encoder.state_dict(), enc_path)
        print(f"Saved: {save_path}  (encoder -> {enc_path})")

    return model, history


