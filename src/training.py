import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from simplearm.robot import RobotInfo

from models import WarmStartPlanner, StateAutoEncoder, EnvAutoEncoder
from losses import (
    compute_trajectory_collision_cost,
    compute_trajectory_joint_limits_cost,
    compute_trajectory_velocity_cost,
    compute_smoothness_cost,
)


def _load(dataset: dict | str) -> dict:
    if isinstance(dataset, str):
        return torch.load(dataset, weights_only=False)
    return dataset


def train(
    train_dataset: dict | str,
    val_dataset: dict | str | None = None,
    # Model
    T: int = 50,
    C: int = 10,
    # Training
    n_epochs: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    # Loss weights
    w_coll: float = 1.0,
    w_joints: float = 0.1,
    w_smooth: float = 0.01,
    w_velocity: float = 0.1,
    # Loss parameters
    collision_eps: float = 0.1,
    dt: float = 0.1,
    # Misc
    log_every: int = 100,
    device: str | None = None,
    save_path: str | None = None,
) -> tuple[WarmStartPlanner, dict]:
    """
    Train a WarmStartPlanner on a pre-generated dataset.

    Returns the trained model and a history dict with keys
    'train' (and 'val' if val_dataset is provided).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────────
    train_ds = _load(train_dataset)
    val_ds   = _load(val_dataset) if val_dataset is not None else None

    meta        = train_ds["metadata"]
    grid_length = meta["grid_length"]
    robot       = RobotInfo.from_linklengths(meta["linklengths"], sphere_rad=meta["sphere_rad"])
    q_min       = torch.tensor(meta["q_min"], dtype=torch.float32, device=device)
    q_max       = torch.tensor(meta["q_max"], dtype=torch.float32, device=device)
    # Joint 0 is a continuous revolute joint with no physical limits — exclude it
    q_min[0]    = float("-inf")
    q_max[0]    = float("inf")
    dof         = meta["dof"]

    sdf     = train_ds["sdf"].to(device)
    q_start = train_ds["q_start"].to(device)
    q_goal  = train_ds["q_goal"].to(device)
    N       = sdf.shape[0]

    if val_ds is not None:
        val_sdf     = val_ds["sdf"].to(device)
        val_q_start = val_ds["q_start"].to(device)
        val_q_goal  = val_ds["q_goal"].to(device)

    # ── Model & optimizer ────────────────────────────────────────────────────
    model     = WarmStartPlanner(dof=dof, T=T, C=C, linklengths=meta["linklengths"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def compute_losses(waypoints, sdf_batch):
        traj = model.trajectory(waypoints)          # [B, T, dof] — full B-spline
        l_coll  = compute_trajectory_collision_cost(traj, sdf_batch, robot, grid_length=grid_length, eps=collision_eps, weight=w_coll)
        l_joint = compute_trajectory_joint_limits_cost(traj, q_min, q_max, weight=w_joints)
        l_vel   = compute_trajectory_velocity_cost(traj, dt=dt, weight=w_velocity)
        l_smooth= compute_smoothness_cost(traj, dt=dt, weight=w_smooth)
        return l_coll + l_joint + l_vel + l_smooth, l_coll, l_joint, l_vel, l_smooth

    # ── Loss scale diagnostics ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _wp   = model(q_start[:min(batch_size, N)], q_goal[:min(batch_size, N)], sdf[:min(batch_size, N)])
        _traj = model.trajectory(_wp)
        _raw  = {
            "coll":   compute_trajectory_collision_cost(_traj, sdf[:min(batch_size, N)], robot, grid_length=grid_length, eps=collision_eps, weight=1.0),
            "joints": compute_trajectory_joint_limits_cost(_traj, q_min, q_max, weight=1.0),
            "vel":    compute_trajectory_velocity_cost(_traj, dt=dt, weight=1.0),
            "smooth": compute_smoothness_cost(_traj, dt=dt, weight=1.0),
        }
        _w = {"coll": w_coll, "joints": w_joints, "vel": w_velocity, "smooth": w_smooth}
    print(f"\n{'Loss':<10} {'Raw':>10}  {'Weight':>10}  {'Weighted':>10}")
    print("─" * 46)
    for name, val in _raw.items():
        weighted = _w[name] * val.item()
        print(f"{name:<10} {val.item():>10.4f}  {_w[name]:>10.4f}  {weighted:>10.4f}")
    print()

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train": [], "val": []}

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()

        if N <= batch_size:
            # Small dataset: use all samples every epoch (no noise from sampling)
            idx = torch.arange(N, device=device)
        else:
            idx = torch.randperm(N, device=device)[:batch_size]
        waypoints = model(q_start[idx], q_goal[idx], sdf[idx])
        loss, l_coll, l_joint, l_vel, l_smooth = compute_losses(waypoints, sdf[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["train"].append(loss.item())

        if val_ds is not None and (epoch + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                val_wp  = model(val_q_start, val_q_goal, val_sdf)
                val_loss, *_ = compute_losses(val_wp, val_sdf)
            history["val"].append(val_loss.item())
            model.train()

        if (epoch + 1) % log_every == 0:
            msg = (f"Epoch {epoch+1:5d} | train={loss.item():.4f}  "
                   f"coll={l_coll.item():.4f}  joints={l_joint.item():.4f}  "
                   f"vel={l_vel.item():.4f}  smooth={l_smooth.item():.4f}")
            if history["val"]:
                msg += f"  val={history['val'][-1]:.4f}"
            tqdm.write(msg)

    # ── Loss plot ────────────────────────────────────────────────────────────
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

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

    return model, history


def train_state_autoencoder(
    linklengths: list[float],
    dof: int = 3,
    latent_dim: int = 64,
    N: int = 120_000,
    batch_size: int = 64,
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

    def _eval(indices):
        model.eval()
        with torch.no_grad():
            rec, features, _ = model(q_start_data[indices].to(device), q_goal_data[indices].to(device))
            return F.mse_loss(rec, features).item()

    history = {"train": [], "val": []}

    for epoch in tqdm(range(epochs), desc="Training State Autoencoder"):
        model.train()
        idx = train_idx[torch.randint(0, len(train_idx), (batch_size,))]
        rec, features, _ = model(q_start_data[idx].to(device), q_goal_data[idx].to(device))
        loss = F.mse_loss(rec, features)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["train"].append(loss.item())
        if epoch % log_every == 0:
            history["val"].append(_eval(val_idx))

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

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        enc_path = os.path.join(os.path.dirname(save_path), "state_encoder.pt")
        torch.save(model.encoder.state_dict(), enc_path)
        print(f"Saved: {save_path}  (encoder → {enc_path})")

    return model, history


def train_env_autoencoder(
    dataset_path: str,
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

    def _eval(indices):
        model.eval()
        with torch.no_grad():
            sdf = sdf_dataset[indices].to(device)
            sdf_rec, _ = model(sdf)
            return F.mse_loss(sdf_rec, sdf).item()

    history = {"train": [], "val": []}

    for epoch in tqdm(range(epochs), desc="Training Env Autoencoder"):
        model.train()
        idx = train_idx[torch.randint(0, len(train_idx), (batch_size,))]
        sdf = sdf_dataset[idx].to(device)
        sdf_rec, _ = model(sdf)
        loss = F.mse_loss(sdf_rec, sdf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["train"].append(loss.item())
        if epoch % log_every == 0:
            history["val"].append(_eval(val_idx))

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

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        enc_path = os.path.join(os.path.dirname(save_path), "env_encoder.pt")
        torch.save(model.encoder.state_dict(), enc_path)
        print(f"Saved: {save_path}  (encoder → {enc_path})")

    return model, history


