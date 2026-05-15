import torch
import torch.nn as nn


# ----------------------------
# B-Spline utilities
# ----------------------------

def _cox_de_boor(t, i, k, knots):
    if k == 0:
        return ((knots[i] <= t) & (t < knots[i + 1])).float()
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    term1 = 0 if denom1 == 0 else (t - knots[i]) / denom1 * _cox_de_boor(t, i, k - 1, knots)
    term2 = 0 if denom2 == 0 else (knots[i + k + 1] - t) / denom2 * _cox_de_boor(t, i + 1, k - 1, knots)
    return term1 + term2


def _bspline_basis(t_vals, C, degree, device="cpu"):
    """
    B-spline basis matrix [len(t_vals), C] with clamped uniform knots.
    The last row is corrected so the curve passes through the last
    control point at t=1 (standard clamped B-spline property).
    """
    knots = torch.linspace(0, 1, C - degree + 1, device=device)
    knots = torch.cat([torch.zeros(degree, device=device), knots, torch.ones(degree, device=device)])

    A = torch.zeros(len(t_vals), C, device=device)
    for ti, t in enumerate(t_vals):
        for i in range(C):
            A[ti, i] = _cox_de_boor(t, i, degree, knots)

    # Cox-de-Boor uses a half-open interval, so t=1 evaluates to 0 for
    # the last basis function. Fix by enforcing the clamped endpoint property.
    A[-1, :] = 0
    A[-1, -1] = 1
    return A


def build_bspline_matrix(T, C, degree=3, device="cpu"):
    """Evaluation matrix [T, C] for B-spline curve evaluation."""
    return _bspline_basis(torch.linspace(0, 1, T, device=device), C, degree, device)


def build_bspline_interpolation_matrix(T, C, degree=3, device="cpu"):
    """
    Returns M = A_eval @ inv(A_interp) ∈ [T, C].

    Given C waypoints P (including fixed start P[0] and goal P[-1]),
    the trajectory traj = M @ P is a B-spline that passes exactly
    through all C waypoints — including the endpoints.
    """
    A_interp = _bspline_basis(torch.linspace(0, 1, C, device=device), C, degree, device)
    A_eval   = _bspline_basis(torch.linspace(0, 1, T, device=device), C, degree, device)
    return A_eval @ torch.linalg.inv(A_interp)


# ----------------------------
# Encoder / Decoder modules
# ----------------------------

class EnvEncoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),   # 128 → 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, latent)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1).squeeze(-1))


class EnvDecoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.fc = nn.Linear(latent, 64 * 16 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),   # 64 → 128
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 16, 16))


class StateEncoder(nn.Module):
    def __init__(self, dof=3, hidden=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dof, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, q_start, q_goal):
        return self.net(torch.cat([q_start, q_goal], dim=-1))


class StateDecoder(nn.Module):
    def __init__(self, dof=3, latent_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 2 * dof),
        )

    def forward(self, z):
        x = self.net(z)
        return x[:, :3], x[:, 3:]


class WaypointDecoder(nn.Module):
    """Predicts C-2 interior waypoints (start and goal are fixed externally)."""
    def __init__(self, latent_env=64, latent_state=64, C=10, dof=3):
        super().__init__()
        self.C = C
        self.dof = dof
        self.mlp = nn.Sequential(
            nn.Linear(latent_env + latent_state, 256),
            nn.ReLU(),
            nn.Linear(256, (C - 2) * dof),
        )

    def forward(self, z_env, z_state):
        return self.mlp(torch.cat([z_env, z_state], dim=-1)).view(-1, self.C - 2, self.dof)


# ----------------------------
# Full Model: Warm Start Planner (B-spline interpolation)
# ----------------------------

class WarmStartPlanner(nn.Module):
    def __init__(self, dof=3, T=50, C=10):
        super().__init__()
        self.T = T
        self.C = C
        self.dof = dof

        self.env_encoder   = EnvEncoder()
        self.state_encoder = StateEncoder(dof=dof, latent_dim=64)
        self.decoder       = WaypointDecoder(dof=dof, C=C)

        # Precompute and store interpolation matrix as a buffer so it
        # moves to the correct device automatically with .to(device).
        self.register_buffer("M", build_bspline_interpolation_matrix(T, C, degree=3))

    def forward(self, q_start, q_goal, sdf):
        inner = self.decoder(self.env_encoder(sdf), self.state_encoder(q_start, q_goal))

        # Interpolation waypoints: start and goal are exact, network fills the interior
        waypoints = torch.cat([q_start.unsqueeze(1), inner, q_goal.unsqueeze(1)], dim=1)

        # traj[b, t] = M[t] @ waypoints[b] — passes exactly through all waypoints
        return torch.einsum("tc,bcd->btd", self.M, waypoints)


# ----------------------------
# Autoencoders
# ----------------------------

class StateAutoEncoder(nn.Module):
    def __init__(self, dof=3, latent_dim=64):
        super().__init__()
        self.encoder = StateEncoder(dof=dof, latent_dim=latent_dim)
        self.decoder = StateDecoder(dof=dof, latent_dim=latent_dim)

    def encode(self, q_start, q_goal):
        return self.encoder(q_start, q_goal)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, q_start, q_goal):
        z = self.encode(q_start, q_goal)
        q_start_rec, q_goal_rec = self.decode(z)
        return q_start_rec, q_goal_rec, z


class EnvAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = EnvEncoder(latent=latent_dim)
        self.decoder = EnvDecoder(latent=latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
