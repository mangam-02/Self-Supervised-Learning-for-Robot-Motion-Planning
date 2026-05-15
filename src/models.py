import torch
import torch.nn as nn


def build_bspline_matrix(T, C, degree=3, device="cpu"):
    """
    Builds a B-spline basis matrix A ∈ [T, C]
    Each row sums weighted cubic B-spline basis functions.
    """

    def cox_de_boor(t, i, k, knots):
        if k == 0:
            return ((knots[i] <= t) & (t < knots[i + 1])).float()
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = (
            0
            if denom1 == 0
            else (t - knots[i]) / denom1 * cox_de_boor(t, i, k - 1, knots)
        )
        term2 = (
            0
            if denom2 == 0
            else (knots[i + k + 1] - t) / denom2 * cox_de_boor(t, i + 1, k - 1, knots)
        )

        return term1 + term2

    # time grid
    t_vals = torch.linspace(0, 1, T, device=device)

    # clamped uniform knots
    knots = torch.linspace(0, 1, C - degree + 1, device=device)
    knots = torch.cat(
        [torch.zeros(degree, device=device), knots, torch.ones(degree, device=device)]
    )

    A = torch.zeros(T, C, device=device)

    for ti, t in enumerate(t_vals):
        for i in range(C):
            A[ti, i] = cox_de_boor(t, i, degree, knots)

    return A


class EnvEncoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(64, latent)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


class EnvDecoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()

        self.fc = nn.Linear(latent, 64 * 16 * 16)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),  # 64 → 128
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 16, 16)  # [B, 64, 16, 16]

        return self.net(x)


class StateEncoder(nn.Module):
    def __init__(self, dof=3, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * dof, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )

    def forward(self, q_start, q_goal):
        x = torch.cat([q_start, q_goal], dim=-1)
        return self.net(x)


class StateDecoder(nn.Module):
    def __init__(self, dof=3, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * dof)
        )

    def forward(self, z):
        x = self.net(z)

        # split back into start / goal
        q_start = x[:, :3]
        q_goal = x[:, 3:]

        return q_start, q_goal


class ControlPointDecoder(nn.Module):
    def __init__(self, latent_env=64, latent_state=128, C=10, dof=3):
        super().__init__()

        self.C = C
        self.dof = dof

        self.mlp = nn.Sequential(
            nn.Linear(latent_env + latent_state, 256),
            nn.ReLU(),
            nn.Linear(256, C * dof),
        )

    def forward(self, z_env, z_state):
        z = torch.cat([z_env, z_state], dim=-1)
        out = self.mlp(z)
        return out.view(-1, self.C, self.dof)


# ----------------------------
# Full Model: Warm Start Planner (B-spline + delta)
# ----------------------------


class WarmStartPlanner(nn.Module):
    def __init__(self, dof=3, T=50, C=10):
        super().__init__()

        self.T = T
        self.C = C
        self.dof = dof

        self.env_encoder = EnvEncoder()
        self.state_encoder = StateEncoder(dof=dof)
        self.decoder = ControlPointDecoder(dof=dof, C=C)

    def forward(self, q_start, q_goal, sdf):
        device = q_start.device

        # Encode inputs
        z_env = self.env_encoder(sdf)
        z_state = self.state_encoder(q_start, q_goal)

        # Predict delta control points
        delta_ctrl = self.decoder(z_env, z_state)
        print(f"Delta control points: {delta_ctrl}")

        # Baseline: straight line control points
        B = torch.linspace(0, 1, self.C, device=device).unsqueeze(-1)
        baseline = q_start.unsqueeze(1) * (1 - B) + q_goal.unsqueeze(1) * B

        # Combine baseline + delta
        ctrl_pts = baseline + delta_ctrl

        A = build_bspline_matrix(self.T, self.C, degree=3, device=device)
        traj = torch.einsum("tc,bcd->btd", A, ctrl_pts)

        # FIX: enforce correct endpoint

        traj[:, -1, :] = ctrl_pts[:, -1, :]

        return traj


class StateAutoEncoder(nn.Module):
    def __init__(self, dof=3, latent_dim=64):
        super().__init__()

        # -------------------------
        # Encoder
        # input: [q_start, q_goal]
        # shape: [B, 2*dof]
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(2 * dof, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # -------------------------
        # Decoder
        # latent -> reconstructed state
        # -------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * dof),
        )

    def encode(self, q_start, q_goal):
        x = torch.cat([q_start, q_goal], dim=-1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_rec = self.decoder(z)

        q_start_rec = x_rec[:, :3]
        q_goal_rec = x_rec[:, 3:]

        return q_start_rec, q_goal_rec

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
        x_rec = self.decode(z)
        return x_rec, z
