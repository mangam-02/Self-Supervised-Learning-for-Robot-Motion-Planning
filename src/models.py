import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from utils import forward_kinematics_torch

# ----------------------------
# B-Spline utilities
# ----------------------------

def _cox_de_boor(t, i, k, knots):
    """
    Cox-de-Boor is an algorithm that calculates the influence of each
    control point over the robot in a certain instant of time. 
    t: Current instant of time
    i: Control point evaluated
    k: Degree of the curve
    knots: Nodes vector
    """
    # 0-degree case
    if k == 0:
        return ((knots[i] <= t) & (t < knots[i + 1])).float()
    
    # 1, 2 and 3-degree case
    # Average the influence of the previous degree
    denom1 = knots[i + k] - knots[i]         # Width of the first block
    denom2 = knots[i + k + 1] - knots[i + 1] # Width of the second block

    # How the influence of this control point grows over time
    term1 = (t - knots[i]) / denom1 * _cox_de_boor(t, i, k - 1, knots) if denom1 != 0 else 0.0
    
    # How the influence of this control point decreases over time
    term2 = (knots[i + k + 1] - t) / denom2 * _cox_de_boor(t, i + 1, k - 1, knots) if denom2 != 0 else 0.0
    
    return term1 + term2

def _bspline_basis(t_vals, C, degree, device="cpu"):
    """
    B-spline basis matrix [len(t_vals), C] with clamped uniform knots.
    The last row is corrected so the curve passes through the last
    control point at t=1 (standard clamped B-spline property).
    t_vals: Instants of time
    C: Control points
    degree: Degree of the curve
    """
    # Configures the timeline in which the control points actuate
    knots = torch.linspace(0, 1, C - degree + 1, device=device)
    knots = torch.cat([torch.zeros(degree, device=device), knots, torch.ones(degree, device=device)])

    # Build the B-spline matrix
    A = torch.zeros(len(t_vals), C, device=device)
    for ti, t in enumerate(t_vals):
        for i in range(C):
            # Compute the influence of the i control point in ti
            A[ti, i] = _cox_de_boor(t, i, degree, knots)

    # Cox-de-Boor uses a half-open interval, so t=1 evaluates to 0 for
    # the last basis function. Fix by enforcing the clamped endpoint property.
    A[-1, :] = 0
    A[-1, -1] = 1
    return A

def build_bspline_matrix(T, C, degree=3, device="cpu"):
    """
    Evaluation matrix [T, C] for B-spline curve evaluation.
    T: Amount of instants of time
    C: Control points
    degree: Degree of the curve
    """
    # Return the matrix that does not force the trajectory to exactly go through
    # every intermediate control point
    return _bspline_basis(torch.linspace(0, 1, T, device=device), C, degree, device)

def build_bspline_interpolation_matrix(T, C, degree=3, device="cpu"):
    """
    Returns M = A_eval @ inv(A_interp) ∈ [T, C].

    Given C waypoints P (including fixed start P[0] and goal P[-1]),
    the trajectory traj = M @ P is a B-spline that passes exactly
    through all C waypoints — including the endpoints.
    T: Amount of instants of time
    C: Control points
    degree: Degree of the curve
    """
    # Return the matrix that force the trajectory to exactly go through every
    # intermediate control point
    A_interp = _bspline_basis(torch.linspace(0, 1, C, device=device), C, degree, device)
    A_eval   = _bspline_basis(torch.linspace(0, 1, T, device=device), C, degree, device)
    return A_eval @ torch.linalg.inv(A_interp)


# ----------------------------
# Encoder / Decoder modules
# ----------------------------

class EnvEncoder(nn.Module):
    """
    ResNet-18 adapted to process the SDF maps and compress them
    into a latent vector
    """
    def __init__(self, latent=64):
        super().__init__()

        # Load ResNet-18
        base_resnet = resnet18(weights=None)

        # Adapted architecture
        base_resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )

        # Extract the convolutional blocks and average pooling
        self.feature_extractor = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
            base_resnet.avgpool
        )
        
        # Linear output layer
        self.fc = nn.Linear(512, latent)

    def forward(self, x):
        # Data flow
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return self.fc(features)


class EnvDecoder(nn.Module):
    """
    CNN that takes the latent vector and rebuilds the SDF map
    """
    def __init__(self, latent=64):
        super().__init__()
        # From the latent vector, rebuild the
        # SDF map (128x128)
        self.fc = nn.Linear(latent, 256 * 4 * 4)
        self.net = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Extra refining
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        # Data flow
        return self.net(self.fc(z).view(-1, 256, 4, 4))


class StateEncoder(nn.Module):
    """
    FCN that processes the robot's movement task
    """
    def __init__(self, input_dim, hidden=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateDecoder(nn.Module):
    """
    FCN that unpackages the robot's movement task information from
    the latent vector
    """
    def __init__(self, output_dim, latent_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class WaypointDecoder(nn.Module):
    """
    Predicts C-2 interior waypoints (start and goal are fixed externally).
    """
    def __init__(self, latent_env=64, latent_state=12, C=10, dof=3):
        super().__init__()
        self.C = C
        self.dof = dof
        self.mlp = nn.Sequential(
            nn.Linear(latent_env + latent_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, (C - 2) * dof),
        )
        # Zero-init output layer so initial offsets are 0 -> trajectory starts as straight line
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_env, z_state):
        return self.mlp(torch.cat([z_env, z_state], dim=-1)).view(-1, self.C - 2, self.dof)


# ----------------------------
# Full Model: Warm Start Planner (B-spline interpolation)
# ----------------------------

class WarmStartPlanner(nn.Module):
    """
    Unifies the previous individual modules into a warm start trajectory
    planner.
    """
    def __init__(self, dof=3, T=50, C=10, linklengths=None):
        super().__init__()
        self.T   = T
        self.C   = C
        self.dof = dof

        ll = torch.tensor(linklengths, dtype=torch.float32) if linklengths is not None \
             else torch.ones(dof, dtype=torch.float32)
        self.register_buffer("linklengths", ll)

        # FK produces (dof+1) joint positions in 2D for each config
        fk_dim         = (dof + 1) * 2
        state_input_dim = 2 * dof + 2 * fk_dim   # q_start + q_goal + fk_start + fk_goal

        self.env_encoder   = EnvEncoder(latent=64)
        self.state_encoder = StateEncoder(input_dim=state_input_dim, latent_dim=12)
        self.decoder       = WaypointDecoder(latent_env=64, latent_state=12, dof=dof, C=C)

        self.register_buffer("M", build_bspline_interpolation_matrix(T, C, degree=3))

    def _state_features(self, q_start, q_goal):
        fk_start = forward_kinematics_torch(q_start, self.linklengths).flatten(1)  # [B, (dof+1)*2]
        fk_goal  = forward_kinematics_torch(q_goal,  self.linklengths).flatten(1)
        return torch.cat([q_start, q_goal, fk_start, fk_goal], dim=-1)

    def forward(self, q_start, q_goal, sdf):
        """Returns C waypoints [B, C, dof] — use .trajectory() to get the full B-spline."""
        state = self._state_features(q_start, q_goal)
        if sdf.ndim == 3:
            sdf = sdf.unsqueeze(1)
        offset = self.decoder(self.env_encoder(sdf), self.state_encoder(state))

        # Linear interpolation as baseline — offset=0 means straight-line trajectory
        t_vals = torch.linspace(0, 1, self.C, device=q_start.device)[1:-1]  # [C-2]
        baseline = q_start.unsqueeze(1) + t_vals.view(1, -1, 1) * (q_goal - q_start).unsqueeze(1)
        inner = baseline + offset

        return torch.cat([q_start.unsqueeze(1), inner, q_goal.unsqueeze(1)], dim=1)

    def trajectory(self, waypoints):
        """Evaluates the B-spline at T timesteps. Input: [B, C, dof] → Output: [B, T, dof]."""
        return torch.einsum("tc,bcd->btd", self.M, waypoints)


# ----------------------------
# Autoencoders
# ----------------------------

class StateAutoEncoder(nn.Module):
    """
    Learns a compressed representation of the robot's movement task.
    Encoder input: [q_start, q_goal, fk_start, fk_goal]
    Decoder output: [q_start, q_goal]  (FK is deterministic from q, no need to reconstruct)
    """
    def __init__(self, dof=3, latent_dim=64, linklengths=None):
        super().__init__()
        ll = torch.tensor(linklengths, dtype=torch.float32) if linklengths is not None \
             else torch.ones(dof, dtype=torch.float32)
        self.register_buffer("linklengths", ll)

        fk_dim   = (dof + 1) * 2
        feat_dim = 2 * dof + 2 * fk_dim   # [q_start, q_goal, fk_start, fk_goal]
        self.encoder = StateEncoder(input_dim=feat_dim, latent_dim=latent_dim)
        self.decoder = StateDecoder(output_dim=feat_dim, latent_dim=latent_dim)

    def _features(self, q_start, q_goal):
        fk_start = forward_kinematics_torch(q_start, self.linklengths).flatten(1)
        fk_goal  = forward_kinematics_torch(q_goal,  self.linklengths).flatten(1)
        return torch.cat([q_start, q_goal, fk_start, fk_goal], dim=-1)

    def encode(self, q_start, q_goal):
        return self.encoder(self._features(q_start, q_goal))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, q_start, q_goal):
        features = self._features(q_start, q_goal)
        z        = self.encoder(features)
        rec      = self.decoder(z)
        return rec, features, z


class EnvAutoEncoder(nn.Module):
    """
    Learns a compressed representation of the environment.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # Define the encoder and decoder
        self.encoder = EnvEncoder(latent=latent_dim)
        self.decoder = EnvDecoder(latent=latent_dim)

    def encode(self, x):
        # SDF map -> latent space
        return self.encoder(x)

    def decode(self, z):
        # latent space -> SDF map
        return self.decoder(z)

    def forward(self, x):
        # Training flow
        z = self.encode(x)
        return self.decoder(z), z
