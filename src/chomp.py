"""
chomp.py — CHOMP trajectory optimizer.

A classical, learning-free trajectory optimizer built from the same differentiable
cost terms (losses.py) the WarmStartPlanner is trained on, so it can be pointed at
that model's exact loss weights and compared head-to-head. It can also refine an
existing trajectory (e.g. the model's output): run the model first, then keep
optimizing. Note the weights are constructor arguments, not defaults — see the
CHOMPOptimizer docstring.

CHOMP (Covariant Hamiltonian Optimization for Motion Planning) minimizes
    U(xi) = collision(xi) + joint_limits(xi) + smoothness(xi)
over a discretized trajectory xi in [B, T, dof]. Its defining feature is the
*covariant* update: rather than a plain Euclidean step xi -= alpha * grad(U),
the gradient is preconditioned by the inverse of a smoothness metric A built
from finite differencing, so a change at one waypoint is spread smoothly over
its neighbours:

    xi_{k+1} = xi_k - (1/eta) * A^{-1} grad(U(xi_k))

A = K^T K, with K the second-difference (acceleration) operator. Start and goal
are held fixed: the metric and update act on the interior waypoints only.

Requires SimpleArm on sys.path before importing this module:
    sys.path.insert(0, "/path/to/SimpleArm/src")
"""

import numpy as np
import torch

from simplearm.robot import RobotInfo

from models import build_bspline_interpolation_matrix
from evaluation import surface_clearance
from losses import (
    compute_trajectory_collision_cost,
    compute_trajectory_max_collision_cost,
    compute_trajectory_joint_limits_cost,
    compute_smoothness_cost,
    compute_exploration_cost,
    compute_waypoint_spacing_cost,
)


def _build_smoothness_metric(T: int, eps_reg: float = 1e-6,
                             device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Inverse smoothness metric A^{-1} restricted to the T-2 interior waypoints.

    K is the second-difference (acceleration) operator of shape [T-2, T] whose
    rows encode q[t-1] - 2 q[t] + q[t+1]. A = K^T K is the [T, T] smoothness
    metric; we keep its interior block [T-2, T-2] (start/goal are fixed) and
    invert it once. eps_reg keeps the matrix well-conditioned.
    """
    K = torch.zeros(T - 2, T, device=device, dtype=dtype)
    for i in range(T - 2):
        K[i, i]     = 1.0
        K[i, i + 1] = -2.0
        K[i, i + 2] = 1.0
    A = K.t() @ K                                   # [T, T]
    A_inner = A[1:-1, 1:-1]                          # [T-2, T-2] — interior only
    A_inner = A_inner + eps_reg * torch.eye(T - 2, device=device, dtype=dtype)
    return torch.linalg.inv(A_inner)                # [T-2, T-2]


class CHOMPOptimizer:
    """
    CHOMP trajectory optimizer over a dense [B, T, dof] trajectory.

    The weights below (`eps`, `w_coll`, `w_smooth`, ...) drive the covariant
    gradient descent. Their defaults are tuned for single-trajectory CHOMP (tight
    collision band, light smoothness) and are NOT copied from any training run.

    To compare CHOMP against a trained model fairly, pass that model's own loss
    weights instead of relying on these defaults — otherwise the two are pursuing
    different objectives and the comparison says little. See chomp_test.ipynb.
    Note that `eta` couples to the weight scale: the step is (1/eta) * A^-1 * grad,
    so unlike Adam this descent is not invariant to scaling the cost.

    Scoring a trajectory is a separate concern — use evaluation.TrajectoryEvaluator.
    """

    def __init__(
        self,
        robot: RobotInfo,
        grid_length: float,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        T: int = 50,
        dt: float = 0.1,
        # --- Optimization hyperparameters (tuned for single-trajectory CHOMP) ---
        # eps (danger-zone half-width) is kept comfortably above sphere_rad so the
        # optimizer pushes sphere SURFACES clear of obstacles, not just centers.
        eps: float = 0.2,
        w_coll: float = 5.0,
        w_joints: float = 0.1,
        w_smooth: float = 0.03,
        ccd: bool = True,
        joint_weight_decay: float = 0.5,
        collision_agg: str = "sum",
        w_explore: float = 0.0,
        w_spacing: float = 0.0,
        explore_threshold: float = 0.5,
        eta: float = 1500.0,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.robot       = robot
        self.grid_length = grid_length
        self.q_min       = q_min.to(device)
        self.q_max       = q_max.to(device)
        self.T           = T
        self.dt          = dt
        self.eps         = eps
        self.w_coll      = w_coll
        self.w_joints    = w_joints
        self.w_smooth    = w_smooth
        self.ccd         = ccd
        self.joint_weight_decay = joint_weight_decay
        # 'sum' penalizes every colliding waypoint -> gradient flows at all of
        # them, which a trajectory optimizer needs. 'max' only carries gradient
        # through the single worst timestep, so it optimizes very slowly.
        if collision_agg not in ("sum", "max"):
            raise ValueError("collision_agg must be 'sum' or 'max'")
        self.collision_agg = collision_agg
        # Optional optimization aids against CHOMP's classic local minima (off by
        # default -> 'pure CHOMP' baseline). exploration pushes the trajectory to
        # deviate MORE from the straight line while it is in collision (so it
        # routes AROUND instead of tunnelling through); spacing penalizes uneven
        # waypoint spacing (so it cannot rush through an obstacle). These are
        # search aids only and are NOT part of the comparison metric.
        self.w_explore   = w_explore
        self.w_spacing   = w_spacing
        self.explore_threshold = explore_threshold
        self.eta         = eta

        # Covariant-update metric — precomputed once, shared across the batch.
        self.A_inv = _build_smoothness_metric(T, device=device)

    @classmethod
    def from_metadata(cls, meta: dict, **overrides) -> "CHOMPOptimizer":
        """
        Build an optimizer straight from a dataset's metadata dict, mirroring the
        robot / joint-limit setup in training.train (including the wider base-joint
        limits). `overrides` forwards any __init__ kwarg (T, eps, eta, weights, ...).
        """
        device = overrides.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        robot  = RobotInfo.from_linklengths(meta["linklengths"], sphere_rad=meta["sphere_rad"])
        robot.sphere_rad = meta["sphere_rad"]

        q_min = torch.tensor(meta["q_min"], dtype=torch.float32, device=device)
        q_max = torch.tensor(meta["q_max"], dtype=torch.float32, device=device)
        # Base joint is a free revolute joint — widen its limits as in training.
        q_min[0] = -2 * torch.pi
        q_max[0] =  2 * torch.pi

        return cls(
            robot=robot,
            grid_length=meta["grid_length"],
            q_min=q_min,
            q_max=q_max,
            device=device,
            **overrides,
        )

    # ── Cost ────────────────────────────────────────────────────────────────

    def _collision_cost(self, traj, sdf, agg, eps=None, weight=None):
        eps    = self.eps    if eps    is None else eps
        weight = self.w_coll if weight is None else weight
        fn = compute_trajectory_collision_cost if agg == "sum" \
            else compute_trajectory_max_collision_cost
        return fn(
            traj, sdf, self.robot, grid_length=self.grid_length, eps=eps,
            weight=weight, ccd=self.ccd, joint_weight_decay=self.joint_weight_decay,
        )

    def _compute_cost(self, traj: torch.Tensor, sdf: torch.Tensor):
        """
        Weighted optimization cost + per-term breakdown for a [B, T, dof]
        trajectory, using the configured collision aggregation. Includes the
        optional exploration / spacing search aids when their weights are > 0.
        Returns (total, terms) where terms always has {"coll", "joints",
        "smooth"} and adds {"explore", "spacing"} when enabled.
        """
        l_coll   = self._collision_cost(traj, sdf, self.collision_agg)
        l_joint  = compute_trajectory_joint_limits_cost(traj, self.q_min, self.q_max, weight=self.w_joints)
        l_smooth = compute_smoothness_cost(traj, dt=self.dt, weight=self.w_smooth)
        total    = l_coll + l_joint + l_smooth
        terms    = {"coll": l_coll, "joints": l_joint, "smooth": l_smooth}

        # The dense trajectory's own endpoints are q_start / q_goal, so the
        # straight-line baseline is reconstructed from traj[:, 0] and traj[:, -1].
        if self.w_explore > 0:
            # Gate with the MAX-based collision (as in training): it is ~10x
            # larger than the sum-based one during a deep collision, so the push
            # to deviate from the straight line is strong enough to escape the
            # local minimum. A sum-based gate is far too weak here.
            coll_gate = (l_coll if self.collision_agg == "max"
                         else self._collision_cost(traj, sdf, "max")).detach()
            l_explore = compute_exploration_cost(
                traj, traj[:, 0, :], traj[:, -1, :], coll_gate,
                threshold=self.explore_threshold, weight=self.w_explore,
            )
            total = total + l_explore
            terms["explore"] = l_explore
        if self.w_spacing > 0:
            l_space = compute_waypoint_spacing_cost(traj, weight=self.w_spacing)
            total = total + l_space
            terms["spacing"] = l_space

        return total, terms

    @torch.no_grad()
    def _surface_clearance(self, traj: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """Sphere-surface clearance [B, T*N] — used by the collision-free stop."""
        return surface_clearance(traj, sdf, self.robot, self.grid_length)

    # ── Initialisation helpers ───────────────────────────────────────────────

    def _as_batched_traj(self, traj: torch.Tensor) -> torch.Tensor:
        traj = traj.to(self.device).float()
        if traj.ndim == 2:           # [T, dof] -> [1, T, dof]
            traj = traj.unsqueeze(0)
        return traj

    def _as_batched_sdf(self, sdf: torch.Tensor, B: int) -> torch.Tensor:
        sdf = sdf.to(self.device).float()
        if sdf.ndim == 2:            # [H, W] -> [1, 1, H, W]
            sdf = sdf.unsqueeze(0).unsqueeze(0)
        elif sdf.ndim == 3:          # [1, H, W] -> [1, 1, H, W]
            sdf = sdf.unsqueeze(0)
        if sdf.shape[0] == 1 and B > 1:
            sdf = sdf.expand(B, -1, -1, -1)
        return sdf

    def init_trajectory(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        init_traj: torch.Tensor | None = None,
        init_waypoints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the [B, T, dof] starting trajectory from one of:
          - init_traj:       a full dense trajectory [B, T, dof] or [T, dof].
          - init_waypoints:  control waypoints [B, C, dof] -> dense trajectory via
                             the same clamped B-spline the model uses (model-output path).
          - neither:         straight-line interpolation between q_start and q_goal.
        Endpoints are always snapped to q_start / q_goal.
        """
        q_start = q_start.to(self.device).float()
        q_goal  = q_goal.to(self.device).float()
        if q_start.ndim == 1:
            q_start = q_start.unsqueeze(0)
            q_goal  = q_goal.unsqueeze(0)
        B = q_start.shape[0]

        if init_traj is not None:
            traj = self._as_batched_traj(init_traj).clone()
        elif init_waypoints is not None:
            wp = init_waypoints.to(self.device).float()
            if wp.ndim == 2:
                wp = wp.unsqueeze(0)
            C = wp.shape[1]
            M = build_bspline_interpolation_matrix(self.T, C, degree=3, device=self.device)
            traj = torch.einsum("tc,bcd->btd", M, wp)
        else:
            t = torch.linspace(0, 1, self.T, device=self.device).view(1, -1, 1)
            traj = q_start.unsqueeze(1) + t * (q_goal - q_start).unsqueeze(1)

        # Pin the endpoints exactly.
        traj[:, 0, :]  = q_start
        traj[:, -1, :] = q_goal
        return traj

    # ── Optimization ──────────────────────────────────────────────────────────

    def optimize(
        self,
        sdf: torch.Tensor,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        init_traj: torch.Tensor | None = None,
        init_waypoints: torch.Tensor | None = None,
        max_iters: int = 500,
        eta: float | None = None,
        max_backtracks: int = 12,
        tol: float = 1e-3,
        patience: int = 15,
        stop_at_collision_free: bool = True,
        check_every: int = 5,
        n_iters: int | None = None,
        verbose: bool = False,
        log_every: int = 25,
        return_history: bool = False,
    ):
        """
        Run CHOMP covariant gradient descent until convergence (not a fixed number
        of steps), so wall-clock time is a meaningful quantity to compare.

        Each iteration takes the covariant step 1/eta * A^-1 grad(U), backtracked
        (halved up to `max_backtracks` times) until the total cost actually decreases,
        so the descent is monotone. Without it a fixed step overshoots on a stiff cost
        and can end worse than it started — in particular it would corrupt a warm-started
        trajectory that is already near a minimum. `eta` therefore sets the *largest*
        step considered, not the step taken. The search uses the batch-summed cost, so
        for B > 1 all samples share one step scale; optimize per sample (B = 1) when the
        step size matters.

        Stopping criteria (whichever fires first):
          * collision-free: every `check_every` iters, if all sphere surfaces are
            clear of obstacles AND `stop_at_collision_free` (the planner's natural
            success condition). Disable for a pure fixed-tolerance run.
          * converged: the relative change in total cost stays below `tol` for
            `patience` consecutive iterations (settled into a local minimum).
          * max_iters: hard cap.

        `n_iters` is a backward-compatible alias: if given, it overrides max_iters
        and disables convergence (a fixed-length run).

        Returns the optimized [B, T, dof] trajectory, or (trajectory, history) when
        return_history=True. `history` has one cost-breakdown dict per iteration
        run (its length = iterations taken), each tagged with "collision_free".
        """
        eta = self.eta if eta is None else eta

        # Backward-compat: a literal n_iters means "exactly this many steps".
        if n_iters is not None:
            max_iters = n_iters
            tol = -1.0                    # never satisfies the plateau test
            stop_at_collision_free = False

        q_start = q_start.to(self.device).float()
        q_goal  = q_goal.to(self.device).float()
        if q_start.ndim == 1:
            q_start = q_start.unsqueeze(0)
            q_goal  = q_goal.unsqueeze(0)

        traj = self.init_trajectory(q_start, q_goal, init_traj, init_waypoints)
        B    = traj.shape[0]
        sdf  = self._as_batched_sdf(sdf, B)

        def _take_step(traj, step, scale):
            """Apply a scaled covariant step; endpoints stay pinned to q_start/q_goal."""
            out = traj.detach().clone()
            out[:, 1:-1, :] = out[:, 1:-1, :] - scale * step
            out[:, 0, :]    = q_start
            out[:, -1, :]   = q_goal
            return out

        history = []
        prev_cost, stalls = None, 0
        for it in range(max_iters):
            traj = traj.detach().requires_grad_(True)
            total, terms = self._compute_cost(traj, sdf)

            grad, = torch.autograd.grad(total, traj)            # [B, T, dof]

            # Covariant step on the interior waypoints only — endpoints stay fixed.
            grad_inner = grad[:, 1:-1, :]                        # [B, T-2, dof]
            step       = torch.einsum("ij,bjd->bid", self.A_inv, grad_inner)

            cur = total.item()

            # Backtracking line search: a fixed 1/eta step overshoots badly on this cost
            # (a strong collision weight over a narrow eps band is stiff), which makes the
            # descent oscillate and can leave a warm-started trajectory worse than it began.
            # Halve the step until the cost actually decreases; give up after max_backtracks
            # and take no step, which then trips the plateau test below.
            with torch.no_grad():
                scale = 1.0 / eta
                for _ in range(max_backtracks):
                    cand = _take_step(traj, step, scale)
                    cand_cost = self._compute_cost(cand, sdf)[0].item()
                    if cand_cost <= cur:
                        break
                    scale *= 0.5
                else:
                    cand, cand_cost = traj.detach().clone(), cur
                traj = cand

            # --- Stopping checks (on the freshly updated trajectory) ---
            free = False
            if stop_at_collision_free and (it % check_every == 0):
                free = bool((self._surface_clearance(traj, sdf) >= 0).all().item())

            if return_history:
                history.append({"total": cur, "collision_free": free,
                                **{k: v.item() for k, v in terms.items()}})
            if verbose and (it % log_every == 0 or it == max_iters - 1):
                print(f"iter {it:4d} | total={cur:.4f}  " +
                      "  ".join(f"{k}={v.item():.4f}" for k, v in terms.items()))

            if free:
                break
            if prev_cost is not None:
                rel = abs(prev_cost - cur) / (abs(prev_cost) + 1e-12)
                stalls = stalls + 1 if rel < tol else 0
                if stalls >= patience:
                    break
            prev_cost = cur

        traj = traj.detach()
        if return_history:
            return traj, history
        return traj
