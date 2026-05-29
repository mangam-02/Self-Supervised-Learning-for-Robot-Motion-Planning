# Warm-Starting Trajectory Optimization with Self-Supervised Learning

> **Neural warm-starts for robot motion planning. No labeled trajectories needed.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Course](https://img.shields.io/badge/TUM-Advanced%20Deep%20Learning%20for%20Robotics-0065BD)

---

## Demo

<table>
  <tr>
    <td align="center"><b>Trained warm-start — validation sample</b></td>
    <td align="center"><b>Baseline (zero output / straight line)</b></td>
  </tr>
  <tr>
    <td><img src="src/resources/val_trajectories_working_trajectory_7.gif" width="360"/></td>
    <td><img src="src/resources/zero_output_trajectory_0.gif" width="360"/></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><b>Multi-obstacle environment</b></td>
    <td align="center"><b>Tunneling failure (before CCD fix)</b></td>
  </tr>
  <tr>
    <td><img src="src/resources/overfit_64_10_2obs_working_trajectory_16.gif" width="360"/></td>
    <td><img src="src/resources/tunnel_trajectory.gif_trajectory_0.gif" width="360"/></td>
  </tr>
</table>

---

## Overview

Trajectory optimization is a fundamental tool in robot motion planning: given start and goal joint configurations, find a smooth, collision-free path through the workspace. Gradient-based optimizers (CHOMP, TrajOpt, GPMP2) are reliable and expressive, but they are **highly sensitive to initialization**. A straight-line warm-start almost always intersects obstacles, forcing the optimizer to spend many iterations escaping infeasible regions or converging to a local minimum that still collides.

**The core tradeoff:**

| Approach | Speed | Reliability | Requires |
|---|---|---|---|
| Classical optimization (cold start) | Slow | High | Good initialization |
| Pure learning-based | Fast | Low at distribution shift | Large labeled dataset |
| **This project** | **Fast** | **High** | **No labeled data** |

This project trains a **self-supervised warm-start predictor**: a neural network that maps (start config, goal config, obstacle SDF) directly to a near-feasible joint-space trajectory, optimized entirely against differentiable planning costs. No expert demonstrations, no labeled trajectory datasets.

At inference, the predicted B-spline trajectory places the optimizer at a configuration already near-feasible, with lower collision cost from the first gradient step.

---

## Project Status

The **neural predictor** (warm-start generation) is fully implemented and trained. The downstream **gradient-based refinement** stage is the planned next step.

| Component | Status |
|---|---|
| Procedural dataset generation (SDF environments, non-trivial pair sampling) | ✅ Done |
| Environment autoencoder pre-training | ✅ Done |
| State autoencoder pre-training | ✅ Done |
| Self-supervised end-to-end training (`WarmStartPlanner`) | ✅ Done |
| Differentiable cost functions (collision, limits, smoothness, spacing, exploration) | ✅ Done |
| Trajectory visualization and GIF export | ✅ Done |
| Gradient-based refinement stage (warm-start → optimizer → collision-free trajectory) | 🚧 Planned |
| Quantitative evaluation (success rate, planning time vs. cold-start baseline) | 🚧 Planned |

---

## Method

### Conceptual Pipeline

```
  Start/Goal Config + Obstacle Environment (SDF)
                       |
                       v
          +------------------------+
          |  Neural Warm-Start     |  <- implemented
          |  Predictor             |
          +------------------------+
                       |
                       v
          Near-feasible trajectory
          (cubic B-spline, T=50 steps)
                       |
                       v
          +------------------------+
          |  Gradient-based        |  <- planned
          |  Refinement            |
          +------------------------+
                       |
                       v
          Collision-free trajectory
```

### Neural Architecture

```
  q_start, q_goal              SDF map (128x128)
        |                             |
        v                             v
  StateEncoder (MLP)          EnvEncoder (CNN)
  [q, FK(q)] -> 64-d          3-layer conv -> 64-d
        |                             |
        +-------------+---------------+
                      |
                      v
              WaypointDecoder (MLP)
            predicts C-2 interior waypoints
                      |
          prepend q_start · append q_goal
                      |
                      v
        B-Spline Interpolation  (T=50 steps)
                      |
        Differentiable Cost Functions
         collision · limits · smoothness
         spacing · exploration
                      |
                      v
               Loss -> Backprop
```

### Trajectory Representation

Trajectories are parameterized as **cubic B-splines**: the network predicts `C = 10` control points in joint space (8 interior waypoints; start and goal are fixed), and the full `T = 50`-step trajectory is evaluated via the Cox-de Boor recursion. This continuous representation provides smooth gradients and prevents discontinuous paths.

### Self-Supervised Training

No reference trajectories are used. The model is trained by minimizing a composite differentiable cost over the predicted B-spline:

| Cost term | Description |
|---|---|
| **Collision** | Three-zone penalty (inside / danger zone / safe) plus exponential repulsion; CCD inflation prevents tunneling through thin obstacles |
| **Joint limits** | Quadratic penalty for joint angle violations |
| **Smoothness** | Penalizes squared joint velocities and accelerations |
| **Waypoint spacing** | Penalizes unequal spacing between consecutive control points to discourage rushing through obstacles |
| **Exploration** | Anti-local-minimum term that encourages larger deviations from the straight line when the path is in collision |

The robot body is approximated by a set of **spheres** placed along each link. Forward kinematics is implemented in PyTorch so gradients flow from sphere world positions back through joint angles into the network weights. Sphere-to-obstacle distances are computed via **bilinear interpolation** on the SDF grid, making the entire pipeline end-to-end differentiable.

---

## Technical Features

- **B-spline trajectory parameterization** with cubic Cox-de Boor basis and clamped endpoints (C=10, T=50)
- **CNN environment encoder** compressing 128x128 SDF maps to a 64-d latent vector
- **MLP state encoder** encoding `[q_start, q_goal, FK_start, FK_goal]` to 64-d
- **Zero-initialized decoder output layer** so that epoch-0 predictions are straight-line interpolations
- **Differentiable sphere-based collision model** with bilinear SDF sampling
- **Continuous collision detection (CCD)** via danger-zone inflation by half travel distance
- **Exploration cost** that breaks the "barely-outside-obstacle" local minimum
- **Pre-training support** for environment and state autoencoders
- **Procedural dataset generation** with non-trivial pair filtering (straight-line path blocked)
- **Interactive Jupyter browsers** for navigating datasets and predicted trajectories
- **GIF export** of trajectory animations via Plotly + Kaleido + Pillow

---

## Results

Trained self-supervised on procedurally generated 2D environments with 1-4 circular obstacles. Validation loss tracks training loss closely, indicating good generalization across unseen obstacle configurations.

**Qualitative results** (from `src/resources/`):

| Asset | Description |
|---|---|
| `val_trajectories_working_trajectory_7.gif` | Predicted trajectory on a held-out validation sample |
| `overfit_64_10_2obs_working_trajectory_16.gif` | Two-obstacle environment, arm routes around both |
| `tunnel_trajectory.gif_trajectory_0.gif` | Early training failure: arm tunnels through an obstacle (fixed with CCD and exploration cost) |
| `zero_output_trajectory_0.gif` | Baseline with output = 0 (straight-line interpolation), collides with obstacles |
| `val_trajectories_working_cost_7.png` | Per-timestep collision and joint-limit costs after training |
| `zero_output_cost_0.png` | Same cost profile for the straight-line baseline |

<table>
  <tr>
    <td align="center"><b>Trained model — cost profile</b></td>
    <td align="center"><b>Baseline — cost profile</b></td>
  </tr>
  <tr>
    <td><img src="src/resources/val_trajectories_working_cost_7.png" width="420"/></td>
    <td><img src="src/resources/zero_output_cost_0.png" width="420"/></td>
  </tr>
</table>

---

## Repository Structure

```
Self-Supervised-Learning-for-Robot-Motion-Planning/
|
+-- src/
|   +-- models.py                    # WarmStartPlanner, encoders, B-spline utilities
|   +-- losses.py                    # Differentiable cost functions
|   +-- training.py                  # Training loops for planner and autoencoders
|   +-- data.py                      # Dataset generation, SDF building, dataset browser
|   +-- utils.py                     # Forward kinematics, sphere FK, differentiable SDF query
|   +-- visualization.py             # Trajectory browser, GIF export
|   |
|   +-- train.ipynb                  # End-to-end training pipeline
|   +-- test.ipynb                   # Evaluation and trajectory visualization
|   +-- create_training_data.ipynb   # Procedural dataset generation
|   +-- process_nvidia_1_sample.ipynb # Single-sample overfitting experiment (GPU)
|   |
|   +-- models/                      # Saved model weights (.pt)
|   +-- data/                        # Generated datasets (.pt) -- gitignored
|   +-- resources/                   # Output GIFs and cost plots
|
+-- external/
|   +-- SimpleArm/                   # Git submodule: 2D planar arm simulator
|
+-- requirements.txt
+-- README.md
```

---

## Installation

### Requirements

- **Python 3.11** (recommended)
- **Git** with submodule support

### Setup

```bash
# 1. Clone repository with submodule
git clone --recurse-submodules https://github.com/mangam-02/Self-Supervised-Learning-for-Robot-Motion-Planning.git
cd Self-Supervised-Learning-for-Robot-Motion-Planning

# 2. Create virtual environment
python3.11 -m venv .venv

# 3. Activate
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows CMD
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# 4. Install dependencies
pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

#### Platform-specific Python 3.11 installation

**macOS (Homebrew)**
```bash
brew install python@3.11
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Windows:** download from [python.org](https://www.python.org/downloads/release/python-3110/) and check *Add Python to PATH*.

### Deactivating

```bash
deactivate
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11: command not found` | `brew link python@3.11 --force` (macOS) or use `python3` |
| Permission denied on activate | `chmod +x .venv/bin/activate` |
| `pip install` fails | `pip install --upgrade pip && pip cache purge` |

---

## Usage

All entry points are Jupyter notebooks under `src/`. Activate your environment first:

```bash
source .venv/bin/activate
jupyter notebook src/
```

### 1 — Generate a dataset

`src/create_training_data.ipynb` calls `data.generate_dataset()`:

```python
from data import generate_dataset
dataset = generate_dataset(
    N_envs=200, pairs_per_env=5,
    save_path="data/dataset.pt",
    split=(0.75, 0.125, 0.125),
)
```

### 2 — Train the planner

`src/train.ipynb` trains the `WarmStartPlanner` end-to-end:

```python
from training import train
model, history = train(
    train_dataset="data/dataset_train.pt",
    val_dataset="data/dataset_val.pt",
    n_epochs=2000,
    batch_size=32,
    save_path="models/warm_start_planner.pt",
)
```

Optional: pre-train encoders separately before end-to-end training:

```python
from training import train_env_autoencoder, train_state_autoencoder
train_env_autoencoder("data/sdf_dataset.pt", ...)
train_state_autoencoder(linklengths=[0.4, 0.3, 0.2], ...)
```

### 3 — Evaluate and visualize

`src/test.ipynb` launches an interactive trajectory browser:

```python
from visualization import browse_trajectories
browse_trajectories(
    model="models/warm_start_planner.pt",
    dataset="data/dataset_val.pt",
    animate=True,
    save_name="val_result",
)
```

Navigate with **Next / Prev** buttons. Click **Save** to export a GIF and cost plot to `src/resources/`.

---

## Implementation Details

| Component | Choice | Rationale |
|---|---|---|
| Trajectory parametrization | Cubic B-spline (C=10, T=50) | Smooth gradients, fewer parameters than raw waypoints |
| Collision model | Sphere decomposition + bilinear SDF sampling | Differentiable and GPU-compatible |
| CCD | Danger-zone inflation by half travel distance | Prevents tunneling through thin obstacles |
| Encoder | 3-layer CNN (16 to 32 to 64 channels) + AdaptiveAvgPool | Compact; SDF maps are spatially smooth |
| State features | `[q, FK(q)]` for start and goal | Exposes Cartesian geometry to the MLP |
| Decoder init | Zero weights on output layer | Epoch-0 output is a straight-line baseline |
| Optimizer | Adam, lr=1e-3, ExponentialLR gamma=0.9995 | Smooth decay without late-stage oscillations |
| Framework | PyTorch 2.x | Autograd through FK, SDF sampling, and B-spline evaluation |

---

## Future Work

- **Gradient-based refinement**: chain the predicted warm-start with a trajectory optimizer (TrajOpt, CHOMP, GPMP2) and measure convergence speed vs. cold-start baseline
- **Quantitative evaluation**: success rate, planning time, and cost comparison on a held-out test set
- **Higher-DOF robots**: extend to 6/7-DOF serial chains and SE(3) task spaces
- **3D environments**: replace 2D SDF grids with voxel grids or neural implicit representations
- **Dynamic obstacles**: condition the encoder on time-varying obstacle states
- **Sim-to-real transfer**: deploy on physical hardware with domain adaptation

---

## References

1. Schulman, J. et al. **Motion Planning with Sequential Convex Optimization and Convex Collision Checking.** IJRR, 2014.
2. Mukadam, M. et al. **Continuous-time Gaussian Process Motion Planning via Probabilistic Inference.** IJRR, 2018.
3. Ratliff, N. et al. **CHOMP: Covariant Hamiltonian Optimization for Motion Planning.** ICRA, 2009.
4. Ha, J. et al. **Learning Sparse Waypoint Representations for Motion Planning.** CoRL, 2022.

---

*Course project — **Advanced Deep Learning for Robotics** (SS26), Prof. Berthold Bäuml, Technical University of Munich (TUM).*
