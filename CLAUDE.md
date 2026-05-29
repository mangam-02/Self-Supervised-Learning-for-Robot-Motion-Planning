# Claude Instructions

## Language
- All code, comments, docstrings, and variable/function names must be in **English**
- Chat communication with the user can be in German

## Project Overview
Self-supervised learning for robot motion planning with a 3-DOF planar robot arm.

### Key files
- `src/train.ipynb` — training pipeline
- `src/test.ipynb` — evaluation and visualization
- `src/create_training_data.ipynb` — dataset generation
- `src/models.py` — WarmStartPlanner and related models
- `src/data.py` — dataset generation and loading utilities
- `src/visualization.py` — RobotViewer GIF export and dataset browser

### Stack
- PyTorch, Plotly, Kaleido, Pillow
- `simplearm` package (local, under `external/SimpleArm/`)
- Python 3.11, venv at `.venv/`
