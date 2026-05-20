import io
import contextlib
import numpy as np
import torch
import plotly.graph_objects as go

from simplearm.robot import RobotInfo
from simplearm.geom import Obstacles
from simplearm.viz import RobotViewer

from models import WarmStartPlanner


def _load(dataset: dict | str) -> dict:
    if isinstance(dataset, str):
        return torch.load(dataset, weights_only=False)
    return dataset


def _load_model(model, dof: int, device: str, linklengths=None):
    """Accept a WarmStartPlanner instance, a state_dict, or a .pt file path."""
    if isinstance(model, str):
        model = torch.load(model, weights_only=False)

    if isinstance(model, dict):
        M    = model["M"]       # shape [T, C]
        T, C = M.shape
        ll   = model.get("linklengths", None)
        if ll is not None:
            linklengths = ll.tolist()
        net  = WarmStartPlanner(dof=dof, T=T, C=C, linklengths=linklengths)
        net.load_state_dict(model)
        model = net

    return model.to(device)


def browse_trajectories(
    model,
    dataset: dict | str,
    device: str | None = None,
    start_idx: int = 0,
    animate: bool = True,
) -> None:
    """
    Interactive browser that runs the model on each dataset sample and
    shows the predicted trajectory with Prev / Next buttons.

    model:      trained WarmStartPlanner (or any model with the same forward signature)
    dataset:    loaded dataset dict or path to a .pt file
    device:     torch device (auto-detected if None)
    start_idx:  sample to show first
    animate:    pass animate=True to RobotViewer for animated playback
    """
    import ipywidgets as widgets
    from IPython.display import display

    ds = _load(dataset)
    N  = ds["metadata"]["N"]

    robot = RobotInfo.from_linklengths(
        ds["metadata"]["linklengths"],
        sphere_rad=ds["metadata"]["sphere_rad"],
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model   = _load_model(model, dof=ds["metadata"]["dof"], device=device, linklengths=ds["metadata"]["linklengths"])
    sdf     = ds["sdf"].to(device)
    q_start = ds["q_start"].to(device)
    q_goal  = ds["q_goal"].to(device)

    idx = [int(start_idx) % N]

    btn_prev   = widgets.Button(description="◄ Prev", layout=widgets.Layout(width="100px"))
    btn_next   = widgets.Button(description="Next ►", layout=widgets.Layout(width="100px"))
    info_label = widgets.HTML()
    out        = widgets.Output()

    def render():
        i = idx[0]

        model.eval()
        with torch.no_grad():
            waypoints = model(q_start[i:i+1], q_goal[i:i+1], sdf[i:i+1])
            traj      = model.trajectory(waypoints)

        q_traj_np = traj.squeeze(0).cpu().numpy()
        q_s_np    = q_start[i].cpu().numpy()
        q_g_np    = q_goal[i].cpu().numpy()
        n_obs     = ds["n_obstacles"][i].item()
        obs_data  = ds["obstacles"][i, :n_obs]

        obstacles_viz = Obstacles(
            x=obs_data[:, 0].numpy(),
            y=obs_data[:, 1].numpy(),
            r=obs_data[:, 2].numpy(),
        )

        info_label.value = (
            f"<b>Sample {i} / {N - 1}</b> &nbsp;|&nbsp; obstacles: {n_obs}<br>"
            f"&nbsp;&nbsp;q_start: {np.round(q_s_np, 3)}<br>"
            f"&nbsp;&nbsp;q_goal:&nbsp; {np.round(q_g_np, 3)}"
        )

        # Build figure: suppress spinner text output and internal fig.show()
        viz = RobotViewer(q_traj_np, robot, obstacles=obstacles_viz, animate=animate)
        _orig_show = go.Figure.show
        go.Figure.show = lambda *a, **kw: None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                viz.plot()
        finally:
            go.Figure.show = _orig_show

        out.clear_output(wait=True)
        with out:
            display(viz.fig)

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
        out,
    ]))
