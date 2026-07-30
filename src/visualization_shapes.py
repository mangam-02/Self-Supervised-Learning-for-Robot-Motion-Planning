import io
import contextlib
from PIL import Image
import numpy as np
import torch
import plotly.graph_objects as go

from simplearm.robot import RobotInfo
from simplearm.geom import SquareGrid
from simplearm.viz_shapes import RobotViewer

from models import WarmStartPlanner
from losses import compute_trajectory_max_collision_cost, compute_trajectory_joint_limits_cost
from evaluation import surface_clearance


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
    save_name: str | None = None,
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
    robot.sphere_rad = ds["metadata"]["sphere_rad"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta    = ds["metadata"]
    model   = _load_model(model, dof=meta["dof"], device=device, linklengths=meta["linklengths"])
    sdf     = ds["sdf"].to(device)
    q_start = ds["q_start"].to(device)
    q_goal  = ds["q_goal"].to(device)
    q_min   = torch.tensor(meta["q_min"], dtype=torch.float32, device=device)
    q_max   = torch.tensor(meta["q_max"], dtype=torch.float32, device=device)


    model.eval()
    
    with torch.no_grad():
        # Generate all trajectories
        all_waypoints = model(q_start, q_goal, sdf)
        all_trajectories = model.trajectory(all_waypoints)

        clearance_matrix = surface_clearance(
            all_trajectories, sdf, robot, 
            grid_length=meta["grid_length"], sphere_rad=meta["sphere_rad"]
        )
        
        # Collision free?
        min_clearance_per_sample = clearance_matrix.min(dim=-1)[0].cpu().numpy()

    # Global counters
    is_clean_mask = min_clearance_per_sample >= 0
    total_clean = int(is_clean_mask.sum())
    total_colliding = N - total_clean
    collision_rate = (total_colliding / N) * 100

    # Summary
    summary_html = widgets.HTML(value=(
        f"<div style='background-color: #f8f9fa; padding: 14px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 12px; font-family: sans-serif;'>"
        f"  <h4 style='margin: 0 0 6px 0; color: #212529;'>📊 Global Test Summary (Geometric Feasibility)</h4>"
        f"  <span style='color: #28a745; font-weight: bold;'>🟢 Clean Trajectories: {total_clean}</span> &nbsp;|&nbsp; "
        f"  <span style='color: #dc3545; font-weight: bold;'>🔴 Colliding Trajectories: {total_colliding}</span> &nbsp;|&nbsp; "
        f"  <b>⚠️ Collision Rate: {collision_rate:.2f}%</b>"
        f"</div>"
    ))
    # -----------------------------------------------------------------

    idx   = [int(start_idx) % N]
    state = {"viz": None, "cost_fig": None}

    btn_prev   = widgets.Button(description="◄ Prev",   layout=widgets.Layout(width="100px"))
    btn_next   = widgets.Button(description="Next ►",   layout=widgets.Layout(width="100px"))
    btn_save   = widgets.Button(description="💾 Save",  layout=widgets.Layout(width="100px"), button_style="info")
    save_dir   = widgets.Text(value="resources", description="Dir:", layout=widgets.Layout(width="200px"))
    info_label = widgets.HTML()
    out        = widgets.Output()

    def render():
        i = idx[0]
        
        # Individual state
        is_sample_clean = is_clean_mask[i]
        sample_min_clearance = min_clearance_per_sample[i]
        
        if is_sample_clean:
            status_tag = f"<span style='background-color: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;'>✔️ GOOD (Clean, Clearance: {sample_min_clearance:.3f})</span>"
        else:
            status_tag = f"<span style='background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;'>❌ BAD (Collision, Penetration: {sample_min_clearance:.3f})</span>"

        q_traj_np = all_trajectories[i].cpu().numpy()
        q_s_np    = q_start[i].cpu().numpy()
        q_g_np    = q_goal[i].cpu().numpy()

        # Get the voxels matrix
        vox_data = ds["voxels"][i, 0].numpy().astype(bool)

        # Rebuild the SquareGrid structure
        voxels_grid = SquareGrid.from_zero_centered(
            limits=(-meta["grid_length"] / 2, meta["grid_length"] / 2),
            data=vox_data.T
        )

        info_label.value = (
            f"<b>Sample {i} / {N - 1}</b> &nbsp;|&nbsp; Voxel Map &nbsp;|&nbsp; {status_tag}<br>"
            f"&nbsp;&nbsp;q_start: {np.round(q_s_np, 3)}<br>"
            f"&nbsp;&nbsp;q_goal:&nbsp; {np.round(q_g_np, 3)}"
        )

        # Instantiate using the voxels
        viz = RobotViewer(q_traj_np, robot, obstacles=None, voxels=voxels_grid)
        
        _orig_show = go.Figure.show
        go.Figure.show = lambda *a, **kw: None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                viz.plot()
        finally:
            go.Figure.show = _orig_show

        # Costs plot
        with torch.no_grad():
            _, coll_per_step = compute_trajectory_max_collision_cost(
                all_trajectories[i:i+1], sdf[i:i+1], robot,
                grid_length=meta["grid_length"], eps=meta.get("collision_eps", 0.1),
                return_per_step=True,
            )
            _, joints_per_step = compute_trajectory_joint_limits_cost(
                all_trajectories[i:i+1], q_min, q_max, return_per_step=True,
            )
            coll_costs   = coll_per_step.squeeze(0).cpu().numpy()
            joints_costs = joints_per_step.squeeze(0).cpu().numpy()

        T_steps = len(coll_costs)
        timesteps = np.arange(T_steps)
        cost_fig = go.Figure()
        cost_fig.add_trace(go.Scatter(
            x=timesteps, y=coll_costs,
            name="Collision (max sphere)", line=dict(color="red"),
        ))
        cost_fig.add_trace(go.Scatter(
            x=timesteps, y=joints_costs,
            name="Joint limits", line=dict(color="orange"),
        ))
        cost_fig.update_layout(
            title=f"Cost per timestep — sample {i}",
            xaxis_title="Timestep",
            yaxis_title="Cost",
            height=250,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", y=1.15),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        cost_fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
        cost_fig.update_yaxes(showgrid=True, gridcolor="lightgrey")

        state["viz"]      = viz
        state["cost_fig"] = cost_fig

        out.clear_output(wait=True)
        with out:
            display(viz.fig)
            display(cost_fig)

    def on_prev(_):
        idx[0] = (idx[0] - 1) % N
        render()

    def on_next(_):
        idx[0] = (idx[0] + 1) % N
        render()

    def on_save(_):
        import os
        d   = save_dir.value.strip() or "resources"
        os.makedirs(d, exist_ok=True)
        i   = idx[0]
        gif_path  = os.path.join(d, f"{save_name}_trajectory_{i}.gif")
        cost_path = os.path.join(d, f"{save_name}_cost_{i}.png")
        with out:
            if state["viz"] is not None:
                save_viewer_as_gif(state["viz"], gif_path)
            if state["cost_fig"] is not None:
                state["cost_fig"].write_image(cost_path, engine="kaleido", scale=3)
                print(f"Cost plot saved: {cost_path}")

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_save.on_click(on_save)

    render()
    display(widgets.VBox([
        summary_html,
        widgets.HBox([btn_prev, btn_next, btn_save, save_dir]),
        info_label,
        out,
    ]))




def save_viewer_as_gif(viewer, path, duration_ms=120, size=600):
    """Save an animated RobotViewer as a GIF file."""
    images = []
    for pf in viewer.fig.frames:
        tmp = go.Figure(data=pf.data, layout=viewer.fig.layout)
        tmp.update_layout(
            updatemenus=[],
            sliders=[],
            margin=dict(l=10, r=10, t=10, b=10),
            width=size,
            height=size,
        )
        png = tmp.to_image(format="png", width=size, height=size, engine="kaleido")
        images.append(Image.open(io.BytesIO(png)))

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"GIF gespeichert: {path}")


