"""Matplotlib figure builders. Uses Figure directly so no pyplot state leaks between runs."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

RESIDUE_COLORS = {"H": "#FF4B4B", "P": "#1C83E1", "K": "#AB63FA", "D": "#FFA15A"}
RESIDUE_LABELS = {"H": "Hydrophobic (H)", "P": "Polar (P)", "K": "Basic (K)", "D": "Acidic (D)"}
BACKBONE_COLOR = "#D3D3D3"
ENERGY_COLOR = "#00CC96"
# Match Streamlit's dark theme (.streamlit/config.toml).
BACKGROUND = "#0E1117"
FOREGROUND = "#FAFAFA"
GRID_COLOR = "#3A3F4B"


def _style_dark(fig: Figure, ax) -> None:
    fig.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.title.set_color(FOREGROUND)
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
    for axis in axes:
        axis.label.set_color(FOREGROUND)
        axis.set_tick_params(colors=FOREGROUND)
        if hasattr(axis, "set_pane_color"):  # 3D panes and their grid lines
            axis.set_pane_color(BACKGROUND)
            axis._axinfo["grid"]["color"] = GRID_COLOR
    for spine in getattr(ax, "spines", {}).values():
        spine.set_color(GRID_COLOR)


def smooth_backbone(coords: np.ndarray, samples_per_bond: int = 12) -> np.ndarray:
    """Catmull-Rom spline through the bead positions, for display only.

    The simulation lives on the cubic lattice; the curve passes exactly through every
    bead (every samples_per_bond-th point) and rounds the 90° lattice corners so the
    backbone reads like a protein cartoon.
    """
    points = coords.astype(float)
    # Duplicate the end beads so the first and last bonds have tangents.
    padded = np.vstack([points[:1], points, points[-1:]])
    t = np.linspace(0.0, 1.0, samples_per_bond, endpoint=False)[:, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    segments = []
    for k in range(1, len(padded) - 2):
        p0, p1 = padded[k], padded[k + 1]
        m0 = (padded[k + 1] - padded[k - 1]) / 2
        m1 = (padded[k + 2] - padded[k]) / 2
        segments.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    segments.append(points[-1:])
    return np.vstack(segments)


def plot_structure(coords: np.ndarray, seq: str, temperature: float, ph: float) -> Figure:
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    curve = smooth_backbone(coords)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=BACKBONE_COLOR, lw=3, zorder=1)
    ax.scatter(
        x, y, z,
        color=[RESIDUE_COLORS[r] for r in seq],
        s=150, edgecolors="black", depthshade=False, zorder=2,
    )
    ax.set_title(f"Final Simulated State (Temp: {temperature}, pH: {ph})")
    ax.set_box_aspect(np.maximum(np.ptp(coords, axis=0), 1))

    present = [r for r in RESIDUE_COLORS if r in seq]
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markeredgecolor="black",
               markerfacecolor=RESIDUE_COLORS[r], label=RESIDUE_LABELS[r])
        for r in present
    ]
    legend = ax.legend(handles=handles, loc="upper left", facecolor=BACKGROUND,
                       edgecolor=GRID_COLOR, labelcolor=FOREGROUND)
    legend.get_frame().set_alpha(0.9)
    _style_dark(fig, ax)
    return fig


def plot_energy(history: list[float]) -> Figure:
    fig = Figure(figsize=(6, 5.3))
    ax = fig.add_subplot(111)
    ax.plot(history, color=ENERGY_COLOR, linewidth=2)
    ax.set_xlabel("Simulation Step / Time")
    ax.set_ylabel("Free Energy (ΔG)")
    ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)
    _style_dark(fig, ax)
    return fig
