"""Matplotlib figure builders. Uses Figure directly so no pyplot state leaks between runs."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

RESIDUE_COLORS = {"H": "#FF4B4B", "P": "#1C83E1", "K": "#AB63FA", "D": "#FFA15A"}
RESIDUE_LABELS = {"H": "Hydrophobic (H)", "P": "Polar (P)", "K": "Basic (K)", "D": "Acidic (D)"}
BACKBONE_COLOR = "#D3D3D3"
ENERGY_COLOR = "#00CC96"


def plot_structure(coords: np.ndarray, seq: str, temperature: float, ph: float) -> Figure:
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.plot(x, y, z, color=BACKBONE_COLOR, lw=3, zorder=1)
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
    ax.legend(handles=handles, loc="upper left")
    return fig


def plot_energy(history: list[float]) -> Figure:
    fig = Figure(figsize=(6, 5.3))
    ax = fig.add_subplot(111)
    ax.plot(history, color=ENERGY_COLOR, linewidth=2)
    ax.set_xlabel("Simulation Step / Time")
    ax.set_ylabel("Free Energy (ΔG)")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig
