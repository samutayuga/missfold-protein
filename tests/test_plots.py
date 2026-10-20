from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex

from misfold.plots import RESIDUE_COLORS, plot_energy, plot_structure

C = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])


def _drawn_colors(fig):
    # 3D scatter colours are only complete after a draw, and then depth-sorted,
    # so compare colour counts rather than order.
    FigureCanvasAgg(fig).draw()
    return Counter(to_hex(c) for c in fig.axes[0].collections[0].get_facecolors())


def test_structure_title():
    ax = plot_structure(C, "HPKD", 1.2, 7.0).axes[0]
    assert ax.name == "3d"
    assert ax.get_title() == "Final Simulated State (Temp: 1.2, pH: 7.0)"


@pytest.mark.parametrize("seq", ["HPKD", "HHHP", "KDDP"])
def test_structure_residue_colors(seq):
    expected = Counter(RESIDUE_COLORS[r].lower() for r in seq)
    assert _drawn_colors(plot_structure(C, seq, 1.2, 7.0)) == expected


def test_legend_only_present_types():
    ax = plot_structure(C[:3], "HPH", 1.2, 7.0).axes[0]
    labels = sorted(t.get_text() for t in ax.get_legend().get_texts())
    assert labels == ["Hydrophobic (H)", "Polar (P)"]


def test_energy_labels():
    ax = plot_energy([0.0, -1.0, -2.0]).axes[0]
    assert ax.get_ylabel() == "Free Energy (ΔG)"
    assert ax.get_xlabel() == "Simulation Step / Time"
    assert list(ax.lines[0].get_ydata()) == [0.0, -1.0, -2.0]


def test_no_pyplot_figures_leak():
    plot_structure(C, "HPKD", 1.0, 7.0)
    plot_energy([0.0])
    assert plt.get_fignums() == []
