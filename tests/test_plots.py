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
    beads = next(c for c in fig.axes[0].collections if type(c).__name__ == "Path3DCollection")
    return Counter(to_hex(c) for c in beads.get_facecolors())


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


def test_figures_use_dark_theme():
    from misfold.plots import BACKGROUND, FOREGROUND

    for fig in (plot_structure(C, "HPKD", 1.2, 7.0), plot_energy([0.0, -1.0])):
        ax = fig.axes[0]
        assert to_hex(fig.get_facecolor()) == BACKGROUND.lower()
        assert to_hex(ax.get_facecolor()) == BACKGROUND.lower()
        assert to_hex(ax.xaxis.label.get_color()) == FOREGROUND.lower()
        assert to_hex(ax.title.get_color()) == FOREGROUND.lower()


def test_smooth_backbone_passes_through_every_bead():
    from misfold.plots import smooth_backbone

    curve = smooth_backbone(C, samples_per_bond=8)
    assert curve.shape == (3 * 8 + 1, 3)
    assert np.allclose(curve[::8], C)


def test_smooth_backbone_keeps_straight_chain_straight():
    from misfold.plots import smooth_backbone

    straight = np.array([[i, 0, 0] for i in range(4)])
    curve = smooth_backbone(straight)
    assert np.allclose(curve[:, 1:], 0)
    assert np.all(np.diff(curve[:, 0]) > 0)


def test_smooth_backbone_rounds_corners():
    from misfold.plots import smooth_backbone

    curve = smooth_backbone(C, samples_per_bond=8)
    mid_first_bond = curve[4]  # halfway between bead 0 (0,0,0) and bead 1 (1,0,0)
    assert not np.isclose(mid_first_bond[1], 0)  # bows away from the straight lattice bond


def test_structure_draws_cartoon_tube():
    fig = plot_structure(C, "HPKD", 1.2, 7.0)
    FigureCanvasAgg(fig).draw()  # 3D polygons are projected at draw time
    ax = fig.axes[0]
    tubes = [c for c in ax.collections if type(c).__name__ == "Poly3DCollection"]
    assert len(tubes) == 1 and len(tubes[0].get_paths()) > 100
    assert [t.get_text() for t in ax.texts] == ["N", "C"]


def test_tube_mesh_rings_at_radius():
    from misfold.plots import smooth_backbone, tube_mesh

    curve = smooth_backbone(C)
    x, y, z = tube_mesh(curve, radius=0.2, sides=12)
    assert x.shape == (len(curve), 13)
    ring = np.stack([x, y, z], axis=-1)
    distances = np.linalg.norm(ring - curve[:, None, :], axis=-1)
    assert np.allclose(distances, 0.2)


def test_backbone_gradient_red_white_blue():
    from misfold.plots import backbone_gradient

    g = backbone_gradient(101)
    start, middle, end = g[0], g[50], g[-1]
    assert start[0] > 0.7 and start[2] < 0.35  # red at the N-terminus
    assert min(middle[:3]) > 0.85  # near white mid-chain
    assert end[2] > 0.7 and end[0] < 0.35  # blue at the C-terminus
