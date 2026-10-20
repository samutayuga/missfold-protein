"""Render the app icon: a protein chain that is half folded (helix) and half unravelled.

Run: .venv/bin/python scripts/make_icon.py  -> assets/icon.png
"""

from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource
from matplotlib.figure import Figure

from misfold.plots import backbone_gradient, tube_mesh

OUT = Path(__file__).resolve().parent.parent / "assets" / "icon.png"


def misfold_path(samples: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)
    helix = t < 0.5
    curve = np.empty((samples, 3))
    # Folded part: a tight alpha-helix-like coil standing upright.
    th = t[helix] / 0.5 * 3.5 * 2 * np.pi
    curve[helix] = np.column_stack([np.cos(th), np.sin(th), t[helix] / 0.5 * 3.2])
    # Unfolded part: the chain leaves the top of the helix and unravels sideways
    # and downwards as a loose, wavy strand.
    s = (t[~helix] - 0.5) / 0.5
    start = curve[helix][-1]
    curve[~helix] = start + np.column_stack(
        [4.2 * s, 0.9 * np.sin(3.0 * np.pi * s), -3.0 * s + 0.7 * np.sin(4.0 * np.pi * s)]
    )
    return curve


def render(path: Path = OUT, size_px: int = 384) -> None:
    fig = Figure(figsize=(4, 4), dpi=size_px / 4)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    curve = misfold_path()
    x, y, z = tube_mesh(curve, radius=0.24, sides=24)
    colors = np.repeat(backbone_gradient(len(curve))[:, None, :], x.shape[1], axis=1)
    ax.plot_surface(x, y, z, facecolors=colors, shade=True, lightsource=LightSource(315, 45),
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.view_init(elev=14, azim=-80)
    ax.set_box_aspect(np.ptp(curve, axis=0) + 0.6)
    fig.subplots_adjust(0, 0, 1, 1)
    FigureCanvasAgg(fig)
    fig.savefig(path, transparent=True)
    _crop_to_content(path)


def _crop_to_content(path: Path, margin: int = 4) -> None:
    """Trim transparent margins (3D axes reserve a lot) to a centred square."""
    img = mpimg.imread(path)
    rows, cols = np.nonzero(img[..., 3] > 0)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    side = max(bottom - top, right - left) + 2 * margin
    cy, cx = (top + bottom) // 2, (left + right) // 2
    square = np.zeros((side, side, 4), dtype=img.dtype)
    y0, x0 = cy - side // 2, cx - side // 2
    src = img[max(y0, 0):y0 + side, max(x0, 0):x0 + side]
    oy, ox = max(-y0, 0), max(-x0, 0)
    square[oy:oy + src.shape[0], ox:ox + src.shape[1]] = src
    mpimg.imsave(path, square)


if __name__ == "__main__":
    render()
    print(f"wrote {OUT}")
