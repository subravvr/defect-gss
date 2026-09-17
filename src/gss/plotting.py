"""Figure style and small 3D helpers for scripts/figures.py."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .mesh import DATA  # noqa: E402

FIGURES = DATA.parent / "figures"
SEVERITY_CMAP = "OrRd"

RC = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 1.4, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.markersize": 5,
}


def apply():
    plt.rcParams.update(RC)


def panel(ax, letter, x=-0.12, y=1.05):
    """Bold 'a)' panel label."""
    ax.text(x, y, f"{letter})", transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem):
    """Write figures/<stem>.pdf and .png (no timestamp in the PDF)."""
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}",
                    **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf")


def outward_normals(verts, faces):
    """Unit face normals pointing out of the enclosed volume, whatever the winding."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    n = cross / np.clip(np.linalg.norm(cross, axis=1, keepdims=True), 1e-15, None)
    signed = (np.einsum("ij,ij->i", verts[faces].mean(axis=1), n)
              * 0.5 * np.linalg.norm(cross, axis=1)).sum()
    return -n if signed < 0 else n


def view_direction(elev, azim):
    """Unit vector toward the camera, in matplotlib's view_init convention."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
