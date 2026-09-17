"""Figures for the test geometries.

    python scripts/figures.py

gamma_severity               Lambda_FEA and K_t against gamma, coloured by GSS
descriptor_degeneracy_pairs  the four superellipsoid pairs with their captions

GSS comes from results/*.csv (scripts/compute_gss.py); Lambda and K_t are the
published reference values in data/reference_values.json.
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gss import load, plotting
from gss.mesh import DATA

RESULTS = DATA.parent / "results"
REFERENCE = json.loads((DATA / "reference_values.json").read_text())

GUIDE = "#B0B0B0"
BRANCH_MARKERS = {"prolate": "^", "oblate": "o", "sphere": "s"}
PAIRS = ("G1", "G2", "G3", "G4")
SURFACE_COLOR = "#0072B2"
ELEV, AZIM = 22.0, -58.0
KEY_LIGHT = np.array([0.45, 0.55, 0.70])
FILL_LIGHT = np.array([-0.60, -0.30, 0.35])

SEP = 1.12           # centre-to-centre offset, in units of the pair radius
PANEL_W, PANEL_H = 3.4, 1.75   # inches per cell; the lenses are flat
ANNOT = 7.0          # caption font size
LABEL_GAP = 0.018    # figure fraction between geometry and caption
CELL_MARGIN = 0.055
FIT_MARGIN = 0.04

# Caption columns, laid out on shared anchors so the "|" line up across panels.
COLS = ("label", "a", "pipe", "b", "delta")
COL_ALIGN = {"label": "right", "a": "right", "pipe": "center",
             "b": "left", "delta": "left"}
COL_GAP = {"label": 0.0, "a": 7.0, "pipe": 4.0, "b": 4.0, "delta": 9.0}  # points
LINE_SP = 1.45

DESCRIPTOR_LABEL = {
    "murakami": r"$\sqrt{A_{\mathrm{proj}}}$",
    "sphericity": r"$\Psi$",
    "aspect_ratio": r"$a_1/a_3$",
    "aspect_ratio_e1e2": r"$a_1/a_2$",
}




def read_csv(name):
    with open(RESULTS / name) as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------ ellipsoids

def branch_of(gamma):
    """Pore-type branch of a spheroid with aspect ratio gamma."""
    return "prolate" if gamma < 1.0 else ("oblate" if gamma > 1.0 else "sphere")


def branch_masks(g):
    """(marker, mask) for each branch present in `g`."""
    for branch, marker in BRANCH_MARKERS.items():
        m = np.array([branch_of(v) == branch for v in g])
        if m.any():
            yield marker, m


def branch_handles():
    """Shape-only legend entries for the three branches."""
    return [plt.Line2D([], [], ls="none", marker=mk, color="#4C4C4C",
                       markersize=6, label=name.capitalize())
            for name, mk in BRANCH_MARKERS.items()]


def ellipsoid_rows():
    """Computed GSS (9-mesh mean if available) with the reference Lambda_FEA and K_t."""
    ref = REFERENCE["ellipsoids"]
    rows = [{"gamma": float(r["gamma"]), "gss": float(r.get("gss_mean") or r["gss"]),
             **ref[r["shape"]]} for r in read_csv("gss_ellipsoids.csv")]
    return sorted(rows, key=lambda r: r["gamma"])


def figure_severity(rows):
    """Lambda_FEA and K_t (meshed) against gamma, coloured by GSS on one scale."""
    rows = [r for r in rows if np.isfinite(r["lambda_fea"])]
    g = np.array([r["gamma"] for r in rows])
    x = np.array([r["gss"] for r in rows])
    norm = Normalize(vmin=float(x.min()), vmax=float(x.max()))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True)
    panels = ((r"$\Lambda_\mathrm{FEA}$", "lambda_fea"),
              (r"$K_t$ (meshed)", "K_t_measured"))
    for ax, (label, key) in zip(axes, panels):
        y = np.array([r[key] for r in rows], float)
        for mk, m in branch_masks(g):
            ax.scatter(g[m], y[m], c=x[m], cmap=plotting.SEVERITY_CMAP, norm=norm,
                       marker=mk, s=62, zorder=4, edgecolors="#4C4C4C",
                       linewidths=0.5)
        ax.axvline(1.0, color=GUIDE, lw=0.9, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma = a_1/a_3$")
        ax.set_ylabel(label)

    axes[0].axhline(1.0, color=GUIDE, lw=0.9, ls="--", zorder=1)
    axes[0].legend(handles=branch_handles(), frameon=False, fontsize=12,
                   loc="upper left")
    for ax, letter in zip(axes, "ab"):
        plotting.panel(ax, letter)

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    cax = fig.add_axes([0.27, 0.075, 0.46, 0.032])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=plotting.SEVERITY_CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label(r"$\mathrm{GSS}$", fontsize=10)
    cb.outline.set_linewidth(0.6)
    return fig



# ------------------------------------------------------- superellipsoids

def superellipsoid_values():
    """Descriptors and GSS (9-mesh mean if available) with the reference Lambda."""
    ref = REFERENCE["superellipsoids"]
    out = {}
    for r in read_csv("gss_superellipsoids.csv"):
        g = r.get("gss_mean") or r["gss"]
        out[r["shape"]] = {**{k: float(r[k]) for k in DESCRIPTOR_LABEL},
                           "gss": float(g), "lambda": ref[r["shape"]]["lambda"]}
    return out


def view_axes(fig, pos):
    """A bare perspective 3D axes at the shared camera."""
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_axis_off()
    ax.set_proj_type("persp", focal_length=1.4)
    ax.view_init(elev=ELEV, azim=AZIM)
    return ax


def shade(normals, base):
    """Lambertian key + fill with a tight specular, on a fixed light rig."""
    k = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    fl = FILL_LIGHT / np.linalg.norm(FILL_LIGHT)
    key = np.clip(normals @ k, 0.0, 1.0)
    inten = 0.42 + 0.62 * key + 0.20 * np.clip(normals @ fl, 0.0, 1.0)
    rgb = np.array(to_rgb(base))[None, :] * inten[:, None] + 0.35 * key[:, None] ** 12
    return np.clip(rgb, 0.0, 1.0)


def draw_pair(ax, va, fa, vb, fb, lim):
    """Both members in one axes, offset along x; returns the drawn vertices."""
    drawn = []
    for v, f, sgn in ((va, fa, -1.0), (vb, fb, +1.0)):
        v, f = np.asarray(v, float), np.asarray(f)
        # Cull at the origin, then translate.
        n = plotting.outward_normals(v, f)
        keep = n @ plotting.view_direction(ELEV, AZIM) > 0
        vt = v.copy()
        vt[:, 0] += sgn * SEP * lim
        ax.add_collection3d(Poly3DCollection(
            vt[f[keep]], facecolors=shade(n[keep], SURFACE_COLOR), shade=False,
            edgecolors=(1, 1, 1, 0.28), linewidths=0.12))
        drawn.append(vt[np.unique(f[keep])])
    return np.vstack(drawn)


def fit_axes(ax, vis):
    """Limits and box aspect set to the drawn extent, so no space is wasted."""
    lo, hi = vis.min(0), vis.max(0)
    pad = FIT_MARGIN * np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))


def content_bbox(fig, ax, vis):
    """(x0, x1, y0, y1) of the drawn geometry, in figure fractions."""
    x2, y2, _ = proj3d.proj_transform(vis[:, 0], vis[:, 1], vis[:, 2], ax.get_proj())
    disp = ax.transData.transform(np.column_stack([x2, y2]))
    frac = fig.transFigure.inverted().transform(disp)
    return frac[:, 0].min(), frac[:, 0].max(), frac[:, 1].min(), frac[:, 1].max()


def fit_cell(fig, ax, vis):
    """Scale the axes rectangle until the geometry fills its grid cell above the caption."""
    pos = ax.get_position()
    cell = ax.get_subplotspec().get_position(fig)
    cx0, cx1, cy0, cy1 = content_bbox(fig, ax, vis)
    cw, ch = max(cx1 - cx0, 1e-9), max(cy1 - cy0, 1e-9)

    line_h = ANNOT * 1.45 / (fig.get_figheight() * 72.0)
    reserve = 4 * line_h + LABEL_GAP
    mx, my = CELL_MARGIN * cell.width, CELL_MARGIN * cell.height
    tx0, tx1 = cell.x0 + mx, cell.x1 - mx
    ty0, ty1 = cell.y0 + reserve, cell.y1 - my
    s = min((tx1 - tx0) / cw, (ty1 - ty0) / ch)
    ty1 -= 0.5 * max(0.0, (ty1 - ty0) - ch * s)

    w, h = pos.width * s, pos.height * s
    rel_x = (0.5 * (cx0 + cx1) - pos.x0) / pos.width
    rel_y = (cy1 - pos.y0) / pos.height
    ax.set_position([0.5 * (tx0 + tx1) - rel_x * w, ty1 - rel_y * h, w, h])


def place_tables(fig, blocks, centers, tops):
    """Draw each caption table under its geometry, columns aligned across all blocks."""
    fw, fh = fig.get_window_extent().x1, fig.get_window_extent().y1
    px = fig.dpi / 72.0

    texts = []
    for b, block in enumerate(blocks):
        for r, row in enumerate(block):
            for c in COLS:
                if row.get(c):
                    t = fig.text(0.0, 0.0, row[c], fontsize=ANNOT, ha=COL_ALIGN[c],
                                 va="baseline", **row.get("kw_" + c, {}))
                    texts.append((b, r, c, t))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    # Measure every cell, then take per-column widths and per-row heights.
    n_rows = max(len(b) for b in blocks)
    width = dict.fromkeys(COLS, 0.0)
    asc, desc = [0.0] * n_rows, [0.0] * n_rows
    for _, r, c, t in texts:
        bb = t.get_window_extent(renderer=rend)
        width[c] = max(width[c], bb.width)
        asc[r] = max(asc[r], bb.y1)
        desc[r] = max(desc[r], -bb.y0)

    anchor, cur = {}, 0.0
    for c in COLS:
        cur += COL_GAP[c] * px
        frac = {"right": 1.0, "center": 0.5, "left": 0.0}[COL_ALIGN[c]]
        anchor[c] = cur + frac * width[c]
        cur += width[c]
    total = cur

    # A tall row (the sqrt label) gets more leading than the nominal spacing.
    line = ANNOT * LINE_SP * px
    base = [-asc[0]]
    for r in range(1, n_rows):
        base.append(base[-1] - max(line, desc[r - 1] + asc[r] + 0.25 * line))

    for b, r, c, t in texts:
        x0 = min(max(centers[b] * fw - 0.5 * total, 0.0), fw - total)
        t.set_position(((x0 + anchor[c]) / fw, (tops[b] * fh + base[r]) / fh))


def caption(pair, a, b, va, vb):
    """Caption rows for one pair: names, tightest descriptor, Lambda, GSS."""
    def rel(x, y):
        return abs(x - y) / (0.5 * abs(x + y))

    def row(label, x, y, dfmt):
        return {"label": label, "a": f"{x:.3f}", "pipe": "|", "b": f"{y:.3f}",
                "delta": r"($\Delta$ " + dfmt.format(rel(x, y)) + ")"}

    tight = min(DESCRIPTOR_LABEL, key=lambda k: rel(va[k], vb[k]))
    return [
        {"label": pair, "a": a, "pipe": "|", "b": b,
         "kw_label": {"fontweight": "bold"}},
        row(f"breaks {DESCRIPTOR_LABEL[tight]}", va[tight], vb[tight], "{:.2%}"),
        row(r"$\Lambda$", va["lambda"], vb["lambda"], "{:.1%}"),
        row("GSS", va["gss"], vb["gss"], "{:.1%}"),
    ]


def figure_pairs(values):
    """2x2 grid of matched pairs with caption tables."""
    meshes = load("superellipsoids")
    fig = plt.figure(figsize=(PANEL_W * 2, PANEL_H * 2))
    gs = fig.add_gridspec(2, 2)
    axes, shown, blocks = [], [], []
    for i, pair in enumerate(PAIRS):
        a, b = f"{pair}A", f"{pair}B"
        va, fa = meshes[a]
        vb, fb = meshes[b]
        lim = 1.02 * max(np.linalg.norm(va, axis=1).max(),
                         np.linalg.norm(vb, axis=1).max())
        ax = view_axes(fig, gs[i // 2, i % 2])
        # Axes are enlarged past their cells; a visible patch would hide captions.
        ax.patch.set_visible(False)
        vis = draw_pair(ax, va, fa, vb, fb, lim)
        fit_axes(ax, vis)
        axes.append(ax)
        shown.append(vis)
        blocks.append(caption(pair, a, b, values[a], values[b]))

    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.075,
                        wspace=0.02, hspace=0.06)
    fig.canvas.draw()
    for ax, vis in zip(axes, shown):
        fit_cell(fig, ax, vis)
    fig.canvas.draw()

    boxes = [content_bbox(fig, ax, vis) for ax, vis in zip(axes, shown)]
    place_tables(fig, blocks, [0.5 * (b[0] + b[1]) for b in boxes],
                 [b[2] - LABEL_GAP for b in boxes])

    # Letters anchored to the grid cell: each axes was scaled by its own factor.
    for k, ax in enumerate(axes):
        cell = ax.get_subplotspec().get_position(fig)
        fig.text(cell.x0 + 0.008, cell.y1 - 0.012, f"{'abcd'[k]})",
                 fontweight="bold", va="top", ha="left")
    return fig



def main():
    plotting.apply()
    plotting.save(figure_severity(ellipsoid_rows()), "gamma_severity")
    # 3D panels only: the house grid and tight bbox would change the layout.
    plt.rcParams.update({"axes.grid": False, "savefig.bbox": None})
    plotting.save(figure_pairs(superellipsoid_values()), "descriptor_degeneracy_pairs")


if __name__ == "__main__":
    main()
