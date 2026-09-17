"""Conventional shape descriptors (paper Sec. 3.2), with the load along e3."""
import numpy as np
from scipy.spatial import ConvexHull

from .mesh import as_mesh, enclosed_volume, surface_area


def sphericity(verts, faces):
    """Wadell sphericity: area of the equal-volume sphere over the actual area."""
    v, a = enclosed_volume(verts, faces), surface_area(verts, faces)
    return float(np.pi ** (1 / 3) * (6.0 * abs(v)) ** (2 / 3) / a)


def principal_frame(verts):
    """Extents along the principal axes of the vertex cloud (descending), and the axes."""
    c = verts - verts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    ext = proj.max(axis=0) - proj.min(axis=0)
    order = np.argsort(ext)[::-1]
    return ext[order], vt[order]


def murakami(verts):
    """sqrt(area of the convex hull of the vertices projected along e3)."""
    return float(np.sqrt(ConvexHull(verts[:, :2]).volume))


def describe(mesh):
    """Sphericity, the two aspect ratios and Murakami's sqrt(area)."""
    verts, faces = as_mesh(mesh)
    ext = np.sort(principal_frame(verts)[0])[::-1]
    return {
        "sphericity": sphericity(verts, faces),
        "aspect_ratio": float(ext[0] / ext[2]),
        "aspect_ratio_e1e2": float(ext[0] / ext[1]),
        "murakami": murakami(verts),
        "area": surface_area(verts, faces),
        "volume": abs(enclosed_volume(verts, faces)),
    }
