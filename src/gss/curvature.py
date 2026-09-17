"""Mean curvature and its spectral smoothing (paper Sec. 2.1, steps 1-2).

H is estimated with the cotangent Laplace-Beltrami operator, projected onto the
generalized eigenbasis L phi = lambda M phi, and truncated at the Nyquist band
limit of the triangulation. The constant mode is kept.
"""
import numpy as np
from scipy.sparse.linalg import eigsh

from .mesh import (angle_defect, as_mesh, corner_angles, cotangent_laplacian,
                   edge_lengths, mass_matrix, vertex_normals)


def operators(verts, faces):
    """(K, L, M): angle-defect curvature, cotangent Laplacian, mass matrix."""
    angles = corner_angles(*edge_lengths(verts, faces))
    n = len(verts)
    return (angle_defect(faces, angles, n),
            cotangent_laplacian(faces, angles, n),
            mass_matrix(verts, faces))


def mean_curvature(verts, faces, L=None, M=None):
    """Scalar mean curvature H_i = h_i . n_i, with h = -1/2 M^-1 L V.

    Sign convention: positive where the surface bulges outward from the cavity,
    for inward-wound faces (negative signed volume), which is how every surface
    in this repository is stored (`shapes.orient_inward`).
    """
    if L is None or M is None:
        _, L, M = operators(verts, faces)
    h = 0.5 * (L @ verts) / M.diagonal()[:, None]
    return 0.5 * np.einsum("ij,ij->i", h, -vertex_normals(verts, faces))


def eigenbasis(L, M, k):
    """The k smallest generalized eigenpairs of (L, M), by shift-invert."""
    return eigsh(L, k=k, M=M, sigma=0.0)


def project(field, eigvecs, M, k):
    """Reconstruction of `field` on the first k modes, constant mode included.

    Keeping the constant mode keeps the mean of the field, so a convex surface
    keeps H > 0 everywhere. Returns (reconstruction, coefficients).
    """
    coeffs = eigvecs[:, :k].T @ (M @ field)
    return eigvecs[:, :k] @ coeffs, coeffs


def band_limit(mesh, samples_per_wavelength=2.0):
    """Truncation index at the Nyquist band limit of the triangulation.

    A mode of wavelength l = 2 pi / sqrt(lambda) is resolved if sampled n times
    per wavelength by the mean edge h; with Weyl's law N(lambda) ~ A lambda / (4 pi)
    this gives k = pi A / (n^2 h^2).
    """
    v, f = as_mesh(mesh)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    h = float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())
    t = v[f]
    area = float(0.5 * np.linalg.norm(
        np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
    k = np.pi * area / (samples_per_wavelength**2 * h**2)
    return int(min(max(round(k), 2), len(v) - 1))


def smoothed_curvature(verts, faces, k=None):
    """H, its truncated reconstruction H_hat, and the truncation index used.

    k=None uses `band_limit` (n = 2 samples per wavelength).
    """
    _, L, M = operators(verts, faces)
    H = mean_curvature(verts, faces, L, M)
    if k is None:
        k = band_limit((verts, faces))
    k = int(min(max(k, 2), len(verts) - 1))
    evals, eigvecs = eigenbasis(L, M, k)
    eigvecs = eigvecs[:, np.argsort(evals)]
    H_hat, _ = project(H, eigvecs, M, k)
    return {"H": H, "H_hat": H_hat, "k": k, "eigvecs": eigvecs, "M": M}
