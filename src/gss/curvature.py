"""Mean curvature and its spectral smoothing (paper Sec. 2.1, steps 1-2).

H is estimated with the cotangent Laplace-Beltrami operator, projected onto the
generalized eigenbasis L phi = lambda M phi, and truncated at a per-defect
index n_e chosen by the elbow of the cumulative spectral energy.
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
    """Reconstruction of `field` on modes 1..k-1 (the constant mode is dropped).

    Returns (reconstruction, coefficients).
    """
    coeffs = eigvecs[:, 1:k].T @ (M @ field)
    return eigvecs[:, 1:k] @ coeffs, coeffs


def elbow_cutoff(field, eigvecs, M):
    """Truncation index by the elbow of the cumulative spectral energy C(m).

    The elbow is the point farthest from the secant joining the first and last
    points of C(m). The returned k is passed straight to `project`, which keeps
    modes 1..k-1.
    """
    _, coeffs = project(field, eigvecs, M, eigvecs.shape[1])
    energy = np.cumsum(coeffs**2) / np.sum(coeffs**2)

    x = np.arange(len(energy))
    line = np.array([x[-1] - x[0], energy[-1] - energy[0]], dtype=float)
    line /= np.linalg.norm(line)
    pts = np.stack([x, energy], axis=1) - np.array([x[0], energy[0]])
    dist = np.linalg.norm(pts - (pts @ line)[:, None] * line, axis=1)
    return int(np.argmax(dist)) + 1


def band_limit(mesh, samples_per_wavelength=4.0):
    """Truncation index from the mesh resolution instead of the elbow.

    Weyl's law with the mean edge h: k = pi A / (n^2 h^2). Used for the
    superellipsoid table, where the elbow is bimodal across re-triangulations.
    """
    v, f = as_mesh(mesh)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    h = float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())
    t = v[f]
    area = float(0.5 * np.linalg.norm(
        np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
    k = np.pi * area / (samples_per_wavelength**2 * h**2)
    return int(min(max(round(k), 2), len(v) - 1))


def smoothed_curvature(verts, faces, k=None, n_eigs=None):
    """H, its truncated reconstruction H_hat, and the truncation index used.

    k=None selects the index with `elbow_cutoff`. n_eigs=None solves the full
    spectrum (n_verts - 1 modes), which is what the published results used.
    """
    _, L, M = operators(verts, faces)
    H = mean_curvature(verts, faces, L, M)
    n = len(verts) - 1 if n_eigs is None else min(n_eigs, len(verts) - 1)
    _, eigvecs = eigenbasis(L, M, n)
    if k is None:
        k = elbow_cutoff(H, eigvecs, M)
    else:
        k = int(min(max(k, 2), n))
    H_hat, _ = project(H, eigvecs, M, k)
    return {"H": H, "H_hat": H_hat, "k": k, "eigvecs": eigvecs, "M": M}
