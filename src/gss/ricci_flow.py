"""Conformal map of a genus-0 surface to the unit sphere (paper Appendix A).

The discrete Ricci flow dw/dt = K* - K(w), with target curvature 4 pi / n_v, is
driven to steady state by Newton's method. Sphere coordinates are then read off
the first three non-constant eigenvectors of L(w*) phi = lambda M(w*) phi.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, spsolve

from .mesh import (angle_defect, conformal_edge_lengths, corner_angles,
                   cotangent_laplacian, edge_lengths, mass_matrix_from_lengths)

MAX_ITER = 20
TOL = 1e-5


def conformal_factor(verts, faces, max_iter=MAX_ITER, tol=TOL):
    """Newton iteration for the conformal factor w; returns (w, residual)."""
    n = len(verts)
    u = np.zeros(n)
    l0 = edge_lengths(verts, faces)
    target = np.full(n, 4 * np.pi / n)
    # L is singular (constant null vector); pinning one diagonal entry fixes it.
    pin = csr_matrix(([1.0], ([0], [0])), shape=(n, n))

    for _ in range(max_iter):
        angles = corner_angles(*conformal_edge_lengths(u, faces, *l0))
        K = angle_defect(faces, angles, n)
        residual = np.linalg.norm(K - target)
        if residual < tol:
            break
        u += spsolve(cotangent_laplacian(faces, angles, n) + pin, target - K)
        u -= np.mean(u)
    else:
        angles = corner_angles(*conformal_edge_lengths(u, faces, *l0))
        residual = np.linalg.norm(angle_defect(faces, angles, n) - target)
    return u, float(residual)


def sphere_embedding(u, verts, faces):
    """Unit-sphere coordinates for each vertex from the flowed metric."""
    n = len(verts)
    lengths = conformal_edge_lengths(u, faces, *edge_lengths(verts, faces))
    L = cotangent_laplacian(faces, corner_angles(*lengths), n)
    M = mass_matrix_from_lengths(*lengths, faces, n)
    try:
        evals, evecs = eigsh(L, k=4, M=M, sigma=0.0)
    except Exception:
        # Rare factorization failure on degenerate meshes; a tiny shift avoids it.
        evals, evecs = eigsh(L, k=4, M=M, sigma=-1e-8)
    # ARPACK does not order its output, so sort before dropping the constant mode.
    evecs = evecs[:, np.argsort(evals)]
    coords = evecs[:, 1:4].copy()
    return coords / np.linalg.norm(coords, axis=1, keepdims=True)


def map_to_sphere(verts, faces, max_iter=MAX_ITER, tol=TOL):
    """(w, sphere_verts, residual) for a closed genus-0 surface."""
    u, residual = conformal_factor(verts, faces, max_iter, tol)
    return u, sphere_embedding(u, verts, faces), residual
