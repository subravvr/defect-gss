"""Directional weighting and optimal transport on the sphere (Sec. 2.1, App. B).

GSS = W2(mu+, mu0): the 2-Wasserstein distance, under the geodesic cost, from
the positive part of the weighted curvature field (as a measure on the sphere)
to the sphere's own area measure.
"""
import numpy as np

from .mesh import mass_matrix

LOADING_DIRS = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

# POT's default (1e5) is silently exceeded on meshes above ~10k vertices, and
# the returned value is then wrong. 1e7 covers the largest meshes used here.
EMD_MAX_ITER = 10_000_000


def loading_weight(normals, loading_dirs=LOADING_DIRS):
    """eta_i = sum_d (1 - |n_i . d|): large where the surface is parallel to the load."""
    w = np.zeros(normals.shape[0])
    for d in np.atleast_2d(loading_dirs):
        w += 1.0 - np.abs(np.dot(normals, d))
    return w


def emd_exact(mu, nu, cost):
    """Exact optimal transport cost; raises if the network simplex did not converge."""
    import ot

    value, log = ot.emd2(mu, nu, cost, numItermax=EMD_MAX_ITER, log=True)
    if log.get("result_code") != 1:
        raise RuntimeError(f"optimal transport did not converge: {log.get('warning')!r}")
    return value


def sphere_measures(field, sphere_verts, faces):
    """(mu_plus, mu_minus, mu0, theta) on the sphere mesh.

    mu+/- weight each vertex by the positive/negative part of `field` times its
    spherical barycentric area; mu0 is the area measure itself; theta is the
    geodesic distance matrix. An empty part is returned as None.
    """
    s = sphere_verts / np.linalg.norm(sphere_verts, axis=1, keepdims=True)
    theta = np.arccos(np.clip(s @ s.T, -1, 1))

    mu0 = mass_matrix(s, faces).diagonal() / (4 * np.pi)
    mu0 /= sum(mu0)

    area = mass_matrix(sphere_verts, faces).diagonal()
    plus = np.maximum(field, 0.0) * area
    minus = np.abs(np.minimum(field, 0.0)) * area
    mu_plus = plus / plus.sum() if plus.sum() > 0 else None
    mu_minus = minus / minus.sum() if minus.sum() > 0 else None
    return mu_plus, mu_minus, mu0, theta


def wasserstein(field, sphere_verts, faces, order=2):
    """(W_p(mu+, mu0), W_p(mu-, mu0)) under the geodesic cost; None for an empty part."""
    mu_plus, mu_minus, mu0, theta = sphere_measures(field, sphere_verts, faces)
    cost = theta if order == 1 else theta ** order

    def dist(mu):
        if mu is None:
            return None
        v = emd_exact(mu, mu0, cost)
        return v if order == 1 else np.sqrt(v) if order == 2 else v ** (1.0 / order)

    return dist(mu_plus), dist(mu_minus)
