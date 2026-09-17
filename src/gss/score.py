"""The geometric severity score: curvature -> sphere -> W2 (paper Sec. 2.1)."""
import time
from dataclasses import dataclass, field

import numpy as np

from . import curvature, ricci_flow, transport
from .mesh import as_mesh, vertex_normals


@dataclass
class Result:
    value: float          # GSS = W2(mu+, mu0)
    k: int                # spectral truncation index used
    w2_minus: float       # W2 of the negative part
    H: np.ndarray         # raw mean curvature
    H_hat: np.ndarray     # truncated mean curvature
    eta: np.ndarray       # loading weight
    u: np.ndarray         # conformal factor
    sphere: np.ndarray    # vertex positions on the unit sphere
    timings: dict = field(default_factory=dict)

    @property
    def weighted(self):
        return self.H_hat * self.eta

    def __float__(self):
        return float(self.value)


def gss(mesh, k=None, loading_dirs=transport.LOADING_DIRS):
    """Score a closed genus-0 surface (a Mesh or a (verts, faces) pair).

    k=None picks the truncation index by the elbow rule; pass an int (e.g.
    `curvature.band_limit(mesh)`) to fix it. Loading is along +/- z by default.
    """
    verts, faces = as_mesh(mesh)
    t = {}

    t0 = time.perf_counter()
    cur = curvature.smoothed_curvature(verts, faces, k=k)
    t["curvature"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u, sphere, _ = ricci_flow.map_to_sphere(verts, faces)
    t["ricci_flow"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    eta = transport.loading_weight(vertex_normals(verts, faces), loading_dirs)
    plus, minus = transport.wasserstein(cur["H_hat"] * eta, sphere, faces)
    t["transport"] = time.perf_counter() - t0

    return Result(plus, cur["k"], minus, cur["H"], cur["H_hat"], eta, u, sphere, t)
