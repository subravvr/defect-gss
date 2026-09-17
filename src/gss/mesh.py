"""Triangle meshes: the Mesh type, stored geometries, and discrete geometry.

Surfaces are closed, genus 0, area-normalized to 4*pi and wound inward
(negative signed volume).
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix, diags

DATA = Path(__file__).resolve().parents[2] / "data"
FAMILIES = ("ellipsoids", "superellipsoids")


class Mesh(NamedTuple):
    verts: np.ndarray
    faces: np.ndarray


def as_mesh(obj):
    """A Mesh from a Mesh, an object with .verts/.faces, or a (verts, faces) pair."""
    verts, faces = (obj.verts, obj.faces) if hasattr(obj, "verts") else obj
    return Mesh(np.asarray(verts, float), np.asarray(faces, np.int64))


def load(family, name=None):
    """A stored geometry, e.g. load("ellipsoids", "g05"); all of them if name is None."""
    family = family if family.endswith("s") else family + "s"
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    with np.load(DATA / f"{family}.npz") as z:
        names = sorted({k.split("__")[0] for k in z.files if k.endswith("__verts")})
        pick = names if name is None else [name]
        out = {n: as_mesh((z[f"{n}__verts"], z[f"{n}__faces"])) for n in pick}
    return out if name is None else out[name]


def edge_lengths(verts, faces):
    """Lengths of the edges (v0,v1), (v1,v2), (v2,v0) of every face."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    l01 = np.linalg.norm(v0 - v1, axis=1)
    l12 = np.linalg.norm(v1 - v2, axis=1)
    l20 = np.linalg.norm(v2 - v0, axis=1)
    return l01, l12, l20


def conformal_edge_lengths(u, faces, l01, l12, l20):
    """Edge lengths under the conformal factor u: l_ij * exp((u_i + u_j) / 2)."""
    return (l01 * np.exp((u[faces[:, 0]] + u[faces[:, 1]]) / 2),
            l12 * np.exp((u[faces[:, 1]] + u[faces[:, 2]]) / 2),
            l20 * np.exp((u[faces[:, 2]] + u[faces[:, 0]]) / 2))


def corner_angles(l01, l12, l20):
    """(3, n_faces) interior angles at v0, v1, v2 by the law of cosines."""
    cos_0 = (l01**2 + l20**2 - l12**2) / (2 * l01 * l20)
    cos_1 = (l01**2 + l12**2 - l20**2) / (2 * l01 * l12)
    cos_2 = (l12**2 + l20**2 - l01**2) / (2 * l12 * l20)
    return np.arccos(np.clip([cos_0, cos_1, cos_2], -1.0, 1.0))


def angle_defect(faces, angles, n_verts):
    """Discrete Gaussian curvature K_i = 2*pi - sum of incident corner angles."""
    K = np.full(n_verts, 2 * np.pi)
    for i in range(3):
        np.add.at(K, faces[:, i], -angles[i])
    return K


def cotangent_laplacian(faces, angles, n_verts):
    """Positive semi-definite cotangent Laplacian L = D - W.

    Note: W carries cot(a) + cot(b) without the conventional factor 1/2, so the
    eigenvalues are twice the textbook ones. Eigenvectors are unaffected.
    """
    cot0 = np.cos(angles[0]) / np.sin(angles[0])
    cot1 = np.cos(angles[1]) / np.sin(angles[1])
    cot2 = np.cos(angles[2]) / np.sin(angles[2])

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0]])
    J = np.concatenate([faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2]])
    W = np.concatenate([cot2, cot2, cot0, cot0, cot1, cot1])

    W = csr_matrix((W, (I, J)), shape=(n_verts, n_verts))
    d = np.asarray(W.sum(axis=1)).ravel()
    D = csr_matrix((d, (np.arange(n_verts), np.arange(n_verts))))
    return D - W


def _lump(areas, faces, n_verts):
    vertex_areas = np.zeros(n_verts)
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], areas / 3.0)
    return diags(vertex_areas)


def mass_matrix(verts, faces):
    """Sparse diagonal barycentric mass matrix (1/3 of incident face area)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return _lump(areas, faces, len(verts))


def heron_areas(l01, l12, l20):
    """Triangle areas from edge lengths (Kahan's numerically stable Heron).

    Needed during Ricci flow, where the metric exists only as edge lengths.
    The parenthesisation matters for needle triangles; do not simplify it.
    """
    s = np.sort(np.stack([l01, l12, l20], axis=1), axis=1)[:, ::-1]
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    radicand = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return 0.25 * np.sqrt(np.maximum(radicand, 0.0))


def mass_matrix_from_lengths(l01, l12, l20, faces, n_verts):
    """Barycentric mass matrix of a metric given only by edge lengths."""
    return _lump(heron_areas(l01, l12, l20), faces, n_verts)


def vertex_normals(verts, faces):
    """Unit angle-weighted vertex normals."""
    V, F = np.asarray(verts), np.asarray(faces)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15, None)

    def angle(a, b):
        cos = np.einsum("ij,ij->i", a, b) / np.clip(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-15, None)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    n = np.zeros_like(V, dtype=float)
    np.add.at(n, F[:, 0], fn * angle(v1 - v0, v2 - v0)[:, None])
    np.add.at(n, F[:, 1], fn * angle(v2 - v1, v0 - v1)[:, None])
    np.add.at(n, F[:, 2], fn * angle(v0 - v2, v1 - v2)[:, None])
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-15, None)


def surface_area(verts, faces):
    p = verts[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def enclosed_volume(verts, faces):
    """Signed enclosed volume (divergence theorem); the sign encodes winding."""
    p = verts[faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def midpoint_subdivide(verts, faces):
    """One 1-to-4 midpoint subdivision. The polyhedron itself is unchanged."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n, m = len(verts), len(faces)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
    new_verts = np.vstack([verts, 0.5 * (verts[uniq[:, 0]] + verts[uniq[:, 1]])])
    ab, bc, ca = n + inv.reshape(3, m)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    new_faces = np.vstack([
        np.stack([a, ab, ca], axis=1),
        np.stack([ab, b, bc], axis=1),
        np.stack([ca, bc, c], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ]).astype(np.int64)
    return new_verts, new_faces


def icosphere(subdivisions):
    """Unit icosphere with outward winding."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], dtype=float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(subdivisions):
        verts, faces = midpoint_subdivide(verts, faces)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, faces
