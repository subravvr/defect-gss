"""Test geometries: the spheroid gamma-family and the modulated superellipsoids.

A superellipsoid is described by a parameter dict
    {"eps1", "eps2", "b_over_a", "c_over_a", "delta"}
with implicit function (a = 1)
    f(q) = (|q_x|^(2/e2) + |q_y/b|^(2/e2))^(e2/e1) + |q_z/c|^(2/e1),
evaluated on q = p / (1 + delta * P4(p_z/|p|)). The surface is f = 1. A spheroid
is the special case e1 = e2 = 1, b = 1, delta = 0.

Ellipsoids are rasterized and meshed through an antialiased voxel route (as a
CT defect would be); superellipsoids are meshed directly on the analytic surface.
Meshing needs pymeshlab and trimesh.
"""
import math

import numpy as np

from .mesh import Mesh, enclosed_volume, mass_matrix

TARGET_EDGE = 0.15


# ------------------------------------------------------------------ meshing

def occupancy_grid(coords, pad=5):
    """Binary grid around (N, 3) voxel indices, with `pad` empty voxels each side."""
    coords = np.asarray(coords)
    lo = np.min(coords, axis=0) - pad
    shifted = coords - lo
    grid = np.zeros((np.max(coords, axis=0) + pad - lo + 1).astype(int), dtype=bool)
    grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True
    return grid


def normalize_area(verts, faces):
    """Scale so the barycentric area is (almost exactly) 4*pi."""
    area = mass_matrix(verts, faces).diagonal().sum()
    return verts / (np.sqrt(area / (4 * np.pi)) + 1e-6)


def repair(verts, faces):
    """Drop non-finite vertices; fill holes and fix winding if not watertight."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.remove_infinite_values()
    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
    return mesh.vertices, mesh.faces


def orient_inward(verts, faces):
    """Flip winding, if needed, to a negative signed volume (the stored convention)."""
    if enclosed_volume(verts, faces) > 0:
        faces = faces[:, ::-1].copy()
    return faces


def isotropic_remesh(verts, faces, target_edge=TARGET_EDGE, max_surf_dist=0.1,
                     iterations=10):
    """Botsch-Kobbelt isotropic remesh (pymeshlab).

    `max_surf_dist` is a percentage of the bounding-box diagonal (pymeshlab's unit).
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(verts, dtype=float),
                               np.asarray(faces, dtype=np.int32)))
    ms.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        targetlen=pymeshlab.PureValue(float(target_edge)),
        featuredeg=180.0, iterations=int(iterations), adaptive=False,
        selectedonly=False, splitflag=True, collapseflag=True, swapflag=True,
        smoothflag=True, reprojectflag=True, checksurfdist=True,
        maxsurfdist=pymeshlab.PercentageValue(float(max_surf_dist)),
    )
    m = ms.current_mesh()
    return (np.asarray(m.vertex_matrix(), dtype=float),
            np.asarray(m.face_matrix(), dtype=np.int64))


def antialiased_isosurface(grid, sigma=0.4, supersample=2):
    """Marching cubes on a resampled, low-passed signed distance field.

    `sigma` is in original voxel units. Vertices are returned as float32 in
    original voxel units, as the stored surfaces were built.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
    from skimage.measure import marching_cubes

    grid = np.asarray(grid, bool)
    field = (distance_transform_edt(~grid) - distance_transform_edt(grid)).astype(float)
    if supersample > 1:
        field = zoom(field, supersample, order=3)
    if sigma > 0.0:
        field = gaussian_filter(field, sigma * supersample)
    verts, faces = marching_cubes(-field, level=0.0)[:2]
    return (verts / supersample).astype(np.float32), faces


def antialiased_surface(coords, sigma=0.4, supersample=2, target_edge=TARGET_EDGE,
                        max_surf_dist=0.3):
    """Voxel mask -> remeshed, centred, inward-wound surface of area ~4*pi."""
    grid = occupancy_grid(coords)
    verts, faces = antialiased_isosurface(grid, sigma, supersample)
    verts = normalize_area(verts, faces)
    verts, faces = isotropic_remesh(verts, faces, target_edge, max_surf_dist)
    verts, faces = repair(verts, faces)
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, np.int64)
    return Mesh(verts - verts.mean(axis=0), orient_inward(verts, faces))


# ------------------------------------------------------ spheroid family (Sec. 3.1)

GAMMA_MIN, GAMMA_MAX, N_GAMMA = 0.5, 2.0, 21
RASTER_LONGEST = 15   # voxels across the equal-volume sphere


def gamma_grid(n=N_GAMMA):
    """21 values of gamma = a1/a3, log-uniform on [0.5, 2], gamma = 1 exactly in the middle."""
    return np.exp(np.linspace(np.log(GAMMA_MIN), np.log(GAMMA_MAX), n))


def spheroid_semi_axes(gamma, R=1.0):
    """(a1, a2, a3) of the equal-volume spheroid with a1 = a2 and a1/a3 = gamma."""
    return (R * gamma ** (1 / 3), R * gamma ** (1 / 3), R * gamma ** (-2 / 3))


def spheroid_params(gamma):
    """Semi-axes (1, 1, 1/gamma) as a superellipsoid parameter dict."""
    return {"eps1": 1.0, "eps2": 1.0, "b_over_a": 1.0, "c_over_a": 1.0 / gamma,
            "delta": 0.0}


def rasterize(p, longest, ref_extent, extents):
    """(N, 3) int16 voxel indices of the shape on a lattice of spacing ref/longest.

    `ref_extent` fixes the lattice spacing; `extents` (the shape's own
    bounding-box size) fixes the grid size so nothing is clipped.
    """
    ref = np.asarray(ref_extent, float)
    h = ref.max() / float(longest)
    ns = np.ceil(np.maximum(np.asarray(extents, float), ref) / h).astype(int) + 6
    axes = [(np.arange(n) - (n - 1) / 2.0) * h for n in ns]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occ = (implicit(grid, p) <= 1.0).reshape(ns)
    return np.argwhere(occ).astype(np.int16)


def spheroid_voxels(gamma, longest=RASTER_LONGEST):
    """Voxel mask of the gamma-family member on a lattice fixed by the equal-volume sphere."""
    d = 2.0 * gamma ** (-1 / 3)
    return rasterize(spheroid_params(gamma), longest, (d, d, d), (2.0, 2.0, 2.0 / gamma))


def ellipsoid(gamma, sigma=0.4, supersample=2):
    """Mesh of one family member: voxelized, antialiased and remeshed (area ~4*pi)."""
    return antialiased_surface(spheroid_voxels(gamma), sigma, supersample)


# --------------------------------------------------------- superellipsoids

def _p4(u):
    return (35.0 * u ** 4 - 30.0 * u ** 2 + 3.0) / 8.0


def _dp4(u):
    return (35.0 * u ** 3 - 15.0 * u) / 2.0


def _sp(t, e):
    return np.sign(t) * np.abs(t) ** e


def implicit(pts, p):
    """f(pts); the surface is f = 1 and the interior f < 1."""
    e1, e2 = p["eps1"], p["eps2"]
    b, c, d = p["b_over_a"], p["c_over_a"], p["delta"]
    pts = np.atleast_2d(np.asarray(pts, float))
    r = np.linalg.norm(pts, axis=1)
    r = np.where(r == 0.0, 1e-300, r)
    q = pts / (1.0 + d * _p4(pts[:, 2] / r))[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.abs(q[:, 0]) ** (2 / e2) + np.abs(q[:, 1] / b) ** (2 / e2)) ** (e2 / e1) \
            + np.abs(q[:, 2] / c) ** (2 / e1)
    return np.nan_to_num(f, nan=np.inf)


def project(pts, p):
    """Exact radial projection onto f = 1.

    f is homogeneous of degree 2/e1 along rays, so one scale f^(-e1/2) suffices.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts * (implicit(pts, p) ** (-p["eps1"] / 2.0))[:, None]


def _frame(eta, omega, p):
    """Surface points and both parametric derivatives at (eta, omega)."""
    e1, e2, b, c, d = p["eps1"], p["eps2"], p["b_over_a"], p["c_over_a"], p["delta"]
    ch, sh, co, so = np.cos(eta), np.sin(eta), np.cos(omega), np.sin(omega)
    R0 = np.stack([_sp(ch, e1) * _sp(co, e2), b * _sp(ch, e1) * _sp(so, e2),
                   c * _sp(sh, e1)], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dch = e1 * np.abs(ch) ** (e1 - 1) * (-sh)
        dsh = e1 * np.abs(sh) ** (e1 - 1) * ch
        dco = e2 * np.abs(co) ** (e2 - 1) * (-so)
        dso = e2 * np.abs(so) ** (e2 - 1) * co
    Re = np.stack([dch * _sp(co, e2), b * dch * _sp(so, e2), c * dsh], -1)
    Ro = np.stack([_sp(ch, e1) * dco, b * _sp(ch, e1) * dso, np.zeros_like(ch)], -1)
    if d == 0.0:
        return R0, Re, Ro

    rho = np.linalg.norm(R0, axis=-1)
    u = R0[..., 2] / rho
    m = (1.0 + d * _p4(u))[..., None]

    def dm(dR):
        drho = np.einsum("...i,...i->...", R0, dR) / rho
        du = dR[..., 2] / rho - R0[..., 2] * drho / rho ** 2
        return (d * _dp4(u) * du)[..., None]

    return R0 * m, Re * m + R0 * dm(Re), Ro * m + R0 * dm(Ro)


def _graded_nodes(n, q=5):
    """Gauss-Legendre nodes on [0, 1], graded toward both ends."""
    x, w = np.polynomial.legendre.leggauss(n)
    v, wv = (x + 1.0) / 2.0, w / 2.0
    vq, mq = v ** q, (1.0 - v) ** q
    g = vq / (vq + mq)
    dg = q * v ** (q - 1) * (1.0 - v) ** (q - 1) / (vq + mq) ** 2
    return g, wv * dg


def volume_area(p, n=160):
    """Exact (quadrature) enclosed volume and surface area of {f = 1}."""
    s, ws = _graded_nodes(n)
    eta, weta = s * np.pi / 2, ws * np.pi / 2
    H, O = np.meshgrid(eta, eta, indexing="ij")
    W = np.outer(weta, weta)
    R, Re, Ro = _frame(H, O, p)
    N = np.cross(Re, Ro)
    area = 8.0 * float(np.sum(W * np.linalg.norm(N, axis=-1)))
    vol = 8.0 * float(np.sum(W * np.abs(np.einsum("ijk,ijk->ij", R, N)))) / 3.0
    return vol, area


def _ideal_icosphere(subdivisions):
    """Unit icosphere. Vertex ordering matters for bitwise reproduction of the
    stored superellipsoid meshes, so this is kept separate from
    `geometry.icosphere`."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    for _ in range(int(subdivisions)):
        mid, vlist, new = {}, list(verts), []

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                mid[key] = len(vlist)
                vlist.append((verts[a] + verts[b]) / 2.0)
            return mid[key]

        for i, j, k in faces:
            a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            new += [[i, a, c], [j, b, a], [k, c, b], [a, b, c]]
        verts, faces = np.asarray(vlist, float), np.asarray(new, np.int64)
    return verts / np.linalg.norm(verts, axis=1)[:, None], faces


def _rotation(seed):
    """Random rotation for a re-triangulation replicate; None is the identity."""
    if seed is None:
        return np.eye(3)
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q * (-1.0 if np.linalg.det(q) < 0 else 1.0)


def superellipsoid(p, seed=None, target_edge=TARGET_EDGE,
                           subdivisions=6, rounds=3):
    """Mesh the analytic surface directly, without voxelization (Sec. 3.3).

    A fine icosphere is projected onto the surface (scaled to area 4*pi), then
    isotropically remeshed and re-projected `rounds` times. `seed` rotates the
    starting icosphere, which gives a different triangulation of the same shape.
    """
    s = math.sqrt(4.0 * math.pi / volume_area(p)[1])

    def on_surface(v):
        return s * project(np.asarray(v, float) / s, p)

    v0, faces = _ideal_icosphere(subdivisions)
    verts = on_surface((v0 @ _rotation(seed).T) * s)
    for _ in range(rounds):
        verts, faces = isotropic_remesh(verts, faces, target_edge)
        verts = on_surface(verts)

    verts, faces = repair(np.asarray(verts, float), np.asarray(faces, np.int64))
    faces = np.asarray(faces, np.int64)
    # The shape is centred at the origin by symmetry, so it is not re-centred.
    verts = on_surface(verts)
    return Mesh(verts, orient_inward(verts, faces))
