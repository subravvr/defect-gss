import json

import numpy as np
import pytest

from gss import band_limit, gss, load, ricci_flow, transport
from gss.mesh import DATA, enclosed_volume, icosphere

REF = json.loads((DATA / "reference_values.json").read_text())


def test_structureless_field_scores_zero():
    verts, faces = icosphere(3)
    plus, minus = transport.wasserstein(np.ones(len(verts)), verts, faces)
    assert plus < 1e-6 and minus is None


def test_ricci_flow_maps_to_unit_sphere():
    verts, faces = icosphere(3)
    _, sphere, residual = ricci_flow.map_to_sphere(verts * [1.5, 1.0, 0.6], faces)
    assert residual < ricci_flow.TOL
    assert np.allclose(np.linalg.norm(sphere, axis=1), 1.0)


@pytest.mark.parametrize("family", ["ellipsoids", "superellipsoids"])
def test_load(family):
    meshes = load(family)
    assert len(meshes) == {"ellipsoids": 21, "superellipsoids": 8}[family]
    for m in meshes.values():
        assert m.faces.dtype == np.int64 and enclosed_volume(*m) < 0


def test_published_values():
    assert abs(gss(load("ellipsoids", "g10")).value - REF["ellipsoids"]["g10"]["gss"]) < 1e-10
    mesh, ref = load("superellipsoids", "G1A"), REF["superellipsoids"]["G1A"]
    assert band_limit(mesh) == ref["k_replicates"][0]
    assert abs(gss(mesh, k=band_limit(mesh)).value - ref["gss_replicates"][0]) < 1e-10
