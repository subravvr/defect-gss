"""Rebuild data/ellipsoids.npz and data/superellipsoids.npz (needs pymeshlab).

    python scripts/generate_geometries.py [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gss import shapes
from gss.mesh import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    ell = {}
    for sid, s in json.loads((DATA / "ellipsoids.json").read_text())["shapes"].items():
        mesh = shapes.ellipsoid(s["gamma"])
        ell.update({f"{sid}__verts": mesh.verts, f"{sid}__faces": mesh.faces,
                    f"{sid}__voxels": shapes.spheroid_voxels(s["gamma"])})
        print(sid, len(mesh.verts), flush=True)
    np.savez_compressed(out / "ellipsoids.npz", **ell)

    sup = {}
    for name, p in json.loads((DATA / "superellipsoids.json").read_text())["shapes"].items():
        mesh = shapes.superellipsoid(p)
        sup.update({f"{name}__verts": mesh.verts, f"{name}__faces": mesh.faces})
        print(name, len(mesh.verts), flush=True)
    np.savez_compressed(out / "superellipsoids.npz", **sup)


if __name__ == "__main__":
    main()
