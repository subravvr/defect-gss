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

    for name, build in (("ellipsoids", lambda s: shapes.ellipsoid(s["gamma"])),
                        ("superellipsoids", shapes.superellipsoid)):
        arrays = {}
        for sid, s in json.loads((DATA / f"{name}.json").read_text())["shapes"].items():
            mesh = build(s)
            arrays[f"{sid}__verts"], arrays[f"{sid}__faces"] = mesh
            print(sid, len(mesh.verts), flush=True)
        np.savez_compressed(out / f"{name}.npz", **arrays)


if __name__ == "__main__":
    main()
