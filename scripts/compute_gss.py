"""GSS and descriptors for every stored geometry -> results/*.csv.

    python scripts/compute_gss.py [--replicates 9]

`gss` is the score on the stored mesh. With --replicates N, each shape is also
re-meshed N-1 times (rotated starting icosphere, needs pymeshlab) and
`gss_mean` / `gss_sd` summarize the N meshes; the paper reports `gss_mean`.
Replicates whose spherical embedding fails are skipped and listed.
"""
import argparse
import csv
import json

import numpy as np

from gss import gss, load, shapes
from gss.descriptors import describe
from gss.mesh import DATA

RESULTS = DATA.parent / "results"


def write(name, rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name}")


def replicate(mesh, remesh, n):
    """Mean, sd and failures of the GSS over the stored mesh and n-1 re-meshings."""
    values, failed = [], []
    for seed in [None] + list(range(1, n)):
        m = mesh if seed is None else remesh(seed)
        try:
            values.append(gss(m).value)
        except RuntimeError:
            failed.append(str(seed))
    return {"gss_mean": float(np.mean(values)), "gss_sd": float(np.std(values, ddof=1)),
            "n_replicates": len(values), "failed_seeds": " ".join(failed)}


def family(name, meta, remesh, extra, replicates):
    rows = []
    for sid, mesh in load(name).items():
        r = gss(mesh)
        row = {"shape": sid, **extra(sid), "n_verts": len(mesh.verts), "k": r.k,
               "gss": r.value, **describe(mesh)}
        if replicates:
            row.update(replicate(mesh, lambda seed: remesh(sid, seed), replicates))
        rows.append(row)
        print(f"{sid}  GSS {r.value:.4f}"
              + (f"  mean {row['gss_mean']:.4f} +/- {row['gss_sd']:.4f}" if replicates else ""),
              flush=True)
    write(f"gss_{name}.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=0)
    n = ap.parse_args().replicates

    ell = json.loads((DATA / "ellipsoids.json").read_text())["shapes"]
    family("ellipsoids", ell, lambda s, seed: shapes.ellipsoid(ell[s]["gamma"], seed),
           lambda s: {"gamma": ell[s]["gamma"]}, n)

    sup = json.loads((DATA / "superellipsoids.json").read_text())["shapes"]
    family("superellipsoids", sup, lambda s, seed: shapes.superellipsoid(sup[s], seed),
           lambda s: dict(sup[s]), n)


if __name__ == "__main__":
    main()
