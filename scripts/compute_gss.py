"""GSS and descriptors for every stored geometry -> results/*.csv.

    python scripts/compute_gss.py [--replicates 9]

With --replicates N, each superellipsoid is also scored on N-1 re-triangulations
(rotated starting icosphere, needs pymeshlab) with the band-limit truncation;
the mean is the value in the paper's Table 2. Replicates whose spherical
embedding fails are skipped and listed.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gss import band_limit, gss, load, shapes
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


def ellipsoids():
    meta = json.loads((DATA / "ellipsoids.json").read_text())["shapes"]
    rows = []
    for sid, mesh in load("ellipsoids").items():
        r = gss(mesh)
        rows.append({"shape": sid, "gamma": meta[sid]["gamma"], "n_verts": len(mesh.verts),
                     "k": r.k, "gss": r.value, **describe(mesh)})
        print(f"{sid}  gamma {meta[sid]['gamma']:.4f}  GSS {r.value:.4f}", flush=True)
    write("gss_ellipsoids.csv", rows)


def superellipsoids(replicates):
    params = json.loads((DATA / "superellipsoids.json").read_text())["shapes"]
    rows = []
    for name, mesh in load("superellipsoids").items():
        row = {"shape": name, "n_verts": len(mesh.verts), **params[name],
               **describe(mesh), "gss_elbow": gss(mesh).value}
        if replicates:
            values, failed = [], []
            for seed in [None] + list(range(1, replicates)):
                m = mesh if seed is None else shapes.superellipsoid(params[name], seed=seed)
                try:
                    values.append(gss(m, k=band_limit(m)).value)
                except RuntimeError:
                    failed.append(str(seed))
            row.update({"gss_bandlimit_mean": float(np.mean(values)),
                        "gss_bandlimit_sd": float(np.std(values, ddof=1)),
                        "n_replicates": len(values), "failed_seeds": " ".join(failed)})
        rows.append(row)
        print(f"{name}  GSS elbow {row['gss_elbow']:.4f}"
              + (f"  band-limit mean {row['gss_bandlimit_mean']:.4f}" if replicates else ""),
              flush=True)
    write("gss_superellipsoids.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=0)
    args = ap.parse_args()
    ellipsoids()
    superellipsoids(args.replicates)


if __name__ == "__main__":
    main()
