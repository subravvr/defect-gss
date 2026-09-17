# Geometric severity score for defects in laser powder bed fusion

This repository holds the geometric severity score (GSS) and the test geometries from *Assessment of geometric severity for defects in laser powder bed fusion* (V. Subraveti, C. Oskay). You can cite this repo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817560.svg)](https://doi.org/10.5281/zenodo.22817560).

The GSS ranks a defect by the shape of its surface. It works in four steps:
1. Smooth the mean curvature spectrally, truncating at the Nyquist band limit of the triangulation and keeping the constant (mean) mode.
2. Weight it by the surface orientation relative to the load.
3. Map the surface onto the unit sphere by discrete Ricci flow.
4. Compare the resulting measure with the uniform measure using the 2-Wasserstein distance.

## Install

```bash
conda env create -f environment.yml    # or: pip install -e ".[meshing,test]"
conda activate defect-gss
```

Only NumPy, SciPy, POT and matplotlib are needed to score meshes. Rebuilding the geometries also needs `pymeshlab` and `trimesh`. Install the package in editable mode (`-e`) so that `gss.load` can find `data/`.

## Usage

```python
from gss import load, gss, band_limit

mesh = load("ellipsoids", "g05")        # Mesh(verts, faces)
r = gss(mesh)                           # band-limit truncation, n = 2
r.value, r.k, r.timings

m = load("superellipsoids", "G1A")
band_limit(m), gss(m, k=300).value      # the default index, or a fixed one

gss((verts, faces))                     # any closed genus-0 surface
```

Input surfaces must be closed and genus 0, area-normalized to 4π, and wound inward (negative signed volume). The load is along ±z unless you pass `loading_dirs=`.

The returned `Result` also holds the intermediate fields: `H`, `H_hat`, `eta`, `u` (conformal factor) and `sphere` (vertex positions on the unit sphere).

The Bayesian severity model (paper Sec. 2.3) fits Λ against GSS with a conjugate normal-inverse-gamma prior. It then assigns severity tiers from the posterior predictive distribution:

```python
from gss import blr
out = blr.fit(x_cal, y_cal, x_val, y_val)   # GSS and Lambda arrays per split
out["posterior"], out["tier_edges"], out["validation"]["confusion"]
```

`fit` returns the posterior on the Λ scale, the quartile tier edges, and the following for each split:
- predictions and scores;
- tier probabilities and median-tier calls;
- the confusion matrix, with accuracy and quadratic-weighted κ;
- a 95% predictive band.

To mesh new shapes:

```python
from gss import shapes
shapes.ellipsoid(gamma=1.5, seed=None)                # analytic surface, remeshed
shapes.superellipsoid({"eps1": 0.8, "eps2": 2.5, "b_over_a": 0.76,
                       "c_over_a": 0.39, "delta": 0.0}, seed=None)
```

## Layout

```
src/gss/
  mesh.py          Mesh, load, discrete geometry (cotangent Laplacian, mass matrix, ...)
  curvature.py     mean curvature, eigenbasis, band-limit truncation
  ricci_flow.py    conformal factor (Newton) and spherical embedding
  transport.py     loading weight, measures on the sphere, exact W2 (POT)
  score.py         gss(mesh) -> Result
  shapes.py        ellipsoid family and superellipsoid meshing
  descriptors.py   sphericity, aspect ratios, Murakami sqrt(area)
  blr.py           conjugate Bayesian linear regression and severity tiers
  plotting.py      figure style
scripts/
  compute_gss.py           GSS and descriptors for all stored geometries -> results/
  generate_geometries.py   rebuild data/*.npz
  figures.py               gamma_severity, descriptor_degeneracy_pairs -> figures/
data/
  ellipsoids.npz/.json        21 spheroids (g00-g20): surfaces and gamma
  superellipsoids.npz/.json   pairs G1A-G4B: surfaces and shape parameters
  reference_values.json       published Lambda and K_t (for the figures), GSS (for the tests)
results/, figures/            outputs of the scripts
```

**Truncation.** The curvature is expanded in the Laplace-Beltrami eigenbasis and kept up to n_e = πA/(n²h̄²) modes, with n = 2 samples per wavelength and h̄ the mean edge length. This is the Nyquist band limit given by Weyl's law. The constant mode is kept, so a convex surface keeps H > 0 everywhere.

**Ellipsoids.** These are 21 equal-volume spheroids with γ = a₁/a₃ log-uniform on [0.5, 2] and the load along a₃. Each is meshed on its analytic surface: a fine icosphere is projected onto it, then remeshed isotropically to edge length 0.15. GSS rises monotonically with γ, in the same order as Λ.

**Superellipsoids.** These are four pairs whose members agree on Murakami's √area, sphericity and both aspect ratios. They are meshed the same way as the ellipsoids.

## Reproducing the paper's values

```bash
python scripts/compute_gss.py --replicates 9   # ~10 min
python scripts/figures.py
pytest
```

- The paper reports `gss_mean`: the mean over the stored mesh and 8 re-meshings from rotated starting icospheres. One G3A re-meshing (seed 6) fails to embed and is skipped.
- `generate_geometries.py` reproduces the stored meshes bit for bit, on the package versions in `environment.yml`.
- ARPACK starts from a random vector, so GSS reproduces to about 1e-13.

## Data availability

This repository contains all data in the paper except the XCT defects:
- the ellipsoid and superellipsoid geometries;
- their GSS values;
- the reference strain energy ratios and K_t values.

The 92 segmented XCT defect surfaces and voxel masks analyzed in the paper come from X-ray computed tomography reconstructions provided by collaborators at Carnegie Mellon University. They are not ours to redistribute. They will be made publicly available with a forthcoming publication, and a link to the data source will be added here when it is released.

## Citation

See `CITATION.cff`.

## AI Usage
The original code was developed with minimal AI assistance (Copilot). Claude was used to generate plotting code, write tests, and package the code into a GitHub-friendly repo.
