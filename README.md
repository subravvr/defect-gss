# defect-gss

This repository holds the geometric severity score (GSS) and the test geometries from *Assessment of geometric severity for defects in laser powder bed fusion* (V. Subraveti, C. Oskay).

The GSS ranks a defect by the shape of its surface. It works in four steps:
1. Smooth the mean curvature spectrally.
2. Weight it by the surface orientation relative to the load.
3. Map the surface onto the unit sphere by discrete Ricci flow.
4. Compare the resulting measure with the uniform measure using the 2-Wasserstein distance.

## Install

```bash
conda env create -f environment.yml    # or: pip install -e ".[meshing,test]"
conda activate defect-gss
```

Only NumPy, SciPy, POT and matplotlib are needed to score meshes. Rebuilding the geometries also needs `pymeshlab`, `trimesh` and `scikit-image`. Install the package in editable mode (`-e`) so that `gss.load` can find `data/`.

## Usage

```python
from gss import load, gss, band_limit

mesh = load("ellipsoids", "g05")        # Mesh(verts, faces)
r = gss(mesh)                           # elbow-rule truncation
r.value, r.k, r.timings

m = load("superellipsoids", "G1A")
gss(m, k=band_limit(m)).value           # band-limit truncation

gss((verts, faces))                     # any closed genus-0 surface
```

Input surfaces must be closed and genus 0, area-normalized to 4π, and wound inward (negative signed volume). The load is along ±z unless you pass `loading_dirs=`.

The returned `Result` also holds the intermediate fields: `H`, `H_hat`, `eta`, `u` (conformal factor) and `sphere` (vertex positions on the unit sphere).

To mesh new shapes:

```python
from gss import shapes
shapes.ellipsoid(gamma=1.5)                           # voxelized, antialiased, remeshed
shapes.superellipsoid({"eps1": 0.8, "eps2": 2.5, "b_over_a": 0.76,
                       "c_over_a": 0.39, "delta": 0.0}, seed=None)
```

## Layout

```
src/gss/
  mesh.py          Mesh, load, discrete geometry (cotangent Laplacian, mass matrix, ...)
  curvature.py     mean curvature, eigenbasis, elbow and band-limit truncation
  ricci_flow.py    conformal factor (Newton) and spherical embedding
  transport.py     loading weight, measures on the sphere, exact W2 (POT)
  score.py         gss(mesh) -> Result
  shapes.py        ellipsoid family and superellipsoid meshing
  descriptors.py   sphericity, aspect ratios, Murakami sqrt(area)
  plotting.py      figure style
scripts/
  compute_gss.py           GSS and descriptors for all stored geometries -> results/
  generate_geometries.py   rebuild data/*.npz
  figures.py               gamma_severity, descriptor_degeneracy_pairs -> figures/
data/
  ellipsoids.npz/.json        21 spheroids (g00-g20): surfaces, voxel masks, gamma
  superellipsoids.npz/.json   pairs G1A-G4B: surfaces and shape parameters
  reference_values.json       published Lambda and K_t (for the figures), GSS (for the tests)
results/, figures/            outputs of the scripts
```

**Ellipsoids.** These are 21 equal-volume spheroids. The ratio γ = a₁/a₃ is log-uniform on [0.5, 2], and the load is along a₃. Each one is rasterized at 15 voxels across its equal-volume sphere, then meshed:
1. Resample the signed distance field 2× (cubic).
2. Apply a Gaussian filter with σ = 0.4 voxel.
3. Extract the surface with marching cubes.
4. Remesh isotropically with edge length 0.15.

**Superellipsoids.** These are four pairs whose members agree on Murakami's √area, sphericity and both aspect ratios. The analytic surface is meshed directly: a fine icosphere is projected onto it, then remeshed and re-projected three times.

## Reproducing the paper's values

```bash
python scripts/compute_gss.py --replicates 9   # ~5 min
python scripts/figures.py
pytest
```

- The superellipsoid GSS in Table 2 is `gss_bandlimit_mean`. It is the mean over 9 re-triangulations (the stored mesh plus 8 rotated starting icospheres), each truncated with `band_limit`. One G3A replicate (seed 6) fails to embed and is skipped, as it was in the published run.
- `generate_geometries.py` reproduces the stored meshes bit for bit, on the package versions in `environment.yml`.
- ARPACK starts from a random vector, so GSS reproduces to about 1e-13.

## Citation

See `CITATION.cff`.
