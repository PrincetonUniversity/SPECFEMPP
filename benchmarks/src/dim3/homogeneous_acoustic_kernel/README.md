# Homogeneous Acoustic Halfspace — 3-D Kernel Benchmark

## Overview

This benchmark computes cross-correlation traveltime sensitivity kernels
(Fréchet derivatives) for a 3-D homogeneous acoustic halfspace following
the adjoint method of **Tromp, Tape & Liu (2005)**.

The workflow runs four stages driven by Snakemake:

1. **Mesh generation** — `xmeshfem3D` discretises the domain.
2. **Forward simulation** — SPECFEM++ propagates a Ricker-wavelet force
   source and records pressure seismograms at a surface array.
3. **Adjoint source computation** — a Python script converts the
   cross-correlation traveltime misfit into an adjoint source
   (Eq. 45 of Tromp et al. 2005).
4. **Combined (adjoint) simulation** — SPECFEM++ back-propagates the adjoint
   wavefield and accumulates the Fréchet kernels on-the-fly.

## Domain & Numerical Parameters

| Parameter | Value |
|---|---|
| Domain (x × y × z) | 134 444 × 134 444 × 60 000 m |
| Element count (NEX_XI × NEX_ETA × vertical) | 18 × 18 × 8 |
| GLL order | 4 (5 × 5 × 5 GLL points / element) |
| Medium | Acoustic isotropic — ρ = 2 300 kg/m³, Vp = 2 800 m/s |
| Time step | dt = 0.2 s |
| Steps | 500 (T_total = 100 s) |

## Source & Receivers

* **Forward source** — body-force at (67 222, 67 222, −30 000) m,
  Ricker wavelet f₀ = 0.1 Hz, amplitude = 10¹⁰ N.
* **Receivers** — surface array along y at x = 67 000 m, depths ≈ 50 m:

| Station | y (m) |
|---|---|
| DB.X20 | 22 732 |
| DB.X30 | 34 696 |
| DB.X40 | 46 661 |
| DB.X50 | 58 625 |

The P-wave traveltime from source to receiver X20 is approximately 19.2 s.

## Adjoint Source Window

The cross-correlation traveltime adjoint source is computed using a
Hanning taper over **t ∈ [14.0, 24.0] s**, centred on the direct P arrival.

## Kernel Output

The combined simulation writes the following sensitivity kernels to
`OUTPUT_FILES/Kernels/acoustic_isotropic/`:

| File | Description |
|---|---|
| `X.npy`, `Y.npy`, `Z.npy` | GLL-point coordinates (m) |
| `rho.npy` | Density kernel Kρ |
| `kappa.npy` | Bulk-modulus kernel Kκ |
| `rhop.npy` | Density-perturbation kernel Kρ′ |
| `alpha.npy` | P-wave speed kernel Kα |

All arrays have shape `(nelem, ngllz, nglly, ngllx)`.

## Reference

Tromp J., Tape C. & Liu Q. (2005). **Seismic tomography, adjoint methods,
time reversal and banana-doughnut kernels.** *Geophys. J. Int.*, 160, 195–216.
<https://doi.org/10.1111/j.1365-246X.2004.02453.x>
