# Medium Physics, Algorithms, and Compute Kernels

## Medium Physics (`specfem::medium_physics`)

**Files:** `core/specfem/medium_physics/`

This namespace provides **stateless, templated compute functions** that encode the physics for each medium type. These are called inside the solver's time-stepping loop. Functions are templated on element attributes so the compiler can inline and optimize per-medium.

| Function | Computes |
|---|---|
| `compute_stress` | Cauchy stress tensor from displacement gradients |
| `compute_wavefield` | Observable wavefield from intrinsic fields |
| `compute_source_contribution` | Force/moment-tensor source terms |
| `compute_frechet_derivatives` | Sensitivity kernel integrands |
| `compute_coupling` | Flux terms at fluid-solid/solid-solid interfaces |
| `compute_damping_force` | Viscous damping force (poroelastic) |
| `compute_cosserat_stress` | Cosserat (micropolar) stress tensor |
| `compute_cosserat_couple_stress` | Cosserat couple-stress tensor |
| `mass_matrix_component` | Per-point mass matrix contribution |

---

## Algorithms (`specfem::algorithms`)

**Files:** `core/specfem/algorithms/`

Reusable mathematical building blocks for SEM, all operating on `specfem::point` data types and dispatched through Kokkos parallel patterns:

| Algorithm | Description |
|---|---|
| `gradient` | Gradient of a vector field via Lagrange derivative polynomials |
| `divergence` | Divergence of a stress tensor |
| `interpolate` | Interpolate field to arbitrary point |
| `coupling_integral` | Boundary integral for coupled media |
| `transfer` | Field data transfer (e.g., scatter/gather) |
| `locate_point` | Find spectral element containing a physical coordinate |

---

## Compute Kernels (`specfem::compute`)

**Files:** `core/specfem/compute/`

High-level orchestration functions that wire together `algorithms`, `medium_physics`, and `assembly` data. These are called by the solver:

| Function | Description |
|---|---|
| `compute_derivatives` | Compute field spatial derivatives across all elements |
| `compute_seismograms` | Sample the wavefield at receiver GLL points |
| `initialize_mass_matrix` | Assemble the lumped mass matrix |

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
