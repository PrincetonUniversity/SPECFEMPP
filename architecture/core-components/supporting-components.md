# Supporting Components

## MPI Wrapper (`specfem::MPI`)

**Files:** `core/specfem/mpi/mpi.hpp`

`specfem::MPI` is a **static** class (no instantiation) providing a thin wrapper around MPI:

```cpp
int rank = specfem::MPI::get_rank();
int size = specfem::MPI::get_size();
specfem::MPI::sync();
specfem::MPI::reduce(value, specfem::sum);
```

MPI is initialized and finalized exclusively by `program::Context`, which prevents accidental double-init. When compiled without `SPECFEM_ENABLE_MPI`, all methods become no-ops.

### Face Communication Patterns

**Files:** `core/specfem/assembly/mpi/dim3/`

Building on the static MPI wrapper, `mpi<DimensionTag>` manages **face-level communication for distributed simulations**. This class:

- **Groups connections by rank**: Analyzes the mesh adjacency graph to identify which element faces are shared across an MPI partition boundary
- **Filters face-only interactions**: Keeps only face adjacencies; excludes edges and corners
- **Computes face rotation**: Uses anchor points (face corners) to determine the discrete rotation index (0–3) between communicating elements
- **Stores compact metadata**: Face normals and rotation indices are stored as `Kokkos::View`s for efficient GPU access

The rotation index (`theta`) is stored as `unsigned char` rather than a floating-point angle, reducing per-face memory from 8 bytes to 1 byte while keeping semantics clear: `actual_angle = theta * π/2`.

---

## Boundary Conditions (`specfem::boundary_conditions`)

**Files:** `core/specfem/boundary_conditions/`

Boundary conditions are applied as **per-GLL-point corrections** to the acceleration field after the wavefield update. They are selected at compile-time via `element::boundary_tag`:

| Boundary tag | Description |
|---|---|
| `none` | No boundary condition (interior or free surface for elastic) |
| `acoustic_free_surface` | Zero-pressure condition for acoustic media |
| `stacey` | Stacey absorbing boundary (first-order approximate PML) |
| `composite_stacey_dirichlet` | Combined absorbing + Dirichlet (corner treatment) |

---

## Attenuation (`specfem::attenuation`)

**Files:** `core/specfem/attenuation/`

Implements **Standard Linear Solid (SLS) / Maxwell solid** attenuation using the nearly-constant-Q model. Key routines:

| Function | Description |
|---|---|
| `compute_tau_sigma` | Compute SLS stress-relaxation times |
| `compute_tau_eps` | Compute SLS strain-relaxation times |
| `compute_factors` | Compute complex modulus factors M₁, M₂ |
| `maxwell` | Maxwell solid constitutive model |

The `specfem qplots` subcommand uses these routines to let users visualize the achieved Q⁻¹ vs. frequency before running a simulation.

---

## Quadrature (`specfem::quadrature`)

**Files:** `core/specfem/quadrature/`

Provides **Gauss-Lobatto-Legendre (GLL)** quadrature rules. Each quadrature object stores:

- `xi` — quadrature points in reference element [-1, 1]
- `w` — quadrature weights
- `hprime` — derivatives of Lagrange interpolating polynomials at quadrature points (NGLL × NGLL matrix)

All arrays are stored as `Kokkos::View`s with both device and host mirrors.

---

## Point Data (`specfem::point`)

**Files:** `core/specfem/point/`

The `specfem::point` namespace contains **small, stack-allocated structs** representing data at a single GLL quadrature point. These are the building blocks passed between algorithms and physics kernels:

| Struct | Holds |
|---|---|
| `point::displacement` | Displacement vector components |
| `point::velocity` | Velocity vector components |
| `point::acceleration` | Acceleration vector components |
| `point::stress` | Stress tensor components |
| `point::field_derivatives` | Spatial derivatives of displacement |
| `point::properties` | Local material properties (ρ, κ, μ, …) |
| `point::kernels` | Fréchet kernel accumulators |
| `point::source` | Source force/moment at a GLL point |
| `point::jacobian_matrix` | Coordinate transform Jacobian |
| `point::index` | Local element + GLL index |
| `point::global_coordinates` | Physical (x, y, z) coordinate |
| `point::mass_inverse` | Diagonal mass matrix inverse |

The design ensures that physics kernels operate on minimal, type-safe bundles of data rather than raw array indices.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
