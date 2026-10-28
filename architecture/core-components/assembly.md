# Assembly

**Files:** `core/specfem/assembly/`

`assembly<DimensionTag>` is the **central data hub** for a running simulation. It takes the raw mesh and computes all per-GLL-point data needed by the solver, storing everything in `Kokkos::View`s that live on the active execution device (GPU or CPU).

## Contents

```
mesh + quadrature + sources + receivers
         │
         ▼
  assembly<DimensionTag>
    ├── mesh             (GLL coords, Jacobians, mass matrices, plus reference GLL coords for Globe3D model sampling)
    ├── properties       (ρ, vp, vs, κ, μ, … at every GLL point)
    ├── fields           (simulation_field for forward/adjoint/backward/buffer)
    ├── jacobian_matrix  (element Jacobians for coordinate transforms)
    ├── sources          (source-array precomputed on GLL grid)
    ├── receivers        (receiver-array precomputed on GLL grid)
    ├── boundaries       (boundary condition data per GLL edge/face)
    ├── kernels          (storage for Fréchet derivative accumulators)
    ├── conforming_interfaces     (coupled-medium continuity data)
    ├── nonconforming_interfaces  (non-conforming mesh interface data)
    ├── mpi_interfaces   (MPI face communication patterns)
    ├── element_types    (per-element physics classification, plus globe region context for Globe3D)
    └── boundary_values  (stored boundary data for adjoint reconstructions)
```

All heavy arrays (`Kokkos::View`) are managed here. The assembly struct is passed by value into the solver — Kokkos views are reference-counted, so this is cheap.

## Coordinate Frames

**Files:** `core/specfem/coordinate_systems/`, `core/specfem/assembly/coordinate_conversion.hpp`

`coordinate_systems` is pure math: `rotation_matrix<CartesianFrame, GeocentricFrame>(theta, phi)` gives the local `(theta, phi, r)` axes in mesh components (`basis_rotation.hpp`), and `east_north_up.hpp` is the single owner of the relabel East = +e_phi, North = -e_theta, Up = +e_r. `assembly::coordinate_conversion` dispatches on the mesh projection through `to_cartesian` and `local_frame_rotation(point, mesh)` (a globe evaluates the rotation at the point; a regional mesh uses the exact `flat_mesh_rotation`, x = East, y = North, z = Up). Sources rotate their moment tensor / force with that rotation; `local_east_north_up_basis(point, mesh)` turns it into the 3x3 matrix each 3-D receiver stores (rows East, North, Up, in mesh components), applied in the seismogram kernel on the owning rank for vector seismograms only.

---

## MPI Interfaces

**Files:** `core/specfem/assembly/mpi/dim3/`

`mpi<DimensionTag>` manages **inter-process face communication patterns** for distributed memory simulations. Constructed from the mesh adjacency graph and GLL parameters, it:

- Groups MPI connections by neighboring rank
- Extracts face-only interactions (excludes edge/corner adjacencies)
- Computes anchor-point-based rotation indices for face-to-face coordinate alignment
- Stores face orientations (mesh entity types) and discrete rotation indices as compact `Kokkos::View`s
- Retrieves the current MPI rank internally via `specfem::MPI::get_rank()` at construction time

Each `communication_group` (one per neighboring MPI process) contains:
- `my_orientation[nfaces]` — Mesh entity type of each face in the local element
- `neighbor_orientation[nfaces]` — Mesh entity type of each face in the neighboring element
- `theta[nfaces]` — Rotation index r ∈ [0,3] for face alignment (stored as `unsigned char` to reduce memory to 1 byte per face)
- `ngll` — GLL points per direction

---

## Simulation Fields

**Files:** `core/specfem/assembly/fields/`

`simulation_field<DimensionTag, FieldType>` stores displacement, velocity, and acceleration arrays for one wavefield type:

| `field_type` | Purpose |
|---|---|
| `forward` | Standard forward wavefield |
| `adjoint` | Adjoint wavefield (driven by adjoint sources at receivers) |
| `backward` | Backward-reconstructed forward wavefield (for kernels) |
| `buffer` | Checkpoint buffer for boundary values |

---

## Properties

**Files:** `core/specfem/assembly/properties/`

`properties<DimensionTag>` holds material parameters at every GLL point (density, elastic constants, anisotropy coefficients, Q factors, etc.) as `Kokkos::View`s. Data is loaded to device via `load_on_device()` and retrieved via `store_on_host()`.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
