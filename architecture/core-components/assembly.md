# Assembly

**Files:** `core/specfem/assembly/`

`assembly<DimensionTag>` is the **central data hub** for a running simulation. It takes the raw mesh and computes all per-GLL-point data needed by the solver, storing everything in `Kokkos::View`s that live on the active execution device (GPU or CPU).

## Contents

```
mesh + quadrature + sources + receivers
         │
         ▼
  assembly<DimensionTag>
    ├── mesh             (GLL coords, Jacobians, mass matrices)
    ├── properties       (ρ, vp, vs, κ, μ, … at every GLL point)
    ├── fields           (simulation_field for forward/adjoint/backward/buffer)
    ├── jacobian_matrix  (element Jacobians for coordinate transforms)
    ├── sources          (source-array precomputed on GLL grid)
    ├── receivers        (receiver-array precomputed on GLL grid)
    ├── boundaries       (boundary condition data per GLL edge/face)
    ├── kernels          (storage for Fréchet derivative accumulators)
    ├── conforming_interfaces     (coupled-medium continuity data)
    ├── nonconforming_interfaces  (non-conforming mesh interface data)
    ├── element_types    (per-element physics classification)
    └── boundary_values  (stored boundary data for adjoint reconstructions)
```

All heavy arrays (`Kokkos::View`) are managed here. The assembly struct is passed by value into the solver — Kokkos views are reference-counted, so this is cheap.

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
