---
paths:
  - "core/**"
  - "tests/**"
---

# Kokkos Coding Patterns

## View handling

- `Kokkos::View` copy constructors perform atomic reference-count operations.
  Avoid capturing views or view-containing structs by value in hot loops.
- Return view-containing structs by `const&` from getters, not by value.
- Use `const auto&` when storing view references to avoid refcount overhead.
- Device views: `assembly.mesh.coord` (DefaultExecutionSpace)
- Host mirrors: `assembly.mesh.h_coord`
- Deep-copy between host and device explicitly: `Kokkos::deep_copy(dst, src)`

## Lambda captures

- Use `KOKKOS_LAMBDA` for device-compatible lambdas.
- **No nested C++ lambdas** inside `KOKKOS_LAMBDA` -- CUDA does not support them.
  Inline the logic instead.
- Copy `std::array` or struct member variables to plain local variables before
  the lambda -- CUDA cannot capture `this`.
- Mark any function called from inside `KOKKOS_LAMBDA` with `KOKKOS_INLINE_FUNCTION`.

## Math functions

- Use `Kokkos::min`, `Kokkos::max`, `Kokkos::fabs`, `Kokkos::sqrt` in device
  lambdas. Do NOT use `std::min`, `std::max`, etc. in device code.

## Parallel patterns

- Use `specfem::execution::ChunkedDomainIterator` and `specfem::execution::for_all`
  for per-element GLL loops (see `core/specfem/assembly/info.tpp` for canonical example).
- Use `ScatterMinMax` scatter views for reductions
  (see `core/specfem/assembly/info/impl/scatter_minmax.hpp`).
- Multi-reducer syntax:
  ```cpp
  Kokkos::parallel_reduce("name", RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i, float& lmin, float& lmax) { ... },
    Kokkos::Min<float>(result_min), Kokkos::Max<float>(result_max));
  ```

## Scratch memory

- The project uses `ChunkElementFieldType`, `ChunkStressIntegrandType`, and
  `ElementQuadratureType` for team-level scratch memory. Follow existing patterns
  rather than inventing new scratch allocation schemes.

## SIMD path

- On CPU, the project uses SIMD chunking. `MDSpan mapping(ispec, iz, ix)` computes
  flat indices -- avoid calling it multiple times for the same index.
- Chunk sizes are defined in `core/specfem/parallel_configuration/chunk_config.hpp`.
