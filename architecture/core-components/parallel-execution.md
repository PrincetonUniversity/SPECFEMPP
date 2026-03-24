# Parallel Execution

## `specfem::parallel_configuration`

**Files:** `core/specfem/parallel_configuration/`

Provides compile-time constants for chunk and tile sizes adapted to the active Kokkos backend:

| Backend | `chunk_size` | Notes |
|---|---|---|
| CUDA | 32 | Warp size |
| HIP | 64 | Wavefront size |
| OpenMP | `1 × simd_size` | SIMD-vectorized |
| Serial | `1 × simd_size` | SIMD-vectorized |

`chunk_config<DimensionTag, ChunkSize, TileSize, NumThreads, VectorLanes, SIMD, ExecSpace>` packages these into a type used by execution iterators.

---

## `specfem::execution`

**Files:** `core/specfem/execution/`

Kokkos-based parallel iterators that expose SEM iteration patterns while hiding backend details:

| Iterator | Iterates over |
|---|---|
| `ChunkedDomainIterator` | Chunks of spectral elements (bulk computation) |
| `ChunkedEdgeIterator` | Element edges (boundary conditions) |
| `ChunkedFaceIterator` | Element faces (3D boundary conditions) |
| `ChunkedIntersectionIterator` | Pairs of elements sharing a face (interfaces) |
| `RangeIterator` | Simple linear range of GLL points |

These iterators are the **primary way physics kernels are parallelized** — they abstract over CUDA thread blocks, OpenMP threads, and scalar serial loops so that physics code is backend-agnostic.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
