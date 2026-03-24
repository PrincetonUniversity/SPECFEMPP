# Hardware Portability via Kokkos

All performance-critical arrays are `Kokkos::View<...>` and all parallel loops use `Kokkos::parallel_for` / `Kokkos::parallel_reduce`. This means the exact same source code runs on:

| Target | CMake Preset | Kokkos Backend |
|---|---|---|
| CPU (serial) | `serial` | `Kokkos::Serial` |
| CPU (OpenMP) | `omp` | `Kokkos::OpenMP` |
| NVIDIA GPU | `cuda` | `Kokkos::Cuda` |
| AMD GPU | `hip` | `Kokkos::HIP` |
| Intel GPU | `sycl` | `Kokkos::SYCL` |

Compile-time backend detection in `parallel_configuration` adjusts chunk sizes and SIMD widths to match the hardware. SIMD vectorization on CPUs uses `Kokkos::Experimental::simd<type_real>`.

## Precision

The floating-point precision is selectable at configure time:

| CMake Option | Precision |
|---|---|
| Default | Single precision (`float`) |
| `SPECFEM_ENABLE_DOUBLE_PRECISION=ON` | Double precision (`double`) |

## Design Principle

No platform-specific code appears in physics kernels. All portability is handled by:

1. **`Kokkos::View`** — automatically allocates on device (GPU HBM or CPU RAM) based on the active backend.
2. **`Kokkos::parallel_for`** — maps to CUDA kernels, OpenMP threads, or scalar loops transparently.
3. **`parallel_configuration`** — provides the correct chunk/tile sizes so memory access patterns are efficient on each target.

---

← [Back to Infrastructure](build-system.md) | [Back to Index](../index.md)
