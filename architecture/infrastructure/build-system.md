# Build System

SPECFEM++ uses CMake with [CMakePresets](../../CMakePresets.json).

## CMake Options

| CMake Option | Default | Description |
|---|---|---|
| `SPECFEM_ENABLE_MPI` | `OFF` | Enable MPI for distributed-memory parallelism |
| `SPECFEM_ENABLE_HDF5` | `OFF` | Enable HDF5 I/O backend |
| `SPECFEM_ENABLE_ADIOS2` | `OFF` | Enable ADIOS2 I/O backend |
| `SPECFEM_ENABLE_NPZ` | `OFF` | Enable NumPy NPZ backend |
| `SPECFEM_ENABLE_VTK` | `ON` | Enable VTK visualization output |
| `SPECFEM_ENABLE_SIMD` | `OFF` | Enable Kokkos SIMD vectorization |
| `SPECFEM_ENABLE_DOUBLE_PRECISION` | `OFF` | Use double instead of float |
| `SPECFEM_BUILD_TESTS` | `OFF` | Build unit tests |
| `SPECFEM_BINDING_PYTHON` | `OFF` | Build Python bindings |
| `SPECFEM_ENABLE_UNITY_BUILD` | `ON` | Unity build for faster compilation |

## Quick Start

```bash
# CPU serial build
cmake --preset serial -B build
cmake --build build

# NVIDIA GPU build
cmake --preset cuda -B build
cmake --build build
```

## Dependency Management

Dependencies are fetched automatically via CMake `FetchContent` if not found on the system:

- Kokkos
- yaml-cpp
- CLI11
- Boost

Optional dependencies (HDF5, ADIOS2, VTK, MPI) must be installed on the system and are detected via `find_package`.

---

← [Back to Index](../index.md)
