# Key Dependencies

| Library | Purpose | Required? |
|---|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | Performance portability (CPU/GPU) | **Yes** |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | YAML parameter file parsing | **Yes** |
| [CLI11](https://github.com/CLIUtils/CLI11) | Command-line argument parsing | **Yes** |
| [Boost](https://www.boost.org/) | Utilities (math, filesystem) | **Yes** |
| HDF5 | HDF5 I/O backend | Optional |
| ADIOS2 | ADIOS2 I/O backend | Optional |
| zlib | NPZ compression | Optional |
| VTK | Wavefield visualization | Optional (ON by default) |
| MPI | Distributed memory parallelism | Optional |
| nanobind/pybind11 | Python bindings | Optional |
| [TensorOperations](https://github.com/Rohit-Kakodkar/TensorOperations) | Declarative element stiffness kernels (einsum level graphs) | Optional |

Required dependencies are fetched automatically by CMake if not found on the system. Optional dependencies must be pre-installed and are enabled via the corresponding `SPECFEM_ENABLE_*` CMake options (see [Build System](build-system.md)).

TensorOperations is header-only but is never vendored or fetched (the repository has no license yet, and its own CMake pulls in a second Kokkos): a checkout must be provided through the `SPECFEM_TENSOROPS_ROOT` cache or environment variable, and `cmake/tensorops.cmake` builds an interface target over its `include/` directory against SPECFEM++'s own Kokkos.

---

← [Back to Index](../index.md)
