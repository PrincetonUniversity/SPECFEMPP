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

Required dependencies are fetched automatically by CMake if not found on the system. Optional dependencies must be pre-installed and are enabled via the corresponding `SPECFEM_ENABLE_*` CMake options (see [Build System](build-system.md)).

---

← [Back to Index](../index.md)
