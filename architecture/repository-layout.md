# Repository Layout

```
SPECFEMPP/
├── core/                   # All C++ library code
│   ├── specfem.cpp         # Main executable entry point
│   ├── specfem.hpp         # Top-level namespace documentation
│   └── specfem/            # Header-only + implementation library
│       ├── assembly/       # SEM data assembly (GLL-point data)
│       ├── algorithms/     # SEM mathematical operators
│       ├── attenuation/    # Attenuation (SLS/Maxwell solid)
│       ├── boundary_conditions/  # Stacey, Dirichlet, etc.
│       ├── compute/        # Top-level compute kernels
│       ├── element/        # Element type tags and attributes
│       ├── enums/          # Enumeration types
│       ├── execution/      # Kokkos parallel iterators
│       ├── io/             # I/O backends and mesh readers
│       ├── medium/         # Medium containers (dim2/dim3)
│       ├── medium_physics/ # Physics compute functions
│       ├── mesh/           # Mesh data structures
│       ├── mpi/            # MPI wrapper
│       ├── parallel_configuration/  # Chunk/tile sizes per backend
│       ├── periodic_tasks/ # Tasks run periodically during time-stepping
│       ├── point/          # Per-quadrature-point data types
│       ├── program/        # Lifecycle, context, 2D/3D program logic
│       ├── quadrature/     # GLL quadrature rules
│       ├── runtime_configuration/  # YAML config parsing
│       ├── solver/         # Solver base and time-marching solvers
│       ├── source/         # Source types
│       ├── source_time_functions/  # STF implementations
│       ├── timescheme/     # Time integration schemes
│       └── ...             # Additional utilities, receivers, etc.
├── src/                    # Fortran mesh I/O (specfem2d / specfem3d)
├── python/                 # Python bindings (specfempp_core)
├── tests/                  # Unit and integration tests
├── examples/               # Runnable example problems
├── docs/                   # Sphinx documentation source
├── cmake/                  # CMake find-modules and helpers
├── CMakeLists.txt          # Top-level build definition
└── CMakePresets.json       # Preset configurations (CPU, CUDA, HIP, …)
```

The `core/` subtree is the primary area developers will work in. Every major concept has its own subdirectory under `core/specfem/`.

## Key Directories

| Directory | Purpose |
|---|---|
| `core/specfem/` | All C++ source — headers and implementations |
| `src/` | Fortran sources for mesh file I/O (legacy compatibility) |
| `python/` | nanobind-based Python bindings (`specfempp_core`) |
| `tests/` | Unit and integration tests (CTest) |
| `examples/` | Runnable example problems with parameter files |
| `docs/` | Sphinx documentation source (RST) |
| `cmake/` | CMake find-modules and utility functions |

---

← [Back to Index](index.md)
