# SPECFEM++ Architecture

SPECFEM++ is a complete C++ rewrite of the SPECFEM suite (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) for simulating seismic (and electromagnetic) wave propagation using the **Spectral Element Method (SEM)**. The primary design goals are:

- **Robustness and flexibility** — a clean, strongly-typed C++17 codebase.
- **Modularity** — physics, geometry, I/O, and execution are fully decoupled.
- **Hardware portability** — runs on CPUs (serial/OpenMP), NVIDIA GPUs (CUDA), AMD GPUs (HIP), and Intel GPUs via the [Kokkos](https://github.com/kokkos/kokkos) performance-portability library.
- **Multi-physics** — supports acoustic, elastic (P/SV/SH, isotropic, anisotropic, Cosserat), poroelastic, and electromagnetic media with coupled interfaces.
- **Inversion support** — computes Fréchet sensitivity kernels via the adjoint method for seismic tomography.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       CLI / main()                           │
│            specfem 2d|3d -p config.yaml                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                 program::Context (RAII)                      │
│          Kokkos::initialize / MPI_Init / MPI_Finalize        │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│           runtime_configuration::setup                       │
│         (Parse YAML → typed C++ configuration)               │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌─────────────────────┐     ┌──────────────────────────┐
│   io::read_2d_mesh  │     │  io::read_2d_sources /   │
│   io::read_3d_mesh  │     │  io::read_2d_receivers   │
│  (Fortran binary)   │     │  (YAML files)            │
└──────────┬──────────┘     └─────────────┬────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│               assembly::assembly<DimensionTag>               │
│   Mesh → per-GLL Kokkos::Views (fields, properties, etc.)    │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│  time_scheme        │    │  periodic_tasks (wavefield I/O, │
│  (Newmark, …)       │    │  plotting, signal checking)     │
└──────────┬──────────┘    └─────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│         solver::time_marching<Simulation, Dim, NGLL>         │
│   Predictor → wavefield update (medium_physics) → Corrector  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              io writers                                      │
│    seismograms / kernels / wavefields / properties           │
└──────────────────────────────────────────────────────────────┘
```

---

## Documentation Map

| Section | Description |
|---|---|
| [Repository Layout](repository-layout.md) | Directory tree and what lives where |
| **Core Components** | |
| [Program Lifecycle](core-components/program-lifecycle.md) | CLI entry point, `Context` RAII, `execute` dispatch |
| [Runtime Configuration](core-components/runtime-configuration.md) | YAML parsing and component factory |
| [Mesh](core-components/mesh.md) | Mesh data structures read from external mesher |
| [Assembly](core-components/assembly.md) | Central data hub: GLL-point arrays, fields, properties |
| [Medium Physics](core-components/medium-physics.md) | Physics kernels, algorithms, compute orchestration |
| [Solver](core-components/solver.md) | Time-marching solver and time scheme |
| [I/O](core-components/io.md) | I/O backends, mesh readers, periodic tasks |
| [Parallel Execution](core-components/parallel-execution.md) | Kokkos iterators, chunk/tile configuration |
| [Supporting Components](core-components/supporting-components.md) | MPI, boundary conditions, attenuation, quadrature, point data |
| **Type System** | |
| [Element Type System](type-system/index.md) | Tag-based compile-time type system overview |
| [Tags](type-system/tags.md) | Dimension, medium, property, boundary, attenuation tags |
| [Element Attributes](type-system/element-attributes.md) | `element::attributes` traits class |
| [Template Patterns](type-system/template-patterns.md) | Tag dispatch, policy-based design, `if constexpr` |
| **Simulation** | |
| [Workflow](simulation/workflow.md) | End-to-end 14-step simulation sequence |
| [Modes](simulation/modes.md) | Forward and combined (adjoint) simulation modes |
| **Infrastructure** | |
| [Hardware Portability](infrastructure/hardware-portability.md) | Kokkos backends, SIMD, precision |
| [Build System](infrastructure/build-system.md) | CMake options and presets |
| [Python Bindings](infrastructure/python-bindings.md) | `specfempp_core` Python package |
| [Dependencies](infrastructure/dependencies.md) | Required and optional third-party libraries |

---

*For API reference documentation, see the [online Doxygen docs](https://specfem2d-kokkos.readthedocs.io/en/latest/). For usage examples, see the [`examples/`](../examples/) directory.*
