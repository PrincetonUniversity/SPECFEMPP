# Core Components

This section covers the major C++ components that make up SPECFEM++. Each component lives in its own subdirectory under `core/specfem/`.

| Component | Description |
|---|---|
| [Program Lifecycle](program-lifecycle.md) | CLI entry point, `Context` RAII, simulation dispatch |
| [Runtime Configuration](runtime-configuration.md) | YAML parsing and component factory |
| [Mesh](mesh.md) | Passive mesh data structures read from external mesher |
| [Assembly](assembly.md) | Central GLL-point data hub for a running simulation |
| [Medium Physics](medium-physics.md) | Stateless physics kernels, algorithms, compute orchestration |
| [Solver](solver.md) | Time-marching solver loop and time integration scheme |
| [I/O](io.md) | File I/O backends, mesh readers, periodic tasks |
| [Parallel Execution](parallel-execution.md) | Kokkos iterators and hardware-adaptive chunk/tile sizing |
| [Supporting Components](supporting-components.md) | MPI, boundary conditions, attenuation, quadrature, point data |

---

← [Back to Index](../index.md)
