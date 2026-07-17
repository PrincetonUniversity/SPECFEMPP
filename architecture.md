# SPECFEM++ Architecture

## Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [High-Level Architecture](#high-level-architecture)
4. [Core Namespaces and Components](#core-namespaces-and-components)
   - [Entry Point and CLI](#entry-point-and-cli)
   - [Program Lifecycle (`specfem::program`)](#program-lifecycle-specfemprogram)
   - [Runtime Configuration (`specfem::runtime_configuration`)](#runtime-configuration-specfemruntime_configuration)
   - [Mesh (`specfem::mesh`)](#mesh-specfemmesh)
   - [Assembly (`specfem::assembly`)](#assembly-specfemassembly)
   - [Fields and Properties](#fields-and-properties)
   - [Medium Physics (`specfem::medium_physics`)](#medium-physics-specfemmedium_physics)
   - [Solver (`specfem::solver`)](#solver-specfemsolver)
   - [Time Scheme (`specfem::time_scheme`)](#time-scheme-specfemtime_scheme)
   - [Algorithms (`specfem::algorithms`)](#algorithms-specfemalgorithms)
   - [Compute Kernels (`specfem::compute`)](#compute-kernels-specfemcompute)
   - [I/O Subsystem (`specfem::io`)](#io-subsystem-specfemio)
   - [Periodic Tasks (`specfem::periodic_tasks`)](#periodic-tasks-specfemperiodic_tasks)
   - [Parallel Execution (`specfem::execution`, `specfem::parallel_configuration`)](#parallel-execution)
   - [MPI Wrapper (`specfem::MPI`)](#mpi-wrapper-specfemmpi)
   - [Boundary Conditions (`specfem::boundary_conditions`)](#boundary-conditions-specfemboundary_conditions)
   - [Attenuation (`specfem::attenuation`)](#attenuation-specfemattenuation)
   - [Quadrature (`specfem::quadrature`)](#quadrature-specfemquadrature)
   - [Point Data (`specfem::point`)](#point-data-specfempoint)
5. [Element Type System](#element-type-system)
6. [Template Design Patterns](#template-design-patterns)
7. [Simulation Workflow](#simulation-workflow)
8. [Simulation Modes](#simulation-modes)
9. [Hardware Portability via Kokkos](#hardware-portability-via-kokkos)
10. [Build System](#build-system)
11. [Python Bindings](#python-bindings)
12. [Key Dependencies](#key-dependencies)

---

## Overview

SPECFEM++ is a complete C++ rewrite of the SPECFEM suite (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) for simulating seismic (and electromagnetic) wave propagation using the **Spectral Element Method (SEM)**. The primary design goals are:

- **Robustness and flexibility** — a clean, strongly-typed C++17 codebase.
- **Modularity** — physics, geometry, I/O, and execution are fully decoupled.
- **Hardware portability** — runs on CPUs (serial/OpenMP), NVIDIA GPUs (CUDA), AMD GPUs (HIP), and Intel GPUs via the [Kokkos](https://github.com/kokkos/kokkos) performance-portability library.
- **Multi-physics** — supports acoustic, elastic (P/SV/SH, isotropic, anisotropic, Cosserat), poroelastic, and electromagnetic media with coupled interfaces.
- **Inversion support** — computes Fréchet sensitivity kernels via the adjoint method for seismic tomography.

---

## Repository Layout

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

## Core Namespaces and Components

### Entry Point and CLI

**File:** `core/specfem.cpp`

The `main()` function uses [CLI11](https://github.com/CLIUtils/CLI11) to parse command-line arguments and dispatch to one of three subcommands:

| Subcommand | Description |
|---|---|
| `2d` | Run a 2D simulation, requires `-p <config.yaml>` |
| `3d` | Run a 3D simulation, requires `-p <config.yaml>` |
| `qplots` | Generate Q-attenuation diagnostic plots |

All logging options (`--log-file`, `--log-level`, etc.) are accepted here and forwarded to `specfem::Logger`. After the CLI parses, control is handed to `specfem::program::execute(dimension, parameter_dict)`.

---

### Program Lifecycle (`specfem::program`)

**Files:** `core/specfem/program/`

#### `program::Context`

`Context` is an RAII guard that manages the lifetime of **Kokkos** and **MPI**. It must be the first object created in `main()`.

```cpp
specfem::program::Context context(argc, argv);
// Kokkos::initialize() and MPI_Init() called here
// ... simulation runs ...
// ~Context(): Kokkos::finalize() and MPI_Finalize() called automatically
```

It is non-copyable and non-movable; only one `Context` should exist at a time.

#### `program::execute`

`program::execute(dimension, parameter_dict)` dispatches to `program_2d(...)` or `program_3d(...)`, which implement the full simulation workflow (described in [Simulation Workflow](#simulation-workflow)).

---

### Runtime Configuration (`specfem::runtime_configuration`)

**Files:** `core/specfem/runtime_configuration/`

The `setup` class parses a YAML parameter file and holds all configuration as typed C++ objects. It acts as a **factory** for simulation components:

| Method | Returns |
|---|---|
| `instantiate_quadrature()` | `specfem::quadrature::quadratures` |
| `instantiate_timescheme(fields)` | `shared_ptr<time_scheme>` |
| `instantiate_solver<NGLL, Dim>(...)` | `shared_ptr<solver::solver>` |
| `instantiate_seismogram_writer()` | `shared_ptr<io::writer>` |
| `instantiate_wavefield_writer<Dim>()` | `shared_ptr<periodic_task>` |
| `instantiate_kernel_writer()` | `shared_ptr<io::writer>` |
| `instantiate_property_reader/writer()` | `shared_ptr<io::reader/writer>` |

Key configuration sections and their corresponding classes:

| YAML section | C++ class |
|---|---|
| `header` | `runtime_configuration::header` |
| `simulation-setup.quadrature` | `runtime_configuration::quadrature` |
| `simulation-setup.solver` | `runtime_configuration::solver` |
| `simulation-setup.solver.time-scheme` | `runtime_configuration::time_scheme` |
| `simulation-mode` | Determines `specfem::simulation::type` |
| `receivers` | `runtime_configuration::receivers` |
| `sources` | `runtime_configuration::sources` |
| `databases` | `runtime_configuration::database_configuration` |

---

### Mesh (`specfem::mesh`)

**Files:** `core/specfem/mesh/`

The `mesh<DimensionTag>` struct stores everything read from the **external mesher database** (a Fortran binary file produced by MESHFEM2D/MESHFEM3D). It is a passive data container — no computation happens here.

Key sub-structs:

| Sub-struct | Contents |
|---|---|
| `mesh::parameters` | Global mesh parameters (nspec, nglob, …) |
| `mesh::coordinates` | GLL-point coordinates |
| `mesh::control_nodes` | Corner node coordinates |
| `mesh::mapping` | Local ↔ global DOF mapping |
| `mesh::materials` | Per-element material assignments |
| `mesh::tags` | Per-element medium/property/boundary tags |
| `mesh::boundaries` | Absorbing, free-surface, forcing boundary lists |
| `mesh::coupled_interfaces` | Fluid-solid / solid-solid coupling lists |
| `mesh::coloring` | Graph-coloring for race-free parallel updates |
| `mesh::inner_outer` | MPI inner/outer element classification |
| `mesh::adjacency` | Element adjacency graph |
| `mesh::mpi` | Shared DOFs between MPI ranks |

The mesh is **read once** from disk and then consumed by the `assembly` constructor.

---

### Assembly (`specfem::assembly`)

**Files:** `core/specfem/assembly/`

`assembly<DimensionTag>` is the **central data hub** for a running simulation. It takes the raw mesh and computes all per-GLL-point data needed by the solver, storing everything in `Kokkos::View`s that live on the active execution device (GPU or CPU).

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

All heavy arrays (`Kokkos::View`) are managed here. The assembly struct is passed by value into the solver (Kokkos views are reference-counted, so this is cheap).

---

### Fields and Properties

**Files:** `core/specfem/assembly/fields/`, `core/specfem/assembly/properties/`

#### Simulation Fields

`simulation_field<DimensionTag, FieldType>` stores displacement, velocity, and acceleration arrays for one wavefield type:

| `field_type` | Purpose |
|---|---|
| `forward` | Standard forward wavefield |
| `adjoint` | Adjoint wavefield (driven by adjoint sources at receivers) |
| `backward` | Backward-reconstructed forward wavefield (for kernels) |
| `buffer` | Checkpoint buffer for boundary values |

#### Properties

`properties<DimensionTag>` holds material parameters at every GLL point (density, elastic constants, anisotropy coefficients, Q factors, etc.) as `Kokkos::View`s. Data is loaded to device via `load_on_device()` and retrieved via `store_on_host()`.

---

### Medium Physics (`specfem::medium_physics`)

**Files:** `core/specfem/medium_physics/`

This namespace provides **stateless, templated compute functions** that encode the physics for each medium type. These are called inside the solver's time-stepping loop. Functions are templated on element attributes so the compiler can inline and optimize per-medium.

| Function | Computes |
|---|---|
| `compute_stress` | Cauchy stress tensor from displacement gradients |
| `compute_wavefield` | Observable wavefield from intrinsic fields |
| `compute_source_contribution` | Force/moment-tensor source terms |
| `compute_frechet_derivatives` | Sensitivity kernel integrands |
| `compute_coupling` | Flux terms at fluid-solid/solid-solid interfaces |
| `compute_damping_force` | Viscous damping force (poroelastic) |
| `compute_cosserat_stress` | Cosserat (micropolar) stress tensor |
| `compute_cosserat_couple_stress` | Cosserat couple-stress tensor |
| `mass_matrix_component` | Per-point mass matrix contribution |

---

### Solver (`specfem::solver`)

**Files:** `core/specfem/solver/`

The abstract base class `solver::solver` has a single virtual method `run()`. The concrete implementation is `solver::time_marching<Simulation, DimensionTag, NGLL>`, templated on:

- `Simulation` — `specfem::simulation::type::forward` or `::combined`
- `DimensionTag` — `dim2` or `dim3`
- `NGLL` — number of GLL points per element per dimension (currently fixed at 5)

#### Forward Solver (`simulation::type::forward`)

Each timestep follows a **predictor-corrector** pattern. Media are processed in a specific order to ensure correct multi-physics coupling at interfaces:

1. **Predictor phase** (all media) — extrapolate displacement/velocity to half-step.
2. **Acoustic wavefield computation** → corrector phase.
3. **Elastic wavefield computation** (elastic_psv, elastic_sh, elastic_psv_t) → corrector phase.
4. **Poroelastic wavefield computation** → corrector phase.

#### Combined Solver (`simulation::type::combined`)

Used for **adjoint simulations** to compute Fréchet kernels:

- Runs the **adjoint wavefield** forward in time (driven by adjoint sources at receiver locations).
- Simultaneously runs the **backward wavefield** in reverse-time (reconstructed from saved checkpoints).
- At each step, correlates adjoint and backward fields to accumulate **Fréchet derivative kernels**.

---

### Time Scheme (`specfem::time_scheme`)

**Files:** `core/specfem/timescheme/`

`time_scheme` is the abstract base class for time integration. The current implementation is `newmark` — the classic **Newmark-beta predictor-corrector** scheme.

Key interface:

```cpp
for (const auto [istep, dt] : ts.iterate_forward()) {
    ts.apply_predictor_phase_forward(medium_tag);
    // ... compute accelerations ...
    ts.apply_corrector_phase_forward(medium_tag);
}
```

The `iterate_forward()` / `iterate_backward()` helper ranges make the time loop direction-agnostic and clean.

---

### Algorithms (`specfem::algorithms`)

**Files:** `core/specfem/algorithms/`

Reusable mathematical building blocks for SEM, all operating on `specfem::point` data types and dispatched through Kokkos parallel patterns:

| Algorithm | Description |
|---|---|
| `gradient` | Gradient of a vector field via Lagrange derivative polynomials |
| `divergence` | Divergence of a stress tensor |
| `interpolate` | Interpolate field to arbitrary point |
| `coupling_integral` | Boundary integral for coupled media |
| `transfer` | Field data transfer (e.g., scatter/gather) |
| `locate_point` | Find spectral element containing a physical coordinate |

---

### Compute Kernels (`specfem::compute`)

**Files:** `core/specfem/compute/`

High-level orchestration functions that wire together `algorithms`, `medium_physics`, and `assembly` data. These are called by the solver:

| Function | Description |
|---|---|
| `compute_derivatives` | Compute field spatial derivatives across all elements |
| `update_wavefields` | Apply computed forces to update acceleration fields |
| `compute_seismograms` | Sample the wavefield at receiver GLL points |
| `initialize_mass_matrix` | Assemble the lumped mass matrix |

---

### I/O Subsystem (`specfem::io`)

**Files:** `core/specfem/io/`

The I/O layer is organized around abstract `reader` and `writer` base classes with multiple format backends:

#### Mesh Readers

| Function | Format |
|---|---|
| `io::read_2d_mesh` | Fortran binary (MESHFEM2D output) |
| `io::read_3d_mesh` | Fortran binary (MESHFEM3D output) |

#### Format Backends

| Backend | Description |
|---|---|
| `io::ASCII` | Plain text (default for seismograms) |
| `io::HDF5` | HDF5 binary (requires `SPECFEM_ENABLE_HDF5=ON`) |
| `io::ADIOS2` | ADIOS2 streaming/file I/O (requires `SPECFEM_ENABLE_ADIOS2=ON`) |
| `io::NPY` | NumPy `.npy` single-array binary |
| `io::NPZ` | NumPy `.npz` archive (requires `SPECFEM_ENABLE_NPZ=ON`) |

#### Data-Type Writers/Readers

| Data type | Description |
|---|---|
| `io::seismogram::writer/reader` | Synthetic seismograms at receivers |
| `io::wavefield::writer/reader` | Full wavefield snapshots |
| `io::property::writer/reader` | Material property fields |
| `io::kernel::writer` | Fréchet sensitivity kernels |

---

### Periodic Tasks (`specfem::periodic_tasks`)

**Files:** `core/specfem/periodic_tasks/`

`periodic_task<DimensionTag>` is an abstract base class for work that must be executed at regular intervals **during the time-stepping loop** (e.g., every N steps). Concrete tasks:

| Task | Description |
|---|---|
| `wavefield_checkpoint` | Define fixed-stride checkpoint and replay windows |
| `wavefield_writer` | Write wavefield snapshots to disk at configured intervals |
| `wavefield_reader` | Read pre-computed wavefield snapshots (adjoint setup) |
| `plot_wavefield` | Real-time or file-based wavefield visualization (VTK/PNG/JPG) |
| `check_signal` | Catch `SIGINT`/`SIGTERM` for graceful shutdown |

Tasks are collected in a `std::vector<shared_ptr<periodic_task>>` and executed by the solver at each configured step interval.

---

### Parallel Execution

**Files:** `core/specfem/execution/`, `core/specfem/parallel_configuration/`

#### `specfem::parallel_configuration`

Provides compile-time constants for chunk and tile sizes adapted to the active Kokkos backend:

| Backend | `chunk_size` | Notes |
|---|---|---|
| CUDA | 32 | Warp size |
| HIP | 64 | Wavefront size |
| OpenMP | `1 × simd_size` | SIMD-vectorized |
| Serial | `1 × simd_size` | SIMD-vectorized |

`chunk_config<DimensionTag, ChunkSize, TileSize, NumThreads, VectorLanes, SIMD, ExecSpace>` packages these into a type used by iterators.

#### `specfem::execution`

Kokkos-based parallel iterators that expose SEM iteration patterns while hiding backend details:

| Iterator | Iterates over |
|---|---|
| `ChunkedDomainIterator` | Chunks of spectral elements (bulk computation) |
| `ChunkedEdgeIterator` | Element edges (boundary conditions) |
| `ChunkedFaceIterator` | Element faces (3D boundary conditions) |
| `ChunkedIntersectionIterator` | Pairs of elements sharing a face (interfaces) |
| `RangeIterator` | Simple linear range of GLL points |

These iterators are the **primary way physics kernels are parallelized** — they abstract over CUDA thread blocks, OpenMP threads, and scalar serial loops.

---

### MPI Wrapper (`specfem::MPI`)

**Files:** `core/specfem/mpi/mpi.hpp`

`specfem::MPI` is a **static** class (no instantiation) providing a thin wrapper around MPI:

```cpp
int rank = specfem::MPI::get_rank();
int size = specfem::MPI::get_size();
specfem::MPI::sync();
specfem::MPI::reduce(value, specfem::sum);
```

MPI is initialized and finalized exclusively by `program::Context`, which prevents accidental double-init. When compiled without `SPECFEM_ENABLE_MPI`, all methods become no-ops.

---

### Boundary Conditions (`specfem::boundary_conditions`)

**Files:** `core/specfem/boundary_conditions/`

Boundary conditions are applied as **per-GLL-point corrections** to the acceleration field after the wavefield update. They are selected at compile-time via `element::boundary_tag`:

| Boundary tag | Description |
|---|---|
| `none` | No boundary condition (interior or free surface for elastic) |
| `acoustic_free_surface` | Zero-pressure condition for acoustic media |
| `stacey` | Stacey absorbing boundary (first-order approximate PML) |
| `composite_stacey_dirichlet` | Combined absorbing + Dirichlet (corner treatment) |

---

### Attenuation (`specfem::attenuation`)

**Files:** `core/specfem/attenuation/`

Implements **Standard Linear Solid (SLS) / Maxwell solid** attenuation using the nearly-constant-Q model. Key routines:

| Function | Description |
|---|---|
| `compute_tau_sigma` | Compute SLS stress-relaxation times |
| `compute_tau_eps` | Compute SLS strain-relaxation times |
| `compute_factors` | Compute complex modulus factors M₁, M₂ |
| `maxwell` | Maxwell solid constitutive model |

The `specfem qplots` subcommand uses these routines to let users visualize the achieved Q⁻¹ vs. frequency before running a simulation.

---

### Quadrature (`specfem::quadrature`)

**Files:** `core/specfem/quadrature/`

Provides **Gauss-Lobatto-Legendre (GLL)** quadrature rules. Each quadrature object stores:

- `xi` — quadrature points in reference element [-1, 1]
- `w` — quadrature weights
- `hprime` — derivatives of Lagrange interpolating polynomials at quadrature points (NGLL × NGLL matrix)

All arrays are stored as `Kokkos::View`s with both device and host mirrors.

---

### Point Data (`specfem::point`)

**Files:** `core/specfem/point/`

The `specfem::point` namespace contains **small, stack-allocated structs** representing data at a single GLL quadrature point. These are the building blocks passed between algorithms and physics kernels:

| Struct | Holds |
|---|---|
| `point::displacement` | Displacement vector components |
| `point::velocity` | Velocity vector components |
| `point::acceleration` | Acceleration vector components |
| `point::stress` | Stress tensor components |
| `point::field_derivatives` | Spatial derivatives of displacement |
| `point::properties` | Local material properties (ρ, κ, μ, …) |
| `point::kernels` | Fréchet kernel accumulators |
| `point::source` | Source force/moment at a GLL point |
| `point::jacobian_matrix` | Coordinate transform Jacobian |
| `point::index` | Local element + GLL index |
| `point::global_coordinates` | Physical (x, y, z) coordinate |
| `point::mass_inverse` | Diagonal mass matrix inverse |

The design ensures that physics kernels operate on minimal, type-safe bundles of data rather than raw array indices.

---

## Element Type System

SPECFEM++ uses a **tag-based** compile-time type system to represent element physics without runtime branching.

### Dimension Tags

```cpp
enum class dimension_tag { dim2, dim3 };
```

All major data structures and algorithms are templated on `dimension_tag`. The compiler generates separate, optimized code paths for 2D and 3D.

### Medium Tags

```cpp
enum class medium_tag {
    elastic_psv,        // 2D P + SV waves (2 DOF: u_x, u_z)
    elastic_sh,         // 2D SH waves (1 DOF: u_y)
    elastic_psv_t,      // 2D PSV + Cosserat spin (3 DOF: u_x, u_z, ω_y)
    acoustic,           // Pressure waves (1 DOF: φ)
    poroelastic,        // Biot poroelastic (fluid+solid DOFs)
    electromagnetic_te, // 2D TE electromagnetic mode
    elastic,            // 3D elastic (3 DOF: u_x, u_y, u_z)
    elastic_spin,       // 3D elastic with spin
    electromagnetic,    // 3D electromagnetic
};
```

### Property Tags

```cpp
enum class property_tag {
    isotropic,          // Scalar λ, μ, ρ
    anisotropic,        // Full elastic tensor Cijkl
    isotropic_cosserat  // Isotropic + micropolar constants
};
```

### Boundary Tags

```cpp
enum class boundary_tag {
    none,
    acoustic_free_surface,
    stacey,
    composite_stacey_dirichlet
};
```

### Attenuation Tags

```cpp
enum class attenuation_tag {
    none,
    constant_isotropic  // Constant Q over frequency band
};
```

### Element Attributes

`element::attributes<DimensionTag, MediumTag>` is a **traits class** providing compile-time information about each medium:

```cpp
// Example: 2D elastic PSV
using attrs = element::attributes<dim2, elastic_psv>;
static_assert(attrs::components == 2);      // u_x, u_z
static_assert(attrs::dimension == 2);
static_assert(attrs::has_cosserat_stress == false);
```

This zero-cost abstraction allows the compiler to eliminate dead code and specialize physics for each medium type.

---

## Template Design Patterns

SPECFEM++ makes extensive use of C++17 templates to achieve zero-cost abstraction:

1. **Tag dispatch** — dimension, medium, property, and boundary tags are template parameters that route to correct physics at compile time.

2. **Policy-based design** — `chunk_config` and `parallel_configuration` encode hardware policies as types; execution iterators are instantiated from these policies.

3. **CRTP / traits classes** — `element::attributes<Dim, Medium>` provides static properties without virtual dispatch.

4. **`if constexpr`** — used extensively inside physics functions to conditionally compile Cosserat, damping, or coupling terms.

5. **Kokkos lambda capture** — physics functions are passed as lambdas to `Kokkos::parallel_for` / `Kokkos::parallel_reduce`, enabling transparent GPU offload.

---

## Simulation Workflow

The following is the canonical execution sequence for a forward simulation (both 2D and 3D follow the same pattern):

```
1. Parse YAML config
        │
        ▼
2. Instantiate GLL quadrature
        │
        ▼
3. Read mesh database (Fortran binary)
        │
        ▼
4. Read sources (YAML) → compute t₀ from source STF
        │
        ▼
5. Read receivers (stations file or YAML)
        │
        ▼
6. Build assembly<DimensionTag>
   ├── Locate sources and receivers in mesh elements
   ├── Compute Jacobians at all GLL points
   ├── Compute and assemble mass matrix
   ├── Initialize material properties at GLL points
   ├── Set up boundary condition data
   └── Allocate forward (+ adjoint/backward) field arrays
        │
        ▼
7. [Optional] Load saved material properties from disk
        │
        ▼
8. [Optional] Early exit: write material properties to disk
        │
        ▼
9. Instantiate time scheme (Newmark)
        │
        ▼
10. Register periodic tasks
    ├── wavefield_reader (if adjoint)
    ├── wavefield_writer (if saving checkpoints)
    ├── wavefield_plotter (if visualization enabled)
    └── check_signal
        │
        ▼
11. Instantiate solver (time_marching<Forward|Combined, Dim, NGLL>)
        │
        ▼
12. solver.run()  ← main time loop
    for each timestep:
        ├── time_scheme.apply_predictor(all media)
        ├── compute_derivatives + update_wavefields (acoustic)
        ├── time_scheme.apply_corrector(acoustic)
        ├── compute_derivatives + update_wavefields (elastic)
        ├── time_scheme.apply_corrector(elastic)
        ├── compute_derivatives + update_wavefields (poroelastic)
        ├── time_scheme.apply_corrector(poroelastic)
        ├── apply boundary conditions
        ├── accumulate seismograms (every nstep_between_samples)
        └── run periodic tasks
        │
        ▼
13. Write seismograms to disk
        │
        ▼
14. [Optional] Write sensitivity kernels to disk
```

---

## Simulation Modes

SPECFEM++ supports two simulation types, controlled by the `simulation-mode` YAML section:

### Forward (`simulation::type::forward`)

Standard forward wave propagation. Sources inject energy, receivers record synthetic seismograms. Optionally saves:
- Seismograms (displacement, velocity, acceleration, pressure)
- Wavefield snapshots at configurable intervals
- Material property files

### Combined (`simulation::type::combined`)

Adjoint + backward simulation for computing **Fréchet sensitivity kernels** used in seismic tomography:

1. The **adjoint wavefield** is propagated forward with time-reversed seismogram residuals injected at receiver locations.
2. The **backward wavefield** reconstructs the original forward wavefield from stored boundary values.
3. The two fields are **cross-correlated** at each timestep to accumulate Fréchet kernels (∂χ/∂m for each material parameter m).

This requires a prior forward run with `wavefield-writer` configured to save boundary checkpoints.

---

## Hardware Portability via Kokkos

All performance-critical arrays are `Kokkos::View<...>` and all parallel loops use `Kokkos::parallel_for` / `Kokkos::parallel_reduce`. This means the exact same source code runs on:

| Target | CMake Preset | Kokkos Backend |
|---|---|---|
| CPU (serial) | `serial` | `Kokkos::Serial` |
| CPU (OpenMP) | `omp` | `Kokkos::OpenMP` |
| NVIDIA GPU | `cuda` | `Kokkos::Cuda` |
| AMD GPU | `hip` | `Kokkos::HIP` |
| Intel GPU | `sycl` | `Kokkos::SYCL` |

Compile-time backend detection in `parallel_configuration` adjusts chunk sizes and SIMD widths to match the hardware. SIMD vectorization on CPUs uses `Kokkos::Experimental::simd<type_real>`.

The floating-point precision is selectable at configure time:
- Default: single precision (`float`)
- `SPECFEM_ENABLE_DOUBLE_PRECISION=ON`: double precision (`double`)

---

## Build System

SPECFEM++ uses CMake with [CMakePresets](CMakePresets.json). Key options:

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

Dependencies are fetched automatically via `FetchContent` if not found on the system (Kokkos, yaml-cpp, CLI11, Boost, etc.).

---

## Python Bindings

**Files:** `python/specfempp_core/`

A Python package `specfempp_core` exposes C++ simulation components to Python via nanobind (or pybind11). This enables configuration and execution of SPECFEM++ simulations from Python without writing YAML files manually. The [specfempp-py](https://github.com/PrincetonUniversity/SPECFEMPP-py) package provides a higher-level Python API on top of these bindings.

Enable with `-DSPECFEM_BINDING_PYTHON=ON` at configure time.

---

## Key Dependencies

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

---

*This document covers the architecture as of the current `devel` branch. For API reference documentation, see the [online Doxygen docs](https://specfem2d-kokkos.readthedocs.io/en/latest/). For usage examples, see the [`examples/`](examples/) directory and the [cookbook documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/sections/cookbooks/index.html).*
