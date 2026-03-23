# Program Lifecycle

## Entry Point and CLI

**File:** `core/specfem.cpp`

The `main()` function uses [CLI11](https://github.com/CLIUtils/CLI11) to parse command-line arguments and dispatch to one of three subcommands:

| Subcommand | Description |
|---|---|
| `2d` | Run a 2D simulation, requires `-p <config.yaml>` |
| `3d` | Run a 3D simulation, requires `-p <config.yaml>` |
| `qplots` | Generate Q-attenuation diagnostic plots |

All logging options (`--log-file`, `--log-level`, etc.) are accepted here and forwarded to `specfem::Logger`. After the CLI parses, control is handed to `specfem::program::execute(dimension, parameter_dict)`.

---

## `program::Context`

**Files:** `core/specfem/program/`

`Context` is an RAII guard that manages the lifetime of **Kokkos** and **MPI**. It must be the first object created in `main()`.

```cpp
specfem::program::Context context(argc, argv);
// Kokkos::initialize() and MPI_Init() called here
// ... simulation runs ...
// ~Context(): Kokkos::finalize() and MPI_Finalize() called automatically
```

It is non-copyable and non-movable; only one `Context` should exist at a time.

---

## `program::execute`

`program::execute(dimension, parameter_dict)` dispatches to `program_2d(...)` or `program_3d(...)`, which implement the full simulation workflow described in [Simulation Workflow](../simulation/workflow.md).

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
