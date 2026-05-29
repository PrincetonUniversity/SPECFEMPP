# SPECFEM++

SPECFEM++ is a C++20 spectral-element code for seismic wave propagation, portable across
CPU (serial/OpenMP), NVIDIA (CUDA), AMD (HIP), and Intel GPUs via Kokkos.

## Repository layout

- `core/specfem/` -- All C++ library code (headers + `.cpp`/`.tpp` implementations)
- `tests/unit-tests/` -- GoogleTest suite with YAML-configured fixtures
- `docs/` -- Sphinx + Breathe + Doxygen documentation (RST format)
- `architecture.md` and `architecture/` -- Full architecture documentation
- `python/` -- Python bindings (`specfempp_core` via pybind11)
- `src/` -- Fortran mesh I/O (legacy specfem2d/specfem3d readers)

## Critical conventions

- **Naming:** `snake_case` for classes, functions, variables, namespaces, and file names.
  `CamelCase` only for template parameters (e.g., `DimensionTag`, `MediumTag`).
- **No `using namespace`** at file or namespace scope. Fully qualify all names.
  Sole exception: `using namespace specfem::units::unit_symbols;` inside function bodies.
- **No anonymous namespaces** -- unity builds are enabled; use `_impl` suffix namespaces
  instead (e.g., `specfem::io::sources_impl`).
- **Max 3 namespace levels:** `specfem::component::sub_component`
- **Header guards:** `#pragma once` (never `#ifndef`/`#define`)
- **Doxygen:** `@brief`, `@param`, `@return`, `@tparam`, `\f$ math \f$`. No `@file` directives.
- **Formatting:** `.clang-format` (LLVM-based, 2-space indent, 80-char limit).
  Run `uv run pre-commit run --all-files` to check.

## Build

```bash
cmake --preset <preset-name>
cmake --build build/<preset>
ctest --test-dir build/<preset>
```

Unity builds enabled by default (batch size 8). See `CMakePresets.json` for available presets.

## Key dependencies

Kokkos, CLI11, yaml-cpp, GoogleTest, Boost (optional), HDF5/ADIOS2 (optional), VTK (optional)
