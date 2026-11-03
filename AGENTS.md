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

- **Naming:** `snake_case` for functions, variables, namespaces, and file names.
  `CamelCase` for classes/structs (new types) and template parameters (e.g.,
  `DimensionTag`, `MediumTag`). Much existing code still uses `snake_case` class names.
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

## Coding-agent configuration

Shared agent configuration lives in `.agents/` and is used by both Claude Code and Codex:

- `AGENTS.md` -- This file (Codex reads it directly; `CLAUDE.md` is a symlink to it).
- `.agents/rules/` -- Detailed conventions. Read the relevant file before working in that area:
  `cpp-style.md`, `namespaces.md`, `kokkos-patterns.md`, `testing.md`, `cmake-build.md`,
  `documentation.md`. Claude Code loads them via the `.claude/rules` symlink.
- `.agents/skills/<name>/SKILL.md` -- Reusable workflows (`review-pr`, `fix-pr-feedback`,
  `audit-changes`, `review-conventions`, `write-docs`, `profile-hotspot`). Codex discovers
  them in `.agents/skills/` (invoke with `$<name>`); Claude Code discovers them via the
  `.claude/skills` symlink (invoke with `/<name>`). Each `SKILL.md` needs `name` (matching
  its directory) and `description` frontmatter. Edit skills in `.agents/skills/`, never
  through the symlink path, and do not add `.claude/commands/` (superseded by skills).
