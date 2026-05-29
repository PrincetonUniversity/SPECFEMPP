# C++ Style Conventions

## Naming

- **Classes and structs**: `snake_case` (e.g., `time_scheme`, `absorbing_boundary`, `tags_container`)
- **Functions and methods**: `snake_case` (e.g., `compute_stiffness`, `load_on_device`)
- **Variables and members**: `snake_case` (e.g., `num_elements`, `medium_tag`)
- **Namespaces**: `snake_case` (e.g., `specfem::mesh`, `specfem::assembly`)
- **File names**: `snake_case` with `.hpp`, `.cpp`, `.tpp` extensions
- **Template parameters**: `CamelCase` (e.g., `DimensionTag`, `MediumTag`, `ParallelConfig`)
- **Enum values**: `snake_case` within scoped enums (e.g., `medium_tag::elastic_psv`)
- **Constants and type aliases**: `snake_case` (e.g., `using type_real = ...`)

## Formatting

Formatting is enforced by `.clang-format` and pre-commit hooks. Key settings:

- LLVM-based style, 2-space indent, 80-character column limit
- `NamespaceIndentation: None`
- `BreakBeforeBraces: Attach`
- Pointer does not bind to type (`int *p`, not `int* p`)
- Do not manually reformat entire files; let clang-format handle it

## Header guards

Always use `#pragma once`. Never use `#ifndef`/`#define` guards.

## `using namespace`

- NEVER use `using namespace` at file scope or namespace scope in any file.
- Fully qualify all names: `std::vector`, `std::string`, `Kokkos::View`, etc.
- **Sole exception:** `using namespace specfem::units::unit_symbols;` is acceptable
  inside function bodies only, for unit-literal readability (e.g., `5.0 * Hz`).

## Anonymous namespaces

- NEVER use anonymous namespaces (`namespace { ... }`).
- The project uses CMake unity builds (batch size 8), and anonymous namespaces
  cause ODR violations when translation units are merged.
- Instead, use a `_impl` suffix namespace for helper functions that should not be
  part of the public API:
  ```cpp
  // CORRECT: use _impl suffix namespace
  namespace specfem::io::sources_impl {
    std::string trim(const std::string &s);
  }

  // WRONG: anonymous namespace
  namespace {
    std::string trim(const std::string &s);
  }
  ```

## Includes

- Project headers: `#include "specfem/..."` (relative to `core/` include path)
- Third-party headers: `#include <Kokkos_Core.hpp>`, `#include <yaml-cpp/yaml.h>`
- Standard library: `#include <vector>`, `#include <string>`
- Order: project headers first, then third-party, then standard library

## General

- Initialize all member variables. Never leave fields in an uninitialized state.
  Use default member initializers or constructor initializer lists.
- Prefer `const auto&` when capturing return values that are references or
  contain Kokkos Views (avoids atomic refcount overhead from copy constructors).
- Use `if constexpr` for compile-time branching on template parameters.
