---
paths:
  - "CMakeLists.txt"
  - "**/CMakeLists.txt"
  - "cmake/**"
  - "CMakePresets.json"
---

# CMake and Build System

## Standards

- Minimum CMake version: 3.17.5
- C++20 standard (`CMAKE_CXX_STANDARD 20`)
- Unity builds enabled by default (`CMAKE_UNITY_BUILD ON`, batch size 8)
  - This means anonymous namespaces cause ODR violations -- never use them
  - Unity builds are disabled automatically for Debug configurations

## Conventions

- Use `FetchContent` for external dependencies.
- Build options follow the `SPECFEM_ENABLE_*` naming pattern
  (e.g., `SPECFEM_ENABLE_MPI`, `SPECFEM_ENABLE_HDF5`).
- Binary output: `${CMAKE_BINARY_DIR}/bin`
- Library output: `${CMAKE_BINARY_DIR}/lib`
- Default install prefix: `${CMAKE_SOURCE_DIR}/bin`

## Presets

- Use `CMakePresets.json` for standard build configurations.
- User-specific overrides go in `CMakeUserPresets.json` (gitignored).
- Available presets can be listed with `cmake --list-presets`.

## Test targets

- Serial tests: `SERIAL_TEST_TARGETS`
- MPI tests: `MPI_TEST_TARGETS` (only when `SPECFEM_ENABLE_MPI` is ON)
- Test finalization: `specfem_finalize_test_targets(ALL_TEST_TARGETS ...)`
