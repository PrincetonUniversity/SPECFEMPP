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

Tests are declared with `specfem_add_test()` (see `tests/test_registration.cmake`), which
defines the executable and registers it with CTest in one call. There are no separate
target lists to keep in sync:

```cmake
specfem_add_test(my_tests
  SOURCES   my/test.cpp
  LIBRARIES specfem::mesh specfem_environment gtest_main
)
```

- Serial tests go in `tests/<suite>/serial.cmake`, MPI tests in `tests/<suite>/mpi.cmake`
  (only included when `SPECFEM_ENABLE_MPI` is ON).
- An MPI test declares its process count inline: `MPI_RANKS 4`.
- Other options: `NO_UNITY`, `TIMEOUT`, `DEFINITIONS`, `INCLUDES`, `PROPERTIES`, `LABELS`.
- Test data directories are exposed to the tests' working directory with
  `specfem_add_test_data(<dir>...)`.

## Test output directory

- Tests run with their working directory set to `SPECFEM_TEST_OUTPUT_DIR`, an absolute path
  computed once by `specfem_init_tests()`. Test binaries stay in the build tree.
- Default: `<build>/tests/run`. Override with `-D SPECFEMPP_TEST_DIR=<path>` (relative paths
  resolve against the source dir; absolute paths are used as-is).
- `ctest --test-dir <build>/tests`, `ctest --test-dir <SPECFEM_TEST_OUTPUT_DIR>`, and
  `cd <SPECFEM_TEST_OUTPUT_DIR> && ctest` all work.
