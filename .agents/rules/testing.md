---
paths:
  - "tests/**"
---

# Testing Conventions

## Framework

- GoogleTest (gtest). All tests use `#include <gtest/gtest.h>`.
- Test executables have a `runner.cpp` with the standard GoogleTest main:
  ```cpp
  #include "../../SPECFEM_Environment.hpp"
  #include "gtest/gtest.h"

  int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
    return RUN_ALL_TESTS();
  }
  ```
- Link against the `specfem++` library target.

## Test fixture pattern

- Tests use YAML-configured fixtures. Configuration lives in `test_config.yaml` per module.
- The fixture class loads test data via `YAML::Node`, specifying mesh databases,
  source files, station files, and expected solutions.
- Pattern: `test_fixture.hpp` (declaration), `test_fixture.tpp` (template impl),
  `test_fixture.cpp` (explicit instantiations).

## Naming

- Test files: `snake_case.cpp`
- Test fixtures: `CamelCase` (GoogleTest convention, e.g., `AcousticElasticCouplingTest`)
- Test cases: descriptive `CamelCase` (e.g., `CouplingCalculation`)

## Requirements

- All new public APIs must have corresponding unit tests.
- Test edge cases and boundary conditions, not just the happy path.
- Use `EXPECT_NEAR` with explicit tolerances for floating-point comparisons.
  Do not use `EXPECT_EQ` for floats.
- Parameterized tests (`TEST_P` / `TestWithParam`) are preferred for testing
  multiple configurations.

## Test data

- Test data lives in `tests/unit-tests/<module>/data/` or alongside test files.
- Binary mesh databases, YAML source/station configs, and STATIONS files.
- Do not commit large binary files without checking with maintainers first.

## Directory structure

Tests mirror the `core/specfem/` structure:
```
tests/unit-tests/
  mesh/dim2/, mesh/dim3/
  assembly/
  algorithms/
  displacement_tests/Newmark/
  ...
```
