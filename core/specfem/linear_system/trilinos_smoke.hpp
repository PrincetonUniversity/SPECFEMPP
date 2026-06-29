#pragma once

namespace specfem {
namespace linear_system {

/**
 * @brief Minimal Trilinos toolchain check (M0 gate for the implicit-solver
 * work).
 *
 * Builds a tiny 1x1 `Tpetra::CrsMatrix`, calls `fillComplete()`, and returns
 * the global row count. Its sole purpose is to prove that SPECFEM++ configures,
 * links, and runs against the Trilinos install selected by `module load
 * trilinos/<variant>` (and shares one Kokkos with it). Replaced by real
 * assembly code in later phases.
 *
 * When SPECFEM++ is built without Trilinos (`SPECFEM_ENABLE_TRILINOS=OFF`),
 * this is a no-op that returns 0.
 *
 * @return Number of global rows in the test matrix (1 with Trilinos, 0
 * without).
 *
 * @note Assumes Kokkos has already been initialized by the calling program.
 */
int trilinos_smoke_test();

} // namespace linear_system
} // namespace specfem
