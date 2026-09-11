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

/**
 * @brief Minimal Belos + Ifpack2 toolchain check (gate for the implicit
 * Newmark solver, issue #1984).
 *
 * Solves a small SPD tridiagonal system \f$ A x = b \f$ (with \f$ b = A
 * \mathbf{1} \f$) through the exact stack the implicit solver uses: an
 * `Ifpack2::Factory`-created RILUK preconditioner applied as a right
 * preconditioner inside `Belos::PseudoBlockGmresSolMgr`, on
 * `scalar_type = type_real`. Its purpose is to prove -- before any solver
 * code lands -- that the Trilinos install provides Belos and Ifpack2
 * instantiations for the build's scalar (the cluster installs are
 * float-only).
 *
 * When SPECFEM++ is built without Trilinos (`SPECFEM_ENABLE_TRILINOS=OFF`),
 * this is a no-op that returns -1.
 *
 * @return Number of GMRES iterations to convergence (>= 0); throws
 * `std::runtime_error` if the solve does not converge or the solution is
 * wrong. Returns -1 without Trilinos.
 *
 * @note Assumes Kokkos has already been initialized by the calling program.
 */
int linear_solver_smoke_test();

} // namespace linear_system
} // namespace specfem
