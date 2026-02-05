#pragma once

#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::program {

/**
 * @brief RAII-based context managing Kokkos and MPI lifecycle
 *
 * This class provides RAII-based management of Kokkos and MPI initialization
 * and proper resource cleanup.
 *
 * Usage:
 * @code
 * int main(int argc, char* argv[]) {
 *     specfem::program::Context context(argc, argv);
 *     // Kokkos & MPI automatically initialized
 *     // ... use context ...
 *     // Automatic cleanup on scope exit
 * }
 * @endcode
 *
 */
class Context {
public:
  /**
   * @brief Initialize context from command line arguments
   * @param argc Argument count
   * @param argv Argument vector
   */
  Context(int argc, char *argv[]);

  /**
   * @brief Initialize context from argument vector
   * @param args Vector of arguments
   */
  explicit Context(const std::vector<std::string> &args);

  /**
   * @brief Finalize Kokkos and MPI
   */
  ~Context();

  /**
   * @name Deleted copy/move constructors and assignment operators
   *
   * Context manages global resources (Kokkos and MPI) with RAII semantics.
   * Copying or moving would violate single-ownership and could lead to
   * double-finalization or resource corruption.
   *
   * @{
   */
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;
  /** @} */

private:
  /**
   * @brief Convert string vector to argc/argv format
   * @param args Vector of string arguments
   * @param argc Output argument count
   * @param argv Output argument vector (dynamically allocated)
   */
  static void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                              char **&argv);

  /**
   * @brief Clean up dynamically allocated argc/argv
   * @param argc Argument count
   * @param argv Argument vector to deallocate
   */
  static void cleanup_argc_argv(int argc, char **argv);
  std::unique_ptr<Kokkos::ScopeGuard> kokkos_guard_; ///< Scope Guard for Kokkos
                                                     ///< lifecycle management
};

} // namespace specfem::program
