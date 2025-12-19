#pragma once

#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::program {

/**
 * @brief Unified SPECFEM++ context class for managing initialization
 * and finalization
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
 */
class Context {
public:
  /**
   * @brief Construct Context with command line arguments
   *
   * @param argc Command line argument count
   * @param argv Command line arguments
   */
  Context(int argc, char *argv[]);

  /**
   * @brief Construct Context with argument vector
   *
   * @param args Vector of command line arguments
   */
  explicit Context(const std::vector<std::string> &args);

  /**
   * @brief Get MPI instance
   *
   * @return Pointer to MPI instance
   */
  specfem::MPI::MPI *get_mpi() const;

  /**
   * @brief Destructor - ensures proper cleanup via RAII
   */
  ~Context();

  /**
   * @brief Delete copy constructor and assignment operator
   *
   * Context cannot be copied or moved to prevent multiple finalization
   */
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;

private:
  /**
   * @brief Internal helper to convert string arguments to argc/argv
   */
  static void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                              char **&argv);
  static void cleanup_argc_argv(int argc, char **argv);

  std::unique_ptr<Kokkos::ScopeGuard> kokkos_guard_;
  std::unique_ptr<specfem::MPI::MPI> mpi_;
};

} // namespace specfem::program
