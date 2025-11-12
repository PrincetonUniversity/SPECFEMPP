#pragma once

#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem {

/**
 * @brief Unified SPECFEM++ context class for managing initialization
 * and finalization
 *
 * This class provides RAII-based management of Kokkos and MPI initialization
 * and proper resource cleanup.
 * Typically used through ContextGuard for scoped lifetime management.
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
   */
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

private:
  Kokkos::ScopeGuard kokkos_guard_;
  std::unique_ptr<specfem::MPI::MPI> mpi_;
};

/**
 * @brief RAII guard for SPECFEM++ context (similar to Kokkos::ScopeGuard)
 *
 * This class provides automatic initialization and finalization of SPECFEM++
 * resources following the RAII pattern. It owns a Context instance and
 * initializes Kokkos and MPI in the constructor, ensuring proper cleanup
 * in the destructor.
 *
 * Usage:
 * @code
 * int main(int argc, char* argv[]) {
 *     specfem::ContextGuard guard(argc, argv);
 *     // Kokkos & MPI automatically initialized
 *     // ... use guard.get_context() ...
 *     // Automatic cleanup on scope exit
 * }
 * @endcode
 */
class ContextGuard {
public:
  /**
   * @brief Constructor - creates and initializes Context with command line
   * arguments
   *
   * @param argc Command line argument count
   * @param argv Command line arguments
   */
  ContextGuard(int argc, char *argv[]);

  /**
   * @brief Constructor - creates and initializes Context with argument vector
   *
   * @param args Vector of command line arguments
   */
  explicit ContextGuard(const std::vector<std::string> &args);

  /**
   * @brief Destructor - automatically destroys Context (triggering cleanup)
   */
  ~ContextGuard();

  /**
   * @brief Get reference to the owned Context instance
   *
   * @return Reference to the managed Context
   */
  Context &get_context();

  /**
   * @brief Get MPI instance for convenience
   *
   * @return Pointer to MPI instance
   */
  specfem::MPI::MPI *get_mpi() const;

  /**
   * @brief Delete copy constructor and assignment operator
   *
   * ContextGuard cannot be copied or moved to prevent multiple finalization
   */
  ContextGuard(const ContextGuard &) = delete;
  ContextGuard &operator=(const ContextGuard &) = delete;
  ContextGuard(ContextGuard &&) = delete;
  ContextGuard &operator=(ContextGuard &&) = delete;

private:
  /**
   * @brief Internal helper to convert string arguments to argc/argv
   */
  static void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                              char **&argv);
  static void cleanup_argc_argv(int argc, char **argv);

  std::unique_ptr<Context> context_;
};

} // namespace specfem
