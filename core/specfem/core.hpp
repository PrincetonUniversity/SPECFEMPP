#pragma once

#include "enumerations/dimension.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem {

/**
 * @brief Unified SPECFEM++ core class for managing initialization, execution,
 * and finalization
 *
 * This class provides a clean singleton interface for managing Kokkos and MPI
 * initialization, dimension-templated execution, and proper resource cleanup.
 * It replaces the global variable approach with a more maintainable RAII-based
 * design.
 */
class Core {
public:
  /**
   * @brief Get the singleton instance
   */
  static Core &instance();

  /**
   * @brief Initialize Kokkos and MPI
   *
   * @param argc Command line argument count
   * @param argv Command line arguments
   * @return true if initialization successful
   */
  bool initialize(int argc, char *argv[]);

  /**
   * @brief Initialize from Python with argument list
   *
   * @param py_argv Python list of command line arguments
   * @return true if initialization successful
   */
  bool initialize_from_python(const std::vector<std::string> &py_argv);

  /**
   * @brief Finalize and cleanup all resources
   *
   * @return true if finalization successful
   */
  bool finalize();

  /**
   * @brief Execute simulation with dimension template
   *
   * @tparam DIM Dimension type (dim2 or dim3)
   * @param parameter_dict YAML parameter configuration
   * @param default_dict YAML default configuration
   * @param tasks Vector of periodic tasks
   * @return true if execution successful
   */
  template <specfem::dimension::type DIM>
  bool
  execute(const YAML::Node &parameter_dict, const YAML::Node &default_dict,
          std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
              &tasks);

  /**
   * @brief Execute simulation with dimension specified as string
   *
   * @param dimension Dimension string ("2d" or "3d")
   * @param parameter_dict YAML parameter configuration
   * @param default_dict YAML default configuration
   * @param tasks Vector of periodic tasks
   * @return true if execution successful
   */
  bool execute_with_dimension(
      const std::string &dimension, const YAML::Node &parameter_dict,
      const YAML::Node &default_dict,
      std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
          &tasks);

  /**
   * @brief Get MPI instance (for backward compatibility)
   *
   * @return Pointer to MPI instance or nullptr if not initialized
   */
  specfem::MPI::MPI *get_mpi() const;

  /**
   * @brief Check if core is initialized
   *
   * @return true if initialized
   */
  bool is_initialized() const;

  /**
   * @brief Check if Kokkos is initialized
   *
   * @return true if Kokkos is initialized
   */
  bool is_kokkos_initialized() const;

  /**
   * @brief Destructor - ensures proper cleanup
   */
  ~Core();

private:
  /**
   * @brief Private constructor for singleton
   */
  Core();

  /**
   * @brief Delete copy constructor and assignment operator
   */
  Core(const Core &) = delete;
  Core &operator=(const Core &) = delete;

  /**
   * @brief Internal helper to convert string arguments to argc/argv
   */
  void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                       char **&argv);
  void cleanup_argc_argv(int argc, char **argv);

  static Core *instance_;
  specfem::MPI::MPI *mpi_;
  bool kokkos_initialized_;
  bool mpi_initialized_;
  bool core_initialized_;
};

} // namespace specfem
