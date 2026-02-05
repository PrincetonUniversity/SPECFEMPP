#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {
template <specfem::element::dimension_tag DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base class for tasks executed periodically during simulation
 *
 */
template <specfem::element::dimension_tag DimensionTag> class periodic_task {
public:
  /**
   * @brief Construct a new periodic task object
   *
   * @param time_interval Time interval between subsequent task executions
   * @param include_last_step Whether or not to include the last step regardless
   * of the time interval
   */
  periodic_task(const int time_interval, const bool include_last_step = true)
      : time_interval(time_interval), include_last_step(include_last_step) {};

  virtual ~periodic_task() = default;

  /**
   * @brief Function to be called periodically.
   *
   */
  virtual void run(specfem::assembly::assembly<DimensionTag> &assembly,
                   const int istep) {
    // Default implementation for dimension-agnostic tasks
    // Derived classes can override for dimension-specific behavior
  };

  /**
   * @brief Functions to be called once at the beginning and once at the end of
   * the simulation.
   *
   */
  virtual void initialize(specfem::assembly::assembly<DimensionTag> &assembly) {
    // Default implementation
  };

  virtual void finalize(specfem::assembly::assembly<DimensionTag> &assembly) {
    // Default implementation
  };

  /**
   * @brief Returns true if the data should be plotted at the current
   * timestep. Updates the internal timestep counter
   *
   * @param istep Current timestep
   * @return true if the data should be plotted at the current timestep
   */
  bool should_run(const int istep) {
    if (include_last_step && istep == -1) {
      return true;
    }

    if (time_interval > 0 && istep % time_interval == 0) {
      return true;
    }

    return false;
  }

protected:
  int time_interval;
  bool include_last_step;
};

} // namespace periodic_tasks
} // namespace specfem
