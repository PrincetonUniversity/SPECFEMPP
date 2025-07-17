#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
template <specfem::dimension::type DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base writer class
 *
 */
class periodic_task {
public:
  /**
   * @brief Construct a new plotter object
   *
   * @param time_interval Time interval between subsequent plots
   * @param include_last_step Whether or not to include the last step regardless
   * of the time interval
   */
  periodic_task(const int time_interval, const bool include_last_step = true)
      : time_interval(time_interval), include_last_step(include_last_step) {};

  /**
   * @brief Function to be called periodically.
   *
   */
  virtual void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep) {};

  /**
   * @brief Functions to be called once at the beginning and once at the end of
   * the simulation.
   *
   */
  virtual void initialize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {};
  virtual void finalize(
      specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {};

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
