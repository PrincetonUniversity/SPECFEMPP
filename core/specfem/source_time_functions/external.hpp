#pragma once
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>
#include <vector>

namespace specfem {
namespace source_time_functions {

/**
 * @brief External source time function loaded from file
 *
 * Allows users to define custom source time functions by reading from
 * external files in various formats (ASCII, HDF5, etc.).
 */
class external : public stf {
public:
  /**
   * @brief Construct an external source time function from YAML configuration
   *
   * @param external YAML node with file paths and format specification
   * @param nsteps Number of time steps
   * @param dt Time step size
   */
  external(const YAML::Node &external, const int nsteps, const type_real dt);

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
          source_time_function) override;

  type_real get_tshift() const override {
    throw std::runtime_error(
        "Time shift not defined for external source time function");
  }

  /**
   * @brief Throw error as time shift update is not supported
   */
  void update_tshift(type_real tshift) override;

  std::string print() const override;

  /**
   * @brief Get the seismogram file format type
   *
   * @return specfem::enums::seismogram::format
   */
  specfem::enums::seismogram::format get_format() const { return format_; }

  /**
   * @brief Get the start time value
   *
   * @return Start time t0
   */
  type_real get_t0() const override { return t0_; }
  type_real get_dt() const { return dt_; }
  int get_nsteps() { return nsteps_; }
  int get_ncomponents() const { return ncomponents_; }

  /**
   * @brief Get the file path for the X-component
   *
   * @return File path as a string
   */
  std::string get_x_component() const { return x_component_; }

  /**
   * @brief Get the file path for the Y-component
   *
   * @return File path as a string
   */
  std::string get_y_component() const { return y_component_; }

  /**
   * @brief Get the file path for the Z-component
   *
   * @return File path as a string
   */
  std::string get_z_component() const { return z_component_; }

  bool operator==(const stf &other) const override;
  bool operator!=(const stf &other) const override;

private:
  int nsteps_;                                ///< Number of time steps
  type_real t0_;                              ///< Start time
  type_real dt_;                              ///< Time step size
  specfem::enums::seismogram::format format_; ///< File format type
  int ncomponents_;                           ///< Number of components
  std::string x_component_ = "";              ///< X-component file path
  std::string y_component_ = "";              ///< Y-component file path
  std::string z_component_ = "";              ///< Z-component file path
};
} // namespace source_time_functions
} // namespace specfem
