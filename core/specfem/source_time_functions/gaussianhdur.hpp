#pragma once

#include "kokkos_abstractions.h"
#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace source_time_functions {
/**
 * @brief Gaussian source time function parameterized by half duration
 *
 * Similar to Gaussian but uses half duration (hdur) instead of frequency for
 * parameterization.
 */
class GaussianHdur : public stf {

public:
  /**
   * @brief Construct a Gaussian source time function with half duration
   * parameterization
   *
   * @param nsteps Number of time steps
   * @param dt Time step size
   * @param hdur Half duration
   * @param tshift Time shift value
   * @param factor Scaling factor
   * @param use_trick_for_better_pressure Use pressure optimization trick
   * @param t0_factor Start time factor (default: 1.5)
   */
  GaussianHdur(const int nsteps, const type_real dt, const type_real hdur,
               const type_real tshift, const type_real factor,
               const bool use_trick_for_better_pressure,
               const type_real t0_factor = 1.5);

  /**
   * @brief Construct a GaussianHdur source time function from YAML
   * configuration
   *
   * @param GaussianNode YAML node with GaussianHdur parameters
   * @param nsteps Number of time steps
   * @param dt Time step size
   * @param use_trick_for_better_pressure Use pressure optimization trick
   * @param t0_factor Start time factor (default: 1.5)
   */
  GaussianHdur(YAML::Node &GaussianNode, const int nsteps, const type_real dt,
               const bool use_trick_for_better_pressure,
               const type_real t0_factor = 1.5);

  /**
   * @brief Compute source time function value at time t
   *
   * @param t Time value
   * @return Source time function value
   */
  type_real compute(type_real t);
  /**
   * @brief Update the time shift value
   *
   * @param tshift New time shift value
   */
  void update_tshift(type_real tshift) override { this->tshift_ = tshift; }
  /**
   * @brief Get start time value
   *
   * @return Start time t0
   */
  type_real get_t0() const override { return this->t0_; }

  type_real get_tshift() const override { return this->tshift_; }

  std::string print() const override;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
          source_time_function) override;

  type_real get_dt() const { return this->dt_; }
  type_real get_factor() const { return this->factor_; }
  type_real get_hdur() const { return this->hdur_; }
  int get_nsteps() const { return this->nsteps_; }
  bool get_use_trick_for_better_pressure() const {
    return this->use_trick_for_better_pressure_;
  }
  int get_ncomponents() const { return 1; }

  bool operator==(const stf &other) const override;
  bool operator!=(const stf &other) const override;

private:
  int nsteps_;                         ///< Number of time steps
  type_real hdur_;                     ///< Half duration
  type_real tshift_;                   ///< Time shift value
  type_real t0_;                       ///< Start time
  type_real t0_factor_;                ///< Start time computation factor
  type_real factor_;                   ///< Scaling factor
  bool use_trick_for_better_pressure_; ///< Pressure optimization flag
  type_real dt_;                       ///< Time step size
};

} // namespace source_time_functions
} // namespace specfem
