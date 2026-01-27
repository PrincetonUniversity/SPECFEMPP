#pragma once
#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace source_time_functions {
/**
 * @brief Derivative of Gaussian source time function
 *
 * The first derivative of a Gaussian pulse, useful for modeling certain types
 * of seismic sources.
 */
class dGaussian : public stf {

public:
  /**
   * @brief Construct a Gaussian derivative source time function
   *
   * @param nsteps Number of time steps
   * @param dt Time step size
   * @param f0 Central frequency
   * @param tshift Time shift value
   * @param factor Scaling factor
   * @param use_trick_for_better_pressure Use pressure optimization trick
   * @param t0_factor Start time factor (default: 1.2)
   */
  dGaussian(const int nsteps, const type_real dt, const type_real f0,
            const type_real tshift, const type_real factor,
            const bool use_trick_for_better_pressure,
            const type_real t0_factor = 1.2);

  /**
   * @brief Construct a dGaussian source time function from YAML configuration
   *
   * @param dGaussian YAML node with dGaussian parameters
   * @param nsteps Number of time steps
   * @param dt Time step size
   * @param use_trick_for_better_pressure Use pressure optimization trick
   * @param t0_factor Start time factor (default: 1.2)
   */
  dGaussian(YAML::Node &dGaussian, const int nsteps, const type_real dt,
            const bool use_trick_for_better_pressure,
            const type_real t0_factor = 1.2);

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
  type_real get_f0() const { return this->f0_; }
  int get_nsteps() const { return this->nsteps_; }
  bool get_use_trick_for_better_pressure() const {
    return this->use_trick_for_better_pressure_;
  }

  bool operator==(const stf &other) const override;
  bool operator!=(const stf &other) const override;

private:
  int nsteps_;                         ///< Number of time steps
  type_real f0_;                       ///< Central frequency
  type_real tshift_;                   ///< Time shift value
  type_real t0_;                       ///< Start time
  type_real t0_factor_;                ///< Start time computation factor
  type_real factor_;                   ///< Scaling factor
  bool use_trick_for_better_pressure_; ///< Pressure optimization flag
  type_real dt_;                       ///< Time step size
};

} // namespace source_time_functions
} // namespace specfem
