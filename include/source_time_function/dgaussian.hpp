#pragma once
#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {
class dGaussian : public stf {

public:
  /**
   * @brief Contruct a dGaussian source time function object
   *
   * @param f0 frequency f0
   * @param tshift tshift value
   * @param factor factor to scale source time function
   * @param use_trick_for_better_pressure
   */
  dGaussian(const int nsteps, const type_real dt, const type_real f0,
            const type_real tshift, const type_real factor,
            const bool use_trick_for_better_pressure);

  dGaussian(YAML::Node &dGaussian, const int nsteps, const type_real dt,
            const bool use_trick_for_better_pressure);

  /**
   * @brief compute the value of stf at time t
   *
   * @param t
   * @return value of source time function at time t
   */
  type_real compute(type_real t);
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  void update_tshift(type_real tshift) override { this->__tshift = tshift; }
  /**
   * @brief Get the t0 value
   *
   * @return t0 value
   */
  type_real get_t0() const override { return this->__t0; }

  type_real get_tshift() const override { return this->__tshift; }

  std::string print() const override;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) override;

private:
  int __nsteps;
  type_real __f0;     ///< frequence f0
  type_real __tshift; ///< value of tshit
  type_real __t0;     ///< t0 value
  type_real __factor; ///< scaling factor
  bool __use_trick_for_better_pressure;
  type_real __dt;
};

} // namespace forcing_function
} // namespace specfem
