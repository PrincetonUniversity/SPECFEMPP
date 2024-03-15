#ifndef _STF_DIRAC_HPP
#define _STF_DIRAC_HPP

#include "kokkos_abstractions.h"
#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {
class Dirac : public stf {

public:
  /**
   * @brief Contruct a Dirac source time function object
   *
   * @param f0 frequency f0
   * @param tshift tshift value
   * @param factor factor to scale source time function
   * @param use_trick_for_better_pressure
   */
  Dirac(const int nsteps, const type_real dt, const type_real f0,
        const type_real tshift, const type_real factor,
        const bool use_trick_for_better_pressure);

  Dirac(YAML::Node &Dirac, const int nsteps, const type_real dt,
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
  void update_tshift(type_real tshift) override { this->tshift = tshift; }
  /**
   * @brief Get the t0 value
   *
   * @return t0 value
   */
  type_real get_t0() const override { return this->t0; }

  std::string print() const override;

  void compute_source_time_function(
      specfem::kokkos::HostView1d<type_real> source_time_function) override;

private:
  int nsteps;
  type_real f0;     ///< frequence f0
  type_real tshift; ///< value of tshit
  type_real t0;     ///< t0 value
  type_real factor; ///< scaling factor
  bool use_trick_for_better_pressure;
  type_real dt;
};

} // namespace forcing_function
} // namespace specfem

#endif
