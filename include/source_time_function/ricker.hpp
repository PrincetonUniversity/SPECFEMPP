#ifndef _STF_RICKER_HPP
#define _STF_RICKER_HPP

#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {
class Ricker : public stf {

public:
  /**
   * @brief Contruct a Ricker source time function object
   *
   * @param f0 frequency f0
   * @param tshift tshift value
   * @param factor factor to scale source time function
   * @param use_trick_for_better_pressure
   */
  KOKKOS_FUNCTION Ricker(type_real f0, type_real tshift, type_real factor,
                         bool use_trick_for_better_pressure);

  /**
   * @brief compute the value of stf at time t
   *
   * @param t
   * @return value of source time function at time t
   */
  KOKKOS_FUNCTION type_real compute(type_real t) override;
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  KOKKOS_FUNCTION void update_tshift(type_real tshift) override {
    this->tshift = tshift;
  }
  /**
   * @brief Get the t0 value
   *
   * @return t0 value
   */
  KOKKOS_FUNCTION type_real get_t0() const override { return this->t0; }

private:
  type_real f0;     ///< frequence f0
  type_real tshift; ///< value of tshit
  type_real t0;     ///< t0 value
  type_real factor; ///< scaling factor
  bool use_trick_for_better_pressure;
};

} // namespace forcing_function
} // namespace specfem

#endif // _STF_RICKER_HPP
