#ifndef SOURCE_TIME_FUNCTION_H
#define SOURCE_TIME_FUNCTION_H

#include "../include/specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {

/**
 * @note The STF function should lie on host and device. Decorate the class
 * functions within using KOKKOS_FUNCTION macro
 *
 */

/**
 * @brief Source time function base class
 *
 */
class stf {
public:
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION stf(){};
  /**
   * @brief compute the value of stf at time t
   *
   * @param t
   * @return value of source time function at time t
   */
  KOKKOS_FUNCTION virtual type_real compute(type_real t) { return 0.0; }
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  KOKKOS_FUNCTION virtual void update_tshift(type_real tshift){};
  /**
   * @brief Get the t0 value
   *
   * @return t0 value
   */
  KOKKOS_FUNCTION virtual type_real get_t0() const { return 0.0; }

  // virtual void print(std::ostream &out) const;
};

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
  KOKKOS_FUNCTION Dirac(type_real f0, type_real tshift, type_real factor,
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

std::ostream &operator<<(std::ostream &out,
                         const specfem::forcing_function::stf &stf);

/**
 * @brief struct used to store source time function pointer on device in a
 * device view
 *
 * There is a syntaxual bug in Kokkos which requires the need of this struct for
 * now.
 *
 */
struct stf_storage {
  specfem::forcing_function::stf *T;
};

} // namespace forcing_function
} // namespace specfem

#endif
