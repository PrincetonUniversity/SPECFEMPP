#ifndef _SOURCE_TIME_FUNCTION_HPP
#define _SOURCE_TIME_FUNCTION_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
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
  stf(){};
  /**
   * @brief compute the value of stf at time t
   *
   * @param t
   * @return value of source time function at time t
   */
  virtual type_real compute(type_real t) { return 0.0; }
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  virtual void update_tshift(type_real tshift){};
  /**
   * @brief
   *
   */
  virtual type_real get_t0() const { return 0.0; }

  virtual std::string print() const = 0;

  // virtual void print(std::ostream &out) const;

  virtual ~stf() = default;

  virtual void compute_source_time_function(
      const int nsteps,
      specfem::kokkos::HostView1d<type_real> source_time_function) = 0;
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

  KOKKOS_FUNCTION type_real compute(const type_real t) { return T->compute(t); }
};

} // namespace forcing_function
} // namespace specfem

#endif
