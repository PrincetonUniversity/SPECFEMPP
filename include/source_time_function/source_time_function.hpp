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
      specfem::kokkos::HostView1d<type_real> source_time_function) = 0;
};

} // namespace forcing_function
} // namespace specfem

#endif
