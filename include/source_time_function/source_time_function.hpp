#pragma once
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
  stf() {};
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  virtual void update_tshift(type_real tshift) {};
  /**
   * @brief
   *
   */
  virtual type_real get_t0() const { return 0.0; }

  virtual type_real get_tshift() const { return 0.0; }

  virtual std::string print() const = 0;

  virtual bool operator==(const stf &other) const {
    // Base implementation might just check type identity
    return typeid(*this) == typeid(other);
  }
  virtual bool operator!=(const specfem::forcing_function::stf &other) const {
    return !(*this == other);
  }

  // virtual void print(std::ostream &out) const;

  virtual ~stf() = default;

  virtual void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) = 0;
};
} // namespace forcing_function
} // namespace specfem
