#pragma once
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {

namespace source_time_functions {

/**
 * @brief Source time function base class
 *
 * Abstract base class for all source time functions. Derived classes implement
 * specific time-dependent behaviors for seismic sources.
 *
 * @note The STF function should lie on host and device. Decorate the class
 * functions within using KOKKOS_FUNCTION macro
 */
class stf {
public:
  /**
   * @brief Default constructor
   */
  stf() {};
  /**
   * @brief Update the time shift value
   *
   * @param tshift New time shift value
   */
  virtual void update_tshift(type_real tshift) {};
  /**
   * @brief Get the start time value
   *
   * @return Start time t0
   */
  virtual type_real get_t0() const { return 0.0; }

  /**
   * @brief Get the time shift value
   *
   * @return Time shift value
   */
  virtual type_real get_tshift() const { return 0.0; }

  /**
   * @brief Get string representation of the source time function
   *
   * @return String describing the STF parameters
   */
  virtual std::string print() const = 0;

  /**
   * @brief Equality operator
   *
   * @param other Another source time function to compare with
   * @return true if equal, false otherwise
   */
  virtual bool operator==(const stf &other) const {
    // Base implementation might just check type identity
    return typeid(*this) == typeid(other);
  }

  /**
   * @brief Inequality operator
   *
   * @param other Another source time function to compare with
   * @return true if not equal, false otherwise
   */
  virtual bool operator!=(const stf &other) const { return !(*this == other); }

  /**
   * @brief Virtual destructor
   */
  virtual ~stf() = default;

  /**
   * @brief Compute source time function values for all time steps
   *
   * @param t0 Start time
   * @param dt Time step size
   * @param nsteps Number of time steps
   * @param source_time_function Output view to store computed values
   */
  virtual void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
          source_time_function) = 0;
};
} // namespace source_time_functions
} // namespace specfem
