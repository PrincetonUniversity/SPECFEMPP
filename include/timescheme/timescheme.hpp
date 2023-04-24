#ifndef TIMESCHEME_H
#define TIMESCHEME_H

#include "domain.h"
#include "specfem_setup.hpp"
#include <ostream>

namespace specfem {
namespace TimeScheme {
/**
 * @brief Base time scheme class.
 *
 */
class TimeScheme {

public:
  /**
   * @brief Return the status of simulation
   *
   * @return false if current step >= number of steps
   * @return true if current step < number of steps
   */
  virtual bool status() const { return false; };
  /**
   * @brief increment by one timestep, also updates the simulation time by dt
   *
   */
  virtual void increment_time(){};
  /**
   * @brief Get the current simulation time
   *
   * @return type_real current time
   */
  virtual type_real get_time() const { return 0.0; }
  /**
   * @brief Get the current timestep
   *
   * @return int current timestep
   */
  virtual int get_timestep() const { return 0; }
  /**
   * @brief reset current time to t0 and timestep to 0
   *
   */
  virtual void reset_time(){};
  /**
   * @brief Get the max timestep (nstep) of the simuation
   *
   * @return int max timestep
   */
  virtual int get_max_timestep() { return 0; }
  /**
   * @brief Apply predictor phase of the timescheme
   *
   * @param domain_class Pointer to domain class to apply predictor phase
   */
  virtual void
  apply_predictor_phase(const specfem::Domain::Domain *domain_class){};
  /**
   * @brief Apply corrector phase of the timescheme
   *
   * @param domain_class Pointer to domain class to apply corrector phase
   */
  virtual void
  apply_corrector_phase(const specfem::Domain::Domain *domain_class){};

  friend std::ostream &operator<<(std::ostream &out, TimeScheme &ts);
  /**
   * @brief Log timescheme information to console
   */
  virtual void print(std::ostream &out) const;
  /**
   * @brief Compute if seismogram needs to be calculated at this timestep
   *
   */
  virtual bool compute_seismogram() const { return false; }
  /**
   * @brief Get the current seismogram step
   *
   * @return int value of the current seismogram step
   */
  virtual int get_seismogram_step() const { return 0; }
  /**
   * @brief Get the max seismogram step
   *
   * @return int maximum value of seismogram step
   */
  virtual int get_max_seismogram_step() const { return 0; }
  /**
   * @brief increment seismogram step
   *
   */
  virtual void increment_seismogram_step(){};
};

} // namespace TimeScheme
} // namespace specfem
#endif
