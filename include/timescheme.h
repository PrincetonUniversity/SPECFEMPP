#ifndef TIMESCHEME_H
#define TIMESCHEME_H

#include "../include/config.h"
// #include "../include/domain.h"
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
  // virtual void update_fields(specfem::Domain::Domain *domain_class){};
  /**
   * @brief Get the max timestep (nstep) of the simuation
   *
   * @return int max timestep
   */
  virtual int get_max_timestep() { return 0; }
  // virtual void apply_predictor_phase(specfem::Domain::Domain *domain_class)
  // {}; virtual void apply_corrector_phase(specfem::Domain::Domain
  // *domain_class) {};
  friend std::ostream &operator<<(std::ostream &out, TimeScheme &ts);
  /**
   * @brief Log timescheme information to console
   */
  virtual void print(std::ostream &out) const;
};

/**
 * @brief Newmark timescheme
 *
 */
class Newmark : public TimeScheme {

public:
  /**
   * @brief Construct a new Newmark timescheme object
   *
   * @param nstep maximum number of timesteps in the simulation
   * @param t0 Simulation start time
   * @param dt delta for the newmark timescheme
   */
  Newmark(int nstep, type_real t0, type_real dt);
  /**
   * @brief Return the status of simulation
   *
   * @return false if current step >= number of steps
   * @return true if current step < number of steps
   */
  bool status() const override { return (this->istep < this->nstep); }
  /**
   * @brief increment by one timestep, also updates the simulation time by dt
   *
   */
  void increment_time() override;
  /**
   * @brief Get the current simulation time
   *
   * @return type_real current time
   */
  type_real get_time() const override { return this->current_time; }
  /**
   * @brief Get the current timestep
   *
   * @return int current timestep
   */
  int get_timestep() const override { return this->istep; }
  /**
   * @brief reset current time to t0 and timestep to 0
   *
   */
  void reset_time() override;
  // void update_fields(specfem::Domain::Domain *domain_class){};
  /**
   * @brief Get the max timestep (nstep) of the simuation
   *
   * @return int max timestep
   */
  int get_max_timestep() override { return this->nstep; }
  // void apply_predictor_phase(specfem::Domain::Domain *domain_class) override;
  // void apply_corrector_phase(specfem::Domain::Domain *domain_class) override;
  /**
   * @brief Log timescheme information to console
   */
  void print(std::ostream &out) const override;

private:
  type_real current_time;
  int istep = 0;
  type_real deltat;
  type_real deltatover2;
  type_real deltatsquareover2;
  int nstep;
  type_real t0;
};

std::ostream &operator<<(std::ostream &out,
                         specfem::TimeScheme::TimeScheme &ts);

} // namespace TimeScheme
} // namespace specfem
#endif
