#ifndef _PARAMETER_TIME_MARCHING_HPP
#define _PARAMETER_TIME_MARCHING_HPP

#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {
namespace time_scheme {
/**
 * @brief time_marching class is used to instantiate a time-marching solver
 *
 */
class time_scheme {

public:
  /**
   * @brief Construct a new time marching object
   *
   * @param timescheme Type of timescheme
   * @param dt delta time of the timescheme
   * @param nstep Number of time steps
   */
  time_scheme(std::string timescheme, type_real dt, type_real nstep,
              specfem::simulation::type simulation)
      : timescheme(timescheme), dt(dt), nstep(nstep), type(simulation) {}
  /**
   * @brief Construct a new time marching object
   *
   * @param Node YAML node describing the time-marching method
   */
  time_scheme(const YAML::Node &Node, specfem::simulation::type simulation);
  /**
   * @brief Update simulation start time.
   *
   * If user has not defined start time then we need to update the simulation
   * start time based on source frequencies and time shift
   *
   * @note This might be specific to only time-marching solvers
   *
   * @param t0 Simulation start time
   */
  void update_t0(type_real t0) { this->t0 = t0; }
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme
   object
   * used in the solver algorithm
   */
  std::shared_ptr<specfem::time_scheme::time_scheme>
  instantiate(const int nstep_between_samples);
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  type_real get_dt() const { return this->dt; }

  type_real get_t0() const { return this->t0; }

private:
  int nstep;              ///< number of time steps
  type_real dt;           ///< delta time for the timescheme
  type_real t0;           ///< simulation start time
  std::string timescheme; ///< Time scheme e.g. Newmark, Runge-Kutta, LDDRK
  specfem::simulation::type type;
};
} // namespace time_scheme
} // namespace runtime_configuration
} // namespace specfem

#endif
