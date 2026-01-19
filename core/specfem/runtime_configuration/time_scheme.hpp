#pragma once

#include "specfem/timescheme/newmark.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {
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
              type_real t0, specfem::simulation::type simulation)
      : timescheme(timescheme), dt(dt), nstep(nstep), t0(t0), type(simulation) {
  }
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
  void update_t0(type_real t0) {
    if (std::abs(this->t0) < 10 * std::numeric_limits<type_real>::epsilon())
      this->t0 = t0;
  }
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme
   object
   * used in the solver algorithm
   */
  template <typename AssemblyFields>
  std::shared_ptr<specfem::time_scheme::time_scheme>
  instantiate(AssemblyFields &fields, const int nstep_between_samples);
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  type_real get_dt() const { return this->dt; }

  type_real get_t0() const { return this->t0; }

  int get_nsteps() const { return this->nstep; }

private:
  int nstep;              ///< number of time steps
  type_real dt;           ///< delta time for the timescheme
  type_real t0 = 0.0;     ///< start time
  std::string timescheme; ///< Time scheme e.g. Newmark, Runge-Kutta, LDDRK
  specfem::simulation::type type; ///< Type of simulation
                                  ///< (forward/adjoint/combined)
};
} // namespace runtime_configuration
} // namespace specfem
