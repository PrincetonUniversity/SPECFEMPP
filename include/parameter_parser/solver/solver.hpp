#ifndef _PARAMETER_SOLVER_HPP
#define _PARAMETER_SOLVER_HPP

#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {
namespace solver {
/**
 * @brief Solver class instantiates solver object which defines solution
 * algorithm for the Spectral Element Method
 *
 * @note Currently solver class is not implemented. Hence the solver class only
 * instantiates a timescheme object (In the future this class will only be
 * specific to time-marching (explicit) SEMs).
 *
 */
class solver {

public:
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme object
   * used in the solver algorithm
   */
  virtual specfem::TimeScheme::TimeScheme *
  instantiate(const int nstep_between_samples);
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
  virtual void update_t0(type_real t0){};
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  virtual type_real get_dt() const {
    throw std::runtime_error("Solver not instantiated properly");
    return 0.0;
  };
  virtual type_real get_t0() const {
    throw std::runtime_error("Solver not instantiated properly");
    return 0.0;
  };
};
} // namespace solver
} // namespace runtime_configuration
} // namespace specfem

#endif
